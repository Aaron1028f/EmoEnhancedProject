import os, sys
sys.path.append('./')
import argparse
from emogene.realtime.emogene_stream import GeneFace2Infer
from utils.commons.hparams import hparams
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import warnings
import torchvision
import tempfile
import shutil

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, Union

import numpy as np
from dotenv import load_dotenv
from livekit import rtc, api
from livekit.agents import JobContext, WorkerOptions, cli

try:
    import av
except ImportError:
    raise RuntimeError("av is required; pip install av")


torchvision.disable_beta_transforms_warning()
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

from dotenv import load_dotenv
load_dotenv(".env.local")
logger = logging.getLogger("video_streamer_agent")

# =================================================================================================
# Fast API
# =================================================================================================
ROOM_NAME = 'playground-hXAr-rAmg'

class GenerateRequest(BaseModel):
    audio_path: str
    room_name: str = ROOM_NAME
    publish_audio: bool = True

# prepare global variables
args = None 
inferer_instance = None

# 佇列與背景工作者
publish_queue: asyncio.Queue | None = None
publisher_task: asyncio.Task | None = None
# 新增：序列化推論的鎖
inference_lock: asyncio.Lock | None = None

@dataclass
class PublishJob:
    room_name: str
    video_path: str
    publish_audio: bool
    done: asyncio.Future  # 若不需等待結果，可以不使用

MODEL_INPUT_MAY = {
    # input output setting
    'out_name': "emogene/DATA/lk_temp.mp4",
    'drv_audio_name': "emogene/DATA/happy.wav",
    
    # model path params
    'audio2secc': 'checkpoints/audio2motion_vae',
    'postnet_dir': '',
    'head_model_dir': '',
    'torso_model_dir': 'checkpoints/motion2video_nerf/may_torso', 
    'use_emotalk': True,
    'device': 'cuda:0',
    
    # emogene settings
    'blend_path': "emotalk/render_testing_92.blend",
    'lm468_bs_np_path': "emotalk/temp_result/lm468_bs_np.npy",
    'bs_lm_area': 8,
    'debug': False,
    'use_emotalk': True,
    'level': 1,
    'person': 3,
    'output_video': False,
    'bs52_level': 2.0,
    
    # GeneFace++ seettings
    'blink_mode': 'none',
    'drv_pose': 'nearest',
    'lle_percent': 1,
    'temperature': 0,
    'mouth_amp': 0.4,
    'raymarching_end_threshold': 0.01,
    'fp16': False,
    'low_memory_usage': False
}


# FastAPI application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # START
    global args, inferer_instance, publish_queue, publisher_task, inference_lock
    
    print("Initializing model...")
    inferer_instance = GeneFace2Infer(
        audio2secc_dir=MODEL_INPUT_MAY['audio2secc'],
        postnet_dir=MODEL_INPUT_MAY['postnet_dir'],
        head_model_dir=MODEL_INPUT_MAY['head_model_dir'],
        torso_model_dir=MODEL_INPUT_MAY['torso_model_dir'],
        use_emotalk=MODEL_INPUT_MAY['use_emotalk'],
        device=MODEL_INPUT_MAY['device']
    )
    publish_queue = asyncio.Queue()
    inference_lock = asyncio.Lock()

    async def publisher_worker():
        logger.info("publisher worker started")
        try:
            while True:
                job: PublishJob = await publish_queue.get()
                try:
                    await publish_video_to_room(job.room_name, job.video_path, publish_audio=job.publish_audio)
                    if not job.done.done():
                        job.done.set_result({"video_path": job.video_path, "error": None, "published": True})
                except Exception as e:
                    if not job.done.done():
                        job.done.set_result({"video_path": job.video_path, "error": f"publish failed: {e}", "published": False})
                finally:
                    publish_queue.task_done()
        except asyncio.CancelledError:
            logger.info("publisher worker cancelled")
            raise

    publisher_task = asyncio.create_task(publisher_worker())

    # 如需啟動時測試播一支影片，也請透過佇列排隊，而不是直接呼叫
    test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/DATA/lk_temp.mp4'
    test_future = asyncio.get_running_loop().create_future()
    await publish_queue.put(PublishJob(ROOM_NAME, test_video_path, True, test_future))


    print("Model loaded.")

    # prewarm: run once with
    print('Start run once to prewarm the model')
    inp = MODEL_INPUT_MAY.copy()
    inp['drv_audio_name'] = "emogene/DATA/happy.wav"
    inferer_instance.infer_once(inp)
    print('Prewarming complete.')

    print("Application startup complete.")
    yield # App running
    
    # END
    print("Application shutting down. Cleaning up resources...")
    if publisher_task:
        publisher_task.cancel()
        try:
            await publisher_task
        except asyncio.CancelledError:
            pass
    inferer_instance = None


app = FastAPI(lifespan=lifespan)

@app.post("/generate_full_video")
async def generate_full_video_api(request: GenerateRequest):
    """
    Generate a full video from the given audio file.
    """
    if not os.path.exists(request.audio_path):
        return {"error": f"Audio file not found: {request.audio_path}", "video_path": None, "accepted": False}
    # return {"error": None, "video_path": None, "accepted": True}
    
    
    # just testing
    # video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/DATA/lk_temp.mp4'
    # await publish_video_to_room(request.room_name, video_path, publish_audio=request.publish_audio)
    
    
    # set the input audio path and output video path
    inp = MODEL_INPUT_MAY.copy()
    inp['drv_audio_name'] = request.audio_path
    inp['out_name'] = f"emogene/DATA/temp/{request.audio_path.split('/')[-1].split('.')[0]}_out.mp4"
    
    
    # check if the path exist
    if not os.path.exists(inp['out_name']):
        os.makedirs(os.path.dirname(inp['out_name']), exist_ok=True)
    
    try:
        if inference_lock is None:
            raise RuntimeError("inference lock not initialized")
        async with inference_lock:
            video_path = await asyncio.to_thread(inferer_instance.infer_once, inp)
    except Exception as e:
        return {"video_path": None, "error": f"inference failed: {e}", "published": False}

    try:
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        if publish_queue is None:
            return {"video_path": video_path, "error": "publish queue not initialized", "published": False}
        await publish_queue.put(PublishJob(request.room_name, video_path, request.publish_audio, done_future))
        # 立即回應，表示已排隊等待發佈；如需等發佈完成，可 await done_future
        return {"video_path": video_path, "error": None, "queued": True, "published": False}
    except Exception as e:
        return {"video_path": video_path, "error": f"enqueue failed: {e}", "published": False}

    # try:
    #     await publish_video_to_room(request.room_name, video_path, publish_audio=request.publish_audio)
    #     return {"video_path": video_path, "error": None, "published": True}
    # except Exception as e:
    #     return {"video_path": video_path, "error": f"publish failed: {e}", "published": False}

    
# @app.post("/generate_streaming_video")
# async def generate_streaming_video(request: GenerateRequest):
#     """
#     Generate a streaming video from the given audio file.
#     """
#     return

#     if not os.path.exists(request.audio_path):
#         return {"error": f"Audio file not found: {request.audio_path}", "video_path": None}

#     # set the input audio path and output video path
#     inp = MODEL_INPUT_MAY.copy()
#     inp['drv_audio_name'] = request.audio_path

#     try:
#         print("Starting inference for API request...")
#         video_path = inferer_instance.infer_once(inp)
#         print(f"API inference successful. Video at: {video_path}")
#         return {"video_path": video_path, "error": None}
#     except Exception as e:
#         print(f"API inference failed: {e}")
#         return {"video_path": None, "error": str(e)}

# =================================================================================================
# Livekit agent
# =================================================================================================

@dataclass
class MediaInfo:
    video_width: int
    video_height: int
    video_fps: float
    audio_sample_rate: int
    audio_channels: int

class MediaFileStreamer:
    """Streams video and audio frames from a media file in an endless loop."""

    def __init__(self, media_file: Union[str, Path]) -> None:
        self._media_file = str(media_file)
        # Create separate containers for each stream
        self._video_container = av.open(self._media_file)
        self._audio_container = av.open(self._media_file)

        # Cache media info
        video_stream = self._video_container.streams.video[0]
        audio_stream = self._audio_container.streams.audio[0]
        self._info = MediaInfo(
            video_width=video_stream.width,
            video_height=video_stream.height,
            video_fps=float(video_stream.average_rate),  # type: ignore
            audio_sample_rate=audio_stream.sample_rate,
            audio_channels=audio_stream.channels,
        )

    @property
    def info(self) -> MediaInfo:
        return self._info

    async def stream_video(self) -> AsyncIterable[tuple[rtc.VideoFrame, float]]:
        """Streams video frames from the media file in an endless loop."""
        for i, av_frame in enumerate(self._video_container.decode(video=0)):
            # Convert video frame to RGBA
            frame = av_frame.to_rgb().to_ndarray()
            frame_rgba = np.ones((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            frame_rgba[:, :, :3] = frame
            yield (
                rtc.VideoFrame(
                    width=frame.shape[1],
                    height=frame.shape[0],
                    type=rtc.VideoBufferType.RGBA,
                    data=frame_rgba.tobytes(),
                ),
                av_frame.time,
            )

    async def stream_audio(self) -> AsyncIterable[tuple[rtc.AudioFrame, float]]:
        """Streams audio frames from the media file in an endless loop."""
        for av_frame in self._audio_container.decode(audio=0):
            # Convert audio frame to raw int16 samples
            frame = av_frame.to_ndarray().T  # Transpose to (samples, channels)
            frame = (frame * 32768).astype(np.int16)
            duration = len(frame) / self.info.audio_sample_rate
            yield (
                rtc.AudioFrame(
                    data=frame.tobytes(),
                    sample_rate=self.info.audio_sample_rate,
                    num_channels=frame.shape[1],
                    samples_per_channel=frame.shape[0],
                ),
                av_frame.time + duration,
            )

    def reset(self):
        self._video_container.seek(0)
        self._audio_container.seek(0)

    async def aclose(self) -> None:
        """Closes the media container and stops streaming."""
        self._video_container.close()
        self._audio_container.close()

async def publish_video_to_room(room_name: str, video_path: str, publish_audio: bool = False):
    token = (
        api.AccessToken()
        .with_identity(f"emogene-publisher-{os.getpid()}")
        .with_name("Emogene Publisher")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                agent=True
            )
        )
        .to_jwt()
    )
    url = os.getenv("LIVEKIT_URL")
    loop = asyncio.get_event_loop()
    room = rtc.Room(loop=loop)
    try:
        await room.connect(url, token)
        print('connected to room %s', room.name)
        logging.info("connected to room %s", room.name)
    except rtc.ConnectError as e:
        print('failed to connect to the room: %s', e)
        logging.error("failed to connect to the room: %s", e)
        return

    # Create media streamer
    streamer = MediaFileStreamer(video_path)
    media_info = streamer.info

    # Create video and audio sources/tracks
    queue_size_ms = 1000
    video_source = rtc.VideoSource(
        width=media_info.video_width,
        height=media_info.video_height,
    )
    logger.info(media_info)
    audio_source = rtc.AudioSource(
        sample_rate=media_info.audio_sample_rate,
        num_channels=media_info.audio_channels,
        queue_size_ms=queue_size_ms,
    )

    video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)
    audio_track = rtc.LocalAudioTrack.create_audio_track("audio", audio_source)

    # Publish tracks
    video_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        video_encoding=rtc.VideoEncoding(
            max_framerate=30,
            max_bitrate=5_000_000,
        ),
    )
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)

    await room.local_participant.publish_track(video_track, video_options)
    await room.local_participant.publish_track(audio_track, audio_options)

    av_sync = rtc.AVSynchronizer(
        audio_source=audio_source,
        video_source=video_source,
        video_fps=media_info.video_fps,
        video_queue_size_ms=queue_size_ms,
    )

    async def _push_frames(
        stream: AsyncIterable[tuple[rtc.VideoFrame | rtc.AudioFrame, float]],
        av_sync: rtc.AVSynchronizer,
    ):
        async for frame, timestamp in stream:
            await av_sync.push(frame, timestamp)
            await asyncio.sleep(0)

    async def _log_fps(av_sync: rtc.AVSynchronizer):
        start_time = asyncio.get_running_loop().time()
        while True:
            await asyncio.sleep(2)
            wall_time = asyncio.get_running_loop().time() - start_time
            diff = av_sync.last_video_time - av_sync.last_audio_time
            logger.info(
                f"fps: {av_sync.actual_fps:.2f}, wall_time: {wall_time:.3f}s, "
                f"video_time: {av_sync.last_video_time:.3f}s, "
                f"audio_time: {av_sync.last_audio_time:.3f}s, diff: {diff:.3f}s"
            )

    try:
        # while True:
        streamer.reset()

        video_stream = streamer.stream_video()
        audio_stream = streamer.stream_audio()

        # read the head frames and push them at the same time
        first_video_frame, video_timestamp = await video_stream.__anext__()
        first_audio_frame, audio_timestamp = await audio_stream.__anext__()
        logger.info(
            f"first video duration: {1 / media_info.video_fps:.3f}s, "
            f"first audio duration: {first_audio_frame.duration:.3f}s"
        )
        await av_sync.push(first_video_frame, video_timestamp)
        await av_sync.push(first_audio_frame, audio_timestamp)

        video_task = asyncio.create_task(_push_frames(video_stream, av_sync))
        audio_task = asyncio.create_task(_push_frames(audio_stream, av_sync))

        log_fps_task = asyncio.create_task(_log_fps(av_sync))

        # wait for both tasks to complete
        await asyncio.gather(video_task, audio_task)
        await av_sync.wait_for_playout()

        # clean up
        av_sync.reset()
        log_fps_task.cancel()
        logger.info("playout finished")
    finally:
        await streamer.aclose()
        await av_sync.aclose()
        await audio_source.aclose()
        await video_source.aclose()
        try:
            await room.disconnect()  # 重要：發布完就斷線，釋放 identity
        except Exception as e:
            logger.warning("room disconnect error: %s", e)        

def main():
    uvicorn.run(app, host="0.0.0.0", port=31000)

if __name__ == "__main__":
    main()


# async def main():
#     config = uvicorn.Config(app, host="0.0.0.0", port=31000)
#     server = uvicorn.Server(config)
    
#     # await server.serve()
#     await asyncio.gather(
#         server.serve(),
#         cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
#     )

# if __name__ == "__main__":
#     asyncio.run(main())