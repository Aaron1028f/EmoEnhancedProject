ROOM_NAME = 'playground-NvXe-6aCq'
NUM_INFERER = 2

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

# 推論並行度（可用環境變數調整）
INFER_CONCURRENCY = int(os.getenv("EMOGENE_INFER_CONCURRENCY", NUM_INFERER))

persistent_room: rtc.Room | None = None
video_source: rtc.VideoSource | None = None
audio_source: rtc.AudioSource | None = None
video_track: rtc.LocalVideoTrack | None = None
audio_track: rtc.LocalAudioTrack | None = None
idle_task: asyncio.Task | None = None
placeholder_rgba: np.ndarray | None = None  # (H,W,4)

# 新增：記錄 persistent source 尺寸，與是否啟用「最後一幀當 idle」
persistent_width: int | None = None
persistent_height: int | None = None
KEEP_LAST_FRAME_ON_IDLE = True

WHITE = (255, 255, 255)
GREY = (200, 200, 200)

def _make_placeholder_rgba(width: int, height: int, color=GREY) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :, 0:3] = np.array(color, dtype=np.uint8)  # RGB
    img[:, :, 3] = 255  # A
    return img

async def ensure_persistent_tracks(room_name: str, width: int = 512, height: int = 512, sr: int = 16000, ch: int = 1):
    global persistent_room, video_source, audio_source, video_track, audio_track, placeholder_rgba
    global persistent_width, persistent_height
    if placeholder_rgba is None:
        placeholder_rgba = _make_placeholder_rgba(width, height)

    url = os.getenv("LIVEKIT_URL")
    token = (
        api.AccessToken()
        .with_identity(f"emogene-persistent-{os.getpid()}")
        .with_name("Emogene Persistent")
        .with_grants(api.VideoGrants(room_join=True, room=room_name, agent=True))
        .to_jwt()
    )
    loop = asyncio.get_running_loop()

    # 確保房間連線
    if persistent_room is None:
        persistent_room = rtc.Room(loop=loop)
    if persistent_room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
        await persistent_room.connect(url, token)
        logger.info("persistent connected to room %s", persistent_room.name)

    # 確保來源/軌跡存在（只建立一次）
    if video_source is None or audio_source is None:
        video_source = rtc.VideoSource(width=width, height=height)
        audio_source = rtc.AudioSource(sample_rate=sr, num_channels=ch, queue_size_ms=1000)
        # 記錄 persistent 尺寸
        persistent_width, persistent_height = width, height

        video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)
        audio_track = rtc.LocalAudioTrack.create_audio_track("audio", audio_source)

        await persistent_room.local_participant.publish_track(
            video_track,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                video_encoding=rtc.VideoEncoding(max_framerate=30, max_bitrate=5_000_000),
            ),
        )
        await persistent_room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        logger.info("persistent tracks published")

async def _idle_loop(fps: float = 10.0):
    """持續送出等待圖，直到被取消。"""
    assert video_source is not None
    assert audio_source is not None
    assert placeholder_rgba is not None
    av_sync = rtc.AVSynchronizer(
        audio_source=audio_source,
        video_source=video_source,
        video_fps=fps,
        video_queue_size_ms=1000,
    )
    t = 0.0
    dt = 1.0 / fps
    try:
        while True:
            vf = rtc.VideoFrame(
                width=placeholder_rgba.shape[1],
                height=placeholder_rgba.shape[0],
                type=rtc.VideoBufferType.RGBA,
                data=placeholder_rgba.tobytes(),
            )
            await av_sync.push(vf, t)
            t += dt
            await asyncio.sleep(dt)
    except asyncio.CancelledError:
        pass
    finally:
        await av_sync.aclose()

async def start_idle():
    """啟動 idle（若未啟動）。"""
    global idle_task
    if idle_task is None or idle_task.done():
        idle_task = asyncio.create_task(_idle_loop())

async def stop_idle():
    """停止 idle（若在跑）。"""
    global idle_task
    if idle_task and not idle_task.done():
        idle_task.cancel()
        try:
            await idle_task
        except asyncio.CancelledError:
            pass
    idle_task = None

async def play_file_persistent(video_path: str, publish_audio: bool = True):
    """在持久 tracks 上播放指定影片，播完後恢復 idle。"""
    await stop_idle()
    assert video_source is not None and audio_source is not None

    streamer = MediaFileStreamer(video_path)
    media_info = streamer.info
    av_sync = rtc.AVSynchronizer(
        audio_source=audio_source,
        video_source=video_source,
        video_fps=media_info.video_fps,
        video_queue_size_ms=1000,
    )

    async def _push_frames(stream, with_audio: bool):
        async for frame, ts in stream:
            # 視訊一定送；音訊依旗標
            if isinstance(frame, rtc.VideoFrame) or with_audio:
                await av_sync.push(frame, ts)
            await asyncio.sleep(0)

    try:
        streamer.reset()

        v_stream = streamer.stream_video()
        a_stream = streamer.stream_audio()

        # 先各取第一幀，對齊時間再送
        first_v, v_ts = await v_stream.__anext__()
        first_a, a_ts = await a_stream.__anext__()
        await av_sync.push(first_v, v_ts)
        if publish_audio:
            await av_sync.push(first_a, a_ts)

        v_task = asyncio.create_task(_push_frames(v_stream, with_audio=False))
        a_task = asyncio.create_task(_push_frames(a_stream, with_audio=publish_audio))

        await asyncio.gather(v_task, a_task)
        await av_sync.wait_for_playout()
        av_sync.reset()
        logger.info("file playback finished")

        # 新增：影片播放完後，若啟用，將最後一幀用作 idle 的 placeholder
        if KEEP_LAST_FRAME_ON_IDLE and streamer.last_rgba is not None:
            try:
                lw, lh = streamer.last_rgba.shape[1], streamer.last_rgba.shape[0]
                if persistent_width is not None and persistent_height is not None and \
                   lw == persistent_width and lh == persistent_height:
                    # 尺寸相符才替換
                    # 注意：需確保為 (H,W,4) uint8 RGBA
                    if streamer.last_rgba.dtype == np.uint8 and streamer.last_rgba.shape[2] == 4:
                        # copy 避免後續被覆寫
                        # 使用 global placeholder_rgba
                        global placeholder_rgba
                        placeholder_rgba = streamer.last_rgba.copy()
                        logger.info("idle placeholder updated to last frame")
                else:
                    logger.info("last frame size %sx%s != persistent %sx%s, keep old placeholder",
                                lw, lh, persistent_width, persistent_height)
            except Exception:
                logger.exception("update placeholder to last frame failed")

    finally:
        await streamer.aclose()
        await av_sync.aclose()
        # 播完恢復 idle
        await start_idle()
# === 新增結束 ===




# =================================================================================================
# Fast API
# =================================================================================================


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
# 新增：序列化推論的鎖（並行版本中不再使用，但保留以相容）
inference_lock: asyncio.Lock | None = None
# 新增：推論佇列與 workers
inference_queue: asyncio.Queue | None = None
# 變更：改為多個推論 task
inference_tasks: list[asyncio.Task] | None = None
# 新增：推論完成佇列與排序 worker
inference_done_queue: asyncio.Queue | None = None
orderer_task: asyncio.Task | None = None
# 新增：序號與多實例
seq_counter: int = 0
inferers: list | None = None

@dataclass
class PublishJob:
    room_name: str
    video_path: str
    publish_audio: bool
    done: asyncio.Future  # 若不需等待結果，可以不使用

# 新增：推論工作（帶有序號）
@dataclass
class InferenceJob:
    seq_id: int
    audio_path: str
    room_name: str
    publish_audio: bool
    done: asyncio.Future | None = None

# 新增：推論完成事件（給排序 worker）
@dataclass
class InferenceDone:
    seq_id: int
    room_name: str
    video_path: str
    publish_audio: bool
    
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

MODEL_INPUT_FENG = {
    # input output setting
    'out_name': "emogene/DATA/lk_temp.mp4",
    'drv_audio_name': "emogene/DATA/happy.wav",
    
    # model path params
    'audio2secc': 'checkpoints/audio2motion_vae',
    'postnet_dir': '',
    'head_model_dir': '',
    'torso_model_dir': 'checkpoints/motion2video_nerf/feng_torso', 
    'use_emotalk': True,
    'device': 'cuda:0',
    
    # emogene settings
    'blend_path': "emotalk/feng_rigged.blend",
    'lm468_bs_np_path': "emotalk/temp_result/lm468_bs_np.npy",
    'bs_lm_area': 9,
    'debug': False,
    'use_emotalk': True,
    'level': 1,
    'person': 3,
    'output_video': False,
    'bs52_level': 1.0,
    
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

MODEL_INPUT = MODEL_INPUT_FENG  # 可改成 MODEL_INPUT_FENG 試試

# FastAPI application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # START
    global args, inferer_instance, publish_queue, publisher_task
    global inference_lock, inference_queue, inference_tasks, inference_done_queue, orderer_task
    global seq_counter, inferers   
     
    print("Initializing model...")
    # 建立第一個推論實例（沿用既有變數以相容）
    inferer_instance = GeneFace2Infer(
        audio2secc_dir=MODEL_INPUT['audio2secc'],
        postnet_dir=MODEL_INPUT['postnet_dir'],
        head_model_dir=MODEL_INPUT['head_model_dir'],
        torso_model_dir=MODEL_INPUT['torso_model_dir'],
        use_emotalk=MODEL_INPUT['use_emotalk'],
        device=MODEL_INPUT['device']
    )

    # 其餘推論實例
    inferers = [inferer_instance]
    for i in range(1, INFER_CONCURRENCY):
        inferers.append(GeneFace2Infer(
            audio2secc_dir=MODEL_INPUT['audio2secc'],
            postnet_dir=MODEL_INPUT['postnet_dir'],
            head_model_dir=MODEL_INPUT['head_model_dir'],
            torso_model_dir=MODEL_INPUT['torso_model_dir'],
            use_emotalk=MODEL_INPUT['use_emotalk'],
            device=MODEL_INPUT['device']
        ))
        
    publish_queue = asyncio.Queue()
    inference_queue = asyncio.Queue()
    inference_done_queue = asyncio.Queue()
    inference_lock = asyncio.Lock()  # 並行版本不使用，保留以相容

    async def publisher_worker():
        logger.info("publisher worker started")
        try:
            await ensure_persistent_tracks(ROOM_NAME)
            await start_idle()
            while True:
                job: PublishJob = await publish_queue.get()
                try:
                    await ensure_persistent_tracks(job.room_name)
                    await play_file_persistent(job.video_path, publish_audio=job.publish_audio)
                    if not job.done.done():
                        job.done.set_result({"video_path": job.video_path, "error": None, "published": True})
                except Exception as e:
                    logger.exception("persistent play failed")
                    if not job.done.done():
                        job.done.set_result({"video_path": job.video_path, "error": f"publish failed: {e}", "published": False})
                finally:
                    publish_queue.task_done()
        except asyncio.CancelledError:
            logger.info("publisher worker cancelled")
            raise

    # 變更：多個推論 workers（每個 worker 擁有自己的 inferer）
    def make_inference_worker(worker_id: int, worker_inferer: GeneFace2Infer):
        async def _worker():
            logger.info(f"inference worker {worker_id} started")
            try:
                while True:
                    job: InferenceJob = await inference_queue.get()
                    try:
                        inp = MODEL_INPUT_MAY.copy()
                        inp['drv_audio_name'] = job.audio_path
                        out_dir = "emogene/DATA/temp"
                        os.makedirs(out_dir, exist_ok=True)
                        inp['out_name'] = f"{out_dir}/{Path(job.audio_path).stem}_out.mp4"

                        # 並行推論（各自實例，無需鎖）
                        video_path = await asyncio.to_thread(worker_inferer.infer_once, inp)

                        # 推論完成，送到排序佇列
                        if inference_done_queue is None:
                            raise RuntimeError("inference_done_queue not initialized")
                        await inference_done_queue.put(InferenceDone(
                            seq_id=job.seq_id,
                            room_name=job.room_name,
                            video_path=video_path,
                            publish_audio=job.publish_audio
                        ))

                        if job.done and not job.done.done():
                            job.done.set_result({"video_path": video_path, "queued_publish": True})
                    except Exception as e:
                        logger.exception("inference failed")
                        if job.done and not job.done.done():
                            job.done.set_result({"video_path": None, "error": f"inference failed: {e}"})
                    finally:
                        inference_queue.task_done()
            except asyncio.CancelledError:
                logger.info(f"inference worker {worker_id} cancelled")
                raise
        return _worker

    # 新增：排序 worker，確保發布按序號先後
    async def orderer_worker():
        logger.info("orderer worker started")
        next_seq = 0
        buffer: dict[int, InferenceDone] = {}
        try:
            while True:
                item: InferenceDone = await inference_done_queue.get()
                try:
                    buffer[item.seq_id] = item
                    # 依序把可發布的項目放入 publish_queue
                    while next_seq in buffer:
                        ready = buffer.pop(next_seq)
                        if publish_queue is None:
                            raise RuntimeError("publish queue not initialized")
                        done_future = asyncio.get_running_loop().create_future()
                        await publish_queue.put(PublishJob(
                            room_name=ready.room_name,
                            video_path=ready.video_path,
                            publish_audio=ready.publish_audio,
                            done=done_future
                        ))
                        next_seq += 1
                finally:
                    inference_done_queue.task_done()
        except asyncio.CancelledError:
            logger.info("orderer worker cancelled")
            raise

    publisher_task = asyncio.create_task(publisher_worker())

    # 啟動多個推論 workers
    inference_tasks = []
    for wid, inf in enumerate(inferers):
        inference_tasks.append(asyncio.create_task(make_inference_worker(wid, inf)()))

    # 啟動排序 worker
    orderer_task = asyncio.create_task(orderer_worker())
    
    # 如需啟動時測試播一支影片，也請透過佇列排隊（可保留或移除）
    test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/DATA/lk_temp.mp4'
    test_future = asyncio.get_running_loop().create_future()
    await publish_queue.put(PublishJob(ROOM_NAME, test_video_path, True, test_future))

    print("Model loaded.")

    # 預熱：仍只用第一個實例
    print('Start run once to prewarm the model')
    inp = MODEL_INPUT_MAY.copy()
    inp['drv_audio_name'] = "emogene/DATA/happy.wav"
    inferer_instance.infer_once(inp)
    print('Prewarming complete.')

    print("Application startup complete.")
    yield  # App running
    
    # END
    print("Application shutting down. Cleaning up resources...")
    if publisher_task:
        publisher_task.cancel()
        try:
            await publisher_task
        except asyncio.CancelledError:
            pass

    if orderer_task:
        orderer_task.cancel()
        try:
            await orderer_task
        except asyncio.CancelledError:
            pass

    if inference_tasks:
        for t in inference_tasks:
            t.cancel()
        for t in inference_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    await stop_idle()
    inferer_instance = None

app = FastAPI(lifespan=lifespan)

@app.post("/generate_full_video")
async def generate_full_video_api(request: GenerateRequest):
    """
    Generate a full video from the given audio file.
    """
    global seq_counter
    if not os.path.exists(request.audio_path):
        return {"error": f"Audio file not found: {request.audio_path}", "video_path": None, "accepted": False}

    try:
        if inference_queue is None:
            return {"error": "inference queue not initialized", "video_path": None, "accepted": False}
        loop = asyncio.get_running_loop()

        # 指派序號，確保發布順序
        seq_id = seq_counter
        seq_counter += 1

        await inference_queue.put(InferenceJob(
            seq_id=seq_id,
            audio_path=request.audio_path,
            room_name=request.room_name,
            publish_audio=request.publish_audio,
            done=None
        ))
        return {"error": None, "video_path": None, "accepted": True, "queued": True, "seq_id": seq_id}
    except Exception as e:
        return {"error": f"enqueue failed: {e}", "video_path": None, "accepted": False}

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
        # 新增：記錄最後一幀 RGBA
        self.last_rgba: np.ndarray | None = None

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
            # 更新最後一幀
            self.last_rgba = frame_rgba
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