# file: agent_with_video_loop.py
import asyncio
import logging
import os
from dotenv import load_dotenv
import asyncio.subprocess as asp

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    RoomInputOptions,
)
from livekit.agents.llm import function_tool
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import noise_cancellation, openai, silero  # 如果需要

from typing import AsyncIterable, Union

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, Union
import sys

import numpy as np
import os
import signal
from livekit import api
from livekit import rtc

try:
    import av
except ImportError:
    raise RuntimeError("av is required to run this example, install with `pip install av`")


# local services
from localLLM import LocalLLM
from localTTS_indextts import LocalTTS

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                # "You are a helpful voice AI assistant. "
                # "Keep responses concise and friendly."
            ),
        )

    # @function_tool
    # async def lookup_weather(self, context, location: str):
    #     logger.info("lookup_weather called for %s", location)
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    # 載入 VAD 等資源供 session 使用
    proc.userdata["vad"] = silero.VAD.load()


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

async def _publish_loop_video(room: rtc.Room, video_path: str):
    # Create media streamer
    media_path = video_path
    streamer = MediaFileStreamer(media_path)
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
        while True:
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

async def entrypoint(ctx: JobContext):
    # 先連上 room（必須的最新用法）
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("connecting to room %s", ctx.room.name)
    # await ctx.connect()  # <-- 正確：讓 framework 處理連線與 token
    logger.info("connected to room %s", ctx.room.name)

    # 建立 agent session（LLM/STT/TTS ...）
    session = AgentSession(
        llm=LocalLLM(),
        stt=openai.STT(model="gpt-4o-transcribe"),
        tts=LocalTTS(),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        preemptive_generation=True,
    )

    # 監聽 false interruptions（示意）
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev):
        logger.info("false interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions)

    # 啟動 agentSession (把 ctx.room 傳進去)
    await session.start(
        agent=Assistant(), 
        room=ctx.room, 
        room_input_options=RoomInputOptions(
                    
        )
    )

    # 啟動單獨的 video loop publisher（永遠循環播放影片）
    VIDEO_PATH = os.getenv("LOOP_VIDEO_PATH", "/home/aaron/project/server/models/GeneFacePlusPlus/tmp.mp4")
    # video_task = asyncio.create_task(_publish_loop_video(ctx.room, VIDEO_PATH))

    # # 等待直到 job 結束（或被框架關閉）
    # try:
    #     await ctx.wait_for_shutdown()
    # finally:
    #     # 清理
    #     video_task.cancel()
    #     try:
    #         await video_task
    #     except Exception:
    #         pass
    #     await session.stop()
    #     logger.info("entrypoint finished.")


if __name__ == "__main__":
    # WorkerOptions 仍然用 framework 的方式啟動
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
