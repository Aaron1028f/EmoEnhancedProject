# // ...existing code...
import asyncio
from lk_webhook_server import serve as webhook_serve
# GeneFace++
from server.models.GeneFacePlusPlus.emogene.realtime.emogene_stream import GeneFace2Infer
import numpy as np
import cv2
# // ...existing code...

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

    # Avatar 任務佇列 + 啟動本地 webhook
    proc.userdata["avatar_queue"] = asyncio.Queue()

    async def _start_webhook():
        await webhook_serve(proc.userdata["avatar_queue"], host="127.0.0.1", port=9901)

    proc.userdata["webhook_task"] = asyncio.create_task(_start_webhook())

    # 可選：在這裡預載 GeneFace++ 模型，避免首發延遲（路徑請按你環境調整）
    # proc.userdata["geneface"] = GeneFace2Infer(
    #     audio2secc_dir="checkpoints/audio2motion_vae",
    #     postnet_dir="",
    #     head_model_dir="",
    #     torso_model_dir="checkpoints/motion2video_nerf/may_torso",
    #     use_emotalk=True,
    # )

async def entrypoint(ctx: JobContext):
    # // ...existing code for AgentSession...

    # 啟動 Avatar 佇列消費者
    avatar_queue: asyncio.Queue = ctx.proc.userdata["avatar_queue"]

    async def avatar_worker():
        # 如未在 prewarm 預載，這裡初始化
        geneface = ctx.proc.userdata.get("geneface")
        if geneface is None:
            geneface = GeneFace2Infer(
                audio2secc_dir="checkpoints/audio2motion_vae",
                postnet_dir="",
                head_model_dir="",
                torso_model_dir="checkpoints/motion2video_nerf/may_torso",
                use_emotalk=True,
            )

        while True:
            job = await avatar_queue.get()
            wav_path = job["wav_path"]
            try:
                await _run_avatar_stream(ctx, geneface, wav_path)
            except Exception as e:
                logger.exception(f"avatar generation failed: {e}")

    asyncio.create_task(avatar_worker())

    // ...existing code to start session and connect...

async def _run_avatar_stream(ctx: JobContext, geneface: GeneFace2Infer, wav_path: str):
    """
    以保存的 wav 驅動 GeneFace++ 逐幀生成。
    TODO: 將 frame_np 推成 LiveKit 視訊軌
    """
    # 準備入參（依你的 pipeline 調整）
    inp = {
        'a2m_ckpt': "checkpoints/audio2motion_vae",
        'postnet_ckpt': "",
        'head_ckpt': "",
        'torso_ckpt': "checkpoints/motion2video_nerf/may_torso",
        'drv_audio_name': wav_path,
        'drv_pose': 'nearest',
        'blink_mode': 'none',
        'temperature': 0.2,
        'mouth_amp': 0.4,
        'lle_percent': 0.2,
        'debug': False,
        'out_name': 'tmp.mp4',
        'raymarching_end_threshold': 0.01,
        'low_memory_usage': False,
        'use_emotalk': True,
        'blend_path': "emotalk/render_testing_92.blend",
        'level': 1,
        'person': 1,
        'output_video': False,
        'bs52_level': 3,
        'bs_lm_area': 1,
    }

    # 先把 audio→motion 等條件算好
    samples = geneface.prepare_batch_from_inp({'drv_audio_name': wav_path, **inp})

    # 準備 LiveKit 視訊軌（TODO：依 LiveKit Python SDK 建立 LocalVideoTrack）
    # 參考概念：
    #   source = rtc.VideoSource(width=512, height=512)
    #   track = rtc.LocalVideoTrack.create_video_track("avatar", source)
    #   await ctx.room.local_participant.publish_track(track)

    # 逐幀產出
    for frame_np in geneface.stream_secc2video(samples, inp):
        # frame_np: (H, W, C) uint8
        # TODO: 將 frame_np 轉成 LiveKit VideoFrame 並經由 source 推送
        # e.g. source.on_captured_frame(frame)
        pass