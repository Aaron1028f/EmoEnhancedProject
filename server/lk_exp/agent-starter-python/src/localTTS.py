from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import Literal

import httpx

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import aio

# 建議用容器格式，避免 wav 串流的 header+raw 交錯問題
MediaType = Literal["webm", "ogg", "aac", "wav", "raw"]

# 根據你的 api_v2.py 預設單聲道，Opus 通常 48000
SAMPLE_RATE = 48000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    base_url: str
    text_lang: str
    prompt_lang: str
    ref_audio_path: str
    prompt_text: str
    media_type: MediaType
    speed_factor: float


class LocalTTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        text_lang: str | None = None,
        prompt_lang: str | None = None,
        ref_audio_path: str | None = None,
        prompt_text: str | None = None,
        # media_type: MediaType = "webm",
        media_type: MediaType = "ogg",
        speed_factor: float = 1.0,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        連到你的本地 TTS API (/tts)。預設使用 streaming_mode=True。
        重要：建議 media_type 使用 webm / ogg / aac 其中之一。
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        
        REF_AUDIO_PATH = '/home/aaron/project/server/models/TTS/GPT-SoVITS/DATA/Feng_EP32/slicer/Feng_live_EP32.wav_0028094080_0028232000.wav'

        self._opts = _TTSOptions(
            base_url=base_url or os.getenv("LOCAL_TTS_URL", "http://127.0.0.1:9880/tts"),
            text_lang=(text_lang or os.getenv("LOCAL_TTS_TEXT_LANG", "zh")).lower(),
            prompt_lang=(prompt_lang or os.getenv("LOCAL_TTS_PROMPT_LANG", "zh")).lower(),
            ref_audio_path=ref_audio_path or os.getenv("LOCAL_TTS_REF_AUDIO", REF_AUDIO_PATH),
            prompt_text=prompt_text or os.getenv("LOCAL_TTS_PROMPT_TEXT", ""),
            media_type=media_type,
            speed_factor=speed_factor,
        )

        self._client = httpx.AsyncClient(
            timeout=timeout or httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
        )
        # self._prewarm_task: aio.AsyncTask | None = None  # 型別註解用，不一定需要
        self._prewarm_task: asyncio.Task | None = None  # 型別註解用，不一定需要

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> _ChunkedStream:
        return _ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def prewarm(self) -> None:
        async def _prewarm() -> None:
            try:
                # 確保 API 活著
                await self._client.get(self._opts.base_url, timeout=5.0)
            except Exception:
                pass

        # self._prewarm_task = aio.create_task(_prewarm())
        self._prewarm_task = asyncio.create_task(_prewarm())

    async def aclose(self) -> None:
        if self._prewarm_task:
            await aio.cancel_and_wait(self._prewarm_task)
        await self._client.aclose()


class _ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: LocalTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = {
            "text": self.input_text,
            "text_lang": self._tts._opts.text_lang,
            "ref_audio_path": self._tts._opts.ref_audio_path,
            "prompt_text": self._tts._opts.prompt_text,
            "prompt_lang": self._tts._opts.prompt_lang,
            "speed_factor": self._tts._opts.speed_factor,
            "media_type": self._tts._opts.media_type,
            "streaming_mode": True,
        }

        try:
            async with self._tts._client.stream(
                "POST",
                self._tts._opts.base_url,
                json=payload,
                timeout=httpx.Timeout(
                    # 讀取用 conn_options.timeout 作為上限
                    connect=min(15.0, float(self._conn_options.timeout or 15.0)),
                    read=self._conn_options.timeout or 60.0,
                    write=15.0,
                    pool=15.0,
                ),
            ) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    # 讀一段 body 當錯誤訊息
                    body = ""
                    try:
                        body = (await resp.aread()).decode(resp.encoding or "utf-8", errors="ignore")
                    except Exception:
                        pass
                    raise APIStatusError(
                        f"TTS server returned {e.response.status_code}",
                        status_code=e.response.status_code,
                        body=body,
                    ) from e

                output_emitter.initialize(
                    request_id="",  # 你的 API 若有回傳可帶上
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type=f"audio/{self._tts._opts.media_type}",
                )

                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    output_emitter.push(chunk)

                output_emitter.flush()

        except httpx.TimeoutException as e:
            raise APITimeoutError() from e
        except httpx.RequestError as e:
            raise APIConnectionError() from e