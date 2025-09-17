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

# IndexTTS 模型的預設取樣率是 22050 Hz
SAMPLE_RATE = 22050
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    """存放 IndexTTS API 所需的參數"""
    base_url: str
    spk_audio_prompt: str
    media_type: MediaType
    # 可以在這裡添加更多 IndexTTS 支援的參數，例如 emo_alpha 等
    # emo_alpha: float


class LocalTTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        spk_audio_prompt: str | None = None,
        media_type: MediaType = "ogg",
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        連到你的本地 IndexTTS API (/tts)。預設使用 streaming_mode=True。
        重要：建議 media_type 使用 ogg, aac, webm 等容器格式。
        """
        super().__init__(
            # IndexTTS 支援串流，所以設定為 True
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        
        # 為 IndexTTS 設定一個預設的參考音訊
        DEFAULT_SPK_AUDIO_PROMPT = '/home/aaron/project/server/models/TTS/index-tts/examples/voice_01.wav'

        self._opts = _TTSOptions(
            base_url=base_url or os.getenv("INDEXTTS_URL", "http://127.0.0.1:40000/tts"),
            spk_audio_prompt=spk_audio_prompt or os.getenv("INDEXTTS_SPK_AUDIO_PROMPT", DEFAULT_SPK_AUDIO_PROMPT),
            media_type=media_type,
        )

        self._client = httpx.AsyncClient(
            timeout=timeout or httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
        )
        self._prewarm_task: asyncio.Task | None = None

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
        # 根據 api_indextts.py 的 TTSRequest 模型建立 payload
        payload = {
            "text": self.input_text,
            "spk_audio_prompt": self._tts._opts.spk_audio_prompt,
            "media_type": self._tts._opts.media_type,
            "streaming_mode": True,
            # 可以在這裡傳遞更多需要的參數
            # "emo_alpha": 1.0,
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
# ```

# ### 如何使用

# 1.  **啟動 IndexTTS API 伺服器**：
#     ```bash
#     # 進入 index-tts 目錄
#     cd /home/aaron/project/server/models/TTS/index-tts/
#     # 執行 api_indextts.py
#     uvicorn indextts.api_indextts:app --host 0.0.0.0 --port 40000
#     ```

# 2.  **在您的 LiveKit Agent 中使用 `localTTS_indextts`**：
#     在您的 agent 主程式 (例如 `agent_feng_vid.py`) 中，將原本的 `LocalTTS` 替換為 `localTTS_indextts`。

#     ```python
#     # 在 agent_feng_vid.py 或類似檔案中
#     # from .localTTS import LocalTTS  <-- 註解掉舊的
#     from .localTTS_indextts import LocalTTS as IndexTTS # <-- 匯入新的，並可重新命名以區分

#     # ...

#     # 在初始化 agent 的地方
#     tts = IndexTTS(
#         # 您可以在這裡覆寫預設值，例如指定不同的參考音訊
#         # spk_audio_prompt="/path/to/your/voice.wav"
#     )
#     ```

# 現在，您的 LiveKit Agent 就會透過新的 `localTTS_indextts.py` 客戶端，與在 `40000` 埠運行的 `api_indextts.py` 伺服器進行通訊，實現文字轉語音的功能。# filepath: /home/aaron/project/server/lk_exp/agent-starter-python/src/localTTS_indextts.py
