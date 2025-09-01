from __future__ import annotations
import os
import json
from typing import Any, AsyncIterator
import httpx

from livekit.agents import llm as lkllm
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError

def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""

class _LocalLLMStream(lkllm.LLMStream):
    def __init__(
        self,
        *,
        llm: "LocalLLM",
        client: httpx.AsyncClient,
        api_url: str,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
    ):
        super().__init__(llm=llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._client = client
        self._api_url = api_url

    async def _run(self) -> None:
        items = list(self._chat_ctx.items)
        user_input = ""
        # 從最後一筆 user 訊息抓文字內容
        for it in reversed(items):
            if getattr(it, "type", None) == "message" and getattr(it, "role", None) == "user":
                user_input = _text_from_content(getattr(it, "content", []))
                break

        try:
            # 你的 API 是 GET + SSE
            async with self._client.stream(
                "GET",
                self._api_url,
                params={"user_input": user_input},
                timeout=self._conn_options.timeout,
            ) as resp:
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = ""
                    try:
                        body = (await resp.aread()).decode(resp.encoding or "utf-8", errors="ignore")
                    except Exception:
                        pass
                    raise APIStatusError(
                        f"Local LLM returned {e.response.status_code}",
                        status_code=e.response.status_code,
                        body=body,
                    ) from e

                async for text in _iter_sse_or_text(resp):
                    if not text:
                        continue
                    self._event_ch.send_nowait(
                        lkllm.ChatChunk(
                            id="local",
                            delta=lkllm.ChoiceDelta(content=text, role="assistant"),
                        )
                    )

        except httpx.TimeoutException as e:
            raise APITimeoutError(retryable=True) from e
        except httpx.RequestError as e:
            raise APIConnectionError(retryable=True) from e
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError(retryable=False) from e

async def _iter_sse_or_text(resp: httpx.Response) -> AsyncIterator[str]:
    async for raw in resp.aiter_lines():
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]" or not data:
                continue
            # 嘗試 JSON {"delta": "..."}，否則當純文字
            try:
                obj = json.loads(data)
                if isinstance(obj, dict):
                    if isinstance(obj.get("delta"), str):
                        yield obj["delta"]
                        continue
                    if isinstance(obj.get("content"), str):
                        yield obj["content"]
                        continue
            except Exception:
                pass
            yield data
        else:
            # 非標準 SSE，就當作純文字片段
            yield line

class LocalLLM(lkllm.LLM):
    def __init__(self, api_url: str | None = None):
        super().__init__()
        self._api_url = api_url or os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:28000/streaming_response")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            follow_redirects=True,
        )

    @property
    def model(self) -> str:
        return "local-roleplay-model"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **_: Any,
    ) -> lkllm.LLMStream:
        return _LocalLLMStream(
            llm=self,
            client=self._client,
            api_url=self._api_url,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass