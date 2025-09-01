# localLLM.py
from __future__ import annotations

import os
import json
from typing import Any, AsyncIterator

import httpx

from livekit.agents import llm as lkllm
from livekit.agents.llm import ChatContext
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
)
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError


def _msg_text_from_content(content: Any) -> str:
    # content 可能是 str、list(part objects)、或其它型態
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # 常見格式: {"type": "input_text"/"output_text", "text": "..."}
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


class _MyLocalLLMStream(lkllm.LLMStream):
    def __init__(
        self,
        *,
        llm: "MyLocalLLM",
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
        # 1) 從 ChatContext 正確取得歷史訊息
        history_messages = list(self._chat_ctx.messages)

        user_input = ""
        if history_messages and history_messages[-1].role == "user":
            user_input = _msg_text_from_content(history_messages[-1].content)
            history_for_api = history_messages[:-1]
        else:
            history_for_api = history_messages

        formatted_history = []
        for msg in history_for_api:
            text = _msg_text_from_content(getattr(msg, "content", None))
            if not text:
                continue
            # role 直接沿用（"system" / "user" / "assistant"）
            formatted_history.append({"role": msg.role, "content": text})

        payload = {
            "input": user_input,
            "chat_history": formatted_history,
            # 視需要加入額外參數
        }

        try:
            # 2) 串流請求（同時兼容非串流）
            async with self._client.stream(
                "POST",
                self._api_url,
                json=payload,
                timeout=self._conn_options.timeout,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise APIStatusError(
                        f"Server returned status {e.response.status_code}",
                        status_code=e.response.status_code,
                        body=await _safe_text(e.response),
                    ) from e

                # 嘗試以行為單位讀取（SSE/NDJSON/純文字都能處理）
                async for text in _iter_text_chunks(response):
                    if not text:
                        continue
                    chunk = lkllm.ChatChunk(
                        delta=lkllm.ChoiceDelta(content=text, role="assistant")
                    )
                    self._event_ch.send_nowait(chunk)

        except httpx.TimeoutException as e:
            raise APITimeoutError(retryable=True) from e
        except httpx.RequestError as e:
            raise APIConnectionError(retryable=True) from e
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError(retryable=False) from e


async def _iter_text_chunks(response: httpx.Response) -> AsyncIterator[str]:
    """
    嘗試解析各種常見串流格式：
    - SSE: "data: <json或純文字>"，以 \n 分隔，可能含 [DONE]
    - NDJSON: 每行一個 JSON；取 "delta" 或 "content" 欄位
    - 純文字：每行即一段
    若伺服器非串流，一次回整塊，也能處理。
    """
    content_type = (response.headers.get("content-type") or "").lower()

    if "text/event-stream" not in content_type and "chunked" not in response.headers.get("transfer-encoding", "").lower():
        full = await _safe_text(response)
        text = _extract_text_from_json_line(full)
        if text is not None:
            yield text
            return
        if full:
            yield full
        return

    async for raw_line in response.aiter_lines():
        if not raw_line:
            continue
        line = raw_line.strip()

        if line.startswith("data:"):
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            text = _extract_text_from_json_line(data)
            if text is not None:
                yield text
                continue
            yield data
            continue

        if line.startswith("{") and line.endswith("}"):
            text = _extract_text_from_json_line(line)
            if text is not None:
                yield text
                continue

        yield line


def _extract_text_from_json_line(s: str) -> str | None:
    try:
        obj = json.loads(s)
    except Exception:
        return None

    for key in ("delta", "content", "text"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val

    if "choices" in obj and isinstance(obj["choices"], list):
        try:
            c0 = obj["choices"][0]
            delta = c0.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                return content
        except Exception:
            pass
    return None


async def _safe_text(resp: httpx.Response) -> str:
    try:
        return (await resp.aread()).decode(resp.encoding or "utf-8", errors="ignore")
    except Exception:
        return ""


class MyLocalLLM(lkllm.LLM):
    def __init__(self, api_url: str | None = None):
        super().__init__()
        self._api_url = api_url or os.getenv("LOCAL_LLM_URL", "http://localhost:28000/streaming_chat")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            follow_redirects=True,
        )

    @property
    def model(self) -> str:
        return "local-fastapi-roleplay-model"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **_: Any,
    ) -> lkllm.LLMStream:
        return _MyLocalLLMStream(
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