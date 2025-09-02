from __future__ import annotations
import asyncio
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class AvatarNotify(BaseModel):
    wav_path: str
    text: str | None = None
    sample_rate: int | None = None

def create_app(queue: "asyncio.Queue[dict[str, Any]]") -> FastAPI:
    app = FastAPI()

    @app.post("/avatar/notify")
    async def avatar_notify(payload: AvatarNotify):
        await queue.put(payload.model_dump())
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app

async def serve(queue: "asyncio.Queue[dict[str, Any]]", host: str = "127.0.0.1", port: int = 9901):
    app = create_app(queue)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)
    await server.serve()