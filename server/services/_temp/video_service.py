# uvicorn video_service:app --port 8004


# video_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import subprocess

app = FastAPI(title="Video Service")

class AudioIn(BaseModel):
    audio_path: str

@app.post("/make_video")
def make_video(body: AudioIn):
    out = f"/tmp/{uuid.uuid4().hex}.mp4"
    # 範例：用 ffmpeg 把音訊疊到一張靜態圖片上，當作影片
    cmd = [
        "ffmpeg", "-loop", "1",
        "-i", "static_bg.png",
        "-i", body.audio_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest", "-y", out
    ]
    subprocess.run(cmd, check=True)
    return {"video_path": out}
