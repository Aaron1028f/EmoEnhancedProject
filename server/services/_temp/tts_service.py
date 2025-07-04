# uvicorn tts_service:app --port 8003


# tts_service.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
from gtts import gTTS   # 或其他 TTS 套件

app = FastAPI(title="TTS Service")

class TextIn(BaseModel):
    text: str

@app.post("/synthesize")
def synthesize(body: TextIn):
    fname = f"/tmp/{uuid.uuid4().hex}.mp3"
    tts = gTTS(body.text, lang="zh-tw")
    tts.save(fname)
    return {"audio_path": fname}
