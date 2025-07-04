# uvicorn orchestrator:app --port 8000



# orchestrator.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Orchestrator")

class UserIn(BaseModel):
    text: str

class Result(BaseModel):
    answer: str
    video_url: str

@app.post("/generate", response_model=Result)
def generate_pipeline(inp: UserIn):
    # 1. RAG
    r = requests.post("http://localhost:8001/retrieve", json={"query": inp.text})
    context = " ".join(r.json()["snippets"])

    # 2. LLM
    llm = requests.post("http://localhost:8002/generate",
                        json={"query": inp.text, "context": context})
    answer = llm.json()["answer"]
    
    
    return Result(answer=answer, video_url="")
    

    # # 3. TTS
    # t = requests.post("http://localhost:8003/synthesize", json={"text": answer})
    # audio_path = t.json()["audio_path"]

    # # 4. Video
    # v = requests.post("http://localhost:8004/make_video",
    #                   json={"audio_path": audio_path})
    # video_path = v.json()["video_path"]

    # return Result(answer=answer, video_url=video_path)
    
