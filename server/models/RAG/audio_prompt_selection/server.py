from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 設定
EMBED_MODEL = "text-embedding-3-small"  # 或 "text-embedding-3-large" 追求更好效果
INDEX_PATH = "faiss_audio_index"
SLICER_LIST = "/home/aaron/project/server/models/RAG/audio_prompt_selection/DATA/slicer.list"

# 初始化嵌入
emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBED_MODEL)

app = FastAPI(title="Audio Transcript RAG Service")

db: Optional[FAISS] = None


def parse_slicer_list(list_path: str) -> List[Document]:
    docs: List[Document] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or "|" not in line:
                continue
            try:
                # 典型格式：/abs/path.wav|slicer|ZH|transcript...
                audio_path, tool, lang, transcript = line.split("|", 3)
                transcript = transcript.strip()
                if not transcript:
                    continue
                meta = {
                    "id": i,
                    "audio_path": audio_path,
                    "tool": tool,
                    "lang": lang,
                    "filename": os.path.basename(audio_path),
                }
                docs.append(Document(page_content=transcript, metadata=meta))
            except Exception:
                # 跳過不合法行
                continue
    return docs


def ensure_index() -> FAISS:
    os.makedirs(INDEX_PATH, exist_ok=True)
    faiss_file = Path(INDEX_PATH) / "index.faiss"
    if faiss_file.exists():
        db = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)
        print(f"已載入索引：{faiss_file}")
        return db

    print(f"索引不存在，開始建立：{faiss_file}")
    docs = parse_slicer_list(SLICER_LIST)
    if not docs:
        raise RuntimeError(f"slicer.list 無資料或解析失敗：{SLICER_LIST}")
    db = FAISS.from_documents(docs, emb)
    db.save_local(INDEX_PATH)
    print(f"已建立並儲存索引：{faiss_file}")
    return db


# 啟動時建立/載入索引
db = ensure_index()


# 輸入/輸出模型
class RetrievePrompt(BaseModel):
    query: str
    k: int = 3


class Hit(BaseModel):
    transcript: str
    audio_path: str
    filename: str
    score: float


class RetrieveResults(BaseModel):
    hits: List[Hit]


@app.post("/retrieve", response_model=RetrieveResults)
def retrieve(body: RetrievePrompt):
    k = max(1, min(20, body.k))
    results = db.similarity_search_with_score(body.query, k=k)
    hits: List[Hit] = []
    for doc, score in results:
        meta = doc.metadata or {}
        hits.append(
            Hit(
                transcript=doc.page_content,
                audio_path=meta.get("audio_path", ""),
                filename=meta.get("filename", ""),
                score=float(score),
            )
        )
    return RetrieveResults(hits=hits)


@app.post("/reindex")
def reindex():
    global db
    db = None
    # 刪現有索引檔
    for p in ["index.faiss", "index.pkl"]:
        fp = Path(INDEX_PATH) / p
        if fp.exists():
            fp.unlink()
    db_local = ensure_index()
    # 重新指派
    db = db_local
    return {"status": "ok", "indexed": True}
# ...existing code...


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=45000)