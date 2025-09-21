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
EMOTION_MAP_JSON = "/home/aaron/project/server/models/RAG/audio_prompt_selection/DATA/emotion_map.json"

# 初始化嵌入
emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBED_MODEL)

app = FastAPI(title="Audio Transcript RAG Service")

db: Optional[FAISS] = None
emotion_map: dict = {}

def load_emotion_map(path: str) -> dict:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # key: 絕對路徑，value: {"emotion": "...", "emotion_score": 0.x}
        return data
    return {}

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
                emo = emotion_map.get(audio_path, {})
                meta = {
                    "id": i,
                    "audio_path": audio_path,
                    "tool": tool,
                    "lang": lang,
                    "filename": os.path.basename(audio_path),
                    "emotion": emo.get("emotion", None),
                    "emotion_score": emo.get("emotion_score", None),
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
emotion_map = load_emotion_map(EMOTION_MAP_JSON)
db = ensure_index()


# 輸入/輸出模型
class RetrievePrompt(BaseModel):
    query: str
    k: int = 3
    desired_emotion: Optional[str] = None  # 例如 "happy"
    emotion_mode: str = "off"              # off | soft | strict

class Hit(BaseModel):
    transcript: str
    audio_path: str
    filename: str
    score: float
    emotion: Optional[str] = None
    emotion_score: Optional[float] = None

class RetrieveResults(BaseModel):
    hits: List[Hit]

def normalize_emotion(e: Optional[str]) -> Optional[str]:
    return e.lower() if isinstance(e, str) else None

@app.post("/retrieve", response_model=RetrieveResults)
def retrieve(body: RetrievePrompt):
    k = max(1, min(20, body.k))
    mode = (body.emotion_mode or "off").lower()
    want = normalize_emotion(body.desired_emotion)

    # 擴大初始召回，以便之後過濾/重排
    fetch_n = k if mode == "off" or not want else max(k * 5, 50)
    results = db.similarity_search_with_score(body.query, k=fetch_n)

    # 整理成簡單清單
    items = []
    for doc, score in results:
        m = doc.metadata or {}
        items.append({
            "doc": doc,
            "score": float(score),
            "emotion": normalize_emotion(m.get("emotion")),
            "emotion_score": m.get("emotion_score"),
        })

    # 嚴格模式：僅保留情緒相符，若為空則回退軟性重排
    def filter_strict(xs):
        return [x for x in xs if x["emotion"] == want]

    if want and mode == "strict":
        filtered = filter_strict(items)
        if not filtered:
            mode = "soft"  # 回退
        else:
            items = filtered

    # 軟性重排：優先相符情緒，再依距離分數
    if want and mode == "soft":
        # 先相符放前面，再依 score 升冪（FAISS 距離越小越好）
        items.sort(key=lambda x: (0 if x["emotion"] == want else 1, x["score"]))
    else:
        # 純語義
        items.sort(key=lambda x: x["score"])

    items = items[:k]

    hits: List[Hit] = []
    for x in items:
        doc = x["doc"]
        m = doc.metadata or {}
        hits.append(Hit(
            transcript=doc.page_content,
            audio_path=m.get("audio_path", ""),
            filename=m.get("filename", ""),
            score=x["score"],
            emotion=x.get("emotion"),
            emotion_score=x.get("emotion_score"),
        ))
    return RetrieveResults(hits=hits)

@app.post("/reindex")
def reindex():
    global db, emotion_map
    db = None
    # 重新載入 emotion_map（允許你先更新 JSON 再重建索引）
    emotion_map = load_emotion_map(EMOTION_MAP_JSON)
    # 刪現有索引檔
    for p in ["index.faiss", "index.pkl"]:
        fp = Path(INDEX_PATH) / p
        if fp.exists():
            fp.unlink()
    db_local = ensure_index()
    db = db_local
    return {"status": "ok", "indexed": True, "emotion_map": bool(emotion_map)}

@app.get("/emotions")
def emotions():
    # 回傳目前索引中可見的情緒統計
    counts = {}
    # 簡單遍歷 emotion_map 即可（較快）
    for _, v in emotion_map.items():
        e = normalize_emotion(v.get("emotion"))
        counts[e] = counts.get(e, 0) + 1
    return {"counts": counts, "available": sorted([k for k in counts.keys() if k])}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=45000)