# uvicorn rag_service:app --port 8001
OPENAI_API_KEY = "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"

# # rag_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import os
import json
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

EMBED_MODEL = "text-embedding-ada-002"
INDEX_PATH  = "faiss_index"
DOCS_PATH   = "QAdata.json"

# client = OpenAI(api_key=OPENAI_API_KEY)
emb   = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBED_MODEL)

if os.path.exists(f"{INDEX_PATH}/index.faiss"):
    db = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)
    print(f"已載入索引：{INDEX_PATH}/index.faiss")
else:
    print(f"索引不存在，開始建立：{INDEX_PATH}.faiss")
    with open(DOCS_PATH, encoding="utf-8") as f:
        items = json.load(f)
    docs = []
    for i, entry in enumerate(items):
        instr = entry["instruction"]
        ctx   = entry.get("input", "")
        out   = entry.get("output", "")
        text = instr + ("\n\nContext: " + ctx if ctx else "") + "\n\nAnswer: " + out
        docs.append(Document(page_content=text, metadata={"id": i}))
    db = FAISS.from_documents(docs, emb)
    db.save_local(INDEX_PATH)


app = FastAPI(title="RAG Service")

# 輸入：使用者文字
class Prompt(BaseModel):
    query: str

# 回傳：相關上下文清單
class Context(BaseModel):
    snippets: List[str]

@app.post("/retrieve", response_model=Context)
def retrieve(prompt: Prompt):
    hits = db.similarity_search(prompt.query, k=3)
    context = [d.page_content for d in hits]
    return Context(snippets=context[:2])