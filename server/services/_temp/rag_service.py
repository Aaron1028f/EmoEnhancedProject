# uvicorn rag_service:app --port 8001

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
from dotenv import load_dotenv
load_dotenv()  # 讀取 .env 檔案中的環境變
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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

#===================================================================


# # 4. RAG 查詢函式（直接用已載入的 db）
# def rag_query(query: str):
#     hits = db.similarity_search(query, k=3)    
#     context = "\n\n".join(d.page_content for d in hits)
#     prompt  = f"以下是相關資訊：\n{context}\n\n問題：{query}\n回答："
#     resp    = client.responses.create(
#         model="gpt-4o-mini",
#         instructions="""
#         你是一位神經內科醫師(方醫師)，喜歡用直白的語氣和生動的舉例，說明艱深的醫學知識，幫助大家更了解自己的身體健康。
#         相關資訊提到的問題與回答都是你自己曾經的回答，請使用這些資訊的語氣和風格來回答問題。
#         """,
#         input=prompt
#     )
#     return resp.output_text

# # 範例呼叫
# print(rag_query("什麼是三分之一活法？"))

