# server.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict

from roleplay_main import prepare_roleplay_chain
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

app = FastAPI()

# 使用 Pydantic 模型來定義請求的結構，更清晰且有自動驗證
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    input: str
    # 將 chat_history 改為可選的，並提供預設空列表
    chat_history: List[ChatMessage] = Field(default_factory=list)

class AppState:
    def __init__(self):
        # roleplay_chain 只需要初始化一次
        self.roleplay_chain = prepare_roleplay_chain()

app_state = AppState()

async def roleplay_streamer(request: ChatRequest):
    """
    這個 streamer 現在接收一個包含 input 和 history 的請求物件
    """
    if not app_state.roleplay_chain:
        yield "Error: Roleplay chain not initialized."
        return

    # 將從請求中收到的 Pydantic 模型轉換為 LangChain 需要的 Message 物件
    langchain_history: List[BaseMessage] = []
    for msg in request.chat_history:
        if msg.role == 'user' or msg.role == 'human':
            langchain_history.append(HumanMessage(content=msg.content))
        elif msg.role == 'assistant' or msg.role == 'ai':
            langchain_history.append(AIMessage(content=msg.content))

    stream_input = {"input": request.input, "chat_history": langchain_history}
    
    # 串流輸出，直接 yield 區塊
    for chunk in app_state.roleplay_chain.stream(stream_input):
        yield chunk

# 將 endpoint 從 GET 改為 POST
@app.post("/streaming_chat")
async def streaming_chat(request: ChatRequest):
    """
    接收 POST 請求，請求主體 (body) 是一個包含 'input' 和 'chat_history' 的 JSON
    """
    return StreamingResponse(roleplay_streamer(request), media_type="text/plain") # media_type 改為 text/plain 更適合原始文字流


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28000)