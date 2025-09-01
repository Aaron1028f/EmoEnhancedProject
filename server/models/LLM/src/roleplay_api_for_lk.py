from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import roleplay_emo
from roleplay_main import prepare_roleplay_chain

from langchain_core.messages import HumanMessage, AIMessage
import json

app = FastAPI()

class AppState:
    def __init__(self):
        self.roleplay_chain = prepare_roleplay_chain()
        self.chat_history = []

app_state = AppState()

async def roleplay_streamer(user_input: str):
    full_response = ""
    stream_input = {"input": user_input, "chat_history": app_state.chat_history}

    if not app_state.roleplay_chain:
        yield "data: {\"error\": \"Roleplay chain not initialized.\"}\n\n"
        return

    for chunk in app_state.roleplay_chain.stream(stream_input):
        full_response += chunk
        # SSE 一個事件一行，空行分隔
        yield f"data: {json.dumps({'delta': chunk})}\n\n"

    # 結束訊號（可選）
    yield "data: [DONE]\n\n"

    # 更新歷史
    app_state.chat_history.append(HumanMessage(content=user_input))
    app_state.chat_history.append(AIMessage(content=full_response))
    if len(app_state.chat_history) > 10:
        app_state.chat_history = app_state.chat_history[-10:]

@app.get("/streaming_response")
async def get_streaming_response(user_input: str):
    return StreamingResponse(roleplay_streamer(user_input), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28000)