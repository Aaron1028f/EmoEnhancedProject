import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from schemas import ChatRequest, ChatResponse
from services import RolePlayRAG
import asyncio

# 初始化 FastAPI 應用
app = FastAPI(
    title="角色扮演 RAG API",
    description="一個使用 LangChain 和 FastAPI 建立的 AI 角色扮演聊天機器人",
    version="1.0.0"
)

# 建立 RolePlayRAG 服務的單例
# 這樣可以確保模型和向量資料庫只在啟動時載入一次
try:
    chatbot_service = RolePlayRAG()
    print("✅ 聊天機器人服務已成功初始化。")
except Exception as e:
    print(f"❌ 初始化聊天機器人服務時發生嚴重錯誤: {e}")
    chatbot_service = None

@app.on_event("startup")
async def startup_event():
    """
    應用程式啟動時執行的事件
    """
    if chatbot_service is None:
        # 如果服務初始化失敗，可以選擇在這裡再次嘗試或直接讓應用程式失敗
        print("❌ 服務未初始化，API 可能無法正常運作。")
    else:
        print("🚀 FastAPI 應用程式已啟動，API 端點已準備就緒。")

@app.post("/chat/", response_model=ChatResponse, summary="標準聊天請求")
async def chat_endpoint(request: ChatRequest):
    """
    接收標準的 RESTful API 請求，並回傳完整的聊天回應。
    這個端點適用於不需要即時串流的場景。
    """
    if chatbot_service is None:
        return JSONResponse(status_code=503, content={"error": "服務暫時不可用，請稍後再試。"})

    try:
        # 使用服務處理請求
        full_response = await chatbot_service.get_response(request.user_input, request.session_id)
        return ChatResponse(response=full_response)
    except Exception as e:
        print(f"處理請求時發生錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": "處理您的請求時發生內部錯誤。"})


@app.websocket("/ws/chat/")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    透過 WebSocket 提供即時串流聊天服務。
    - 建立連線後，客戶端應傳送一個包含 `session_id` 的 JSON 字串。
    - 之後傳送的每個訊息都應為包含 `user_input` 的 JSON 字串。
    - 伺服器會將生成的回應以串流方式即時回傳。
    """
    await websocket.accept()
    session_id = None

    if chatbot_service is None:
        await websocket.send_json({"type": "error", "data": "服務暫時不可用，請稍後再試。"})
        await websocket.close()
        return

    try:
        # 第一個訊息應該是 session_id
        initial_message = await websocket.receive_json()
        if 'session_id' in initial_message:
            session_id = initial_message['session_id']
            await websocket.send_json({"type": "system", "data": f"連線成功，對話 Session ID: {session_id}"})
        else:
            await websocket.send_json({"type": "error", "data": "連線失敗：第一個訊息必須包含 session_id。"})
            await websocket.close()
            return

        # 監聽後續的訊息
        while True:
            data = await websocket.receive_json()
            user_input = data.get("user_input")

            if not user_input:
                continue

            # 使用服務的串流功能
            async for chunk in chatbot_service.stream_response(user_input, session_id):
                await websocket.send_json({"type": "stream", "data": chunk})
            # 標示串流結束
            await websocket.send_json({"type": "stream_end"})

    except WebSocketDisconnect:
        print(f"WebSocket 連線已中斷 (Session: {session_id})")
        if session_id:
            chatbot_service.clear_history(session_id) # 清理對話歷史
    except Exception as e:
        print(f"WebSocket 發生錯誤 (Session: {session_id}): {e}")
        await websocket.send_json({"type": "error", "data": f"發生未預期的錯誤: {e}"})
    finally:
        # 確保連線被關閉
        try:
            await websocket.close()
        except RuntimeError:
            # 連線可能已經被關閉
            pass

if __name__ == "__main__":
    # 使用 uvicorn 啟動伺服器
    # reload=True 可以在程式碼變更時自動重啟，方便開發
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)