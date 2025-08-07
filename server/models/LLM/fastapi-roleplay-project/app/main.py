import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from schemas import ChatRequest, ChatResponse
from services import RolePlayRAG
from contextlib import asynccontextmanager

# --- Lifespan Management ---
# 使用 lifespan 上下文管理器來處理啟動和關閉事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    應用程式的生命週期管理器。
    """
    print("🚀 開始初始化角色扮演引擎...")
    # 將服務的初始化放在這裡
    app.state.chatbot_service = RolePlayRAG()
    print("✅ 聊天機器人服務已成功初始化。")
    print("🚀 FastAPI 應用程式已啟動，API 端點已準備就緒。")
    
    yield # 應用程式在這裡運行
    
    # 應用程式關閉時執行的清理程式碼
    print("👋 應用程式正在關閉...")
    # 可以在這裡添加資源清理的邏輯，例如關閉資料庫連線等
    app.state.chatbot_service = None


# 初始化 FastAPI 應用，並傳入 lifespan 管理器
app = FastAPI(
    title="角色扮演 RAG API",
    description="一個使用 LangChain 和 FastAPI 建立的 AI 角色扮演聊天機器人",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/chat/", response_model=ChatResponse, summary="標準聊天請求")
async def chat_endpoint(request: ChatRequest):
    """
    接收標準的 RESTful API 請求，並回傳完整的聊天回應。
    """
    chatbot_service = app.state.chatbot_service
    if chatbot_service is None:
        return JSONResponse(status_code=503, content={"error": "服務暫時不可用，請稍後再試。"})

    try:
        full_response = await chatbot_service.get_response(request.user_input, request.session_id)
        return ChatResponse(response=full_response)
    except Exception as e:
        print(f"處理請求時發生錯誤: {e}")
        return JSONResponse(status_code=500, content={"error": "處理您的請求時發生內部錯誤。"})


@app.websocket("/ws/chat/")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    透過 WebSocket 提供即時串流聊天服務。
    每個連線處理一次完整的「請求-回應」流程。
    """
    chatbot_service = websocket.app.state.chatbot_service
    await websocket.accept()
    print("INFO:     connection open") # 增加日誌以便觀察
    session_id = None

    if chatbot_service is None:
        await websocket.send_json({"type": "error", "data": "服務暫時不可用，請稍後再試。"})
        await websocket.close()
        return

    try:
        # 步驟 1: 接收包含 session_id 的初始訊息
        initial_message = await websocket.receive_json()
        if 'session_id' in initial_message:
            session_id = initial_message['session_id']
            await websocket.send_json({"type": "system", "data": f"連線成功，對話 Session ID: {session_id}"})
        else:
            await websocket.send_json({"type": "error", "data": "連線失敗：第一個訊息必須包含 session_id。"})
            await websocket.close()
            return

        # 步驟 2: 接收包含使用者輸入的訊息
        data = await websocket.receive_json()
        user_input = data.get("user_input")

        # 步驟 3: 如果有使用者輸入，則開始串流回應
        if user_input:
            async for chunk in chatbot_service.stream_response(user_input, session_id):
                await websocket.send_json({"type": "stream", "data": chunk})
            # 標示串流結束
            await websocket.send_json({"type": "stream_end"})
        
        # 流程結束，準備關閉連線

    except WebSocketDisconnect:
        # 這個例外是正常的，因為客戶端在完成後會主動斷線
        print(f"WebSocket 連線已由客戶端正常關閉 (Session: {session_id})")
        # 注意：由於每個連線都是短期的，我們可能不需要在這裡清理歷史
        # 除非有特定的需求要在連線中斷時立即清除
        # if session_id:
        #     chatbot_service.clear_history(session_id)
    except Exception as e:
        print(f"WebSocket 發生未預期的錯誤 (Session: {session_id}): {e}")
        # 確保在發生錯誤時通知客戶端
        try:
            await websocket.send_json({"type": "error", "data": f"發生未預期的錯誤: {e}"})
        except RuntimeError:
            pass # 連線可能已經中斷
    finally:
        print("INFO:     connection closed") # 增加日誌以便觀察
        # 確保連線被關閉
        try:
            await websocket.close()
        except RuntimeError:
            pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)