import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.roleplay_service import roleplay_service
from app.schemas import WebSocketMessage

router = APIRouter()

@router.websocket("/ws/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            # 等待來自客戶端的訊息
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            user_input = data.get("input")

            if user_input is None:
                await websocket.send_json({"error": "缺少 'input' 欄位"})
                continue

            # 使用服務進行串流聊天
            async for chunk in roleplay_service.stream_chat(user_input, session_id):
                # 將每個字塊作為 token 傳回客戶端
                await websocket.send_json({"token": chunk})
            
            # 可以在這裡傳送一個結束信號，但目前客戶端是透過連線關閉來判斷結束
            # await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        print(f"WebSocket session {session_id} disconnected.")
    except Exception as e:
        print(f"WebSocket Error for session {session_id}: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        if not websocket.client_state.DISCONNECTED:
            await websocket.close()