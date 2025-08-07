# filepath: /home/aaron/project/server/models/LLM/fastapi-roleplay-project/app/api/rest_api.py
from fastapi import APIRouter, Depends
from app.schemas import UserInput, RoleplayResponse
from app.services.roleplay_service import roleplay_service
import uuid

router = APIRouter()

# 簡單的 session 管理，可以用更複雜的依賴注入代替
def get_session_id():
    # 每次請求都產生一個新的 session id，適合無狀態的 REST API
    return str(uuid.uuid4())

@router.post("/chat", response_model=RoleplayResponse)
async def chat_endpoint(request: UserInput, session_id: str = Depends(get_session_id)):
    """
    接收使用者輸入，回傳一次性的完整 AI 回應。
    """
    response_text = await roleplay_service.chat(request.input, session_id)
    # 注意：這裡我們沒有解析情緒標籤，僅回傳純文字
    return RoleplayResponse(response=response_text)