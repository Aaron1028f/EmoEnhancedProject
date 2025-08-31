# Run this program to supply tokens and manage livekit room

# 啟動方法:
# uvicorn lk_server:app --host 0.0.0.0 --port 8000 --reload

# 提供訪問以取得 Access Token:
# http://localhost:8000/get_token?identity=tester&room=demo-room


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

API_KEY = "devkey"
API_SECRET = "secret"

app = FastAPI()

# 可選：允許前端跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_token")
def get_token(identity: str, room: str):
    """
    後端產生 LiveKit AccessToken
    identity: 用戶身份
    room: 房間名稱
    """
    at = api.AccessToken(API_KEY, API_SECRET, identity=identity)
    # 允許進入指定房間
    at.add_grant(api.VideoGrants(room_join=True, room=room))
    return {"token": at.to_jwt()}