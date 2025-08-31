## start local livekit server

```bash
cd server/lk_exp/server_src_bin/

# source: https://docs.livekit.io/home/self-hosting/local/
# 1. 安裝 Binary
wget https://github.com/livekit/livekit/releases/download/v1.9.0/livekit_1.9.0_linux_amd64.tar.gz
tar -xvzf livekit_1.9.0_linux_amd64.tar.gz
# bin file path: /home/aaron/project/server/lk_exp/server_src_bin/livekit-server

# -------------------------------------------------------------------------------------------------
# 2. 啟動 Server
# start server in dev mode (original: livekit-server --dev)
/home/aaron/project/server/lk_exp/server_src_bin/livekit-server --dev
# 小提醒：如果你希望這台 Server 能在本機網路中被其他裝置（例如手機或另一台電腦）連線，則可以加入參數 --bind 0.0.0.0，這樣 LiveKit Server 會綁在所有可用網卡 IP 上。
/home/aaron/project/server/lk_exp/server_src_bin/livekit-server --dev --bind 0.0.0.0

# 測試 generate key
# /home/aaron/project/server/lk_exp/server_src_bin/livekit-server generate-key
# -------------------------------------------------------------------------------------------------

# 2.5 測試是否啟動成功
# 下載 livekit-cli
wget https://github.com/livekit/livekit-cli/releases/download/v2.5.5/lk_2.5.5_linux_amd64.tar.gz
tar -xvzf lk_2.5.5_linux_amd64.tar.gz
# bin file path: /home/aaron/project/server/lk_exp/server_src_bin/lk

# 產生access token (會產生加入時要填的url, 以及access token)
./lk token create \
  --api-key devkey --api-secret secret \
  --join --room test_room --identity test_user \
  --valid-for 24h

# 使用 lk room join 加入測試房間
# /home/aaron/project/server/lk_exp/server_src_bin/lk
./lk room join   --url ws://localhost:7880   --api-key devkey --api-secret secret   --identity tester --publish-demo   demo-room

# 在網頁看結果
打開網頁: https://meet.livekit.io/?tab=custom
url輸入: ws://localhost:7880
token輸入: devkey/secret

```

## python SDK and API key

#### 一個python程式負責房間管理與token產生
Python 後端API產生token
```python
from fastapi import FastAPI
from livekit import api

API_KEY = "devkey"
API_SECRET = "secret"

app = FastAPI()

@app.get("/get_token")
def get_token(identity: str, room: str):
    # 建立一個 AccessToken
    at = api.AccessToken(API_KEY, API_SECRET, identity=identity)
    at.add_grant(api.VideoGrants(room_join=True, room=room))
    return {"token": at.to_jwt()}
```

#### 其他python程式作為Bot進入房間
Python 後端也能作為「參與者」加入，這樣 Python 就能像人一樣加入房間（可訂閱 / 發佈音訊、影像）。
```python
import asyncio
from livekit import rtc

async def main():
    room = rtc.Room()
    await room.connect(
        url="ws://localhost:7880",
        token="你後端生成的 AccessToken",
    )

    @room.on("participant_connected")
    def on_participant_connected(participant):
        print(f"新參與者加入: {participant.identity}")

    await asyncio.Future()  # 永遠不結束，保持運行

asyncio.run(main())

```