# 啟動livekit server整體流程

### 啟動 Local Livekit Server
```bash
cd server/lk_exp/server_src_bin/
/home/aaron/project/server/lk_exp/server_src_bin/livekit-server --dev
# /home/aaron/project/server/lk_exp/server_src_bin/livekit-server --dev --bind 0.0.0.0

```

### FastAPI 後端程式 (專門管理房間與產生token)
```bash
cd server/lk_exp
uvicorn lk_server:app --host 0.0.0.0 --port 8000 --reload
```


### 其他 python Bot ( ASR, TTS 等後端服務 )
```bash
# example
# cd server/lk_exp
# uvicorn asr_bot:app --host 0.0.0.0 --port 8001 --reload

```


### 其他 (必要python套件安裝)
```bash
pip install fastapi uvicorn livekit

```