下面範例示範如何用 FastAPI 把整個「文字→RAG→LLM→TTS→影片」流程拆成幾個微服務，各自啟動並互相呼叫，最後有一個入口（Orchestrator）把結果組成回傳給使用者。

---

## 🏗️ 架構圖

```txt
┌──────────────────┐      HTTP       ┌───────────────────┐
│ 1. Client        │ ─────────────▶ │ 2. RAG Service    │
└──────────────────┘                └───────────────────┘
                                         │  (context)
                                         ▼
                                     ┌───────────────────┐
                                     │ 3. LLM Service    │
                                     └───────────────────┘
                                         │  (回答 text)
                                         ▼
                                     ┌───────────────────┐
                                     │ 4. TTS Service    │
                                     └───────────────────┘
                                         │  (audio.mp3)
                                         ▼
                                     ┌───────────────────┐
                                     │ 5. Video Service  │
                                     └───────────────────┘
                                         │  (video.mp4)
                                         ▼
┌──────────────────┐      HTTP       ┌───────────────────┐
│  Orchestrator    │ ◀────────────── │  Client waits…    │
└──────────────────┘                  └───────────────────┘
```

1. **RAG Service**：搜尋向量庫、回傳相關上下文
2. **LLM Service**：接 context + user prompt 呼叫 OpenAI / 本地 LLM，回傳文字答案
3. **TTS Service**：把文字轉成語音檔（.mp3）
4. **Video Service**：把語音檔合成影片（如 overlay 靜圖或 lip-sync），輸出 .mp4
5. **Orchestrator**：總控，依序呼叫上面各服務，最後把影片與文字一起回給前端

---

## 📂 專案結構

```txt
project/
├─ rag_service.py
├─ llm_service.py
├─ tts_service.py
├─ video_service.py
└─ orchestrator.py
```

---

### 1️⃣ rag\_service.py

```python
# rag_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="RAG Service")

# 輸入：使用者文字
class Prompt(BaseModel):
    query: str

# 回傳：相關上下文清單
class Context(BaseModel):
    snippets: List[str]

@app.post("/retrieve", response_model=Context)
def retrieve(prompt: Prompt):
    # 這裡示範用假資料，實務可接 FAISS/Pinecone 等
    fake_db = [
        "這是一段知識A …",
        "這是一段知識B …",
        "這是一段知識C …"
    ]
    # 擇一或多個最相關段落
    return Context(snippets=fake_db[:2])
```

啟動：

```bash
uvicorn rag_service:app --port 8001
```

---

### 2️⃣ llm\_service.py

```python
# llm_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import openai   # or 本地 LLM wrapper

app = FastAPI(title="LLM Service")

class LLMRequest(BaseModel):
    query: str
    context: str

class LLMResponse(BaseModel):
    answer: str

@app.post("/generate", response_model=LLMResponse)
def generate(req: LLMRequest):
    prompt = req.context + "\nUser: " + req.query
    # 呼叫 OpenAI API
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    text = resp.choices[0].message.content
    return LLMResponse(answer=text)
```

啟動：

```bash
uvicorn llm_service:app --port 8002
```

---

### 3️⃣ tts\_service.py

```python
# tts_service.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
from gtts import gTTS   # 或其他 TTS 套件

app = FastAPI(title="TTS Service")

class TextIn(BaseModel):
    text: str

@app.post("/synthesize")
def synthesize(body: TextIn):
    fname = f"/tmp/{uuid.uuid4().hex}.mp3"
    tts = gTTS(body.text, lang="zh-tw")
    tts.save(fname)
    return {"audio_path": fname}
```

啟動：

```bash
uvicorn tts_service:app --port 8003
```

---

### 4️⃣ video\_service.py

```python
# video_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import subprocess

app = FastAPI(title="Video Service")

class AudioIn(BaseModel):
    audio_path: str

@app.post("/make_video")
def make_video(body: AudioIn):
    out = f"/tmp/{uuid.uuid4().hex}.mp4"
    # 範例：用 ffmpeg 把音訊疊到一張靜態圖片上，當作影片
    cmd = [
        "ffmpeg", "-loop", "1",
        "-i", "static_bg.png",
        "-i", body.audio_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest", "-y", out
    ]
    subprocess.run(cmd, check=True)
    return {"video_path": out}
```

啟動：

```bash
uvicorn video_service:app --port 8004
```

---

### 5️⃣ orchestrator.py

```python
# orchestrator.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Orchestrator")

class UserIn(BaseModel):
    text: str

class Result(BaseModel):
    answer: str
    video_url: str

@app.post("/generate", response_model=Result)
def generate_pipeline(inp: UserIn):
    # 1. RAG
    r = requests.post("http://localhost:8001/retrieve", json={"query": inp.text})
    context = " ".join(r.json()["snippets"])

    # 2. LLM
    llm = requests.post("http://localhost:8002/generate",
                        json={"query": inp.text, "context": context})
    answer = llm.json()["answer"]

    # 3. TTS
    t = requests.post("http://localhost:8003/synthesize", json={"text": answer})
    audio_path = t.json()["audio_path"]

    # 4. Video
    v = requests.post("http://localhost:8004/make_video",
                      json={"audio_path": audio_path})
    video_path = v.json()["video_path"]

    return Result(answer=answer, video_url=video_path)
```

啟動 Orchestrator：

```bash
uvicorn orchestrator:app --port 8000
```

---

## 🚀 使用方式

1. 先分別啟動 8001\~8004 四個微服務
2. 再啟動主控的 Orchestrator (8000)
3. 客戶端呼叫

```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"text":"你好，我想要知道…"}'
```

回傳 JSON：

```json
{
  "answer": "這是一個LLM回覆的範例。",
  "video_url": "/tmp/abc123def456.mp4"
}
```

---

## ✅ 小結

* 完全採用 **FastAPI + uvicorn** 做本地微服務通訊
* 各服務可獨立擴展、語言無關（只要 HTTP API）
* 在開發階段很方便，可以 later containerize (Docker)
* 部署到雲端時，也可以把微服務搬到各自的 VM / Container

這樣的「本地 API pattern」非常適合做 RAG + LLM + TTS + Video 整合，把複雜流程拆成容易維護的小模組。


---



在開發測試階段，確實會希望「一鍵啟動」所有微服務；但在工程化、維運的角度，更推薦使用專門的進程管理或容器編排工具，而不是自己在 Python 裡寫一個 `main.py` 來 spawn 5 個 uvicorn。下面整理幾種常見做法：

---

## 一、Docker Compose（最推薦）

**優點**

* 各服務彼此隔離，環境一致
* 一行指令 `docker-compose up` 就能同時啟動、重啟
* 可以設定重啟策略、網路、環境變數

**範例 `docker-compose.yml`**

```yaml
version: "3.8"
services:
  rag:
    build: ./rag_service
    ports:
      - "8001:8001"
  llm:
    build: ./llm_service
    ports:
      - "8002:8002"
  tts:
    build: ./tts_service
    ports:
      - "8003:8003"
  video:
    build: ./video_service
    ports:
      - "8004:8004"
  orchestrator:
    build: ./orchestrator
    ports:
      - "8000:8000"
    depends_on:
      - rag
      - llm
      - tts
      - video
```

```bash
# 在專案根目錄執行：
docker-compose up --build
```

---

## 二、Supervisor / systemd

如果把每個服務當成系統守護進程（daemon）來跑，可以用：

* **Supervisor**：寫一個 `supervisord.conf`，把 5 個 uvicorn 進程都加進去
* **systemd unit**：為每個服務寫一個 `.service`，最後 `systemctl start rag llm tts video orchestrator`

這種方式適合裸機或 VM 上的微服務管理。

---

## 三、Bash 啟動腳本（簡單直接）

在開發機上可以先做個小腳本：

```bash
#!/usr/bin/env bash
# start_all.sh
uvicorn rag_service:app --port 8001 &
uvicorn llm_service:app --port 8002 &
uvicorn tts_service:app --port 8003 &
uvicorn video_service:app --port 8004 &
uvicorn orchestrator:app --port 8000 &
wait
```

```bash
chmod +x start_all.sh
./start_all.sh
```

> **缺點**：
>
> * 無法自動重啟失敗進程
> * 無法集中管理日誌與監控

---

## 四、在 Python 內程式化啟動（不太建議）

```python
# launch.py
import multiprocessing
import uvicorn

def run(service_module, port):
    uvicorn.run(f"{service_module}:app", host="127.0.0.1", port=port)

if __name__ == "__main__":
    services = [
        ("rag_service",    8001),
        ("llm_service",    8002),
        ("tts_service",    8003),
        ("video_service",  8004),
        ("orchestrator",   8000),
    ]
    procs = []
    for module, port in services:
        p = multiprocessing.Process(target=run, args=(module, port))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
```

> **缺點**：
>
> * 所有 uvicorn workers 都在同一個 Python interpreter 啟動，彼此干擾難除
> * 無法輕易設定不同重啟、日誌策略

---

## 🔑 小結

* **開發測試**：可以先用 Bash 腳本快速啟動
* **正式上線**：強烈建議用 **Docker Compose**（或 Kubernetes）／**Supervisor**／**systemd** 來集中管理、設定重啟策略、日誌收集
* **避免**在業務程式裡硬生生用 `multiprocessing` 去啟動多個 uvicorn

這樣的分層管理不只好維運，也能讓服務更穩定、可觀測。
