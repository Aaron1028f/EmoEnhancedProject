import os
import sys
import traceback
import uuid
import asyncio
import threading
from typing import Generator

# --- 環境設定與路徑 ---
now_dir = os.getcwd()
sys.path.append(now_dir)
# 確保可以找到 indextts 模組
if os.path.basename(now_dir) != 'index-tts':
    # 如果不是在 index-tts 目錄下執行，需要調整路徑
    # 假設 api_indextts.py 在 index-tts 目錄下
    sys.path.append(os.path.join(now_dir, "indextts"))

import argparse
import numpy as np
import soundfile as sf
import httpx
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from pydantic import BaseModel, Field

import wave


# --- 從您的串流版本 infer 檔案中匯入 ---
# 確保您的檔案名為 infer_v2_streaming.py
from indextts.infer_v2_streaming import IndexTTS2

# --- 全域變數與初始化 ---
parser = argparse.ArgumentParser(description="IndexTTS API")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--port", type=int, default=40000, help="Port to run the API on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference")

cmd_args = parser.parse_args()

print("Initializing IndexTTS2 model...")
tts_pipeline = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
)
print("IndexTTS2 model initialized.")

# --- FastAPI 應用設定 ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 請求模型 ---
class TTSRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    spk_audio_prompt: str = Field(..., description="音色參考音訊的路徑")
    emo_audio_prompt: str | None = Field(None, description="情感參考音訊的路徑")
    emo_alpha: float = Field(1.0, description="情感混合比例")
    emo_vector: list[float] | None = Field(None, description="情感向量")
    use_emo_text: bool = Field(False, description="是否使用文本生成情感向量")
    emo_text: str | None = Field(None, description="用於生成情感向量的文本")
    interval_silence: int = Field(200, description="片段間靜音時長 (ms)")
    max_text_tokens_per_segment: int = Field(120, description="每段最大文本 token 數")
    
    # 控制項
    streaming_mode: bool = Field(False, description="是否使用串流模式回傳")
    media_type: str = Field("wav", description="音訊格式 (wav, raw)")

# --- Webhook 通知相關 (類似 api_v2_lk_save_wav.py) ---
AVATAR_SAVE_DIR = "/home/aaron/project/server/models/GeneFacePlusPlus/emogene/DATA/temp"
AVATAR_WEBHOOK_URL = "http://127.0.0.1:31000/generate_full_video"
os.makedirs(AVATAR_SAVE_DIR, exist_ok=True)

def _save_wav(np_data: np.ndarray, sr: int) -> str:
    """將 numpy 音訊資料存成 wav，回傳檔案路徑。"""
    out_path = os.path.join(AVATAR_SAVE_DIR, f"{uuid.uuid4().hex}_{sr}.wav")
    sf.write(str(out_path), np_data, sr, subtype="PCM_16")
    return str(out_path)

def notify_emogene_fire_and_forget(wav_path: str):
    """在背景執行緒發送回呼，不等待回傳。"""
    payload = {"audio_path": wav_path}

    def _worker():
        try:
            with httpx.Client(timeout=10.0) as client:
                client.post(AVATAR_WEBHOOK_URL, json=payload)
                print(f"Successfully notified EmoGene with path: {wav_path}")
        except Exception:
            print(f"Failed to notify EmoGene for path: {wav_path}")
            traceback.print_exc()

    threading.Thread(target=_worker, daemon=True).start()

# --- 音訊處理函式 ---
def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    sf.write(io_buffer, data, rate, format="wav", subtype="PCM_16")
    return io_buffer

def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    # 使用 soundfile 直接寫入 ogg 格式 (通常使用 vorbis 編碼)
    sf.write(io_buffer, data, rate, format="ogg", subtype="VORBIS")
    return io_buffer

def pack_raw(io_buffer: BytesIO, data: np.ndarray):
    io_buffer.write(data.tobytes())
    return io_buffer

# 新增：pack_audio 函式 (從 api_v2_lk_save_wav.py 借鑒)
def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "wav":
        # 注意：串流 wav 時，我們只在第一塊加 header，後續應為 raw
        # 這個邏輯會在 streaming_generator 中處理
        io_buffer = pack_raw(io_buffer, data)
    else: # raw
        io_buffer = pack_raw(io_buffer, data)
    
    io_buffer.seek(0)
    return io_buffer


def wave_header_chunk(sample_rate=22050, channels=1, sample_width=2):
    """生成 WAV 檔案頭"""
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
    wav_buf.seek(0)
    return wav_buf.read()

# --- 核心 TTS 處理邏輯 ---
async def tts_handle(req: dict):
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    # 準備 tts_pipeline.infer 的參數
    infer_params = req.copy()
    # 移除 API 控制參數
    infer_params.pop("streaming_mode", None)
    infer_params.pop("media_type", None)
    # 串流模式下 output_path 必須為 None
    infer_params["output_path"] = None

    try:
        tts_generator = tts_pipeline.infer(**infer_params)

        if streaming_mode:
            def streaming_generator(generator: Generator, requested_media_type: str):
                sr_for_save = None
                accum_np = []
                is_first_chunk = True
                
                # 內部 media_type，用於在 wav 串流時切換
                current_media_type = requested_media_type

                try:
                    for sr, chunk_np in generator:
                        if sr_for_save is None:
                            sr_for_save = sr
                        
                        accum_np.append(chunk_np)

                        if is_first_chunk and media_type == "wav":
                            yield wave_header_chunk(sample_rate=sr)
                            current_media_type = "raw"  # 後續皆為 raw
                            is_first_chunk = False
                        
                        # yield chunk_np.tobytes()
                        # 使用 pack_audio 對每一塊進行編碼
                        yield pack_audio(BytesIO(), chunk_np, sr, current_media_type).getvalue()
                finally:
                    # 串流結束後，合併、存檔、通知
                    if accum_np and sr_for_save:
                        try:
                            full_np = np.concatenate(accum_np, axis=0)
                            wav_path = _save_wav(full_np, sr_for_save)
                            # notify_emogene_fire_and_forget(wav_path)
                        except Exception:
                            traceback.print_exc()
            # 設定正確的 Content-Type
            response_media_type = f"audio/{media_type}"
            if media_type == "raw":
                # 確保客戶端知道這是 raw pcm
                response_media_type = "audio/raw"                            
            return StreamingResponse(streaming_generator(tts_generator, media_type), media_type=response_media_type)

            # return StreamingResponse(streaming_generator(tts_generator), media_type="audio/raw")

        else: # 非串流模式
            # 因為 infer 已被改為生成器，我們需要迭代它來獲取結果
            sr = None
            all_chunks = []
            for current_sr, chunk_np in tts_generator:
                if sr is None:
                    sr = current_sr
                all_chunks.append(chunk_np)
            
            if not all_chunks:
                return JSONResponse(status_code=400, content={"message": "TTS failed to generate any audio"})

            # 合併所有片段
            full_audio_np = np.concatenate(all_chunks, axis=0)
            
            # 存檔並通知 Webhook
            try:
                wav_path = _save_wav(full_audio_np, sr)
                # notify_emogene_fire_and_forget(wav_path)
            except Exception:
                traceback.print_exc()

            # 根據請求的 media_type 回傳
            io_buffer = BytesIO()
            if media_type == "wav":
                pack_wav(io_buffer, full_audio_np, sr)
                response_media_type = "audio/wav"
            elif media_type == "ogg":
                pack_ogg(io_buffer, full_audio_np, sr)
                response_media_type = "audio/ogg"
            else: # raw
                pack_raw(io_buffer, full_audio_np)
                response_media_type = "audio/raw"
            
            io_buffer.seek(0)
            return Response(io_buffer.getvalue(), media_type=response_media_type)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "TTS failed", "exception": str(e)})

# --- FastAPI 端點 ---
@app.post("/tts")
async def tts_post_endpoint(request: TTSRequest):
    return await tts_handle(request.dict())

@app.get("/tts")
async def tts_get_endpoint(request: Request):
    try:
        # 將 GET 請求的查詢參數轉換為 Pydantic 模型
        req_model = TTSRequest(**request.query_params)
        return await tts_handle(req_model.dict())
    except Exception as e:
        # 處理參數驗證失敗等問題
        return JSONResponse(status_code=400, content={"message": "Invalid parameters", "detail": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port, workers=1)

# ```

# #### 如何執行

# 1.  將您修改過的 `infer_v2.py` 重新命名為 `infer_v2_streaming.py`。
# 2.  將上面的程式碼儲存為 `api_indextts.py`。
# 3.  在您的 `indextts` conda 環境中，執行以下指令：

#     ```bash
#     # 進入正確的目錄
#     cd /home/aaron/project/server/models/TTS/index-tts/

#     # 啟動 API 伺服器
#     # 您可以根據需要加上 --deepspeed, --fp16 等參數
#     uvicorn api_indextts:app --host 0.0.0.0 --port 40000
#     ```
#     或者使用您原本的指令：
#     ```bash
#     CUDA_VISIBLE_DEVICES=0 uv run api_indextts.py --port 40000 --deepspeed
#     ```