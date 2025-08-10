import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))) # Add project root to path
sys.path.append('./')

import argparse
import uuid
import traceback
import subprocess
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 舊程式碼中的主要運算類別
from emogene.realtime.gene_stream import GeneFace2Infer

# --- 1. 初始化 FastAPI 應用和模型 ---

app = FastAPI()
infer_obj = None # 我們將在啟動時載入模型

# 允許所有來源的跨域請求，這樣我們的 HTML 檔案才能呼叫 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 定義 API 的請求模型 ---

class StreamParams(BaseModel):
    # 這裡定義所有來自前端的參數
    blink_mode: str = 'none'
    temperature: float = 0.0
    mouth_amp: float = 0.4
    # ... 您可以從舊的 webui.py 中加入所有需要的滑桿和選項 ...

# --- 3. 建立核心推流函數 (從舊程式碼修改而來) ---

def run_inference_and_stream(params: dict):
    """
    這個函數將在背景執行，不會阻塞 API 回應
    """
    try:
        # 準備輸入參數
        inp = params.copy()
        inp['drv_pose'] = 'nearest'
        samples = infer_obj.prepare_batch_from_inp(inp)
        audio_path = infer_obj.wav16k_name
        stream_key = inp['stream_key']
        rtmp_url = f"rtmp://localhost:19350/live/{stream_key}"
        
        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', '512x512', '-pix_fmt', 'rgb24', '-r', '25', '-i', '-',
            '-i', audio_path, 
            '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
            '-c:a', 'aac', '-ar', '44100', '-f', 'flv',
            rtmp_url
        ]

        print(f"Starting FFmpeg for stream key: {stream_key}")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        frame_count = 0
        for frame in infer_obj.stream_forward_system(samples, inp):
            try:
                process.stdin.write(frame.tobytes())
                frame_count += 1
            except (IOError, BrokenPipeError) as e:
                print(f"FFmpeg process pipe broken for stream {stream_key}: {e}")
                break
        
        process.stdin.close()
        stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
        process.wait()
        
        if process.returncode != 0:
            print(f"FFmpeg Error for stream {stream_key}:\n{stderr_output}")
        else:
            print(f"Stream {stream_key} finished successfully. Total frames: {frame_count}")

    except Exception as e:
        print(f"Inference Error for stream {stream_key}: {e}\n{traceback.format_exc()}")

# --- 4. 建立 API 端點 (Endpoint) ---

@app.post("/start_stream")
async def start_stream(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    blink_mode: str = Form('none'),
    temperature: float = Form(0.0),
    mouth_amp: float = Form(0.4)
    # ... 您可以加入所有需要的 Form 參數 ...
):
    """
    這個 API 端點接收請求，立即返回 stream_key，並在背景開始推流
    """
    try:
        # 產生唯一的串流密鑰
        stream_key = str(uuid.uuid4())
        
        # 儲存上傳的音訊檔案
        temp_audio_path = f"/tmp/{stream_key}_{audio_file.filename}"
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        # 準備傳遞給推流函數的參數字典
        params = {
            'drv_audio_name': temp_audio_path,
            'stream_key': stream_key,
            'blink_mode': blink_mode,
            'temperature': temperature,
            'mouth_amp': mouth_amp,
            # 填入所有固定的或從 Form 接收的參數
            'fp16': False, 'low_memory_usage': False, 'debug': True,
            'a2m_ckpt': infer_obj.a2m_dir,
            'postnet_ckpt': infer_obj.postnet_dir,
            'head_ckpt': infer_obj.head_model_dir,
            'torso_ckpt': infer_obj.torso_model_dir,
            'use_emotalk': infer_obj.use_emotalk,
            'blend_path': "emotalk/render_testing_92.blend",
            'level': 1, 'person': 3, 'output_video': False,
            'bs52_level': 2.0, 'bs_lm_area': 8,
            'lle_percent': 1.0, 'raymarching_end_threshold': 0.01,
        }

        # 將真正的運算作為背景任務執行
        background_tasks.add_task(run_inference_and_stream, params)

        # 立即返回 stream_key 給前端
        return JSONResponse(content={"stream_key": stream_key, "message": "Stream started."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {e}"})

# --- 5. 定義啟動伺服器的程式碼 ---

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    # 從舊程式碼複製所有模型路徑參數
    parser.add_argument("--a2m_ckpt", type=str, default='checkpoints/audio2motion_vae/model_ckpt_steps_400000.ckpt')
    parser.add_argument("--postnet_ckpt", type=str, default='')
    parser.add_argument("--head_ckpt", type=str, default='')
    parser.add_argument("--torso_ckpt", type=str, default='checkpoints/motion2video_nerf/may_torso/model_ckpt_steps_250000.ckpt') 
    parser.add_argument("--port", type=int, default=8000) # FastAPI 預設 8000
    parser.add_argument("--use_emotalk", default=True, action='store_true')
    parser.add_argument("--blend_path", type=str, default='emotalk/render_testing_92.blend')
    args = parser.parse_args()

    print("Loading model, please wait...")
    infer_obj = GeneFace2Infer(
        audio2secc_dir=args.a2m_ckpt,
        postnet_dir=args.postnet_ckpt,
        head_model_dir=args.head_ckpt,
        torso_model_dir=args.torso_ckpt,
        use_emotalk=args.use_emotalk,
        device='cuda:0',
    )
    print("Model loaded successfully.")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    


# #### 第四步：新的運行流程

# 1.  **啟動 Nginx 伺服器** (和之前一樣)
#     在一個終端機中，確保您的本地 Nginx 正在運行：
#     ```bash
#     /home/aaron/nginx/sbin/nginx
#     ```

# 2.  **啟動 FastAPI 後端伺服器**
#     在另一個終端機中，進入 `realtime` 目錄並啟動 `server.py`：
#     ```bash
#     # 啟用您的 conda 環境
#     conda activate geneface 
#     # 進入目錄
#     cd /home/aaron/project/server/models/GeneFacePlusPlus/emogene/realtime/
#     # 啟動伺服器
#     python server.py
#     ```
#     您會看到日誌顯示模型正在載入，然後提示 `Uvicorn running on http://0.0.0.0:8000`。

# 3.  **打開前端網頁**
#     直接用您的瀏覽器打開 `index.html` 檔案。

# 4.  **測試**
#     在網頁上選擇一個音訊檔案，點擊 "Generate Stream"。觀察 "狀態" 區域的變化，並在瀏覽器開發者工具的 "主控台" 和 "網路" 分頁中查看日誌和請求。

# 這個架構讓每一個環節都變得清晰可控，是解決您問題的根本之道。// filepath: /home/aaron/project/server/models/GeneFacePlusPlus/emogene/realtime/server.py