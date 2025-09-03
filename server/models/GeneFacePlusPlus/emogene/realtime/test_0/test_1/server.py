import os, sys
sys.path.append('./')

import argparse
import uuid
import traceback
import subprocess
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import asyncio
import time
import httpx

from server.models.GeneFacePlusPlus.emogene.realtime.emogene_stream import GeneFace2Infer

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

@app.get("/")
async def read_index():
    # [修改] 使用絕對路徑來定位 index.html
    # 這可以確保無論您從哪個目錄執行 server.py，都能找到檔案
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, 'index.html')
    
    if not os.path.exists(index_path):
        return JSONResponse(
            status_code=404, 
            content={"message": f"Error: index.html not found at path: {index_path}"}
        )
        
    return FileResponse(index_path)

class StreamParams(BaseModel):
    # 這裡定義所有來自前端的參數
    blink_mode: str = 'none'
    temperature: float = 0.0
    mouth_amp: float = 0.4

# 新增：HLS HTTP 基底（可用環境變數覆蓋）
HLS_HTTP_BASE = os.getenv("HLS_HTTP_BASE", "http://127.0.0.1:11842/hls")

async def wait_until_manifest_ready(stream_key: str, timeout_sec: float = 30.0, interval_sec: float = 0.5) -> str:
    """
    輪詢 HLS m3u8 是否可用（HTTP 200 且內容為 #EXTM3U），回傳完整 hls_url。
    """
    url = f"{HLS_HTTP_BASE}/{stream_key}.m3u8"
    start = time.time()
    headers = {"Cache-Control": "no-cache", "Pragma": "no-cache"}
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                # 加上時間參數避免快取
                r = await client.get(url, headers=headers, params={"t": int(time.time() * 1000)})
                if r.status_code == 200 and r.text.startswith("#EXTM3U"):
                    return url
            except Exception:
                pass
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Timeout waiting for HLS manifest: {url}")
            await asyncio.sleep(interval_sec)

def pump_frames_sync(infer_obj: GeneFace2Infer, samples, inp, process) -> int:
    """
    同步函式：連續把影格寫入 ffmpeg stdin，直到完成或管線中斷。
    """
    frame_count = 0
    try:
        for frame in infer_obj.stream_forward_system(samples, inp):
            try:
                process.stdin.write(frame.tobytes())
                frame_count += 1
            except (IOError, BrokenPipeError):
                break
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass
    return frame_count


async def run_inference_and_stream(params: dict, websocket: WebSocket):
    """
    背景推流：啟動 ffmpeg -> 併發送幀 -> 等 m3u8 可用 -> 通知前端 ready。
    """
    stream_key = params.get('stream_key', 'unknown')
    try:
        inp = params.copy()
        inp['drv_pose'] = 'nearest'
        samples = infer_obj.prepare_batch_from_inp(inp)
        audio_path = infer_obj.wav16k_name
        stream_key = inp['stream_key']
        rtmp_url = f"rtmp://localhost:19350/live/{stream_key}"

        command = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', '512x512', '-pix_fmt', 'rgb24', '-r', '25', '-i', '-',
            '-i', audio_path,
            '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
            '-c:a', 'aac', '-ar', '44100',
            '-f', 'flv', rtmp_url
        ]
        print(f"Starting FFmpeg for stream key: {stream_key}")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # 併發：一邊寫入影格，一邊等待 m3u8 就緒
        pump_task = asyncio.create_task(asyncio.to_thread(pump_frames_sync, infer_obj, samples, inp, process))

        # 等待 m3u8 可用後再通知前端
        try:
            hls_url = await wait_until_manifest_ready(stream_key, timeout_sec=30.0, interval_sec=0.5)
            print(f"HLS manifest is ready: {hls_url}")
            await websocket.send_json({"status": "ready", "stream_key": stream_key, "hls_url": hls_url})
        except TimeoutError as te:
            msg = str(te)
            print(msg)
            await websocket.send_json({"status": "error", "message": msg})
            try:
                process.terminate()
            except Exception:
                pass
            return

        # 等待推流完成
        frame_count = await pump_task
        stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
        process.wait()

        if process.returncode != 0:
            print(f"FFmpeg Error for stream {stream_key}:\n{stderr_output}")
        else:
            print(f"Stream {stream_key} finished successfully. Total frames: {frame_count}")

    except Exception as e:
        print(f"Inference Error for stream {stream_key}: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        print(f"Closing WebSocket for stream {stream_key}")
        try:
            await websocket.close()
        except Exception:
            pass


# just for testing
def run_test_inference(params: dict):
    """
    這個函數只會執行一幀的推論，用於測試模型和API是否正常，不進行推流。
    """
    try:
        print("--- Starting Test Inference ---")
        inp = params.copy()
        inp['drv_pose'] = 'nearest'
        samples = infer_obj.prepare_batch_from_inp(inp)
        
        # 只取第一幀進行測試
        frame_generator = infer_obj.stream_forward_system(samples, inp)
        first_frame = next(frame_generator)
        
        if first_frame is not None:
            print(f"Test inference successful. First frame generated with shape: {first_frame.shape}")
            return True
        else:
            print("Test inference failed: No frame was generated.")
            return False

    except Exception as e:
        print(f"Test Inference Error: {e}\n{traceback.format_exc()}")
        return False

@app.post("/test_inference")
async def test_inference(
    audio_file: UploadFile = File(...)
):
    """
    這個 API 端點只執行一次推論，不推流，用於快速測試。
    """
    try:
        # 儲存上傳的音訊檔案
        temp_audio_path = f"/tmp/test_{uuid.uuid4()}_{audio_file.filename}"
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        # 準備參數字典 (可以簡化，因為很多參數與推流無關)
        params = {
            'drv_audio_name': temp_audio_path,
            'blink_mode': 'none', 'temperature': 0.0, 'mouth_amp': 0.4,
            'fp16': False, 'low_memory_usage': False, 'debug': True,
            'a2m_ckpt': infer_obj.audio2secc_dir,
            'postnet_ckpt': infer_obj.postnet_dir,
            'head_ckpt': infer_obj.head_model_dir,
            'torso_ckpt': infer_obj.torso_model_dir,
            'use_emotalk': infer_obj.use_emotalk,
            'blend_path': "emotalk/render_testing_92.blend",
            'level': 1, 'person': 3, 'output_video': False,
            'bs52_level': 2.0, 'bs_lm_area': 8,
            'lle_percent': 1.0, 'raymarching_end_threshold': 0.01,
        }

        # 直接呼叫測試函數
        success = run_test_inference(params)

        if success:
            return JSONResponse(content={"message": "Test inference successful. Check server console for details."})
        else:
            return JSONResponse(status_code=500, content={"message": "Test inference failed. Check server console for details."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error in test endpoint: {e}"})


# --- 4. 建立 API 端點 (Endpoint) ---

# @app.post("/start_stream")
# async def start_stream(
#     background_tasks: BackgroundTasks,
#     audio_file: UploadFile = File(...),
#     blink_mode: str = Form('none'),
#     temperature: float = Form(0.0),
#     mouth_amp: float = Form(0.4)
#     # ... 您可以加入所有需要的 Form 參數 ...
# ):
#     """
#     這個 API 端點接收請求，立即返回 stream_key，並在背景開始推流
#     """
#     try:
#         # 產生唯一的串流密鑰
#         stream_key = str(uuid.uuid4())
        
#         # 儲存上傳的音訊檔案
#         temp_audio_path = f"/tmp/{stream_key}_{audio_file.filename}"
#         with open(temp_audio_path, "wb") as buffer:
#             buffer.write(await audio_file.read())

#         # 準備傳遞給推流函數的參數字典
#         params = {
#             'drv_audio_name': temp_audio_path,
#             'stream_key': stream_key,
#             'blink_mode': blink_mode,
#             'temperature': temperature,
#             'mouth_amp': mouth_amp,
#             # 填入所有固定的或從 Form 接收的參數
#             'fp16': False, 'low_memory_usage': False, 'debug': True,
#             'a2m_ckpt': infer_obj.audio2secc_dir,
#             'postnet_ckpt': infer_obj.postnet_dir,
#             'head_ckpt': infer_obj.head_model_dir,
#             'torso_ckpt': infer_obj.torso_model_dir,
#             'use_emotalk': infer_obj.use_emotalk,
#             'blend_path': "emotalk/render_testing_92.blend",
#             'level': 1, 'person': 3, 'output_video': False,
#             'bs52_level': 2.0, 'bs_lm_area': 8,
#             'lle_percent': 1.0, 'raymarching_end_threshold': 0.01,
#         }

#         # 將真正的運算作為背景任務執行
#         background_tasks.add_task(run_inference_and_stream, params)

#         # 立即返回 stream_key 給前端
#         return JSONResponse(content={"stream_key": stream_key, "message": "Stream started."})

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"message": f"Error: {e}"})


# [修改] start_stream 端點，現在它只負責儲存檔案和返回 stream_key
@app.post("/start_stream")
async def start_stream(
    audio_file: UploadFile = File(...),
    blink_mode: str = Form('none'),
    temperature: float = Form(0.0),
    mouth_amp: float = Form(0.4)
):
    try:
        stream_key = str(uuid.uuid4())
        temp_audio_path = f"/tmp/{stream_key}_{audio_file.filename}"
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        # 只返回 stream_key 和音訊路徑，真正的推流由 WebSocket 觸發
        return JSONResponse(content={
            "stream_key": stream_key, 
            "audio_path": temp_audio_path,
            "message": "Stream key generated. Connect to WebSocket to start processing."
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {e}"})

# [新增] WebSocket 端點
@app.websocket("/ws/{stream_key}")
async def websocket_endpoint(websocket: WebSocket, stream_key: str):
    await websocket.accept()
    try:
        # 等待前端發送包含所有參數的訊息
        initial_params = await websocket.receive_json()
        
        # 準備傳遞給推流函數的參數字典
        params = {
            'drv_audio_name': initial_params['audio_path'],
            'stream_key': stream_key,
            'blink_mode': initial_params['blink_mode'],
            'temperature': initial_params['temperature'],
            'mouth_amp': initial_params['mouth_amp'],
            # ... (其他參數不變) ...
            'fp16': False, 'low_memory_usage': False, 'debug': True,
            'a2m_ckpt': infer_obj.audio2secc_dir,
            'postnet_ckpt': infer_obj.postnet_dir,
            'head_ckpt': infer_obj.head_model_dir,
            'torso_ckpt': infer_obj.torso_model_dir,
            'use_emotalk': infer_obj.use_emotalk,
            'blend_path': "emotalk/render_testing_92.blend",
            'level': 1, 'person': 3, 'output_video': False,
            'bs52_level': 2.0, 'bs_lm_area': 8,
            'lle_percent': 1.0, 'raymarching_end_threshold': 0.01,
        }

        # [修改] 直接 await 背景任務，這會保持連線直到任務完成
        await run_inference_and_stream(params, websocket)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for stream {stream_key}")
    except Exception as e:
        print(f"WebSocket Error for stream {stream_key}: {e}")
        # 確保在出錯時也關閉連線
        if not websocket.client_state == 'DISCONNECTED':
            await websocket.close()

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    
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