# step1: open the video server
# cd server/models/GeneFacePlusPlus/emogene
# uvicorn api_emogene:app --port 9000 --host 0.0

# import gradio as gr
import requests
import os
import shutil
import uuid
from urllib.parse import urljoin

# --- 設定 ---
# 你的 API 伺服器運行的位址和端口
SERVER_HOST = "http://127.0.0.1:9000"
# API 端點
API_ENDPOINT = "/generate_video"
# 完整的 API URL
SERVER_URL = urljoin(SERVER_HOST, API_ENDPOINT)


def generate_video_from_api(audio_filepath):
    """
    呼叫後端 API 來生成影片。
    
    Args:
        audio_filepath (str): Gradio 提供的使用者上傳音訊的暫存路徑。

    Returns:
        str: 由 API 回傳的生成影片的路徑，或在失敗時返回 None。
    """


    print(f"客戶端：接收到暫存音訊檔位於 {audio_filepath}")

    
    # 2. 準備發送到 API 的 JSON 資料
    payload = {
        "audio_path": audio_filepath,
        "mouth_amp": 0.5 # 你可以在這裡或UI上提供更多參數
    }
    
    print(f"客戶端：正在向 {SERVER_URL} 發送請求...")
    print(f"客戶端：請求內容 (Payload): {payload}")

    # 3. 呼叫 API
    # 設定一個較長的超時時間，因為影片生成需要時間
    response = requests.post(SERVER_URL, json=payload, timeout=600)
    response.raise_for_status()  # 如果狀態碼不是 2xx，則引發異常

    # 4. 解析 API 回應
    result = response.json()
    print(f"客戶端：收到來自伺服器的回應: {result}")
    
    video_path = result.get("video_path")
    error_message = result.get("error")

    print(f"客戶端：成功取得影片路徑: {video_path}")
    
    
    # convert relative path to absolute path
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.getcwd(), video_path)
    return video_path


if __name__ == "__main__":
    # 測試用的音訊檔案路徑
    test_audio_path = "/home/ykwei/project/server/services/demo_audio.wav"

    if os.path.exists(test_audio_path):
        video_path = generate_video_from_api(test_audio_path)
        print(f"生成的影片路徑: {video_path}")
    else:
        print(f"找不到測試音訊檔案: {test_audio_path}")