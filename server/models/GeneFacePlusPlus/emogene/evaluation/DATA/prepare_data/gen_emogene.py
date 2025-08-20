import requests
import os
import shutil
import uuid
from urllib.parse import urljoin

import argparse

SERVER_HOST = "http://127.0.0.1:9000"
API_ENDPOINT = "/generate_video"
SERVER_URL = urljoin(SERVER_HOST, API_ENDPOINT)


def generate_video_from_api(audio_filepath, video_output_filepath):
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
        "mouth_amp": 0.5,  # 你可以在這裡或UI上提供更多參數
        "output_dir": video_output_filepath
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
    parser = argparse.ArgumentParser(description="Generate video from audio using API")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file")
    # ex: "/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
    
    parser.add_argument("--output_base_dir", type=str, required=True, help="Directory to save the output video file")
    # ex: "/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/emogene_ver"

    args = parser.parse_args()

    # get the output file name and path
    abs_audio_path = args.audio_path
    abs_output_base_dir = args.output_base_dir
    
    # testing
    # abs_audio_path = "/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
    # abs_output_base_dir = "/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/emogene_ver"

    audio_filename = os.path.basename(abs_audio_path) # ex: 03-01-01-01-01-01-01.wav
    video_filename = audio_filename.replace('.wav', '.mp4') # ex: 03-01-01-01-01-01-01.mp4
    output_mid_dir = os.path.join(abs_output_base_dir, *abs_audio_path.split(os.sep)[-3:-1])
    final_output_path = os.path.join(output_mid_dir, video_filename)
    
    # input args
    print(f'audio input file path: {abs_audio_path}')
    print(f'video output file path: {final_output_path}')
    
    # run
    if os.path.exists(abs_audio_path):
        # make sure output directory exists
        os.makedirs(output_mid_dir, exist_ok=True)
        
        video_path = generate_video_from_api(abs_audio_path, final_output_path)
        print(f"Generated video path: {video_path}")
    else:
        print(f"Audio input file not found: {abs_audio_path}")

    # test_audio_path = "/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
    # # get audio file name
    # audio_filename = os.path.basename(test_audio_path)
    # video_filename = audio_filename.replace('.wav', '.mp4')
    # test_output_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/temp'
    # test_output_path = os.path.join(test_output_path, video_filename)

    # if os.path.exists(test_audio_path):
    #     video_path = generate_video_from_api(test_audio_path, test_output_path)
    #     print(f"生成的影片路徑: {video_path}")
    # else:
    #     print(f"找不到測試音訊檔案: {test_audio_path}")