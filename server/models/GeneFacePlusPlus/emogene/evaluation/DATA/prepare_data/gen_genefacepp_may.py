import requests
import os
import shutil
import uuid
from urllib.parse import urljoin
import argparse
import time

SERVER_HOST = "http://127.0.0.1:9001"
API_ENDPOINT = "/generate_video"
SERVER_URL = urljoin(SERVER_HOST, API_ENDPOINT)


def generate_video_from_api(audio_filepath, video_output_filepath):
    """
    呼叫後端 API 來生成影片。
    
    Args:
        audio_filepath (str): 輸入音訊的完整路徑。
        video_output_filepath (str): 期望輸出的影片完整路徑。

    Returns:
        str: 由 API 回傳的生成影片的路徑，或在失敗時返回 None。
    """
    payload = {
        "audio_path": audio_filepath,
        "output_dir": video_output_filepath
    }
    
    print(f"  - 發送請求至 API: {SERVER_URL}")
    # print(f"  - 請求內容: {payload}")

    try:
        # 設定一個較長的超時時間，因為影片生成需要時間
        response = requests.post(SERVER_URL, json=payload, timeout=600)
        response.raise_for_status()  # 如果狀態碼不是 2xx，則引發異常

        result = response.json()
        # print(f"  - 收到 API 回應: {result}")
        
        video_path = result.get("video_path")
        error_message = result.get("error")

        if error_message:
            print(f"  - API 錯誤: {error_message}")
            return None
        
        if video_path and os.path.exists(video_path):
            print(f"  - 影片成功生成於: {video_path}")
            return video_path
        else:
            print(f"  - 錯誤: API 回傳了無效的路徑或檔案不存在: {video_path}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"  - 呼叫 API 時發生錯誤: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="透過 API 為音訊檔案列表批次生成影片。")
    parser.add_argument(
        "--audio_list_file", 
        type=str, 
        default="/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/prepare_data/RAVDESS_file_list.txt",
        # default="/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/prepare_data/test.txt",
        help="包含音訊檔案路徑列表的文字檔案路徑。"
    )
    parser.add_argument(
        "--output_base_dir", 
        type=str, 
        default="/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/genefacepp_ver/RAVDESS",
        help="儲存輸出影片的根目錄。"
    )
    args = parser.parse_args()

    # 檢查列表檔案是否存在
    if not os.path.exists(args.audio_list_file):
        print(f"錯誤：找不到音訊列表檔案: {args.audio_list_file}")
        exit(1)

    # 讀取音訊檔案列表
    with open(args.audio_list_file, 'r') as f:
        audio_files = [line.strip() for line in f if line.strip()]

    total_files = len(audio_files)
    print(f"找到 {total_files} 個音訊檔案，準備開始處理...")
    print("==================================================")

    # 逐一處理每個音訊檔案
    for i, audio_path in enumerate(audio_files):
        start_time = time.time()
        print(f"正在處理第 {i+1}/{total_files} 個檔案: {os.path.basename(audio_path)}")

        if not os.path.exists(audio_path):
            print(f"  - 警告：找不到音訊檔案，跳過。路徑: {audio_path}")
            continue

        try:
            # 根據輸入音訊路徑，建立對應的輸出路徑
            audio_filename = os.path.basename(audio_path)
            video_filename = audio_filename.replace('.wav', '.mp4')
            
            # 從原始路徑中提取 'Actor_XX' 這層目錄
            path_parts = audio_path.split(os.sep)
            if 'RAVDESS' in path_parts:
                # 找到 RAVDESS 後面的 Actor 目錄
                actor_dir_index = path_parts.index('RAVDESS') + 1
                actor_dir = path_parts[actor_dir_index]
                output_mid_dir = os.path.join(args.output_base_dir, actor_dir)
            else:
                # 如果路徑結構不同，則直接放在根目錄下
                output_mid_dir = args.output_base_dir

            final_output_path = os.path.join(output_mid_dir, video_filename)

            # 如果影片已存在，則跳過
            if os.path.exists(final_output_path):
                print(f"  - 影片已存在，跳過。路徑: {final_output_path}")
                continue

            # 確保輸出目錄存在
            os.makedirs(output_mid_dir, exist_ok=True)
            
            # 呼叫 API
            generate_video_from_api(audio_path, final_output_path)

        except Exception as e:
            print(f"  - 處理檔案 {audio_path} 時發生未預期的錯誤: {e}")
        
        end_time = time.time()
        print(f"  - 耗時: {end_time - start_time:.2f} 秒")
        print("--------------------------------------------------")


    print("所有音訊檔案處理完畢！")