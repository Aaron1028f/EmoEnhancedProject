# pip install deepface opencv-python

import cv2
from deepface import DeepFace
from collections import Counter
import os

def evaluate_video_emotion_deepface(video_path: str, ground_truth_emotion: str):
    """
    使用 DeepFace 分析影片中的主要情感，並與標準答案比對。

    :param video_path: 影片檔案的路徑
    :param ground_truth_emotion: 您期望影片表現出的情感 (e.g., 'happy', 'sad', 'neutral')
    """
    if not os.path.exists(video_path):
        print(f"錯誤：影片檔案不存在於 {video_path}")
        return

    # 1. 載入影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {video_path}")
        return

    emotion_predictions = []
    frame_count = 0

    print("開始分析影片，請稍候...")
    # 2. 逐幀迴圈
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 影片讀取完畢

        frame_count += 1
        # 每隔幾幀分析一次以加速，例如每5幀
        if frame_count % 5 != 0:
            continue
            
        try:
            # 3. 對當前幀進行情感分析
            # backend 設為 'ssd' 或 'mtcnn' 較快
            analysis = DeepFace.analyze(
                img_path=frame, 
                actions=['emotion'], 
                enforce_detection=True, # 確認偵測到人臉
                detector_backend='ssd' 
            )
            
            # DeepFace 回傳的是一個包含字典的列表
            dominant_emotion = analysis[0]['dominant_emotion']
            emotion_predictions.append(dominant_emotion)

        except Exception as e:
            # 4. 處理偵測不到人臉的幀
            # print(f"在第 {frame_count} 幀未偵測到人臉或發生錯誤: {e}")
            pass
            
    cap.release()
    print("影片分析完畢。")

    # 5. 匯總結果
    if not emotion_predictions:
        print("警告：在整部影片中都未能成功偵測到人臉並分析情感。")
        return

    # 計算出現最頻繁的情感
    most_common_emotion = Counter(emotion_predictions).most_common(1)[0][0]
    
    # 計算符合 ground_truth 的幀數比例 (準確率)
    correct_frames = emotion_predictions.count(ground_truth_emotion)
    total_analyzed_frames = len(emotion_predictions)
    accuracy = (correct_frames / total_analyzed_frames) * 100 if total_analyzed_frames > 0 else 0

    # 6. 輸出評估報告
    print("\n--- DeepFace 情感評估報告 ---")
    print(f"影片路徑: {video_path}")
    print(f"標準答案情感: {ground_truth_emotion}")
    print(f"總共分析的幀數: {total_analyzed_frames}")
    print(f"偵測到的主要情感: {most_common_emotion}")
    print(f"情感準確率: {accuracy:.2f}% ({correct_frames} / {total_analyzed_frames} 幀符合)")
    print("---------------------------------")


# --- 使用範例 ---
# 假設你有一個應該是快樂表情的影片 'generated_happy_video.mp4'
# video_file = 'generated_happy_video.mp4'
# expected_emotion = 'happy'

video_file = 'datas/May/tmp.mp4'
expected_emotion = 'sad'

evaluate_video_emotion_deepface(video_file, expected_emotion)