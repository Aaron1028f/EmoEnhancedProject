# pip install py-feat opencv-python
import cv2
from feat import Detector
from collections import Counter
import os

def evaluate_video_emotion_pyfeat(video_path: str, ground_truth_emotion: str):
    """
    使用 Py-Feat 分析影片中的主要情感，並與標準答案比對。
    
    :param video_path: 影片檔案的路徑
    :param ground_truth_emotion: 您期望影片表現出的情感 (e.g., 'happy', 'sad', 'neutral')
    """
    if not os.path.exists(video_path):
        print(f"錯誤：影片檔案不存在於 {video_path}")
        return

    # 初始化 Detector，可以指定人臉模型和情感模型
    # 將模型物件放在迴圈外，避免重複載入
    detector = Detector(
        face_model="retinaface",
        emotion_model="resmasknet",
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {video_path}")
        return

    emotion_predictions = []
    frame_count = 0
    
    print("開始分析影片 (使用 Py-Feat)，請稍候...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 同樣可以設定隔幾幀分析一次
        if frame_count % 5 != 0:
            continue

        try:
            # 對單張圖片進行偵測
            # Py-Feat 的 detect_image 回傳一個 Fex object
            detected_faces = detector.detect_image(frame)
            
            # 如果偵測到人臉
            if not detected_faces.empty:
                # 取得情感分析結果 (是一個 DataFrame)
                emotion_df = detected_faces.emotions
                # 取得分數最高的情感作為主要情感
                dominant_emotion = emotion_df.idxmax(axis=1).iloc[0]
                emotion_predictions.append(dominant_emotion.lower()) # 轉為小寫以統一格式

        except Exception as e:
            # print(f"在第 {frame_count} 幀發生錯誤: {e}")
            pass
            
    cap.release()
    print("影片分析完畢。")
    
    if not emotion_predictions:
        print("警告：在整部影片中都未能成功偵測到人臉並分析情感。")
        return

    most_common_emotion = Counter(emotion_predictions).most_common(1)[0][0]
    correct_frames = emotion_predictions.count(ground_truth_emotion)
    total_analyzed_frames = len(emotion_predictions)
    accuracy = (correct_frames / total_analyzed_frames) * 100 if total_analyzed_frames > 0 else 0

    print("\n--- Py-Feat 情感評估報告 ---")
    print(f"影片路徑: {video_path}")
    print(f"標準答案情感: {ground_truth_emotion}")
    print(f"總共分析的幀數: {total_analyzed_frames}")
    print(f"偵測到的主要情感: {most_common_emotion}")
    print(f"情感準確率: {accuracy:.2f}% ({correct_frames} / {total_analyzed_frames} 幀符合)")
    print("------------------------------")


# --- 使用範例 ---
video_file = 'generated_happy_video.mp4'
expected_emotion = 'happy'
evaluate_video_emotion_pyfeat(video_file, expected_emotion)