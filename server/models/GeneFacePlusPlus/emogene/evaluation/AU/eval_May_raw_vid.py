from feat import Detector
import os
import pandas as pd

# --- 1. 設定 ---
# 初始化檢測器
detector = Detector(device='cuda')

# 設定您要分析的單一影片路徑
VIDEO_PATH = '/home/aaron/project/server/models/GeneFacePlusPlus/data/raw/videos/May.mp4'
# 設定儲存分析結果的目錄
SAVE_BASE_PATH = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/AU/figures/'
if not os.path.exists(SAVE_BASE_PATH):
    os.makedirs(SAVE_BASE_PATH)


# --- 2. 核心函式 ---
def detect_video(video_path):
    """對單一影片執行 py-feat 偵測。"""
    print(f"Analyzing video: {video_path}...")
    # skip_frames=1000 是一個合理的設定，可以加快處理速度，同時保留足夠的細節
    video_prediction = detector.detect_video(
        video_path, data_type="video", skip_frames=24, face_detection_threshold=0.95
    )
    print("Analysis complete.")
    return video_prediction

def flatten_series_into_dict(results_dict, prefix, series):
    """一個輔助函式，將 Series 的內容攤平後加入字典中。"""
    if series is not None:
        for idx, val in series.items():
            results_dict[f"{prefix}_{idx}"] = val

# def print_single_video_report(stats, video_filename):
#     """
#     為單一影片的統計數據印出格式化的分析報告。
#     """
#     # 定義我們關心的每個情緒類別下的關鍵 AU
#     EMOTION_KEY_AUS = {
#         'happiness': [6, 12], 'sadness': [1, 4, 15], 'anger': [4, 5, 7, 23],
#         'fear': [1, 2, 4, 5, 7, 20, 26], 'disgust': [9, 15], 'surprise': [1, 2, 5, 26]
#     }
#     emotions = list(EMOTION_KEY_AUS.keys())

#     # --- 表格格式設定 ---
#     left_col_width = 18
#     col_width = 9
#     total_width = left_col_width
#     for emotion in emotions:
#         total_width += len(EMOTION_KEY_AUS[emotion]) * col_width + 1

#     # --- 建立報告字串 ---
#     report = f"\n{'='*total_width}\n"
#     report += f"{'Single Video Facial Analysis Report':^{total_width}}\n"
#     report += f"Video File: {video_filename:^{total_width-12}}\n"
#     report += f"{'='*total_width}\n\n"

#     # --- 標頭 ---
#     header_line = f"{'Metric':<{left_col_width}}"
#     au_line = f"{'Key AU':<{left_col_width}}"
#     for emotion in emotions:
#         key_aus = EMOTION_KEY_AUS[emotion]
#         emotion_width = len(key_aus) * col_width
#         header_line += f"|{emotion.capitalize():^{emotion_width}}"
#         au_header = "".join([f"{'AU'+str(au):<{col_width}}" for au in key_aus])
#         au_line += f"|{au_header}"
#     report += f"{header_line}\n{au_line}\n{'-'*total_width}\n"

#     # --- 逐一印出每個指標的數據 ---
#     metrics_to_print = {
#         "Emotion Mean": "emotion_mean", "Emotion Std": "emotion_std",
#         "Emotion Min": "emotion_min", "Emotion Max": "emotion_max",
#         "AU Mean": "au_mean", "AU Std": "au_std",
#         "AU Min": "au_min", "AU Max": "au_max",
#     }

#     for label, key_prefix in metrics_to_print.items():
#         line = f"{label:<{left_col_width}}"
#         for emotion in emotions:
#             line += "|"
            
#             # --- 邏輯修正 ---
#             if "Emotion" in label:
#                 # 對於情緒指標，直接使用外層迴圈的 emotion
#                 col_name = f"{key_prefix}_{emotion}"
#                 val = stats.get(col_name, 0)
#                 emotion_width = len(EMOTION_KEY_AUS[emotion]) * col_width
#                 line += f"{val:^{emotion_width}.3f}"
#             else: # "AU" in label
#                 # 對於 AU 指標，遍歷該情緒下的關鍵 AU
#                 for au_num in EMOTION_KEY_AUS[emotion]:
#                     col_name = f"{key_prefix}_AU{au_num:02d}"
#                     val = stats.get(col_name, 0)
#                     line += f"{val:<{col_width}.3f}"

#         report += f"{line}\n"
#         # 在每個大類別後增加分隔線
#         if "Max" in label and "Emotion" in label:
#              report += "\n" # 在 Emotion Max 後空一行
#         elif "Max" in label and "AU" in label:
#             report += f"{'-'*total_width}\n"

#     report += f"{'='*total_width}\n"
#     print(report)

def print_single_video_report(stats, video_filename):
    """
    為單一影片的統計數據印出格式化的分析報告。
    """
    # 定義我們關心的每個情緒類別下的關鍵 AU
    EMOTION_KEY_AUS = {
        'happiness': [6, 12], 'sadness': [1, 4, 15], 'anger': [4, 5, 7, 23],
        'fear': [1, 2, 4, 5, 7, 20, 26], 'disgust': [9, 15], 'surprise': [1, 2, 5, 26]
    }
    emotions = list(EMOTION_KEY_AUS.keys())

    # --- 表格格式設定 ---
    left_col_width = 18
    col_width = 9
    total_width = left_col_width
    for emotion in emotions:
        total_width += len(EMOTION_KEY_AUS[emotion]) * col_width + 1

    # --- 建立報告字串 ---
    report = f"\n{'='*total_width}\n"
    report += f"{'Single Video Facial Analysis Report':^{total_width}}\n"
    report += f"Video File: {video_filename:^{total_width-12}}\n"
    report += f"{'='*total_width}\n\n"

    # --- 標頭 ---
    header_line = f"{'Metric':<{left_col_width}}"
    au_line = f"{'Key AU':<{left_col_width}}"
    for emotion in emotions:
        key_aus = EMOTION_KEY_AUS[emotion]
        emotion_width = len(key_aus) * col_width
        header_line += f"|{emotion.capitalize():^{emotion_width}}"
        au_header = "".join([f"{'AU'+str(au):<{col_width}}" for au in key_aus])
        au_line += f"|{au_header}"
    report += f"{header_line}\n{au_line}\n{'-'*total_width}\n"

    # --- 逐一印出每個指標的數據 ---
    metrics_to_print = {
        "Emotion Mean": "emotion_mean", "Emotion Std": "emotion_std",
        "Emotion Min": "emotion_min", "Emotion Max": "emotion_max",
        "AU Mean": "au_mean", "AU Std": "au_std",
        "AU Min": "au_min", "AU Max": "au_max",
    }

    for label, key_prefix in metrics_to_print.items():
        line = f"{label:<{left_col_width}}"
        for emotion in emotions:
            line += "|"
            
            if "Emotion" in label:
                col_name = f"{key_prefix}_{emotion}"
                val = stats.get(col_name, 0)
                emotion_width = len(EMOTION_KEY_AUS[emotion]) * col_width
                line += f"{val:^{emotion_width}.3f}"
            else: # "AU" in label
                for au_num in EMOTION_KEY_AUS[emotion]:
                    col_name = f"{key_prefix}_AU{au_num:02d}"
                    val = stats.get(col_name, 0)
                    line += f"{val:<{col_width}.3f}"

        report += f"{line}\n"
        if "Max" in label and "Emotion" in label:
             report += "\n"
        elif "Max" in label and "AU" in label:
            report += f"{'-'*total_width}\n"

    # --- 新增區塊：所有 AU 的整體統計數據 ---
    report += "\n"
    
    # 1. 從 stats 字典中獲取所有偵測到的 AU 名稱
    all_au_names = sorted([key.replace('au_mean_', '') for key in stats.keys() if key.startswith('au_mean_AU')])
    
    # 2. 計算新表格的寬度
    overall_table_width = left_col_width + len(all_au_names) * col_width
    report += f"--- Overall Statistics for All Detected AUs {'-'*max(0, overall_table_width - 44)}\n"

    # 3. 定義此區塊要顯示的指標
    overall_metrics = {
        "Mean": "au_mean",
        "Std": "au_std",
        "Min": "au_min",
        "Max": "au_max",
    }
    
    # 第 1 列：所有 AU 的名稱
    au_header_line = f"{'All AUs':<{left_col_width}}"
    for au_name in all_au_names:
        au_header_line += f"{au_name:<{col_width}}"
    report += f"{au_header_line}\n"
    
    # 第 2-5 列：對應的統計數據
    for label, key_prefix in overall_metrics.items():
        line = f"{label:<{left_col_width}}"
        for au_name in all_au_names:
            col_name = f"{key_prefix}_{au_name}"
            val = stats.get(col_name, 0)
            line += f"{val:<{col_width}.3f}"
        report += f"{line}\n"

    report += f"{'='*max(total_width, len(au_header_line))}\n"
    print(report)



def load_and_print_from_csv(csv_path, video_filename):
    """
    從已儲存的 CSV 檔案載入分析結果並印出報告。
    """
    print(f"Loading analysis from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        # 因為 CSV 只有一行，我們讀取第一行並將其轉換為字典
        video_stats = df.iloc[0].to_dict()
        print_single_video_report(video_stats, video_filename)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run with 'run_full_detection = True' first to generate the file.")
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")

def main():
    """
    主執行函式：分析單一影片並印出報告。
    可選擇執行完整分析或從 CSV 載入。
    """
    # --- 開關：設為 True 以執行完整分析，設為 False 以從 CSV 載入 ---
    run_full_detection = True

    video_filename = os.path.basename(VIDEO_PATH)
    save_path = os.path.join(SAVE_BASE_PATH, f"analysis_{video_filename.replace('.mp4', '.csv')}")

    if run_full_detection:
        if not os.path.exists(VIDEO_PATH):
            print(f"Error: Video file not found at {VIDEO_PATH}")
            return

        # 1. 執行偵測
        vid_prediction = detect_video(VIDEO_PATH)
        
        if vid_prediction is None or vid_prediction.empty:
            print("Could not detect any faces in the video. Exiting.")
            return

        # 2. 收集所有統計數據到一個字典中
        video_stats = {}
        flatten_series_into_dict(video_stats, 'emotion_mean', vid_prediction.emotions.mean())
        flatten_series_into_dict(video_stats, 'emotion_std', vid_prediction.emotions.std())
        flatten_series_into_dict(video_stats, 'emotion_min', vid_prediction.emotions.min())
        flatten_series_into_dict(video_stats, 'emotion_max', vid_prediction.emotions.max())
        
        flatten_series_into_dict(video_stats, 'au_mean', vid_prediction.aus.mean())
        flatten_series_into_dict(video_stats, 'au_std', vid_prediction.aus.std())
        flatten_series_into_dict(video_stats, 'au_min', vid_prediction.aus.min())
        flatten_series_into_dict(video_stats, 'au_max', vid_prediction.aus.max())

        # 3. 將結果儲存到一個扁平的 CSV 檔案中
        results_df = pd.DataFrame([video_stats])
        results_df.to_csv(save_path, index=False)
        print(f"Detailed analysis saved to: {save_path}")

        # 4. 印出格式化的報告
        print_single_video_report(video_stats, video_filename)
    
    else: # 從 CSV 載入
        load_and_print_from_csv(save_path, video_filename)


if __name__ == '__main__':
    main()
# ```

# ### 如何使用

# 現在您的程式有兩種操作模式，由 `main` 函式中的 `run_full_detection` 變數控制：

# 1.  **模式一：執行完整分析並儲存結果**
#     *   **設定**: `run_full_detection = True`
#     *   **操作**: 執行 `python eval_May_raw_vid.py`。
#     *   **結果**: 程式會像以前一樣，完整分析 `VIDEO_PATH` 指定的影片，將詳細的統計數據儲存到 `figures/analysis_May.csv`，並印出最終的報告。

# 2.  **模式二：從 CSV 快速載入並產生報告**
#     *   **前提**: 您必須已經執行過模式一，並成功產生了 `analysis_May.csv` 檔案。
#     *   **設定**: `run_full_detection = False`
#     *   **操作**: 再次執行 `python eval_May_raw_vid.py`。
#     *   **結果**: 程式會跳過耗時的影片分析，直接讀取 `analysis_May.csv` 的內容，並在瞬間產生完全相同的報告。

# ### 程式碼修改重點

# 1.  **`main` 函式中的開關**：加入了 `run_full_detection` 變數和一個 `if/else` 結構來引導程式流程。
# 2.  **新增 `load_and_print_from_csv` 函式**：
#     *   這個函式負責處理從 CSV 讀取的所有邏輯。
#     *   它使用 `pd.read_csv()` 讀取檔案。
#     *   最關鍵的一步是 `df.iloc[0].to_dict()`，它能非常簡單地將 CSV 中的那一行數據轉換回 `print_single_video_report` 所需的字典格式。
#     *   包含了錯誤處理，如果找不到 CSV 檔案會給予提示。
# 3.  **路徑變數的共用**：我將 `video_filename` 和 `save_path` 的定義移到了 `if/else` 結構之前，這樣兩種模式都可以共用這些路徑變數，避免了程式碼重複。# filepath: /home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/AU/eval_May_raw_vid.py