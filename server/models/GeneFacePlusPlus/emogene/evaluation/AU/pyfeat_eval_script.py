from feat import Detector
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

detector = Detector(device='cuda')
# =================================================================================================
# fixed settings
EMOGENE_BASE_DIR = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/emogene_ver/RAVDESS'
GENEFACEPP_BASE_DIR = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/genefacepp_ver/RAVDESS'
VIDEO_DIR = '/Actor_{actor_id:02d}/03-01-0{emo_id}-02-01-02-{actor_id:02d}.mp4'

EMOTION_MAP = {
    3: 'happiness', 4: 'sadness', 5: 'anger', 6: 'fear', 7: 'disgust', 8: 'surprise'
}    
SAVE_FIG_BASE_DIR = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/AU/figures/'
if not os.path.exists(SAVE_FIG_BASE_DIR):
    os.makedirs(SAVE_FIG_BASE_DIR)
    
# =================================================================================================
# other settings

# EMOTION_ID_LIST = [i for i in range(3, 9)] # emotion id: 3~8 (py-feat only support 3~8)
# ACTOR_ID_LIST = [i for i in range(1, 25)] # actor id: 1~24
EMOTION_ID_LIST = [i for i in range(3, 9)] # emotion id: 3~8 (py-feat only support 3~8)
ACTOR_ID_LIST = [20] # actor id: 1~24
SKIP_FRAME_NUM = 3
SAVE_CSV_FILENAME = 'temp_a20_{src}.csv'
# SAVE_CSV_FILENAME = 'results_{src}_RAVDESS_May_all_flat.csv'
SAVE_CSV_PATH = f'{SAVE_FIG_BASE_DIR}/{SAVE_CSV_FILENAME}'

# -------------------------------------------------------------------------------------------------
# --- Set to True to run detection, False to load from CSV ---
RUN_FULL_DETECTION = True  
EXISTING_CSV_FILE_PATH_EMOGENE = f'{SAVE_FIG_BASE_DIR}/results_emogene_RAVDESS_May_raw_flat.csv'
EXISTING_CSV_FILE_PATH_GENEFACEPP = f'{SAVE_FIG_BASE_DIR}/results_genefacepp_RAVDESS_May_raw_flat.csv'

# =================================================================================================

def detect_videos(video_path):
    out_name = video_path.replace('.mp4', '.csv')
    video_prediction = detector.detect_video(
        video_path, data_type="video", skip_frames=SKIP_FRAME_NUM, face_detection_threshold=0.95, save=out_name
    )
    return video_prediction, out_name

# ... (All plotting functions like plot_emotion_means, etc. remain the same) ...
def plot_emotion_means(genefacepp_emotion_means, emogene_emotion_means, title="Mean Emotion Scores Comparison", fig_id='xxx'):
    # plot emotion means using bar graph
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    x = np.arange(len(genefacepp_emotion_means))

    plt.bar(x - bar_width/2, genefacepp_emotion_means.values, color='blue', width=bar_width, label='GeneFace++')
    plt.bar(x + bar_width/2, emogene_emotion_means.values, color='red', width=bar_width, label='EmoGene')

    plt.title("Mean Emotion Scores Comparison:           " + title)
    plt.xlabel("Emotion")
    plt.ylabel("Mean Score")
    plt.xticks(x, genefacepp_emotion_means.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{SAVE_FIG_BASE_DIR}/mean_emotion_scores_comparison_{fig_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_AU_means(genefacepp_AU_means, emogene_AU_means, title="Mean AU Scores Comparison", fig_id='xxx'):
    # plot AU means using bar graph
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    x = np.arange(len(genefacepp_AU_means))

    plt.bar(x - bar_width/2, genefacepp_AU_means.values, color='blue', width=bar_width, label='GeneFace++')
    plt.bar(x + bar_width/2, emogene_AU_means.values, color='red', width=bar_width, label='EmoGene')

    plt.title("Mean AU Scores Comparison:           " + title)
    plt.xlabel("AU")
    plt.ylabel("Mean Score")
    plt.xticks(x, genefacepp_AU_means.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{SAVE_FIG_BASE_DIR}/mean_AU_scores_comparison_{fig_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_emotion_over_time(video_prediction, title="Emotion Detection Over Time", fig_id='xxx'):
    # plot emotion over time
    plt.figure(figsize=(10, 5))
    for emotion in video_prediction.emotions.columns:
        plt.plot(video_prediction.emotions.index, video_prediction.emotions[emotion], label=emotion)
        
    plt.title("Emotion Detection Over Time:           " + title)
    plt.xlabel("Frame")
    plt.ylabel("Emotion Score")
    plt.legend()
    plt.savefig(f'{SAVE_FIG_BASE_DIR}/emotion_detection_over_time_{fig_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_AU_over_time(video_prediction, title="AU Detection Over Time", fig_id='xxx'):
    # plot AU over time
    plt.figure(figsize=(10, 5))
    for au in video_prediction.aus.columns:
        plt.plot(video_prediction.aus.index, video_prediction.aus[au], label=au)

    plt.title("AU Detection Over Time:           " + title)
    plt.xlabel("Frame")
    plt.ylabel("AU Score")
    plt.legend()
    plt.savefig(f'{SAVE_FIG_BASE_DIR}/au_detection_over_time_{fig_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_emotion_comparison(actor_id, emo_id, emogene_prediction, genefacepp_prediction):
    emogene_emotion_means = emogene_prediction.emotions.mean()
    genefacepp_emotion_means = genefacepp_prediction.emotions.mean()
    plot_emotion_means(
        genefacepp_emotion_means, emogene_emotion_means, title=f"Actor {actor_id} - {EMOTION_MAP[emo_id]}", fig_id=f"a{actor_id:02d}e{emo_id}"
    )
    # plot emotion detection over time
    plot_emotion_over_time(
        emogene_prediction, title=f"EmoGene - Actor {actor_id} - {EMOTION_MAP[emo_id]}", fig_id=f"a{actor_id:02d}e{emo_id}_emogene"
    )
    plot_emotion_over_time(
        genefacepp_prediction, title=f"GeneFace++ - Actor {actor_id} - {EMOTION_MAP[emo_id]}", fig_id=f"a{actor_id:02d}e{emo_id}_genefacepp"
    )

def plot_AU_comparison(actor_id, emo_id, emogene_prediction, genefacepp_prediction):
    emogene_AU_means = emogene_prediction.aus.mean()
    genefacepp_AU_means = genefacepp_prediction.aus.mean()
    plot_AU_means(
        genefacepp_AU_means, emogene_AU_means, title=f"Actor {actor_id} - {EMOTION_MAP[emo_id]}", fig_id=f"a{actor_id:02d}e{emo_id}"
    )
    # plot AU detection over time
    plot_AU_over_time(
        emogene_prediction, title=f"EmoGene - Actor {actor_id} - {EMOTION_MAP[emo_id]}", fig_id=f"a{actor_id:02d}e{emo_id}_emogene"
    )
    plot_AU_over_time(
        genefacepp_prediction, title=f"GeneFace++ - Actor {actor_id} - {EMOTION_MAP[emo_id]}", fig_id=f"a{actor_id:02d}e{emo_id}_genefacepp"
    )

def process_results(results_input, src):
    """
    Processes video results from a list of dicts or a DataFrame.
    The data is expected to be in a "flat" format.
    """
    if isinstance(results_input, list):
        results_df = pd.DataFrame(results_input)
        # Save the new, clean CSV format
        # results_df.to_csv(f'{SAVE_FIG_BASE_DIR}/results_{src}_RAVDESS_May_all_flat.csv', index=False)
        results_df.to_csv(SAVE_CSV_PATH.format(src=src), index=False)
    else:
        # Input is already a DataFrame from a CSV
        results_df = results_input

    # The aggregation is now a simple, standard pandas operation
    return results_df.groupby('emotion').mean()

def load_and_print_from_csv(emogene_csv_path, genefacepp_csv_path):
    """
    Loads raw results from two FLAT CSV files, processes them, and prints the final report.
    """
    print(f"Loading EmoGene results from: {emogene_csv_path}")
    emogene_df = pd.read_csv(emogene_csv_path)
    
    print(f"Loading GeneFace++ results from: {genefacepp_csv_path}")
    genefacepp_df = pd.read_csv(genefacepp_csv_path)

    # Process the loaded DataFrames
    emogene_avg = process_results(emogene_df, 'emogene')
    genefacepp_avg = process_results(genefacepp_df, 'genefacepp')

    # Print the final formatted report
    print_final_result(emogene_avg, genefacepp_avg)

def print_final_result(emogene_avg, genefacepp_avg):
    """
    Print the final comparison report.
    This version reads from a DataFrame where columns are already aggregated means.
    """
    EMOTION_KEY_AUS = {
        'happiness': [6, 12], 'sadness': [1, 4, 15], 'anger': [4, 5, 7, 23],
        'fear': [1, 2, 4, 5, 7, 20, 26], 'disgust': [9, 15], 'surprise': [1, 2, 5, 26]
    }
    emotions = list(EMOTION_KEY_AUS.keys())

    left_col_width = 30
    col_width = 8
    total_width = left_col_width
    for emotion in emotions:
        total_width += len(EMOTION_KEY_AUS[emotion]) * col_width + 1

    report = f"\n{'='*total_width}\n"
    report += f"{'Final Emotion & AU Analysis Report':^{total_width}}\n"
    report += f"{'='*total_width}\n\n"

    header_line = f"{'Metric':<{left_col_width}}"
    au_line = f"{'Key AU':<{left_col_width}}"
    for emotion in emotions:
        key_aus = EMOTION_KEY_AUS[emotion]
        emotion_width = len(key_aus) * col_width
        header_line += f"|{emotion.capitalize():^{emotion_width}}"
        au_header = "".join([f"{'AU'+str(au):<{col_width}}" for au in key_aus])
        au_line += f"|{au_header}"
    report += f"{header_line}\n{au_line}\n{'-'*total_width}\n"

    # --- REVISED: Loop through all emotion metrics ---
    emotion_metrics_to_print = {
        "Avg Emotion Mean": "emotion_mean",
        "Avg Emotion Std": "emotion_std",
        "Avg Emotion MAX": "emotion_max",
        "Avg Emotion MIN": "emotion_min",
    }

    for label, key_prefix in emotion_metrics_to_print.items():
        g_line = f"{label + ' (GeneFace++)':<{left_col_width}}"
        e_line = f"{label + ' (EmoGene)':<{left_col_width}}"
        
        for emotion in emotions:
            emotion_width = len(EMOTION_KEY_AUS[emotion]) * col_width
            col_name = f"{key_prefix}_{emotion}"
            
            g_val = genefacepp_avg.loc[emotion, col_name] if col_name in genefacepp_avg.columns else 0
            e_val = emogene_avg.loc[emotion, col_name] if col_name in emogene_avg.columns else 0
            
            # Use centered alignment for the values
            g_line += f"|{f'{g_val:.3f}':^{emotion_width}}"
            e_line += f"|{f'{e_val:.3f}':^{emotion_width}}"
            
        report += f"{g_line}\n{e_line}\n"
    report += "\n"


    report += f"--- INTENSITY {'-'*(total_width-14)}\n"
    metrics_intensity = {
        'Key AUs Mean (GeneFace++)': ('au_mean', genefacepp_avg), 'Key AUs Mean (EmoGene)': ('au_mean', emogene_avg),
        'Key AUs MAX (GeneFace++)': ('au_max', genefacepp_avg), 'Key AUs MAX (EmoGene)': ('au_max', emogene_avg),
        'Key AUs MIN (GeneFace++)': ('au_min', genefacepp_avg), 'Key AUs MIN (EmoGene)': ('au_min', emogene_avg),
    }
    for label, (key, data) in metrics_intensity.items():
        line = f"{label:<{left_col_width}}"
        for emotion in emotions:
            line += "|"
            for au_num in EMOTION_KEY_AUS[emotion]:
                col_name = f"{key}_AU{au_num:02d}"
                val = data.loc[emotion, col_name] if col_name in data.columns else 0
                line += f"{val:<{col_width}.3f}"
        report += f"{line}\n"
    report += "\n"

    report += f"--- VARIABILITY {'-'*(total_width-16)}\n"
    metrics_variability = {
        'Key AUs Std (GeneFace++)': ('au_std', genefacepp_avg), 'Key AUs Std (EmoGene)': ('au_std', emogene_avg),
    }
    for label, (key, data) in metrics_variability.items():
        line = f"{label:<{left_col_width}}"
        for emotion in emotions:
            line += "|"
            for au_num in EMOTION_KEY_AUS[emotion]:
                col_name = f"{key}_AU{au_num:02d}"
                val = data.loc[emotion, col_name] if col_name in data.columns else 0
                line += f"{val:<{col_width}.3f}"
        report += f"{line}\n"
    report += f"{'='*total_width}\n"
    print(report)

def flatten_series_into_dict(results_dict, prefix, series):
    """Adds items from a Series to a dict with a given prefix."""
    if series is not None:
        for idx, val in series.items():
            results_dict[f"{prefix}_{idx}"] = val

def main():
    if RUN_FULL_DETECTION:
        emogene_results_list = []
        genefacepp_results_list = []

        for emo_id in EMOTION_ID_LIST:
            for actor_id in ACTOR_ID_LIST:
                print(f"Processing Actor {actor_id}, Emotion {EMOTION_MAP[emo_id]}...")
                # Emogene
                emogene_video_path = EMOGENE_BASE_DIR + VIDEO_DIR.format(actor_id=actor_id, emo_id=emo_id)
                emogene_prediction, _ = detect_videos(emogene_video_path)
                
                emogene_video_results = {'actor_id': actor_id, 'emotion': EMOTION_MAP[emo_id]}
                flatten_series_into_dict(emogene_video_results, 'emotion_mean', emogene_prediction.emotions.mean())
                flatten_series_into_dict(emogene_video_results, 'emotion_std', emogene_prediction.emotions.std())
                flatten_series_into_dict(emogene_video_results, 'emotion_max', emogene_prediction.emotions.max())
                flatten_series_into_dict(emogene_video_results, 'emotion_min', emogene_prediction.emotions.min())
                flatten_series_into_dict(emogene_video_results, 'au_mean', emogene_prediction.aus.mean())
                flatten_series_into_dict(emogene_video_results, 'au_std', emogene_prediction.aus.std())
                flatten_series_into_dict(emogene_video_results, 'au_max', emogene_prediction.aus.max())
                flatten_series_into_dict(emogene_video_results, 'au_min', emogene_prediction.aus.min())
                emogene_results_list.append(emogene_video_results)

                # GeneFace++
                genefacepp_video_path = GENEFACEPP_BASE_DIR + VIDEO_DIR.format(actor_id=actor_id, emo_id=emo_id)
                genefacepp_prediction, _ = detect_videos(genefacepp_video_path)

                genefacepp_video_results = {'actor_id': actor_id, 'emotion': EMOTION_MAP[emo_id]}
                flatten_series_into_dict(genefacepp_video_results, 'emotion_mean', genefacepp_prediction.emotions.mean())
                flatten_series_into_dict(genefacepp_video_results, 'emotion_std', genefacepp_prediction.emotions.std())
                flatten_series_into_dict(genefacepp_video_results, 'emotion_max', genefacepp_prediction.emotions.max())
                flatten_series_into_dict(genefacepp_video_results, 'emotion_min', genefacepp_prediction.emotions.min())
                flatten_series_into_dict(genefacepp_video_results, 'au_mean', genefacepp_prediction.aus.mean())
                flatten_series_into_dict(genefacepp_video_results, 'au_std', genefacepp_prediction.aus.std())
                flatten_series_into_dict(genefacepp_video_results, 'au_max', genefacepp_prediction.aus.max())
                flatten_series_into_dict(genefacepp_video_results, 'au_min', genefacepp_prediction.aus.min())
                genefacepp_results_list.append(genefacepp_video_results)
            
        emogene_avg = process_results(emogene_results_list, 'emogene')
        genefacepp_avg = process_results(genefacepp_results_list, 'genefacepp')
        print_final_result(emogene_avg, genefacepp_avg)

    else:
        # emogene_file = f'{SAVE_FIG_BASE_DIR}/results_emogene_RAVDESS_May_raw_flat.csv'
        # genefacepp_file = f'{SAVE_FIG_BASE_DIR}/results_genefacepp_RAVDESS_May_raw_flat.csv'

        if os.path.exists(EXISTING_CSV_FILE_PATH_EMOGENE) and os.path.exists(EXISTING_CSV_FILE_PATH_GENEFACEPP):
            load_and_print_from_csv(EXISTING_CSV_FILE_PATH_EMOGENE, EXISTING_CSV_FILE_PATH_GENEFACEPP)
        else:
            print("CSV files not found. Please run with 'RUN_FULL_DETECTION = True' first to generate them.")

if __name__ == '__main__':
    main()