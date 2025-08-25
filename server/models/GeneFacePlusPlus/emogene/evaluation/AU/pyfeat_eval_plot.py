# source: https://github.com/cosanlab/py-feat/blob/main/docs/basic_tutorials/02_detector_vids.ipynb
from feat import Detector
import matplotlib.pyplot as plt
from feat.utils.io import get_test_data_path
import os
import numpy as np
import pandas as pd

detector = Detector(device='cuda')

# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/emogene_ver/RAVDESS/Actor_20/03-01-07-02-01-02-20.mp4'
EMOGENE_BASE_DIR = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/emogene_ver/RAVDESS'
GENEFACEPP_BASE_DIR = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/genefacepp_ver/RAVDESS'

# Always test with the sencond try of "Kids are talking by the door." using strong intensity
VIDEO_DIR = '/Actor_{actor_id}/03-01-0{emo_id}-02-01-02-{actor_id}.mp4'

SAVE_FIG_BASE_DIR = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/AU/figures/'
if not os.path.exists(SAVE_FIG_BASE_DIR):
    os.makedirs(SAVE_FIG_BASE_DIR)

# ACTOR_ID_LIST = [i for i in range(1, 25)] # actor id: 1~24
EMOTION_ID_LIST = [i for i in range(3, 9)] # emotion id: 3~8 (py-feat only support 3~8)

ACTOR_ID_LIST = [20] # actor id: 1~24
# EMOTION_ID_LIST = [8] # emotion id: 1~8

EMOTION_MAP = {
    # 1: 'neutral',
    # 2: 'calm',
    
    3: 'happiness',
    4: 'sadness',
    5: 'anger',
    6: 'fear',
    7: 'disgust',
    8: 'surprise'
}


def detect_videos(video_path):
    out_name = video_path.replace('.mp4', '.csv')

    video_prediction = detector.detect_video(
        video_path, data_type="video", skip_frames=100, face_detection_threshold=0.95, save=out_name
    )
    # write video_prediction
    # video_prediction.to_csv(out_name, index=False)
    
    return video_prediction, out_name

    # print(video_prediction.head())
    # print(video_prediction.shape)
    # print(video_prediction.columns)
    # print(video_prediction.emotions)

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
    Processes video results. Handles two types of input:
    1. A list of dictionaries (from a live run).
    2. A pandas DataFrame (loaded from a CSV file).
    """
    if isinstance(results_input, list):
        results_df = pd.DataFrame(results_input)
        # Save the raw results from the live run
        results_df.to_csv(f'{SAVE_FIG_BASE_DIR}/results_{src}_RAVDESS_May_raw.csv', index=False)
    else:
        # Input is already a DataFrame, likely from a CSV
        results_df = results_input

    avg = {}
    grouped = results_df.groupby('emotion')
    
    # # For each column that contains Series-like data
    # for col in ['emotion_mean', 'au_mean', 'au_max', 'au_std']:
    #     # This lambda function is robust enough to handle both
    #     # actual Series objects and string representations from CSVs.
    #     # It uses pd.concat which can parse the string representation.
    #     avg[col] = grouped[col].apply(lambda s: pd.concat(
    #         [pd.read_csv(pd.io.common.StringIO(item), header=None, index_col=0).iloc[:, 0] if isinstance(item, str) else item for item in s]
    #     ).groupby(level=0).mean())
    
    # For each column that contains Series, aggregate them correctly
    for col in ['emotion_mean', 'au_mean', 'au_max', 'au_std']:
        # Apply a function to each group
        # The function concatenates all Series in the group and then calculates the mean
        avg[col] = grouped[col].apply(lambda s: pd.concat(list(s)).groupby(level=0).mean())

    return avg
# def process_results(results_list, src):
#     """
#     Processes a list of video result dictionaries.
#     Correctly handles aggregation of pandas Series stored in the list.
#     """
#     results_df = pd.DataFrame(results_list)
#     results_df.to_csv(f'{SAVE_FIG_BASE_DIR}/results_{src}_RAVDESS_May_raw.csv', index=False)
    
#     avg = {}
#     # Group by the 'emotion' column (e.g., 'happy', 'sad')
#     grouped = results_df.groupby('emotion')
    
#     # For each column that contains Series, aggregate them correctly
#     for col in ['emotion_mean', 'au_mean', 'au_max', 'au_std']:
#         # Apply a function to each group
#         # The function concatenates all Series in the group and then calculates the mean
#         avg[col] = grouped[col].apply(lambda s: pd.concat(list(s)).groupby(level=0).mean())

#     return avg

def load_and_print_from_csv(emogene_csv_path, genefacepp_csv_path):
    """
    Loads raw results from two CSV files, processes them, and prints the final report.
    This allows re-generating the report without re-running video detection.

    Args:
        emogene_csv_path (str): Path to the EmoGene raw results CSV.
        genefacepp_csv_path (str): Path to the GeneFace++ raw results CSV.
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
    Print the final comparison report between EmoGene and GeneFace++.
    """
    EMOTION_KEY_AUS = {
        'happiness': [6, 12],
        'sadness': [1, 4, 15],
        'anger': [4, 5, 7, 23],
        'fear': [1, 2, 4, 5, 7, 20, 26],
        'disgust': [9, 15],
        'surprise': [1, 2, 5, 26]
    }
    emotions = list(EMOTION_KEY_AUS.keys())

    # set column widths
    left_col_width = 30
    col_width = 8  # Set fixed width for each AU data

    # Calculate total width
    total_width = left_col_width
    for emotion in emotions:
        total_width += len(EMOTION_KEY_AUS[emotion]) * col_width + 1 # +1 for "|"

    # create header string
    report = "\n" + "="*total_width + "\n"
    report += f"{'Final Emotion & AU Analysis Report':^{total_width}}\n"
    report += "="*total_width + "\n\n"

    # header
    header_line = f"{'Metric':<{left_col_width}}"
    au_line = f"{'Key AU':<{left_col_width}}"
    for emotion in emotions:
        key_aus = EMOTION_KEY_AUS[emotion]
        emotion_width = len(key_aus) * col_width
        header_line += f"|{emotion.capitalize():^{emotion_width}}"
        
        au_header = "".join([f"{'AU'+str(au):<{col_width}}" for au in key_aus])
        au_line += f"|{au_header}"
        
    report += header_line + "\n"
    report += au_line + "\n"
    report += "-"*total_width + "\n"

    # mean emotion scores
    g_emo_means = [f"{genefacepp_avg['emotion_mean'][emo][emo]:.3f}" for emo in emotions]
    e_emo_means = [f"{emogene_avg['emotion_mean'][emo][emo]:.3f}" for emo in emotions]
    
    g_emo_line = f"{'GeneFace++ Avg Emotion Mean':<{left_col_width}}"
    e_emo_line = f"{'EmoGene Avg Emotion Mean':<{left_col_width}}"

    for i, emotion in enumerate(emotions):
        emotion_width = len(EMOTION_KEY_AUS[emotion]) * col_width
        g_emo_line += f"|{g_emo_means[i]:^{emotion_width}}"
        e_emo_line += f"|{e_emo_means[i]:^{emotion_width}}"

    report += g_emo_line + "\n"
    report += e_emo_line + "\n\n"

    # INTENSITY
    report += f"--- INTENSITY {'-'*(total_width-14)}\n"
    metrics_intensity = {
        'GeneFace++ Key AUs Mean': ('au_mean', genefacepp_avg),
        'EmoGene Key AUs Mean': ('au_mean', emogene_avg),
        'GeneFace++ Key AUs Peak': ('au_max', genefacepp_avg),
        'EmoGene Key AUs Peak': ('au_max', emogene_avg),
    }
    for label, (key, data) in metrics_intensity.items():
        line = f"{label:<{left_col_width}}"
        for emotion in emotions:
            line += "|"
            for au_num in EMOTION_KEY_AUS[emotion]:
                au_name = f'AU{au_num:02d}'
                val = data[key][emotion].get(au_name, 0)
                line += f"{val:<{col_width}.3f}"
        report += line + "\n"
    report += "\n"

    # VARIABILITY
    report += f"--- VARIABILITY {'-'*(total_width-16)}\n"
    metrics_variability = {
        'GeneFace++ Key AUs Std': ('au_std', genefacepp_avg),
        'EmoGene Key AUs Std': ('au_std', emogene_avg),
    }
    for label, (key, data) in metrics_variability.items():
        line = f"{label:<{left_col_width}}"
        for emotion in emotions:
            line += "|"
            for au_num in EMOTION_KEY_AUS[emotion]:
                au_name = f'AU{au_num:02d}'
                val = data[key][emotion].get(au_name, 0)
                line += f"{val:<{col_width}.3f}"
        report += line + "\n"
        
    report += "="*total_width + "\n"
    
    print(report)

def main():
    emogene_results_list = []
    genefacepp_results_list = []

    # for all actors and all emotions
    for emo_id in EMOTION_ID_LIST:
        for actor_id in ACTOR_ID_LIST:
            # emogene detection
            emogene_video_path = EMOGENE_BASE_DIR + VIDEO_DIR.format(actor_id=actor_id, emo_id=emo_id)
            emogene_prediction, emogene_out_name = detect_videos(emogene_video_path)

            # genefacepp detection
            genefacepp_video_path = GENEFACEPP_BASE_DIR + VIDEO_DIR.format(actor_id=actor_id, emo_id=emo_id)
            genefacepp_prediction, genefacepp_out_name = detect_videos(genefacepp_video_path)

            # plot graph of single video analysis
            # plot_emotion_comparison(actor_id, emo_id, emogene_prediction, genefacepp_prediction)
            # plot_AU_comparison(actor_id, emo_id, emogene_prediction, genefacepp_prediction)
            
            # save the results
            emogene_video_results = {
                'actor_id': actor_id,
                'emotion_id': emo_id,
                'emotion': EMOTION_MAP[emo_id],
                'emotion_mean': emogene_prediction.emotions.mean(),
                'au_mean': emogene_prediction.aus.mean(),
                'au_std': emogene_prediction.aus.std(),
                'au_max': emogene_prediction.aus.max(),
            }
            genefacepp_video_results = {
                'actor_id': actor_id,
                'emotion_id': emo_id,
                'emotion': EMOTION_MAP[emo_id],
                'emotion_mean': genefacepp_prediction.emotions.mean(),
                'au_mean': genefacepp_prediction.aus.mean(),
                'au_std': genefacepp_prediction.aus.std(),
                'au_max': genefacepp_prediction.aus.max(),
            }

            emogene_results_list.append(emogene_video_results)
            genefacepp_results_list.append(genefacepp_video_results)
            
    # process the results
    emogene_avg = process_results(emogene_results_list, 'emogene')
    genefacepp_avg = process_results(genefacepp_results_list, 'genefacepp')

    # plot the graph
    print_final_result(emogene_avg, genefacepp_avg)



    # --- OPTION 2: Load from existing CSV files and print results ---
    # emogene_file = f'{SAVE_FIG_BASE_DIR}/results_emogene_RAVDESS_May_raw.csv'
    # genefacepp_file = f'{SAVE_FIG_BASE_DIR}/results_genefacepp_RAVDESS_May_raw.csv'
    
    # if os.path.exists(emogene_file) and os.path.exists(genefacepp_file):
    #     load_and_print_from_csv(emogene_file, genefacepp_file)
    # else:
    #     print("CSV files not found. Please run with 'run_full_detection = True' first.")



if __name__ == '__main__':
    # main()
    
    # just run from existing CSV files
    
    # --- OPTION 2: Load from existing CSV files and print results ---
    emogene_file = f'{SAVE_FIG_BASE_DIR}/results_emogene_RAVDESS_May_raw.csv'
    genefacepp_file = f'{SAVE_FIG_BASE_DIR}/results_genefacepp_RAVDESS_May_raw.csv'
    
    if os.path.exists(emogene_file) and os.path.exists(genefacepp_file):
        load_and_print_from_csv(emogene_file, genefacepp_file)
    else:
        print("CSV files not found. Please run with 'run_full_detection = True' first.")
    
    
    
    
    
# -------------------------------  | happy     | sad            | angry                | fearful                             | disgust   | surprised           |
# key AU names                     | AU6| AU12 | AU1| AU4| AU15 | AU4| AU5| AU7| AU23  | AU1| AU2| AU4| AU5| AU7| AU20| AU26 | AU9| AU15 | AU1| AU2| AU5| AU26 |
# -------------------------------
# genefacepp avg emotion means     |           |                |                      |                                     |           |                     |
# emogene avg emotion means        |           |                |                      |                                     |           |                     |

# ----------------------------intensity-----------------------------
# genefacepp emotion key AUs means |    |      |    |    |      |    |    |    |       |    |    |    |    |    |     |      |    |      |    |    |    |      |
# emogene emotion AUs means        |    |      |    |    |      |    |    |    |       |    |    |    |    |    |     |      |    |      |    |    |    |      |

# genefacepp emotion key AUs peak  |    |      |    |    |      |    |    |    |       |    |    |    |    |    |     |      |    |      |    |    |    |      |
# emogene emotion AUs peak         |    |      |    |    |      |    |    |    |       |    |    |    |    |    |     |      |    |      |    |    |    |      |

# ----------------------------variability---------------------------
# genefacepp emotion key AUs std   |    |      |    |    |      |    |    |    |       |    |    |    |    |    |     |      |    |      |    |    |    |      |
# emogene emotion key AUs std      |    |      |    |    |      |    |    |    |       |    |    |    |    |    |     |      |    |      |    |    |    |      |