# source: https://github.com/cosanlab/py-feat/blob/main/docs/basic_tutorials/02_detector_vids.ipynb
from feat import Detector
import matplotlib.pyplot as plt
from feat.utils.io import get_test_data_path
import os
import numpy as np

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
# EMOTION_ID_LIST = [i for i in range(1, 9)] # emotion id: 1~8

ACTOR_ID_LIST = [20] # actor id: 1~24
EMOTION_ID_LIST = [5] # emotion id: 1~8

EMOTION_MAP = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fearful',
    7: 'disgust',
    8: 'surprised'
}

print(EMOTION_MAP[1])


def detect_videos(video_path):
    out_name = video_path.replace('.mp4', '.csv')

    video_prediction = detector.detect_video(
        video_path, data_type="video", skip_frames=3, face_detection_threshold=0.95, save=out_name
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

def main():
    # for all actors and all emotions
    for actor_id in ACTOR_ID_LIST:
        for emo_id in EMOTION_ID_LIST:
            # emogene detection
            emogene_video_path = EMOGENE_BASE_DIR + VIDEO_DIR.format(actor_id=actor_id, emo_id=emo_id)
            emogene_prediction, emogene_out_name = detect_videos(emogene_video_path)

            # genefacepp detection
            genefacepp_video_path = GENEFACEPP_BASE_DIR + VIDEO_DIR.format(actor_id=actor_id, emo_id=emo_id)
            genefacepp_prediction, genefacepp_out_name = detect_videos(genefacepp_video_path)

            # ============================================== PLOT EMOTION ===========================================================================
            # calculate emotion mean and plot comparison bar graph
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
            # ============================================== PLOT EMOTION ===========================================================================

            # ============================================== PLOT AU=================================================================================
            # calculate AU mean and plot comparison bar graph
            emogene_AU_means = emogene_prediction.aus.mean()
            genefacepp_AU_means = genefacepp_prediction.aus.mean()
            plot_AU_means(
                genefacepp_AU_means, emogene_AU_means, title=f"Actor {actor_id} - {EMOTION_MAP[emo_id]}", fig_id=f"a{actor_id:02d}e{emo_id}"
            )

            plot_AU_over_time(
                emogene_prediction, title=f"EmoGene - Actor {actor_id} - {EMOTION_MAP[emo_id]}", fig_id=f"a{actor_id:02d}e{emo_id}_emogene"
            )
            plot_AU_over_time(
                genefacepp_prediction, title=f"GeneFace++ - Actor {actor_id} - {EMOTION_MAP[emo_id]}", fig_id=f"a{actor_id:02d}e{emo_id}_genefacepp"
            )
            # ============================================== PLOT AU=================================================================================

if __name__ == '__main__':
    main()

