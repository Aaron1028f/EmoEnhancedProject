# https://github.com/cosanlab/py-feat/blob/main/docs/basic_tutorials/02_detector_vids.ipynb

from feat import Detector
import matplotlib.pyplot as plt

detector = Detector(device='cuda')

# detector

from feat.utils.io import get_test_data_path
import os

# test_data_dir = get_test_data_path()
# test_video_path = os.path.join(test_data_dir, "WolfgangLanger_Pexels.mp4")

# test_video_path = 'datas/May/tmp.mp4'
# test_video_path = 'datas/May/tmp2.mp4'

# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/emogene_ver/RAVDESS/Actor_20/03-01-03-02-01-02-20.mp4' # happy
# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/emogene_ver/RAVDESS/Actor_20/03-01-05-02-01-02-20.mp4' # angry
test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/emogene_ver/RAVDESS/Actor_20/03-01-07-02-01-02-20.mp4' # disgust

# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/emogene_ver/RAVDESS/Actor_20/03-01-08-02-01-02-20.mp4' # surprise -> 


# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/genefacepp_ver/RAVDESS/Actor_04/03-01-04-01-02-02-04.mp4' # sad
# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/genefacepp_ver/RAVDESS/Actor_04/03-01-05-01-02-02-04.mp4' # 
# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/genefacepp_ver/RAVDESS/Actor_04/03-01-06-01-02-02-04.mp4' # fear -> sad and fear
# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/genefacepp_ver/RAVDESS/Actor_04/03-01-07-01-02-02-04.mp4' # disgust -> sad
# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/genefacepp_ver/RAVDESS/Actor_04/03-01-03-01-02-02-04.mp4' # happy -> sad
# test_video_path = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/May/genefacepp_ver/RAVDESS/Actor_04/03-01-08-01-02-02-04.mp4' # surprise -> 




# Show video
# from IPython.core.display import Video

# Video(test_video_path, embed=False)
out_name = test_video_path.replace('.mp4', '.csv')
print(out_name)
video_prediction = detector.detect_video(
    test_video_path, data_type="video", skip_frames=3, face_detection_threshold=0.95, save=out_name
)
# print(video_prediction.head())

print(video_prediction.shape)
print(video_prediction.columns)

# write video_prediction
video_prediction.to_csv(out_name, index=False)

print(video_prediction.emotions)

# calculate emotion mean
emotion_means = video_prediction.emotions.mean()
print(emotion_means)


# plot emotion means using bar graph
plt.figure(figsize=(10, 5))
plt.bar(emotion_means.index, emotion_means.values)
plt.title("Mean Emotion Scores")
plt.xlabel("Emotion")
plt.ylabel("Mean Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('mean_emotion_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# plot emotion over time
plt.figure(figsize=(10, 5))
for emotion in video_prediction.emotions.columns:
    plt.plot(video_prediction.emotions.index, video_prediction.emotions[emotion], label=emotion)
plt.title("Emotion Detection Over Time")
plt.xlabel("Frame")
plt.ylabel("Emotion Score")
plt.legend()
plt.savefig('emotion_detection_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

# AU
AUs = video_prediction.aus
print(AUs.shape)
print(AUs.columns)
print(AUs)

# AU mean
au_means = AUs.mean()
print(au_means)

# plot AU mean
plt.figure(figsize=(10, 5))
plt.bar(au_means.index, au_means.values)
plt.title("Mean AU Scores")
plt.xlabel("AU")
plt.ylabel("Mean Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('mean_au_scores.png', dpi=300, bbox_inches='tight')
plt.close()
