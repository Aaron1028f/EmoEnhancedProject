# https://github.com/cosanlab/py-feat/blob/main/docs/basic_tutorials/02_detector_vids.ipynb

from feat import Detector

detector = Detector(device='cuda')

# detector

from feat.utils.io import get_test_data_path
import os

# test_data_dir = get_test_data_path()
# test_video_path = os.path.join(test_data_dir, "WolfgangLanger_Pexels.mp4")

# test_video_path = 'datas/May/tmp.mp4'
test_video_path = 'datas/May/tmp2.mp4'


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
# # Frame 48 = ~0:02
# # Frame 408 = ~0:14
# video_prediction.query("frame in [48, 100]").plot_detections(
#     faceboxes=False, add_titles=False
# )

print(video_prediction.emotions)


# plot video_prediction.emotions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for emotion in video_prediction.emotions.columns:
    plt.plot(video_prediction.emotions.index, video_prediction.emotions[emotion], label=emotion)
plt.title("Emotion Detection Over Time")
plt.xlabel("Frame")
plt.ylabel("Emotion Score")
plt.legend()
# plt.show()
plt.savefig('emotion_detection_over_time.png', dpi=300, bbox_inches='tight')
