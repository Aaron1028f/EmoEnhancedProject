# https://github.com/cosanlab/py-feat/blob/main/docs/basic_tutorials/02_detector_vids.ipynb

from feat import Detector

detector = Detector()

# detector

from feat.utils.io import get_test_data_path
import os

test_data_dir = get_test_data_path()
# test_video_path = os.path.join(test_data_dir, "WolfgangLanger_Pexels.mp4")

test_video_path = 'datas/May/tmp.mp4'

# Show video
# from IPython.core.display import Video

# Video(test_video_path, embed=False)

video_prediction = detector.detect(
    test_video_path, data_type="video", skip_frames=24, face_detection_threshold=0.95
)
print(video_prediction.head())

print(video_prediction.shape)

# Frame 48 = ~0:02
# Frame 408 = ~0:14
video_prediction.query("frame in [48, 408]").plot_detections(
    faceboxes=False, add_titles=False
)

axes = video_prediction.emotions.plot()
