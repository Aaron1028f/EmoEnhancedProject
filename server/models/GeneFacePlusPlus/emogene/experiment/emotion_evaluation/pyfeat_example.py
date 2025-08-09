# github: https://github.com/cosanlab/py-feat

# src: https://colab.research.google.com/drive/1ObfVE7u0DduEHdJD9QbT6oHIqhUPbIsG?usp=sharing#scrollTo=Z_5l62TTA4zR

import os
from PIL import Image
import matplotlib.pyplot as plt

# from feat.tests.utils import get_test_data_path
from feat import Detector

# define the models
face_model = "retinaface"
landmark_model = "mobilenet"
# au_model = "rf"
au_model = 'svm'
emotion_model = "resmasknet"
detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)

# load and visualize the image
# test_image = os.path.join('/content/', "home_alone4.jpg")
test_image = 'img4.jpg'
f, ax = plt.subplots()
im = Image.open(test_image)
ax.imshow(im)

# get prediction
image_prediction = detector.detect_image(test_image)

# Show results
print(image_prediction)

image_prediction.plot_detections()