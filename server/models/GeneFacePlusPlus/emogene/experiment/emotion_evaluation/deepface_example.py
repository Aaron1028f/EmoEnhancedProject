# https://github.com/serengil/deepface

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

check_img = cv2.imread('img4.jpg')
# plt.figure(figsize=(7,7))
# plt.imshow(check_img[:, :, ::-1])

analyze_face = DeepFace.analyze(check_img)

print('-'*100)
print(analyze_face[0]['emotion'])
print('-'*100)
# print(analyze_face)

for emotion, score in analyze_face[0]['emotion'].items():
    print(f"情感: {emotion}, 分數: {score}")
print('-'*100)

# plot the emotion score in bar chart
plt.bar(analyze_face[0]['emotion'].keys(), analyze_face[0]['emotion'].values())
plt.xlabel('Emotion')
plt.ylabel('Score')
plt.title('Emotion Analysis')
plt.show()
plt.savefig('datas/emotion_analysis.png')