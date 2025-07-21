# source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker?hl=zh-tw

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import urllib.request
import os

from mediapipe import solutions

# Function to draw landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

# Function to plot face blendshapes bar graph
def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
#   plt.show()
  plt.tight_layout()
  plt.savefig("face_blendshapes.png")
  print("Face blendshapes plot saved to face_blendshapes.png")
  plt.close() # Close the figure to free up memory
  
# --- Main script starts here ---

# Download the face landmarker model bundle
model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
model_filename = "face_landmarker_v2_with_blendshapes.task"
if not os.path.exists(model_filename):
    print(f"Downloading {model_filename}...")
    urllib.request.urlretrieve(model_url, model_filename)
    print("Download complete.")
else:
    print(f"{model_filename} already exists. Skipping download.")


# Download the test image
image_url = "https://storage.googleapis.com/mediapipe-assets/business-person.png" # Using the working image URL
image_filename = "test_image.png"
if not os.path.exists(image_filename):
    print(f"Downloading {image_filename}...")
    urllib.request.urlretrieve(image_url, image_filename)
    print("Download complete.")
else:
     print(f"{image_filename} already exists. Skipping download.")

# STEP 1: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path=model_filename)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 2: Load the input image.
image = mp.Image.create_from_file(image_filename)

# STEP 3: Detect face landmarks from the input image.
detection_result = detector.detect(image)

# STEP 4: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

# Display the annotated image using OpenCV (will open a window locally)
# cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the annotated image to a file instead of displaying it
output_image_path = "annotated_image.png"
cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
print(f"Annotated image saved to {output_image_path}")


# STEP 5: Plot a bar graph of the face blendshapes.
if detection_result.face_blendshapes and detection_result.face_blendshapes[0]:
    plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
else:
    print("No face blendshapes detected.")


# STEP 6: Print the facial transformation matrix.
print("\nFacial Transformation Matrix:")
if detection_result.facial_transformation_matrixes:
    for matrix in detection_result.facial_transformation_matrixes:
        print(matrix)
else:
    print("No facial transformation matrix detected.")