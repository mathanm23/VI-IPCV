import cv2
import mediapipe as mp
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "model", "face_landmarker.task")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# Create dummy image
img = cv2.imread(r"C:\Users\Mathankumar\Downloads\IPCVPJT\drowsiness_detection\model\face_landmarker.task") # just to get a shape? no that's a binary file.
img = __import__('numpy').zeros((480, 640, 3), dtype=__import__('numpy').uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
result = landmarker.detect(mp_image)

# We can't detect face in a black image.
# We'll just look at the docstring or create a test.
