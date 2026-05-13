import cv2
import mediapipe as mp
import urllib.request
import os

class LandmarkDetector:
    def __init__(self, model_path=None):
        """
        Initialize the MediaPipe FaceLandmarker module using the tasks API (modern way).
        """
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(base_dir, "model", "face_landmarker.task")
        else:
            self.model_path = model_path
            
        self._ensure_model_exists()
        
        # Load the new tasks API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Indices for Left and Right Eye landmarks (for drawing if needed)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"[INFO] Downloading MediaPipe FaceLandmarker model to {self.model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("[INFO] Download complete.")

    def extract_eye_landmarks(self, image):
        """
        Extract eye landmarks and blendshape scores for blinking.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            left_eye_points, right_eye_points, left_blink_score, right_blink_score
        """
        # Convert BGR to RGB and create a MediaPipe Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect landmarks
        detection_result = self.landmarker.detect(mp_image)
        
        left_eye_points = []
        right_eye_points = []
        left_blink_score = 0.0
        right_blink_score = 0.0
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            ih, iw, _ = image.shape
            
            # Extract left eye points
            for idx in self.LEFT_EYE_INDICES:
                x = int(landmarks[idx].x * iw)
                y = int(landmarks[idx].y * ih)
                left_eye_points.append((x, y))
                
            # Extract right eye points
            for idx in self.RIGHT_EYE_INDICES:
                x = int(landmarks[idx].x * iw)
                y = int(landmarks[idx].y * ih)
                right_eye_points.append((x, y))
                
            # Extract blendshapes
            if detection_result.face_blendshapes:
                for category in detection_result.face_blendshapes[0]:
                    if category.category_name == "eyeBlinkLeft":
                        left_blink_score = category.score
                    elif category.category_name == "eyeBlinkRight":
                        right_blink_score = category.score
                
            return left_eye_points, right_eye_points, left_blink_score, right_blink_score
            
        return None, None, 0.0, 0.0
