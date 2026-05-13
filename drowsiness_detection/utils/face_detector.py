import cv2

class FaceDetector:
    def __init__(self):
        """
        Initialize the OpenCV Haar Cascade Face Detection module.
        """
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_face(self, image):
        """
        Detect faces in the image.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            bbox: (x, y, w, h) of the primary detected face, or None if no face found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50)
        )
        
        if len(faces) > 0:
            # Return the largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            return tuple(faces[0])
            
        return None
