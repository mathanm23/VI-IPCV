import cv2
import numpy as np
import tensorflow as tf

def preprocess_eye(eye_img, target_size=(224, 224)):
    """
    Preprocesses the eye image for the MobileNetV3 model.
    
    Args:
        eye_img: Cropped BGR eye image from OpenCV
        target_size: Tuple (width, height) for the model input
        
    Returns:
        Processed image tensor ready for prediction.
    """
    if eye_img is None or eye_img.size == 0:
        return None
        
    # Resize to the model's expected input size
    resized_img = cv2.resize(eye_img, target_size)
    
    # Convert BGR to RGB (models are typically trained on RGB)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # Add batch dimension: (1, 224, 224, 3)
    batch_img = np.expand_dims(rgb_img, axis=0)
    
    # Scale to [0, 1] - this is the most common preprocessing for custom models
    preprocessed_img = batch_img / 255.0
    
    return preprocessed_img
