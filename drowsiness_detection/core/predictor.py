import os
import tensorflow as tf
import numpy as np

class EyeStatePredictor:
    def __init__(self, model_path="model/mobilenetv3_model.h5"):
        """
        Loads the pre-trained MobileNetV3 model once.
        
        Args:
            model_path: Path to the .h5 model file.
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Loads the TensorFlow/Keras model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please add the model file.")
            
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully.")

    def predict(self, preprocessed_eye):
        """
        Predicts whether the eye is Open or Closed.
        
        Args:
            preprocessed_eye: Numpy array of shape (1, 224, 224, 3)
            
        Returns:
            (state, confidence): ("Open"/"Closed"/"Unknown", float)
        """
        if self.model is None or preprocessed_eye is None:
            return "Unknown", 0.0
            
        # Perform inference
        try:
            # Using model() directly is faster than model.predict() for single frames
            predictions = self.model(preprocessed_eye, training=False).numpy()
        except Exception as e:
            print(f"Inference error: {e}")
            return "Unknown", 0.0
        
        if predictions.shape[1] == 1:
            # Single sigmoid output (0=Closed, 1=Open)
            prob = float(predictions[0][0])
            if prob >= 0.5:
                return "Open", prob
            else:
                return "Closed", 1.0 - prob
        else:
            # Categorical output (softmax)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            if class_idx == 1:
                return "Open", confidence
            else:
                return "Closed", confidence
