# Driver Drowsiness Detection using Facial Landmark Analysis

A real-time driver drowsiness detection system utilizing facial landmark analysis with MediaPipe and an eye-state classification model using MobileNetV3.

## Project Description

This system continuously monitors a driver's eyes using a webcam feed. It detects the face and extracts precise eye regions using MediaPipe Face Mesh. The cropped eye images are then preprocessed and passed to a lightweight, pre-trained MobileNetV3 model to classify whether the eyes are "Open" or "Closed". By tracking consecutive closed-eye frames, the system determines the driver's state and displays a "DROWSY" warning when a defined threshold is exceeded. 

## Folder Structure

```
drowsiness_detection/
│
├── model/
│   └── mobilenetv3_model.h5       # Place your pre-trained model here
│
├── utils/
│   ├── __init__.py
│   ├── face_detector.py           # Detects primary face bounding box
│   ├── landmark_detector.py       # Extracts 468 facial landmarks
│   ├── eye_extractor.py           # Crops left and right eye regions
│   ├── preprocessing.py           # Resizes to 224x224 and normalizes
│
├── core/
│   ├── __init__.py
│   ├── predictor.py               # Loads .h5 model and predicts eye state
│   ├── drowsiness_logic.py        # Maintains frame counter for threshold
│
├── main.py                        # Main application script
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

## Setup Instructions

1. **Install Python 3.10+**.
2. **Install Dependencies**:
   Open a terminal in the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Add the Model**:
   You must provide your own pre-trained `.h5` MobileNetV3 model. Place the file inside the `model/` folder and name it EXACTLY `mobilenetv3_model.h5`. The model should accept `(224, 224, 3)` inputs and output probabilities or binary classes (0 for Closed, 1 for Open). If your labels are swapped, adjust the logic in `core/predictor.py`.

## How to Run

Execute the main script:
```bash
python main.py
```
Press `q` at any time while the webcam window is active to exit the application.

## Sample Output

- Upon starting, your webcam will activate.
- A bounding box will be drawn around your face.
- Above your face, `L: Open` or `L: Closed` (and `R: ...` for the right eye) will track the real-time state of your eyes.
- The `FPS` will be displayed in the top-left corner.
- The state text (`STATUS: AWAKE` in green) will be displayed below the FPS. If you close your eyes for more than 15 consecutive frames, the text will turn red and display `STATUS: DROWSY`.
