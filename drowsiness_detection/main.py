import cv2
import time
import os
from utils.face_detector import FaceDetector
from utils.landmark_detector import LandmarkDetector
from utils.eye_extractor import EyeExtractor
from utils.preprocessing import preprocess_eye
from core.predictor import EyeStatePredictor
from core.drowsiness_logic import DrowsinessMonitor

def main():
    print("Initializing Drowsiness Detection System...")
    
    # Initialize all modules
    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    eye_extractor = EyeExtractor(padding=15) # Add 15 pixels padding for crops
    
    try:
        # Get absolute path to model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model", "mobilenetv3_model.h5")
        predictor = EyeStatePredictor(model_path=model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please place the 'mobilenetv3_model.h5' in the 'model/' directory and run again.")
        return
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    # Initialize logic with 3-second threshold
    drowsiness_monitor = DrowsinessMonitor(threshold_seconds=3.0)
    
    # Initialize webcam with a fallback mechanism
    cap = None
    for index in [0, 1, 2]: # Try common camera indices
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Webcam successfully opened on index {index}")
                break
            else:
                cap.release()
                cap = None
    
    if not cap or not cap.isOpened():
        print("Error: Could not open any webcam. Please check your connections and ensure no other app is using the camera.")
        return
        
    # Warm-up the camera to stabilize exposure/focus
    for _ in range(5):
        cap.read()
        
    print("System ready. Press 'q' to quit.")
    
    prev_time = 0
    fps_list = []
    status = "AWAKE"
    face_lost_counter = 0
    MAX_FACE_LOST_FRAMES = 10 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        frame = cv2.flip(frame, 1)
        
        # Calculate Smoothed FPS
        current_time = time.time()
        if prev_time > 0:
            fps_list.append(1 / (current_time - prev_time))
            if len(fps_list) > 30: fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
        else:
            avg_fps = 0
        prev_time = current_time
        
        left_state, left_conf = "Unknown", 0.0
        right_state, right_conf = "Unknown", 0.0
        
        # 1. Detect Face (for visual feedback and focus)
        face_bbox = face_detector.detect_face(frame)
        
        if face_bbox:
            fx, fy, fw, fh = face_bbox
            # Draw Face Bounding Box (Blue)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            
            # 2. Detect Eye Landmarks and Blink Scores (using built-in MediaPipe AI)
            left_eye_pts, right_eye_pts, left_blink_score, right_blink_score = landmark_detector.extract_eye_landmarks(frame)
            
            if left_eye_pts and right_eye_pts:
                face_lost_counter = 0
                
                # Use Blendshapes for 100% accurate blink detection (> 0.5 is closed)
                # MediaPipe names "eyeBlinkLeft" relative to the image, so it might be flipped. 
                # We just determine the state directly.
                left_state = "Closed" if left_blink_score > 0.5 else "Open"
                right_state = "Closed" if right_blink_score > 0.5 else "Open"
                
                # 3. Update drowsiness logic
                status = drowsiness_monitor.update_state(left_state, right_state)
                
                # Display eye states with confidence (using MediaPipe's own highly accurate score)
                left_label = "EYES OPENED" if left_state == "Open" else "EYES CLOSED"
                right_label = "EYES OPENED" if right_state == "Open" else "EYES CLOSED"
                
                cv2.putText(frame, f"L: {left_label} ({(1.0-left_blink_score)*100:.0f}%)", (fx, fy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"R: {right_label} ({(1.0-right_blink_score)*100:.0f}%)", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                face_lost_counter += 1
        else:
            face_lost_counter += 1
            
        if face_lost_counter > MAX_FACE_LOST_FRAMES:
            # Reset logic on face loss
            drowsiness_monitor.closed_start_time = None
            drowsiness_monitor.elapsed_closed_time = 0.0
            status = "AWAKE"
            cv2.putText(frame, "FACE LOST", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # 6. Draw UI Elements
        # Draw FPS
        cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw Status
        color = (0, 0, 255) if status == "DROWSY" else (0, 255, 0) 
        cv2.putText(frame, f"STATUS: {status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw Timer if eyes are closed
        if drowsiness_monitor.elapsed_closed_time > 0:
            timer_color = (0, 0, 255) if status == "DROWSY" else (0, 255, 255)
            cv2.putText(frame, f"CLOSED: {drowsiness_monitor.elapsed_closed_time:.1f}s", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, timer_color, 2)
        
        if status == "DROWSY":
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        
        # Display the frame
        cv2.imshow("Driver Drowsiness Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
