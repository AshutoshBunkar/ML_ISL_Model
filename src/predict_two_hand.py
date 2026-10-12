"""
Two-Hand ISL Real-Time Prediction

Real-time Indian Sign Language recognition supporting two hands (126 features).
Automatically detects up to 2 hands with MediaPipe, normalizes features,
and predicts gestures in real time using a webcam.

Usage:
    python src/predict_two_hand.py
    python src/predict_two_hand.py --camera 0
"""

import sys
import os
import subprocess
import time
from collections import deque, Counter

# Auto-redirect to venv python if needed
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
_venv_python = os.path.join(_root_dir, "venv", "Scripts", "python.exe")

if os.path.exists(_venv_python):
    _curr = os.path.normcase(os.path.abspath(sys.executable))
    _target = os.path.normcase(os.path.abspath(_venv_python))
    if _curr != _target:
        sys.exit(subprocess.run([_venv_python] + sys.argv).returncode)

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

MODEL_PATH = "models/isl_model.keras"
HAND_MODEL_PATH = "models/hand_landmarker.task"
LABELS_PATH = "models/labels.npy"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please train your 126-feature two-hand model first: python src/train_model.py")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
expected_features = model.input_shape[1]
print(f"Loaded model expecting {expected_features} input features.")

labels = np.load(LABELS_PATH, allow_pickle=True)
print(f"Loaded {len(labels)} classes: {labels}")


# --------------------------------------------------
# MEDIAPIPE (num_hands=2)
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.35,
    min_hand_presence_confidence=0.35,
    min_tracking_confidence=0.35
)
detector = HandLandmarker.create_from_options(options)


# --------------------------------------------------
# FEATURE EXTRACTION (126 features)
# --------------------------------------------------

def extract_features(result):
    """
    Extract and normalize up to 2 detected hands into a 126-length feature vector
    matching the relative-coordinate training format.

    - 0 hands: returns None.
    - 1 hand: wrist-relative, scaled to [-1, 1], placed in Slot 1 (indices 0..62) with Slot 2 as 0.0s (63..125).
    - 2+ hands:
        * Deterministic screen-space sort:
            Hand 1 = screen-left hand (smaller wrist x).
            Hand 2 = screen-right hand (larger wrist x).
        * Hand 1 centered at Hand 1 wrist (P1 - W1). Hand 1 wrist is at (0, 0, 0).
        * Hand 2 centered relative to Hand 1 wrist (P2 - W1). Hand 2 wrist is at (W2 - W1).
        * Joint scale: normalized by max absolute coordinate across all 126 features.
    """
    if not result or not result.hand_landmarks:
        return None

    num_hands = len(result.hand_landmarks)

    if num_hands == 1:
        h = result.hand_landmarks[0]
        wx, wy, wz = h[0].x, h[0].y, h[0].z
        coords = []
        for lm in h:
            coords.extend([lm.x - wx, lm.y - wy, lm.z - wz])
        coords = np.array(coords, dtype=np.float32)
        m = np.max(np.abs(coords))
        if m > 0:
            coords = coords / m
        return np.concatenate([coords, np.zeros(63, dtype=np.float32)])

    # 2 or more hands: sort by wrist x (screen-left vs screen-right)
    hA = result.hand_landmarks[0]
    hB = result.hand_landmarks[1]

    if hA[0].x <= hB[0].x:
        h1, h2 = hA, hB
    else:
        h1, h2 = hB, hA

    w1x, w1y, w1z = h1[0].x, h1[0].y, h1[0].z

    h1_coords = []
    for lm in h1:
        h1_coords.extend([lm.x - w1x, lm.y - w1y, lm.z - w1z])

    h2_coords = []
    for lm in h2:
        h2_coords.extend([lm.x - w1x, lm.y - w1y, lm.z - w1z])

    feat = np.array(h1_coords + h2_coords, dtype=np.float32)
    m = np.max(np.abs(feat))
    if m > 0:
        feat = feat / m
    return feat


# --------------------------------------------------
# CAMERA SETUP
# --------------------------------------------------

camera_idx = 0
for i, a in enumerate(sys.argv):
    if a in ("--camera", "-c") and i + 1 < len(sys.argv):
        try:
            camera_idx = int(sys.argv[i + 1])
        except ValueError:
            pass

camera = cv2.VideoCapture(camera_idx)
if not camera.isOpened():
    print(f"Could not open camera {camera_idx}. Trying default camera 0...")
    camera = cv2.VideoCapture(0)


# --------------------------------------------------
# MAIN REAL-TIME LOOP
# --------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  TWO-HAND ISL REAL-TIME PREDICTION STARTED")
    print("  Hold both hands in frame to predict signs.")
    print("  Press 'Q' to quit.")
    print("=" * 60)

    start_time = time.monotonic()
    history = deque(maxlen=10)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - start_time) * 1000)

        result = detector.detect_for_video(mp_image, ts)

        predicted_sign = "---"
        confidence = 0.0
        hands_count = len(result.hand_landmarks) if result.hand_landmarks else 0

        # Draw hand landmarks
        if result.hand_landmarks:
            for hand_idx, lms in enumerate(result.hand_landmarks):
                # Color code: Left Hand in Cyan, Right Hand in Green
                color = (255, 255, 0) if hand_idx == 0 else (0, 255, 128)
                for lm in lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, color, -1)

            features = extract_features(result)
            if features is not None and len(features) == expected_features:
                probs = model.predict(features.reshape(1, expected_features), verbose=0)[0]
                best_idx = np.argmax(probs)
                conf = probs[best_idx] * 100.0

                if conf >= 65.0:
                    history.append(labels[best_idx])
                    most_common = Counter(history).most_common(1)[0][0]
                    predicted_sign = most_common
                    confidence = conf
                else:
                    history.append("---")

        # UI Overlay
        cv2.rectangle(frame, (20, 20), (380, 140), (20, 20, 30), -1)
        cv2.rectangle(frame, (20, 20), (380, 140), (0, 212, 255), 2)

        cv2.putText(frame, f"Sign: {predicted_sign}", (35, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (35, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        cv2.putText(frame, f"Hands in View: {hands_count}/2", (35, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100) if hands_count == 2 else (0, 165, 255), 2)

        cv2.imshow("Two-Hand ISL Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
