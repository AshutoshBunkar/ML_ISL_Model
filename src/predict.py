"""
ISL Real-Time Prediction (OpenCV-only)

Lightweight real-time ISL alphabet recognition using webcam.
For the full GUI application, use: python src/app.py

Usage:
    python src/predict.py
"""
import sys
import os
import subprocess

# Auto-redirect to venv python if run with global python
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
import time

from collections import deque, Counter


sys.path.insert(0, _script_dir)


# ==================================================
# FILE PATHS
# ==================================================

MODEL_PATH = os.path.join(_root_dir, "models", "isl_model.keras")
HAND_MODEL_PATH = os.path.join(_root_dir, "models", "hand_landmarker.task")
LABELS_PATH = os.path.join(_root_dir, "models", "labels.npy")


# ==================================================
# LOAD MODEL
# ==================================================

model = tf.keras.models.load_model(MODEL_PATH)

labels = np.load(
    LABELS_PATH,
    allow_pickle=True
)

print("AI model loaded successfully!")
print(f"Classes: {labels}")
print(f"Number of classes: {len(labels)}")


# ==================================================
# MEDIAPIPE SETUP
# ==================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


expected_features = model.input_shape[1]
print(f"Model expects {expected_features} input features.")

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2 if expected_features == 126 else 1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

detector = HandLandmarker.create_from_options(options)


# ==================================================
# NORMALIZE LANDMARKS
# ==================================================

def normalize_landmarks(hand_landmarks):
    landmarks = []
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    wrist_z = hand_landmarks[0].z

    for landmark in hand_landmarks:
        x = landmark.x - wrist_x
        y = landmark.y - wrist_y
        z = landmark.z - wrist_z
        landmarks.extend([x, y, z])

    landmarks = np.array(landmarks, dtype=np.float32)
    max_value = np.max(np.abs(landmarks))
    if max_value > 0:
        landmarks = landmarks / max_value
    return landmarks


def normalize_two_hands(result):
    """
    Normalize up to 2 detected hands into 126 features matching the relative-coordinate training format.

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


# ==================================================
# CAMERA
# ==================================================

# Determine camera index
camera_idx = 0
for i, a in enumerate(sys.argv):
    if a in ("--camera", "-c") and i + 1 < len(sys.argv):
        try:
            camera_idx = int(sys.argv[i + 1])
        except ValueError:
            pass

from camera_utils import open_best_camera
camera, active_cam_info = open_best_camera(preferred_idx=camera_idx)
if not camera:
    print("ERROR: Could not open camera. Please check camera connections and permissions.")
    sys.exit(1)
current_camera_idx = active_cam_info["index"]


# ==================================================
# SETTINGS
# ==================================================

start_time = time.monotonic()

prediction_history = deque(
    maxlen=10
)

CONFIDENCE_THRESHOLD = 0.70


# ==================================================
# TEXT BUFFER AND LETTER CONFIRMATION
# ==================================================

text_buffer = ""

# Letter confirmation system
CONFIRM_FRAMES = 15        # Frames of stable prediction to confirm
COOLDOWN_FRAMES = 20       # Cooldown frames after confirmation
SAME_LETTER_CHANGE = True  # Require sign change before same letter

stable_count = 0
last_stable_label = None
cooldown_counter = 0
last_confirmed_label = None
need_change_before_repeat = False


# ==================================================
# CURRENT PREDICTION
# ==================================================

predicted_label = "..."

confidence = 0.0


# ==================================================
# MAIN LOOP
# ==================================================

print()
print("=" * 50)
print("ISL Real-Time Recognition")
print("=" * 50)
print()
print("Controls:")
print("  SPACE     = Add current letter")
print("  -         = Add space to text")
print("  BACKSPACE = Delete last character")
print("  C         = Clear text")
print("  Q         = Quit")
print()
print("Auto-confirm: Hold a sign steady to auto-add it.")
print()

WINDOW_NAME = "Indian Sign Language Recognition"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 720)

while True:

    # Check if window was closed via [X] button
    try:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            break
    except Exception:
        pass

    success, frame = camera.read()

    if not success or frame is None or frame.size == 0:
        time.sleep(0.02)
        continue


    # --------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape


    # --------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------
    # MEDIAPIPE IMAGE
    # --------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------

    timestamp_ms = int(
        (
            time.monotonic()
            - start_time
        ) * 1000
    )


    # --------------------------------------------------
    # DETECT HAND
    # --------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ==================================================
    # HAND DETECTED
    # ==================================================

    hand_detected = False

    if result.hand_landmarks:
        hand_detected = True

        # Extract features according to model requirements
        if expected_features == 126:
            features = normalize_two_hands(result)
            if features is not None:
                features = features.reshape(1, 126)
        else:
            features = normalize_landmarks(result.hand_landmarks[0])
            features = features.reshape(1, 63)

        if features is not None:
            prediction = model.predict(
                features,
                verbose=0
            )

        predicted_index = np.argmax(
            prediction[0]
        )

        confidence = float(
            prediction[0][predicted_index]
        )

        current_label = str(
            labels[predicted_index]
        )

        # --------------------------------------------------
        # CONFIDENCE FILTER + SMOOTHING
        # --------------------------------------------------

        if confidence >= CONFIDENCE_THRESHOLD:

            prediction_history.append(
                current_label
            )

            predicted_label = Counter(
                prediction_history
            ).most_common(1)[0][0]

        else:

            prediction_history.clear()
            predicted_label = "..."

        # --------------------------------------------------
        # LETTER CONFIRMATION (Auto-confirm)
        # --------------------------------------------------

        if cooldown_counter > 0:
            cooldown_counter -= 1

        elif predicted_label != "..." and predicted_label != "Unknown":

            if predicted_label == last_stable_label:
                stable_count += 1
            else:
                stable_count = 1
                last_stable_label = predicted_label
                need_change_before_repeat = False

            # Auto-confirm after stable frames
            if stable_count >= CONFIRM_FRAMES:
                if not need_change_before_repeat:
                    text_buffer += predicted_label
                    print(f"Auto-confirmed: {predicted_label} | Text: {text_buffer}")

                    last_confirmed_label = predicted_label
                    need_change_before_repeat = True
                    stable_count = 0
                    cooldown_counter = COOLDOWN_FRAMES

        # --------------------------------------------------
        # DRAW LANDMARKS & SKELETON FOR ALL DETECTED HANDS
        # --------------------------------------------------
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]

        for hand_idx, h_lms in enumerate(result.hand_landmarks):
            line_color = (0, 255, 0) if hand_idx == 0 else (255, 200, 0)
            for landmark in h_lms:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            for start_idx, end_idx in connections:
                start = h_lms[start_idx]
                end = h_lms[end_idx]
                x1 = int(start.x * width)
                y1 = int(start.y * height)
                x2 = int(end.x * width)
                y2 = int(end.y * height)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)


    else:

        prediction_history.clear()
        predicted_label = "..."
        confidence = 0.0
        stable_count = 0
        last_stable_label = None
        need_change_before_repeat = False


    # ==================================================
    # DISPLAY
    # ==================================================

    # Dark overlay bar at top
    cv2.rectangle(frame, (0, 0), (width, 280), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (width, 280), (50, 50, 50), 2)

    # Title
    cv2.putText(
        frame, "ISL Recognition System",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 200, 255), 2
    )

    # Current sign
    sign_color = (0, 255, 0) if hand_detected else (100, 100, 100)
    cv2.putText(
        frame, f"Sign: {predicted_label}",
        (15, 70), cv2.FONT_HERSHEY_SIMPLEX,
        1.3, sign_color, 3
    )

    # Confidence
    conf_color = (0, 255, 255)
    if confidence < 0.7:
        conf_color = (0, 100, 255)
    cv2.putText(
        frame, f"Confidence: {confidence * 100:.1f}%",
        (15, 105), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, conf_color, 2
    )

    # Status
    if float(np.mean(frame)) < 0.5:
        status = "⚠️ SHUTTER CLOSED (Open slider / Fn+F10)"
        status_color = (0, 165, 255)
        cv2.putText(
            frame, "WEBCAM LENS SHUTTER APPEARS CLOSED",
            (15, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.putText(
            frame, "Slide open the physical privacy cover on your laptop bezel or press Fn+F10",
            (15, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1
        )
    else:
        status = "Hand Detected" if hand_detected else "Show your hand"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(
        frame, f"Status: {status}",
        (15, 135), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, status_color, 2
    )

    # Confirmation progress
    if stable_count > 0 and not need_change_before_repeat:
        progress = min(stable_count / CONFIRM_FRAMES, 1.0)
        bar_width = 200
        cv2.rectangle(frame, (15, 150), (15 + bar_width, 165), (50, 50, 50), -1)
        cv2.rectangle(frame, (15, 150), (15 + int(bar_width * progress), 165), (0, 200, 0), -1)
        cv2.putText(
            frame, "Confirming...",
            (220, 163), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (200, 200, 200), 1
        )

    # Text buffer
    cv2.putText(
        frame, f"Text: {text_buffer}",
        (15, 200), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255, 255, 255), 2
    )

    # Controls
    cv2.putText(
        frame,
        "SPACE=Add  -=Space  BS=Delete  C=Clear  Q=Quit",
        (15, 240), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (160, 160, 160), 1
    )

    # Supported classes
    cv2.putText(
        frame, f"Supported: {', '.join(labels)}",
        (15, 268), cv2.FONT_HERSHEY_SIMPLEX,
        0.4, (120, 120, 120), 1
    )

    # Show camera frame
    cv2.imshow(
        "Indian Sign Language Recognition",
        frame
    )


    # ==================================================
    # KEYBOARD INPUT
    # ==================================================

    key = cv2.waitKey(1) & 0xFF

    # ADD LETTER (Space)
    if key == 32:
        if predicted_label not in ("...", "Unknown"):
            text_buffer += predicted_label
            print(f"Added: {predicted_label} | Text: {text_buffer}")

    # ADD SPACE (- key)
    elif key == ord("-"):
        text_buffer += " "
        print(f"Space added | Text: {text_buffer}")

    # BACKSPACE
    elif key == 8:
        if len(text_buffer) > 0:
            text_buffer = text_buffer[:-1]
            print(f"Backspace | Text: {text_buffer}")

    # CLEAR
    elif key == ord("c"):
        text_buffer = ""
        print("Text cleared.")

    # SWITCH CAMERA
    elif key == ord("s"):
        new_idx = 1 if current_camera_idx == 0 else 0
        print(f"Switching to camera {new_idx}...")
        camera.release()
        new_cam, new_idx = open_camera(new_idx)
        if new_cam:
            camera = new_cam
            current_camera_idx = new_idx
        else:
            camera, current_camera_idx = open_camera(current_camera_idx)

    # QUIT
    elif key == ord("q"):
        break


# ==================================================
# CLEANUP
# ==================================================

camera.release()
cv2.destroyAllWindows()
detector.close()

print()
print("Final text:", text_buffer)