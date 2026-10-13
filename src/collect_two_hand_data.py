"""
Two-Hand ISL Dataset Collection Tool

Collects standard two-hand Indian Sign Language alphabet data
using a webcam and MediaPipe Hand Landmarker (num_hands=2).
Saves 126 landmark features (63 Left Hand + 63 Right Hand) + label.

Usage:
    python src/collect_two_hand_data.py
    python src/collect_two_hand_data.py --letters ABCDE
    python src/collect_two_hand_data.py --samples 300
    python src/collect_two_hand_data.py --resume
"""

import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
import sys


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = "models/hand_landmarker.task"
CSV_FILE = "dataset/isl_two_hand_landmarks.csv"

ALL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAMPLES_PER_LABEL = 300


# --------------------------------------------------
# AUTHENTIC TWO-HAND ISL SIGN DESCRIPTIONS
# --------------------------------------------------

TWO_HAND_SIGN_DESCRIPTIONS = {
    "A": "Dominant index touches non-dominant thumb tip",
    "B": "Both hands form circles / touching index and thumbs together",
    "C": "Curved hand forming C shape (framed with non-dominant base)",
    "D": "Dominant index touches non-dominant thumb and index (forms D)",
    "E": "Dominant index touches non-dominant index tip",
    "F": "Dominant index & middle rest across non-dominant index & middle",
    "G": "Dominant fist placed over non-dominant fist",
    "H": "Dominant flat palm placed across non-dominant flat palm",
    "I": "Dominant index touches non-dominant middle finger tip",
    "J": "Dominant index traces from non-dominant middle down palm",
    "K": "Dominant hooked index hooks into non-dominant index",
    "L": "Dominant index placed flat across non-dominant open palm",
    "M": "Dominant 3 fingers (index, middle, ring) rest on non-dominant palm",
    "N": "Dominant 2 fingers (index, middle) rest on non-dominant palm",
    "O": "Dominant index touches non-dominant ring finger tip",
    "P": "Dominant index & thumb circle touches non-dominant index tip",
    "Q": "Non-dominant thumb & index circle, dominant index hooks in",
    "R": "Dominant curled index finger rests on non-dominant palm",
    "S": "Dominant pinky finger hooks into non-dominant pinky",
    "T": "Dominant index touches edge/base of non-dominant index",
    "U": "Dominant index touches non-dominant pinky tip",
    "V": "Dominant V (peace sign) placed on non-dominant palm",
    "W": "Fingers of both hands interlocked or 3 fingers on palm",
    "X": "Both index fingers crossed forming an X shape",
    "Y": "Non-dominant index placed between dominant thumb and index",
    "Z": "Dominant bent hand placed at right angles against non-dominant palm"
}


# --------------------------------------------------
# CSV SETUP (126 Features + Label)
# --------------------------------------------------

def ensure_csv_exists():
    """Create CSV file with 126 landmark headers if missing."""
    os.makedirs("dataset", exist_ok=True)
    if not os.path.exists(CSV_FILE):
        header = []
        # Left Hand: 21 landmarks (63 features)
        for i in range(21):
            header.extend([f"lh_x{i}", f"lh_y{i}", f"lh_z{i}"])
        # Right Hand: 21 landmarks (63 features)
        for i in range(21):
            header.extend([f"rh_x{i}", f"rh_y{i}", f"rh_z{i}"])
        header.append("label")

        with open(CSV_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)


def count_existing_samples():
    """Count existing samples per label in the CSV."""
    counts = {}
    if not os.path.exists(CSV_FILE):
        return counts

    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 127:
                label = row[-1]
                counts[label] = counts.get(label, 0) + 1
    return counts


# --------------------------------------------------
# PARSE ARGUMENTS
# --------------------------------------------------

def parse_args():
    labels_to_collect = ALL_LABELS[:]
    samples = SAMPLES_PER_LABEL

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--letters" and i + 1 < len(sys.argv):
            letters = sys.argv[i + 1].upper()
            labels_to_collect = [ch for ch in letters if ch in ALL_LABELS]
            i += 2
        elif arg == "--samples" and i + 1 < len(sys.argv):
            samples = int(sys.argv[i + 1])
            i += 2
        elif arg == "--resume":
            counts = count_existing_samples()
            labels_to_collect = [lbl for lbl in ALL_LABELS if counts.get(lbl, 0) < samples]
            i += 1
        elif arg == "--help":
            print("Two-Hand ISL Data Collector")
            print("Usage: python src/collect_two_hand_data.py [--letters XYZ] [--samples N] [--resume]")
            sys.exit(0)
        else:
            i += 1

    return labels_to_collect, samples


# --------------------------------------------------
# TWO-HAND LANDMARK NORMALIZATION (126 Features)
# --------------------------------------------------

def normalize_two_hands(result):
    """
    Extracts, orders, and normalizes up to two hands into 126 features.
    Guarantees consistent ordering:
      - Slot 0..62:  Left Hand landmarks
      - Slot 63..125: Right Hand landmarks
    If only one hand is detected, the missing hand is zero-padded.
    """
    if not result.hand_landmarks:
        return None

    lh_landmarks = None
    rh_landmarks = None

    # Classify detected hands by MediaPipe handedness or horizontal position
    for idx, landmarks in enumerate(result.hand_landmarks):
        label = "Right"
        if result.handedness and idx < len(result.handedness):
            label = result.handedness[idx][0].category_name

        # Note: If mirrored webcam, left/right may be flipped
        if label == "Left" and lh_landmarks is None:
            lh_landmarks = landmarks
        elif label == "Right" and rh_landmarks is None:
            rh_landmarks = landmarks
        else:
            if lh_landmarks is None:
                lh_landmarks = landmarks
            elif rh_landmarks is None:
                rh_landmarks = landmarks

    def extract_and_normalize_single(hand_lms):
        if hand_lms is None:
            return [0.0] * 63  # Zero padding for missing hand

        coords = []
        wrist_x, wrist_y, wrist_z = hand_lms[0].x, hand_lms[0].y, hand_lms[0].z
        for lm in hand_lms:
            coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

        coords = np.array(coords, dtype=np.float32)
        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = coords / max_val
        return coords.tolist()

    lh_features = extract_and_normalize_single(lh_landmarks)
    rh_features = extract_and_normalize_single(rh_landmarks)

    return lh_features + rh_features  # Exactly 126 elements


# --------------------------------------------------
# MAIN COLLECTION LOOP
# --------------------------------------------------

def main():
    labels_to_collect, samples_per_label = parse_args()
    ensure_csv_exists()
    existing_counts = count_existing_samples()

    print()
    print("=" * 60)
    print("  TWO-HAND ISL DATASET COLLECTION TOOL")
    print("  Feature Dimension: 126 (2 x 21 landmarks x 3D coordinates)")
    print("=" * 60)
    print(f"Target Samples per Class: {samples_per_label}")
    print(f"Classes to Collect: {', '.join(labels_to_collect)}")
    print()

    # Configure MediaPipe for 2 hands
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,  # <--- CRUCIAL: Detects up to 2 hands
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )

    detector = HandLandmarker.create_from_options(options)
    from camera_utils import open_best_camera
    camera, _ = open_best_camera(preferred_idx=0)

    if not camera:
        print("ERROR: Could not open webcam.")
        return

    start_time = time.monotonic()
    total_new_collected = 0

    for idx, label in enumerate(labels_to_collect):
        existing = existing_counts.get(label, 0)
        needed = max(0, samples_per_label - existing)
        if needed == 0:
            print(f"Skipping {label}: already has {existing} samples")
            continue

        desc = TWO_HAND_SIGN_DESCRIPTIONS.get(label, "")
        print(f"\nLetter {idx + 1}/{len(labels_to_collect)}: [{label}]")
        print(f"  Description: {desc}")
        print(f"  Need {needed} more samples. Position both hands and press SPACE.")

        ready = False
        while not ready:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, f"Sign: {label} (2-Handed)", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Action: {desc}", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)
            cv2.putText(frame, "Press SPACE to start recording | Q to quit", (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Two-Hand ISL Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                ready = True
            elif key == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                detector.close()
                return

        collected = 0
        while collected < needed:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)

            num_detected = len(result.hand_landmarks) if result.hand_landmarks else 0

            # Draw visual landmarks
            if result.hand_landmarks:
                h, w, _ = frame.shape
                for hand_lms in result.hand_landmarks:
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Record if at least one hand is detected
            if num_detected > 0:
                features_126 = normalize_two_hands(result)
                if features_126 and len(features_126) == 126:
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(features_126 + [label])
                    collected += 1
                    total_new_collected += 1

            status_color = (0, 255, 0) if num_detected == 2 else (0, 165, 255)
            cv2.putText(frame, f"Recording: {label} [{collected}/{needed}]", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Hands Tracked: {num_detected}/2", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow("Two-Hand ISL Collector", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                detector.close()
                return

    camera.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\nCollection Complete! Total samples added: {total_new_collected}")
    print(f"Saved to: {CSV_FILE}")


if __name__ == "__main__":
    main()
