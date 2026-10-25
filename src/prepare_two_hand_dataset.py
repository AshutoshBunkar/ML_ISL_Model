"""
Two-Hand ISL Dataset Preparation from Image Folders

Extracts MediaPipe hand landmarks (num_hands=2) from ISL image directories
and compiles them into a 126-feature normalized CSV dataset.

Expected directory structure:
    dataset/raw/A/*.jpg (or .png)
    dataset/raw/B/*.jpg
    ...
    dataset/raw/Z/*.jpg

Output:
    dataset/isl_two_hand_landmarks.csv (126 features + 1 label column)

Usage:
    python src/prepare_two_hand_dataset.py --input dataset/raw
    python src/prepare_two_hand_dataset.py --input path/to/dataset --output dataset/isl_two_hand_landmarks.csv
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
import csv
import numpy as np
from pathlib import Path


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = os.path.join(_root_dir, "models", "hand_landmarker.task")
DEFAULT_INPUT = os.path.join(_root_dir, "dataset", "raw")
OUTPUT_FILE = os.path.join(_root_dir, "dataset", "isl_two_hand_landmarks.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------
# TWO-HAND NORMALIZATION
# --------------------------------------------------

def normalize_two_hands(hand_landmarks_list, handedness_list=None):
    """
    Extracts up to 2 hands and returns a 126-feature normalized vector.
    Left Hand: features 0..62
    Right Hand: features 63..125
    Missing hands are padded with 0.0.
    """
    if not hand_landmarks_list:
        return None

    lh_landmarks = None
    rh_landmarks = None

    for idx, landmarks in enumerate(hand_landmarks_list):
        category = "Right"
        if handedness_list and idx < len(handedness_list):
            category = handedness_list[idx][0].category_name

        if category == "Left" and lh_landmarks is None:
            lh_landmarks = landmarks
        elif category == "Right" and rh_landmarks is None:
            rh_landmarks = landmarks
        else:
            if lh_landmarks is None:
                lh_landmarks = landmarks
            elif rh_landmarks is None:
                rh_landmarks = landmarks

    def normalize_single_hand(landmarks):
        if landmarks is None:
            return [0.0] * 63

        coords = []
        wrist_x = landmarks[0].x
        wrist_y = landmarks[0].y
        wrist_z = landmarks[0].z

        for lm in landmarks:
            coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

        coords = np.array(coords, dtype=np.float32)
        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = coords / max_val
        return coords.tolist()

    lh_norm = normalize_single_hand(lh_landmarks)
    rh_norm = normalize_single_hand(rh_landmarks)

    return lh_norm + rh_norm  # Exactly 126 features


# --------------------------------------------------
# MAIN PROCESSOR
# --------------------------------------------------

def main():
    input_dir = DEFAULT_INPUT
    output_path = OUTPUT_FILE
    max_per_class = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--input" and i + 1 < len(sys.argv):
            input_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--max-per-class" and i + 1 < len(sys.argv):
            max_per_class = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    print()
    print("=" * 60)
    print("  TWO-HAND ISL DATASET PREPARATION TOOL")
    print("=" * 60)
    print(f"Source Directory: {input_dir}")
    print(f"Output CSV:      {output_path}")
    print()

    if not os.path.isdir(input_dir):
        print(f"Directory not found: {input_dir}")
        print("Please extract your raw two-hand ISL image dataset into:")
        print(f"  {input_dir}/A/, {input_dir}/B/, ... {input_dir}/Z/")
        return

    class_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and len(d) == 1 and d.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ])

    if not class_dirs:
        print(f"ERROR: No single-letter class directories (A-Z) found inside {input_dir}.")
        return

    print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")

    # MediaPipe setup (IMAGE mode, num_hands=2)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    detector = HandLandmarker.create_from_options(options)

    # Prepare header: 126 features + label
    header = []
    for idx in range(21):
        header.extend([f"lh_x{idx}", f"lh_y{idx}", f"lh_z{idx}"])
    for idx in range(21):
        header.extend([f"rh_x{idx}", f"rh_y{idx}", f"rh_z{idx}"])
    header.append("label")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_extracted = 0
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for class_name in class_dirs:
            label = class_name.upper()
            dir_path = os.path.join(input_dir, class_name)
            images = [f for f in os.listdir(dir_path) if Path(f).suffix.lower() in IMAGE_EXTENSIONS]

            if max_per_class:
                images = images[:max_per_class]

            extracted_for_class = 0
            for img_file in images:
                img_path = os.path.join(dir_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)

                if res.hand_landmarks:
                    feat_126 = normalize_two_hands(res.hand_landmarks, res.handedness)
                    if feat_126:
                        writer.writerow(feat_126 + [label])
                        extracted_for_class += 1
                        total_extracted += 1

            print(f"Class [{label}]: Processed {len(images)} images -> Extracted {extracted_for_class} landmark rows")

    detector.close()
    print()
    print("=" * 60)
    print(f"SUCCESS: Two-hand landmark dataset created with {total_extracted} samples!")
    print(f"Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
