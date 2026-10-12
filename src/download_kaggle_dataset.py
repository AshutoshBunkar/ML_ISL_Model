"""
Kaggle ISL Landmark Dataset Downloader & Adapter

Automates fetching and formatting the Indian Sign Language Hand Landmarks
Dataset from Kaggle: shreyaskhare/indian-sign-language-hand-landmarks-dataset

Usage:
    python src/download_kaggle_dataset.py
    python src/download_kaggle_dataset.py --file path/to/downloaded.csv
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np

DATASET_SLUG = "shreyaskhare/indian-sign-language-hand-landmarks-dataset"
TARGET_CSV = "dataset/isl_two_hand_landmarks.csv"
KAGGLE_DIR = os.path.expanduser("~/.kaggle")
KAGGLE_JSON = os.path.join(KAGGLE_DIR, "kaggle.json")


def format_and_save(source_csv_path):
    """Inspects and standardizes the Kaggle CSV into the project dataset format."""
    print(f"\nProcessing CSV: {source_csv_path}")
    df = pd.read_csv(source_csv_path)
    print(f"Original Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Locate the label column
    label_col = None
    for col in ["label", "Label", "class", "Class", "target", "Target"]:
        if col in df.columns:
            label_col = col
            break

    if not label_col:
        # Check last column
        label_col = df.columns[-1]

    print(f"Identified Label Column: '{label_col}'")
    unique_labels = sorted(df[label_col].astype(str).unique())
    print(f"Classes ({len(unique_labels)}): {', '.join(unique_labels[:10])}... ({len(unique_labels)} total)")

    os.makedirs("dataset", exist_ok=True)
    df.to_csv(TARGET_CSV, index=False)
    print(f"\nSUCCESS: Standardized dataset saved to: {TARGET_CSV}")
    print(f"Total Samples: {len(df)}")
    print(f"Features: {df.shape[1] - 1}")
    print("\nYou can now train your model using:")
    print(f"  venv\\Scripts\\python src/train_model.py --dataset {TARGET_CSV} --epochs 50")


def try_download_via_kagglehub():
    """Attempts downloading via kagglehub if authenticated."""
    try:
        import kagglehub
        print(f"Connecting to Kaggle to download '{DATASET_SLUG}'...")
        path = kagglehub.dataset_download(DATASET_SLUG)
        print(f"Downloaded to cache: {path}")

        # Find CSV file inside the downloaded directory
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            print("No CSV found in the downloaded package.")
            return False

        source_csv = os.path.join(path, csv_files[0])
        format_and_save(source_csv)
        return True
    except Exception as e:
        print(f"\nAutomated download could not proceed: {e}")
        return False


def main():
    # If user provided a local file path
    for i, arg in enumerate(sys.argv):
        if arg == "--file" and i + 1 < len(sys.argv):
            format_and_save(sys.argv[i + 1])
            return

    # Check for existing downloaded file in dataset folder
    candidates = [
        "dataset/data.csv",
        "dataset/indian_sign_language.csv",
        "dataset/gesture_dataset.csv",
        "dataset/landmarks.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"Found existing CSV at: {c}")
            format_and_save(c)
            return

    print("=" * 65)
    print("  KAGGLE ISL TWO-HAND LANDMARKS DATASET HELPER")
    print(f"  Dataset: {DATASET_SLUG}")
    print("=" * 65)

    # Try kagglehub if kaggle.json exists
    if os.path.exists(KAGGLE_JSON):
        print(f"Found Kaggle API key at {KAGGLE_JSON}.")
        if try_download_via_kagglehub():
            return

    print("\nTo obtain this dataset from Kaggle, follow these simple steps:\n")
    print("  1. Open the Kaggle dataset page in your browser:")
    print(f"     -> https://www.kaggle.com/datasets/{DATASET_SLUG}")
    print()
    print("  2. Click the 'Download' button (top right of the dataset page).")
    print()
    print("  3. Extract the downloaded ZIP file.")
    print()
    print("  4. Run this script pointing to the extracted CSV file:")
    print("     venv\\Scripts\\python src/download_kaggle_dataset.py --file path\\to\\downloaded.csv")
    print()
    print("  (Or place the extracted CSV directly into 'dataset/data.csv' and re-run)")
    print("=" * 65)


if __name__ == "__main__":
    main()
