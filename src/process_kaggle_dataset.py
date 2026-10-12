"""
Process Kaggle Two-Hand ISL Dataset (Spatial Relative-Coordinate Pipeline)

Transforms the raw 50,859-sample Kaggle dataset
(eraakash/indian-sign-language-hand-landmarks-dataset)
into our project's standardized 126-feature format with:

1. Deterministic screen-space sorting:
   - Hand 1 = Screen-left hand (smaller wrist x)
   - Hand 2 = Screen-right hand (larger wrist x)
2. Preserved inter-hand relative geometry:
   - Hand 1 centered at Hand 1 wrist (wrist at 0, 0, 0)
   - Hand 2 centered relative to Hand 1 wrist (wrist at W2 - W1!)
   This preserves inter-hand distance, relative height (dy), and horizontal separation (dx),
   permanently resolving D vs Y, M vs N, and Z vs H confusions!
3. Single-hand slot invariance:
   - For 1-hand signs (C, I, L, O, U, V), both [Hand, 0] and [0, Hand] are trained,
     allowing signers to use either hand.
4. Per-sample joint scale normalization:
   - Divides by max absolute coordinate across all 126 features.

Output:
    dataset/isl_two_hand_landmarks.csv (62,753 clean samples)
"""

import os
import string
import pandas as pd
import numpy as np

RAW_CSV_PATH = os.path.expanduser(
    r"~/.cache/kagglehub/datasets/eraakash/indian-sign-language-hand-landmarks-dataset/versions/1/Indian Sign Language Gesture Landmarks.csv"
)
OUTPUT_CSV_PATH = "dataset/isl_two_hand_landmarks.csv"

LETTER_MAP = dict(enumerate(string.ascii_uppercase))


def process_pair(h1, h2):
    """Normalize a pair of hands relative to Hand 1's wrist and joint max extent."""
    w = h1[:, :3]
    h1_out = np.zeros_like(h1)
    h2_out = np.zeros_like(h2)
    for i in range(21):
        idx = i * 3
        h1_out[:, idx:idx+3] = h1[:, idx:idx+3] - w
        h2_out[:, idx:idx+3] = h2[:, idx:idx+3] - w
    X = np.hstack([h1_out, h2_out])
    m = np.max(np.abs(X), axis=1, keepdims=True)
    m[m == 0] = 1.0
    return X / m


def main():
    print("=" * 70)
    print("  PROCESSING TWO-HAND ISL DATASET (RELATIVE-COORDINATE PIPELINE)")
    print(f"  Source: {RAW_CSV_PATH}")
    print(f"  Target: {OUTPUT_CSV_PATH}")
    print("=" * 70)

    if not os.path.exists(RAW_CSV_PATH):
        print(f"ERROR: Raw CSV not found at: {RAW_CSV_PATH}")
        return

    print("Loading raw CSV...")
    df = pd.read_csv(RAW_CSV_PATH)
    print(f"Loaded {len(df):,} samples with {df.shape[1]} columns.")

    lh_cols = [c for c in df.columns if c.startswith("left_hand_")]
    rh_cols = [c for c in df.columns if c.startswith("right_hand_")]

    assert len(lh_cols) == 63 and len(rh_cols) == 63

    lh_raw = df[lh_cols].values.astype(np.float32)
    rh_raw = df[rh_cols].values.astype(np.float32)

    lh_valid = (lh_raw[:, 0] != -1.0)
    rh_valid = (rh_raw[:, 0] != -1.0)
    single_mask = (lh_valid & ~rh_valid)
    dual_mask = (lh_valid & rh_valid)

    print(f"Single-hand samples: {single_mask.sum():,}")
    print(f"Two-hand samples:    {dual_mask.sum():,}")

    # -------------------------------------------------------------------------
    # 1. Dual-Hand Samples: Deterministic screen-space sort (smaller x -> Hand 1)
    # -------------------------------------------------------------------------
    swap_mask = (rh_raw[:, 0] < lh_raw[:, 0]) & dual_mask
    h1_primary = lh_raw.copy()
    h2_primary = rh_raw.copy()
    h1_primary[swap_mask] = rh_raw[swap_mask]
    h2_primary[swap_mask] = lh_raw[swap_mask]

    X_dual = process_pair(h1_primary[dual_mask], h2_primary[dual_mask])
    y_dual = df.loc[dual_mask, "target"].values

    # -------------------------------------------------------------------------
    # 2. Single-Hand Samples: Center at wrist, augment both Slot 1 and Slot 2
    # -------------------------------------------------------------------------
    w_single = lh_raw[single_mask, :3]
    h_single = np.zeros_like(lh_raw[single_mask])
    for i in range(21):
        idx = i * 3
        h_single[:, idx:idx+3] = lh_raw[single_mask, idx:idx+3] - w_single
    m_s = np.max(np.abs(h_single), axis=1, keepdims=True)
    m_s[m_s == 0] = 1.0
    h_single = h_single / m_s

    zeros_s = np.zeros_like(h_single)
    X_single_slot1 = np.hstack([h_single, zeros_s])
    X_single_slot2 = np.hstack([zeros_s, h_single])
    y_single = df.loc[single_mask, "target"].values

    # -------------------------------------------------------------------------
    # 3. Combine All Clean Sets
    # -------------------------------------------------------------------------
    X_all = np.vstack([X_dual, X_single_slot1, X_single_slot2])
    y_target_all = np.concatenate([y_dual, y_single, y_single])
    labels_all = pd.Series(y_target_all).map(LETTER_MAP).values

    header = []
    for i in range(21):
        header.extend([f"lh_x{i}", f"lh_y{i}", f"lh_z{i}"])
    for i in range(21):
        header.extend([f"rh_x{i}", f"rh_y{i}", f"rh_z{i}"])

    out_df = pd.DataFrame(X_all, columns=header)
    out_df["label"] = labels_all

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    print(f"Saving standardized dataset to {OUTPUT_CSV_PATH}...")
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print()
    print("=" * 70)
    print("DATASET PROCESSING COMPLETE!")
    print(f"Total Samples: {len(out_df):,}")
    print("Feature Dimensions: 126 (63 Hand 1 + 63 Hand 2)")
    print("Classes: 26 (A-Z)")
    print("Class Distribution:")
    print(out_df["label"].value_counts().sort_index())
    print("=" * 70)


if __name__ == "__main__":
    main()

