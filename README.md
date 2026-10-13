# 🇮🇳 Indian Sign Language (ISL) Recognition System
### Real-Time Two-Hand Relative-Coordinate Pipeline, Deep Neural Network & Assistive Desktop Interface

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-007ACC?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-99.72%25-00C853)](file:///results/classification_report.txt)
[![Test Suite](https://img.shields.io/badge/Test_Suite-11%2F11_Passed-brightgreen)](file:///test_all_features.py)

---

## 📌 Executive Summary

The **Indian Sign Language (ISL) Recognition System** is an authentic, vision-based assistive communication application designed for real-time translation of the manual ISL alphabet into written text and spoken speech. 

Unlike American Sign Language (ASL), which is primarily single-handed, **over 75% of Indian Sign Language alphabet signs are two-handed** (e.g., `A, B, D, G, H, M, N, P, Q, R, S, T, W, X, Y`). This project rejects the conventional single-hand paradigm and introduces a **Deterministic Spatial Relative-Coordinate Normalization Engine** that models up to 42 hand keypoints (126 dimensions across $x, y, z$).

### 🌟 Headline Achievements
- **99.72% Test Accuracy** evaluated on **12,551 completely unseen held-out test samples** (35 total errors across the entire test benchmark).
- **Macro Average F1-Score: 99.72%** | **Weighted F1-Score: 99.72%**.
- **19 of 26 Alphabet Classes** achieve flawless **100.0% precision and recall**.
- **Historical Ambiguity Resolution:** Completely eliminates prior single-hand confusions (`X` recall: 15% $\to$ 99.4%, `N` recall: 32% $\to$ 100%, `M` recall: 42% $\to$ 100%, `D` vs `Y` separated).
- **Multi-Backend Camera Engine:** Seamless support for Windows DirectShow (`CAP_DSHOW`) and Media Foundation (`CAP_MSMF`) with pixel-level active validation and privacy shutter detection.
- **Production Assistive Desktop GUI:** Full text editing suite (Add, Space, Delete Letter, Delete Word, Clear All, and Undo with stack history), bidirectional in-place typing sync, asynchronous Text-to-Speech (TTS), clipboard export, timestamped file logging, and an interactive 26-letter gesture guide.
- **Automated Verification:** 11-step end-to-end integration test suite (`test_all_features.py`) with a **100% pass rate**.

---

## 📐 System Architecture & Workflow

```
                        Webcam Video Feed
                   (Dual-Backend: DirectShow / MSMF)
                                │
                                ▼
                   MediaPipe Hand Landmarker
                 (Tracks up to 2 Hands, 42 Keypoints)
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ Spatial Relative-Coordinate Normalization    │
         │ 1. Deterministic Screen-x Ordering (h1, h2)  │
         │ 2. Hand 1 (Anchor): P1 - W1 (W1 = 0)         │
         │ 3. Hand 2 (Relative Vector): P2 - W1         │
         │ 4. Joint Max-Abs Scale Factor (126 Features) │
         │ 5. Ambidextrous Dual-Slot 1-Hand Routing     │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ Deep Neural Classifier (TensorFlow/Keras)    │
         │ 126 In -> Dense(256) + BatchNorm + Dropout   │
         │        -> Dense(256) + BatchNorm + Dropout   │
         │        -> Dense(128) + BatchNorm + Dropout   │
         │        -> Dense(64)  + BatchNorm             │
         │        -> Dense(26, Softmax) Output          │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ Temporal Smoothing & Hold Engine             │
         │ - Dynamic Confidence Filter (50% - 95%)      │
         │ - 10-Frame Majority Vote Queue               │
         │ - Hold-Speed Auto-Confirm (8 - 30 frames)    │
         │ - Post-Confirm Cooldown & State Lock         │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ Assistive Graphical User Interface (GUI)     │
         │ - Live Video Feed (640x480 @ 40+ FPS)        │
         │ - Text Editing (Add, Space, Del, Undo)       │
         │ - Bidirectional In-Place Text Box Typing     │
         │ - Offline Text-to-Speech (pyttsx3)           │
         │ - Clipboard & Timestamped File Export        │
         │ - Interactive 26-Letter Sign Guide Modal     │
         └──────────────────────────────────────────────┘
```

---

## 🔤 Supported Alphabet (26 Classes)

The system supports all 26 uppercase alphabet letters (`A` through `Z`) according to standardized Indian Sign Language manual conventions:

| Category | Letters | Visual Structure |
|---|---|---|
| **Two-Handed Signs (20 Classes)** | `A, B, D, E, F, G, H, J, K, M, N, P, Q, R, S, T, W, X, Y, Z` | Coordinated interaction between dominant and base hand (touch, stacking, intersection) |
| **Single-Handed Signs (6 Classes)** | `C, I, L, O, U, V` | Standard single-hand finger configurations (supported ambidextrously for left/right hands) |

---

## 📊 Model Evaluation & Benchmark Results

The model was trained on **62,753 landmark samples** using an 80/20 stratified split and evaluated on **12,551 completely unseen test samples** ([`results/test_data.npz`](file:///results/test_data.npz)).

### Headline Metrics
| Metric | Value | Significance |
|---|---|---|
| **Test Accuracy** | **99.72%** | **12,516 / 12,551 correct classifications** |
| **Precision (Weighted)** | **99.72%** | Exceptional false positive rejection |
| **Recall (Weighted)** | **99.72%** | Near-zero false negatives across all classes |
| **F1-Score (Weighted)** | **99.72%** | Balanced precision-recall performance |
| **Categorical Test Loss** | **0.0112** | Calibrated posterior probabilities |
| **Total Test Errors** | **35 / 12,551** | **Total error rate of only 0.28%** |
| **Inference Latency** | **< 2.0 ms** | Real-time classification on CPU |

### Per-Class Performance Summary
From [`results/classification_report.txt`](file:///results/classification_report.txt):

```text
              precision    recall  f1-score   support
           A       1.00      1.00      1.00       400
           B       1.00      1.00      1.00       393
           C       0.98      0.99      0.99       798
           D       1.00      1.00      1.00       392
           E       1.00      1.00      1.00       399
           F       1.00      1.00      1.00       400
           G       1.00      1.00      1.00       374
           H       1.00      1.00      1.00       366
           I       0.99      1.00      0.99       798
           J       1.00      1.00      1.00       394
           K       1.00      1.00      1.00       400
           L       1.00      0.99      0.99       800
           M       1.00      1.00      1.00       382
           N       1.00      1.00      1.00       386
           O       1.00      1.00      1.00       780
           P       1.00      1.00      1.00       400
           Q       1.00      1.00      1.00       399
           R       1.00      1.00      1.00       382
           S       1.00      1.00      1.00       400
           T       1.00      1.00      1.00       400
           U       0.99      0.98      0.98       794
           V       1.00      1.00      1.00       789
           W       1.00      1.00      1.00       381
           X       1.00      0.99      1.00       351
           Y       1.00      1.00      1.00       393
           Z       1.00      1.00      1.00       400

    accuracy                           1.00     12551
   macro avg       1.00      1.00      1.00     12551
weighted avg       1.00      1.00      1.00     12551
```

---

## 🖥️ Graphical User Interface (GUI) Features

The application GUI ([`src/app.py`](file:///src/app.py)) provides a comprehensive assistive communication interface:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 🤟 INDIAN SIGN LANGUAGE RECOGNITION SYSTEM        Camera: [ Camera 0 (DSHOW) ▼] [🔄]  │
├────────────────────────────────────────┬───────────────────────────────────────────────┤
│           LIVE VIDEO FEED              │            CURRENT SIGN DETECTION             │
│ ┌────────────────────────────────────┐ │ ┌───────────────────────────────────────────┐ │
│ │                                    │ │ │   [  A  ]    Confidence: 99.4%            │ │
│ │         640 x 480 Video            │ │ │              Status: ✋ 2 Hands Detected  │ │
│ │     (FPS: 41.2 | HD 1280x720)      │ │ │   Hold Progress: [████████████░░░░]       │ │
│ │                                    │ │ └───────────────────────────────────────────┘ │
│ └────────────────────────────────────┘ │            RECOGNIZED SENTENCE                │
│  [X] Mirror Video   [X] Show Skeleton  │ ┌───────────────────────────────────────────┐ │
│  [X] Auto-Add Held  [⏸ Pause Detect]   │ │ SIGN LANGUAGE RECOGNITION SYSTEM          │ │
│  Min Confidence: [───────O────] 70%    │ │                                           │ │
│  Hold Speed:     [────O───────] 15 fms │ └───────────────────────────────────────────┘ │
│  Camera: Device 0 [DirectShow]         │  [✓ ADD SIGN]            [⎵ SPACE]            │
│                                        │  [⌫ DELETE LETTER]       [🗑️ DELETE WORD]     │
│                                        │  [✕ CLEAR ALL]           [↩ UNDO]             │
│                                        │  [📋 COPY TEXT]          [💾 SAVE TO FILE]    │
│                                        │  [🔊 SPEAK SENTENCE (TTS)]                    │
│                                        │  [📖 SIGN GUIDE]         [✕ EXIT]             │
└────────────────────────────────────────┴───────────────────────────────────────────────┘
```

### 1. Camera & Video Viewport Controls
- **Live Canvas Display:** 640x480 live stream with smooth 30–45 FPS rendering.
- **Dynamic Camera Switcher & Rescan:** Dropdown menu listing all probed devices with a one-click rescan button to detect newly connected webcams on the fly.
- **Mirror Video Toggle:** Horizontally flips video for natural signer perspective.
- **Landmarks / Skeleton Toggle:** Enables or hides green/yellow skeletal overlays.
- **Pause / Resume Detection:** Pauses inference without interrupting video playback.
- **Min Confidence Slider (50% – 95%):** Dynamically filters low-confidence predictions.
- **Hold Speed Slider (8 – 30 frames):** Customizes sign hold duration (~0.3s to 1.0s).

### 2. Auto-Confirmation & Duplicate Prevention
- **Visual Progress Bar:** Real-time hold progress fills from 0% to 100%.
- **State Lock & Cooldown:** Appends exactly one character upon hold completion and activates a 20-frame cooldown, preventing accidental repeated insertions (e.g., preventing `"AAAA"`).

### 3. Complete Text Editing Suite
- **`✓ ADD SIGN`:** Manually commits current predicted letter.
- **`⎵ SPACE`:** Inserts word space.
- **`⌫ DELETE LETTER`:** Backspace removing the last character.
- **`🗑️ DELETE WORD`:** Removes the last space-separated word in one click.
- **`✕ CLEAR ALL`:** Clears buffer with automatic safety backup.
- **`↩ UNDO`:** Dedicated undo history stack (up to 30 past states).
- **Bidirectional In-Place Text Box:** Interactive `tk.Text` widget allowing direct physical keyboard typing, selecting, and deleting with real-time buffer synchronization.
- **Live Character & Word Counter:** Displays dynamic sentence metrics.

### 4. Assistive Output & Accessibility
- **Multi-Threaded Text-to-Speech (TTS):** Speaks formed sentences aloud via `pyttsx3` in an asynchronous background daemon thread without stalling video frames.
- **Clipboard Export:** One-click copy to Windows system clipboard.
- **Session File Export:** Appends recognized sentences with ISO timestamps into [`isl_history.txt`](file:///isl_history.txt).
- **Interactive Gesture Guide Modal:** Popup dialog showing the full 26-letter ISL alphabet grid and best practices.

---

## ⌨️ Global Keyboard Shortcuts

| Shortcut | Action | Description |
|---|---|---|
| `Space` | **Add Sign / Space** | Adds current detected sign (or space if no sign is active) |
| `-` (Minus) | **Add Space** | Inserts a space character |
| `Backspace` | **Delete Character** | Removes the last character |
| `Ctrl + Backspace` | **Delete Word** | Deletes the last word |
| `Delete` | **Clear Text** | Clears the entire text buffer |
| `Ctrl + Z` | **Undo** | Reverts the last edit action |
| `Ctrl + C` | **Copy Text** | Copies recognized text to clipboard |
| `S` | **Speak (TTS)** | Reads the current text buffer aloud |
| `P` | **Pause / Resume** | Toggles sign detection pause |
| `Esc` | **Exit** | Gracefully closes the application |

---

## 🧪 Automated Verification Suite (`test_all_features.py`)

The project includes an automated 11-step integration test suite verifying every component from hardware probing to GUI widgets:

```bash
python test_all_features.py
```

### Verification Results (11/11 Passed):
- `[TEST 1/11]` Probing Camera Hardware & Validating Physical Webcam **[PASS]**
- `[TEST 2/11]` Testing AI Model & Input Feature Architecture (126 features, 26 classes) **[PASS]**
- `[TEST 3/11]` Initializing Camera Feed inside Application **[PASS]**
- `[TEST 4/11]` Executing live `process_frame()` cycle with image rendering **[PASS]**
- `[TEST 5/11]` Testing All Text Editing Controls (Add, Backspace, Word Del, Clear, Undo) **[PASS]**
- `[TEST 6/11]` Testing Direct In-Place Text Box Editing & Event Sync **[PASS]**
- `[TEST 7/11]` Testing Auto-Confirmation Hold Engine & Cooldown **[PASS]**
- `[TEST 8/11]` Testing Feature Checkboxes (Mirror, Landmarks, Auto-Add) **[PASS]**
- `[TEST 9/11]` Testing Pause/Resume Button Invocations **[PASS]**
- `[TEST 10/11]` Testing Interactive Sign Reference Guide Modal **[PASS]**
- `[TEST 11/11]` Testing Clipboard Copy and File Save (`isl_history.txt`) **[PASS]**

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- **Operating System:** Windows 10 or Windows 11
- **Python Version:** Python 3.11 (Recommended for optimal TensorFlow and MediaPipe compatibility)
- **Hardware:** Standard USB or built-in webcam, 4 GB RAM, dual-core CPU (GPU not required)

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/AshutoshBunkar/ML_ISL_Model.git
cd ML_ISL_Model

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Launching the System
You can launch the system using any of the following methods:

```bash
# Option A: One-click Batch launcher (Windows)
run.bat

# Option B: Interactive GUI launcher with camera selector
python run.py

# Option C: Direct application launch
python src/app.py

# Option D: Specify preferred camera index
python src/app.py --camera 0
```

---

## 📂 Project Directory Structure

```text
ML_ISL_Model/
├── dataset/
│   ├── isl_two_hand_landmarks.csv     # 62,753 clean 126-feature samples
│   └── isl_dataset.csv                # Historical single-hand baseline data
├── models/
│   ├── isl_model.keras                # Trained two-hand neural network (1.71 MB)
│   ├── hand_landmarker.task           # MediaPipe vision landmarker model
│   └── labels.npy                     # 26 alphabet labels (A through Z)
├── results/
│   ├── classification_report.txt      # 99.72% test classification report
│   ├── test_confusion_matrix.png      # 26x26 test confusion matrix plot
│   ├── test_data.npz                  # 12,551 unseen test vectors & labels
│   ├── training_accuracy.png          # Training vs validation accuracy curve
│   ├── training_loss.png              # Training vs validation loss curve
│   └── training_history.json          # Epoch-by-epoch metrics
├── src/
│   ├── app.py                         # Production assistive Tkinter GUI application
│   ├── camera_utils.py                # Multi-backend camera probe & active validator
│   ├── camera_test.py                 # Standalone diagnostic camera utility
│   ├── evaluate_model.py              # Offline test benchmark script
│   ├── train_model.py                 # Neural network training pipeline
│   ├── collect_two_hand_data.py       # Dual-hand landmark acquisition tool
│   ├── predict_two_hand.py            # Standalone OpenCV two-hand HUD
│   └── hand_detection.py              # MediaPipe HandLandmarker wrapper
├── test_all_features.py               # Comprehensive 11-step automated test suite
├── run.py                             # Interactive GUI launcher & camera selector
├── run.bat                            # Windows one-click batch startup script
├── MODEL_ACCURACY_REPORT.md           # Deep statistical accuracy documentation
├── PROJECT_REPORT.md                  # Comprehensive academic supervisor report
├── PRESENTATION_CONTENT.md            # Slide-by-slide presentation deck & defense Q&A
├── DEMO_GUIDE.md                      # Live demonstration walkthrough script
├── README.md                          # Master repository documentation
└── requirements.txt                   # Dependency manifest
```

---

## 🛠️ Verification & Reproduction Commands

```bash
# 1. Run the comprehensive full-feature test suite
python test_all_features.py

# 2. Evaluate model accuracy on 12,551 unseen test samples
python src/evaluate_model.py

# 3. Test camera hardware diagnostics and switch backends
python src/camera_test.py

# 4. Run standalone OpenCV HUD without GUI
python src/predict_two_hand.py
```

---

## 📜 Documentation Suite

For exhaustive technical evaluations and presentation materials, refer to:
- **Academic Project Report:** [`PROJECT_REPORT.md`](file:///PROJECT_REPORT.md)
- **Deep Statistical Model Evaluation:** [`MODEL_ACCURACY_REPORT.md`](file:///MODEL_ACCURACY_REPORT.md)
- **Presentation Deck & Defense Q&A:** [`PRESENTATION_CONTENT.md`](file:///PRESENTATION_CONTENT.md)
- **Live Demonstration Walkthrough:** [`DEMO_GUIDE.md`](file:///DEMO_GUIDE.md)

---

## ⚖️ License & Acknowledgements

Developed as a Major Laboratory Project in Machine Learning & Assistive Computer Vision. Built upon Google MediaPipe, TensorFlow, and the Indian Sign Language Research and Training Centre (ISLRTC) manual alphabet guidelines.