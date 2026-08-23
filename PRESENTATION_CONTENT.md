# ISL Recognition System — Presentation Content

## Approximately 15 slides for MCA project presentation

---

## Slide 1: Title Slide

**Indian Sign Language Recognition System**

- Real-Time ISL Alphabet Recognition using Computer Vision and Deep Learning
- MCA Project
- Technologies: Python, TensorFlow, MediaPipe, OpenCV

> *Explain:* Introduce the project title and your name. Briefly mention this is a real-time system that recognizes ISL hand signs and converts them to text.

---

## Slide 2: Introduction

**What is this project?**

- A real-time system that recognizes Indian Sign Language alphabet signs (A-Z)
- Uses a webcam to detect hand gestures
- Translates hand signs into text on screen
- Supports text-to-speech for communication

**Key Highlights:**
- 26 classes (A-Z)
- Professional desktop GUI
- Offline operation (no internet needed)

> *Explain:* Give a high-level overview. Emphasize that the system works in real time using just a webcam, making it accessible and practical.

---

## Slide 3: Problem Statement

**Communication Barrier**

- 18 million hearing-impaired people in India (Census)
- Most hearing individuals do not understand ISL
- Human interpreters are not always available
- Limited technology solutions for ISL specifically
- Need for automated sign-to-text translation

> *Explain:* Highlight the real-world problem. Emphasize the scale of the hearing-impaired population and the lack of ISL-specific technology.

---

## Slide 4: Motivation

**Why this project?**

- Bridge the communication gap between deaf and hearing communities
- ISL has less research coverage compared to ASL
- Demonstrate practical application of AI in assistive technology
- Contribute to making technology more accessible
- Leverage recent advances in hand tracking (MediaPipe)

> *Explain:* Explain why this problem is worth solving and why now is a good time (availability of MediaPipe and TensorFlow making it practical).

---

## Slide 5: Objectives

**Project Goals**

1. Build a robust hand detection pipeline using MediaPipe
2. Collect/prepare a dataset of ISL alphabet landmarks (A-Z)
3. Train a neural network to classify 26 signs
4. Achieve high accuracy on unseen test data
5. Implement real-time webcam recognition
6. Build a desktop GUI for text formation
7. Integrate text-to-speech output

> *Explain:* Walk through each objective. These map directly to the project deliverables.

---

## Slide 6: Existing System

**Current Approaches and Limitations**

| Approach | Limitation |
|----------|-----------|
| Image-based CNN | Sensitive to background, lighting, skin tone |
| Glove-based sensors | Requires special hardware, expensive |
| Depth camera (Kinect) | Expensive, not portable |
| ASL recognition systems | Not applicable to ISL |

**Our approach:** Landmark-based recognition using MediaPipe
- Works with any webcam
- Invariant to skin tone and background
- Lightweight model (runs on CPU)

> *Explain:* Compare existing approaches and explain why the landmark-based approach is better for this application.

---

## Slide 7: Proposed System

**Our Solution**

```
Webcam → MediaPipe → Landmarks → Neural Network → Text → Speech
```

**Key Components:**
1. MediaPipe Hand Landmarker for 21-point hand detection
2. Wrist-referenced normalization for position/scale invariance
3. Feedforward neural network for classification
4. Prediction smoothing for stability
5. Letter confirmation for reliable text formation
6. Tkinter GUI with text-to-speech

> *Explain:* Walk through the pipeline from camera input to speech output. Emphasize the end-to-end nature of the system.

---

## Slide 8: System Architecture

**Architecture Diagram**

```
Webcam → OpenCV → MediaPipe → 21 Landmarks
                                    ↓
                              Normalization
                              (63 features)
                                    ↓
                           Neural Network
                           (128→64→32→26)
                                    ↓
                        Confidence Threshold
                              (70%)
                                    ↓
                        Prediction Smoothing
                        (10-frame majority vote)
                                    ↓
                        Letter Confirmation
                        (15-frame stability)
                                    ↓
                         GUI + Text-to-Speech
```

> *Explain:* Walk through each stage of the pipeline. Highlight the normalization step that makes it work regardless of hand position.

---

## Slide 9: Dataset

**Data Collection and Preparation**

- **Collection tool:** Custom webcam-based data collector
- **Features:** 21 landmarks × 3 coordinates = 63 features per sample
- **Classes:** 26 (A-Z)
- **Format:** CSV with 63 features + 1 label column
- **Split:** 80% training / 20% testing (stratified)

**Normalization:**
- Wrist as reference point (position invariant)
- Max-abs scaling to [-1, 1] (scale invariant)

> *Explain:* Explain how data is collected and preprocessed. Show a sample of the CSV format if possible.

---

## Slide 10: Methodology

**Training Process**

1. Load and validate A-Z dataset
2. Normalize landmarks
3. Encode labels
4. 80/20 stratified split
5. Data augmentation (if needed)
   - Small translation, scaling, noise
   - No mirroring (preserves sign meaning)
6. Train with early stopping (patience=5)
7. Save best model by validation accuracy

**Prediction Process:**
- 70% confidence threshold
- 10-frame majority voting
- 15-frame auto-confirmation with 20-frame cooldown

> *Explain:* Walk through both the training and real-time prediction methodology. Emphasize the safeguards against unstable predictions.

---

## Slide 11: Machine Learning Model

**Neural Network Architecture**

| Layer | Type | Size |
|-------|------|------|
| Input | - | 63 |
| Hidden 1 | Dense + ReLU | 128 |
| Dropout 1 | - | 30% |
| Hidden 2 | Dense + ReLU | 64 |
| Dropout 2 | - | 20% |
| Hidden 3 | Dense + ReLU | 32 |
| Output | Dense + Softmax | 26 |

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Total params:** ~19,386
- **Callbacks:** Early stopping + Model checkpointing

> *Explain:* Explain why this architecture works — small enough to be fast, large enough to learn 26 classes from 63 features. Dropout prevents overfitting.

---

## Slide 12: Results

**Model Evaluation on Unseen Test Data**

| Metric | Value |
|--------|-------|
| Test Accuracy | 72.77% |
| Precision (weighted) | 74.50% |
| Recall (weighted) | 72.77% |
| F1-Score (weighted) | 71.69% |
| Test Samples | 830 |

- Confusion matrix at `results/test_confusion_matrix.png`
- Classification report at `results/classification_report.txt`

> *Explain:* Present the actual measured results. Be honest about any classes that perform poorly (like X at 15.4% and N/J at ~32-34%) and explain why (similar static hand shapes or small sample sizes in the original dataset).

---

## Slide 13: Real-Time Application

**Desktop GUI**

- Live webcam feed with hand skeleton overlay
- Current detected sign with confidence
- Text buffer for forming words/sentences
- Buttons: Add Letter, Space, Backspace, Clear, Speak, Exit
- Keyboard shortcuts for fast interaction

**Letter Confirmation:**
- Hold sign steady for ~0.5 seconds to auto-add
- Prevents accidental duplicates
- Cooldown between confirmations

> *Explain:* Demonstrate the GUI. Show how the letter confirmation system prevents "AAAA..." from holding one sign.

---

## Slide 14: Text-to-Speech

**Communication Feature**

- Engine: pyttsx3 (offline, Windows SAPI5)
- User forms text by signing letters
- Press SPEAK to hear the text
- Runs in background thread (camera doesn't freeze)

**Example workflow:**
1. Sign H → E → L → L → O
2. Press Space for word break
3. Sign W → O → R → L → D
4. Press SPEAK → System says "HELLO WORLD"

> *Explain:* Walk through a complete text formation and speech example. This is the most impressive part of the demo.

---

## Slide 15: Conclusion & Future Scope

**Conclusion**

- Successfully built a real-time ISL alphabet recognition system
- Supports all 26 letters (A-Z)
- Professional GUI with text formation and speech
- Practical demonstration of CV + DL in assistive technology

**Future Scope:**

1. Dynamic gesture recognition (LSTM/Transformer)
2. Word-level ISL recognition
3. Multi-user models (diverse hands)
4. Facial expression integration
5. Mobile app (Android/iOS)
6. Hindi/Devanagari script support
7. Bidirectional communication (text → ISL animation)

**Thank You!**

> *Explain:* Summarize what was achieved and the exciting possibilities for future work. Emphasize that this is a foundation for more comprehensive ISL systems.
