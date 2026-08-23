# Indian Sign Language Recognition System — Project Report

---

## ABSTRACT

This project presents a real-time Indian Sign Language (ISL) static alphabet recognition system using computer vision and deep learning. The system captures hand gestures through a webcam, extracts 21 hand landmarks using Google's MediaPipe framework, normalizes them into 63 features, and classifies them into 26 alphabet classes (A-Z) using a trained feedforward neural network. The application features a professional desktop GUI built with Tkinter that enables users to form words and sentences from recognized letters and convert the text to speech using an offline text-to-speech engine. The system employs prediction smoothing via majority voting and a letter confirmation mechanism to ensure stable, reliable text formation. This project demonstrates the practical application of computer vision and deep learning in assistive technology for the hearing-impaired community.

**Keywords:** Indian Sign Language, Hand Gesture Recognition, MediaPipe, TensorFlow, Computer Vision, Deep Learning, Assistive Technology

---

## 1. INTRODUCTION

Sign language is the primary mode of communication for the hearing-impaired community worldwide. In India, approximately 18 million people are hearing-impaired (Census of India), and Indian Sign Language (ISL) serves as their primary communication medium. However, the majority of the hearing population does not understand ISL, creating a significant communication barrier.

Recent advances in computer vision and deep learning have made it possible to develop automated sign language recognition systems that can translate hand gestures into text in real time. This project leverages these technologies to create a practical ISL alphabet recognition system.

The system uses a webcam to capture hand gestures, Google's MediaPipe framework to detect and extract hand landmarks, and a neural network built with TensorFlow/Keras to classify these landmarks into the 26 English alphabet letters as represented in ISL.

---

## 2. BACKGROUND

### 2.1 Indian Sign Language

Indian Sign Language (ISL) is a visual-gestural language used by the deaf community in India. It has its own grammar and syntax, distinct from spoken Hindi or English. The ISL alphabet consists of hand shapes representing letters A through Z, used primarily for fingerspelling proper nouns, technical terms, and words that do not have established signs.

### 2.2 Hand Gesture Recognition

Hand gesture recognition is a subfield of computer vision that involves detecting and interpreting hand poses and movements. Modern approaches typically use one of two strategies:

1. **Image-based methods:** Using CNNs on raw images or segmented hand regions
2. **Landmark-based methods:** Extracting hand keypoints and using their coordinates as features

This project uses the landmark-based approach, which offers several advantages including lower computational cost, invariance to skin tone and background, and smaller model size.

### 2.3 MediaPipe

MediaPipe is an open-source framework by Google that provides optimized, production-ready computer vision pipelines. The MediaPipe Hand Landmarker model detects 21 key points on each hand with x, y, z coordinates, running efficiently on CPU.

### 2.4 Related Work

Several systems for sign language recognition have been developed:

- American Sign Language (ASL) recognition using CNNs (various authors)
- ISL recognition using image classification with deep learning
- Real-time sign language detection using MediaPipe and machine learning
- Multi-modal sign language recognition combining hand, face, and body pose

This project builds upon these approaches with a focus on practical usability, stable prediction, and integrated text-to-speech.

---

## 3. PROBLEM STATEMENT

The hearing-impaired community in India faces significant communication barriers because most hearing individuals do not understand Indian Sign Language. While sign language interpreters exist, they are not always available, creating dependency and limiting independent communication.

There is a need for an automated system that can recognize ISL hand gestures and translate them into text in real time, enabling communication without the need for a human interpreter.

---

## 4. MOTIVATION

1. **Social Impact:** Bridging the communication gap between hearing-impaired and hearing communities
2. **Accessibility:** Making communication tools available using just a webcam
3. **Technology Application:** Demonstrating practical use of computer vision and deep learning
4. **Academic Contribution:** Contributing to ISL recognition research, which has less coverage compared to ASL
5. **Assistive Technology:** Creating a tool that can serve as a foundation for more comprehensive ISL translation systems

---

## 5. OBJECTIVES

1. Develop a robust hand landmark detection and extraction pipeline using MediaPipe
2. Create a comprehensive dataset of ISL alphabet hand landmarks (A-Z)
3. Train and evaluate a neural network classifier for 26 ISL alphabet classes
4. Implement real-time sign recognition with prediction smoothing and stability
5. Build a desktop GUI application with text formation capabilities
6. Integrate text-to-speech for converting formed text to spoken words
7. Evaluate the system rigorously and document results transparently

---

## 6. SCOPE

### In Scope
- Recognition of 26 static ISL alphabet signs (A-Z)
- Real-time webcam-based detection
- Text formation (letters → words → sentences)
- Text-to-speech output
- Desktop GUI application for Windows

### Out of Scope
- Dynamic gestures and signs involving motion
- Word-level and sentence-level ISL recognition
- ISL grammar and syntax processing
- Facial expression recognition
- Multi-hand detection
- Mobile application development

---

## 7. PROPOSED SYSTEM

The proposed system consists of four main components:

### 7.1 Data Collection Module
- Uses webcam and MediaPipe to collect hand landmark data
- Supports all 26 ISL alphabet signs
- Produces normalized CSV dataset with 63 features per sample

### 7.2 Training Module
- Loads and validates the landmark dataset
- Trains a feedforward neural network with early stopping
- Generates training plots and saves the best model

### 7.3 Evaluation Module
- Evaluates the trained model on unseen test data
- Generates classification report and confusion matrix
- Reports per-class accuracy, precision, recall, and F1-score

### 7.4 Recognition Application
- Real-time webcam-based sign recognition
- Professional Tkinter GUI
- Letter confirmation with auto-confirm and cooldown
- Text formation with add, space, backspace, and clear
- Text-to-speech using pyttsx3

---

## 8. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────┐
│                     WEBCAM INPUT                     │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              OPENCV FRAME CAPTURE                    │
│              (BGR → RGB Conversion)                  │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│           MEDIAPIPE HAND LANDMARKER                  │
│    • 21 landmarks detected per hand                  │
│    • x, y, z coordinates extracted                   │
│    • Confidence threshold: 0.3                       │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│            LANDMARK NORMALIZATION                    │
│    • Wrist (landmark 0) as reference point           │
│    • Max-abs scaling to [-1, 1]                      │
│    • Output: 63 features                             │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│          NEURAL NETWORK CLASSIFIER                   │
│    • Dense(128) → Dropout → Dense(64)                │
│    • → Dropout → Dense(32) → Dense(26)               │
│    • Softmax output → 26 class probabilities         │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│            PREDICTION PROCESSING                     │
│    • Confidence threshold (70%)                      │
│    • 10-frame majority voting                        │
│    • 15-frame stable confirmation                    │
│    • 20-frame cooldown after confirmation             │
└────────────────────────┬────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              TKINTER GUI OUTPUT                      │
│    • Live camera feed                                │
│    • Current sign & confidence                       │
│    • Text buffer display                             │
│    • Control buttons & TTS                           │
└─────────────────────────────────────────────────────┘
```

---

## 9. DATASET

### 9.1 Data Collection

Hand landmark data is collected using the project's data collection tool (`src/collect_data.py`), which:

1. Opens the webcam
2. Displays the current sign to perform
3. Detects the hand using MediaPipe
4. Extracts and normalizes 21 landmarks
5. Saves 63 features + label to CSV

### 9.2 Dataset Format

| Column | Description |
|--------|-------------|
| x0, y0, z0 | Wrist landmark (always 0, 0, 0 after normalization) |
| x1, y1, z1 | Thumb CMC |
| ... | ... |
| x20, y20, z20 | Pinky tip |
| label | Class label (A-Z) |

### 9.3 Dataset Split

- **Training:** 80% (stratified)
- **Testing:** 20% (stratified, completely unseen during training)
- **Random seed:** 42 (for reproducibility)

---

## 10. DATA PREPROCESSING

### 10.1 Hand Landmark Extraction

MediaPipe Hand Landmarker extracts 21 key points from each detected hand. Each landmark has:
- **x:** Horizontal position (normalized to image width)
- **y:** Vertical position (normalized to image height)
- **z:** Depth (relative to wrist)

### 10.2 Wrist Normalization

All landmarks are translated so the wrist (landmark 0) is at the origin:
```
x_normalized = x - wrist_x
y_normalized = y - wrist_y
z_normalized = z - wrist_z
```

This makes the features invariant to hand position in the frame.

### 10.3 Max-Abs Scaling

After wrist normalization, all 63 values are divided by the maximum absolute value:
```
features = features / max(|features|)
```

This scales all values to the range [-1, 1], making the features invariant to hand distance from the camera.

### 10.4 Data Augmentation

When the dataset is small, safe augmentation is applied to training data:

| Transformation | Range | Purpose |
|----------------|-------|---------|
| Translation | ±0.02 | Simulate position variation |
| Scaling | 0.95–1.05 | Simulate distance variation |
| Gaussian noise | σ=0.01 | Improve generalization |

**Not applied:** Horizontal mirroring (would change the semantic meaning of ISL signs).

---

## 11. MODEL ARCHITECTURE

The classifier is a feedforward neural network with the following architecture:

| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
| Input | Input | (63,) | 0 |
| Hidden 1 | Dense + ReLU | (128,) | 8,192 |
| Dropout 1 | Dropout (0.3) | (128,) | 0 |
| Hidden 2 | Dense + ReLU | (64,) | 8,256 |
| Dropout 2 | Dropout (0.2) | (64,) | 0 |
| Hidden 3 | Dense + ReLU | (32,) | 2,080 |
| Output | Dense + Softmax | (26,) | 858 |

**Total parameters:** ~19,386

### Compilation

- **Optimizer:** Adam (default learning rate 0.001)
- **Loss function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy

### Training Callbacks

- **Early Stopping:** Monitor val_loss, patience=5, restore best weights
- **Model Checkpoint:** Save best model by validation accuracy

---

## 12. TRAINING METHODOLOGY

1. Load landmark CSV dataset
2. Validate labels (confirm all 26 classes present)
3. Check class balance
4. Convert features to float32 and normalize
5. Encode labels using sklearn LabelEncoder
6. Split: 80% training, 20% testing (stratified, seed=42)
7. Apply data augmentation to training set if needed
8. Train with 20% of training data as validation
9. Apply early stopping and model checkpointing
10. Save best model, labels, and training history
11. Generate accuracy and loss plots

---

## 13. TESTING METHODOLOGY

- The test set (20%) is **completely held out** during training
- Test data is saved to `results/test_data.npz` for reproducible evaluation
- The trained model is evaluated on this unseen test set
- Results include accuracy, precision, recall, F1-score, and confusion matrix

---

## 14. EVALUATION METRICS

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions / total predictions |
| Precision | True positives / (True positives + False positives) per class |
| Recall | True positives / (True positives + False negatives) per class |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Visual matrix showing per-class classification results |

---

## 15. RESULTS

The neural network model has been trained and evaluated on the full A-Z dataset. The overall test results on completely unseen data are:

- **Overall Test Accuracy:** 72.77%
- **Weighted Precision:** 74.50%
- **Weighted Recall:** 72.77%
- **Weighted F1-Score:** 71.69%
- **Training Samples:** 3,320 (augmented to 9,960)
- **Testing Samples:** 830

Detailed classification metrics for each class are saved in [classification_report.txt](file:///d:/Indian-Sign-Language-Detection/results/classification_report.txt).

---

## 16. CONFUSION MATRIX ANALYSIS

The confusion matrix (saved to `results/test_confusion_matrix.png`) shows:
- Diagonal elements: Correctly classified samples per class
- Off-diagonal elements: Misclassifications between classes

Common confusions may occur between visually similar signs (e.g., letters with similar finger configurations).

---

## 17. REAL-TIME RECOGNITION

The real-time recognition pipeline processes webcam frames continuously:

1. **Frame capture:** OpenCV reads frames from the webcam
2. **Hand detection:** MediaPipe identifies and tracks the hand
3. **Feature extraction:** 21 landmarks normalized to 63 features
4. **Classification:** Neural network produces 26 class probabilities
5. **Confidence filtering:** Predictions below 70% confidence are rejected
6. **Prediction smoothing:** Majority vote over the last 10 frames

---

## 18. TEXT FORMATION

The text formation system includes:

- **Auto-confirmation:** Hold a sign steady for 15+ frames to auto-add
- **Manual addition:** Press Space to add the current letter
- **Space insertion:** Press `-` or click SPACE button
- **Backspace:** Press Backspace to remove the last character
- **Clear:** Press Delete to clear all text
- **Cooldown:** 20-frame cooldown after auto-confirmation prevents duplicates

---

## 19. TEXT-TO-SPEECH

- **Engine:** pyttsx3 (offline, no internet required)
- **Platform:** Windows (SAPI5 voice engine)
- **Activation:** User explicitly presses SPEAK button or `S` key
- **Threading:** Speech runs in a background thread to avoid freezing the GUI
- **Error handling:** Graceful fallback if TTS is unavailable

---

## 20. LIMITATIONS

1. **Static signs only:** Cannot recognize dynamic gestures or signs involving motion
2. **Single hand:** Processes one hand at a time
3. **Lighting sensitivity:** Performance varies with lighting conditions
4. **User-specific:** Model may need retraining for different users
5. **English alphabet only:** Maps to English letters, not Hindi script
6. **No grammar:** Does not process ISL grammar or syntax
7. **No facial expressions:** ISL uses facial expressions as grammatical markers

---

## 21. FUTURE SCOPE

1. Dynamic gesture recognition using temporal models (LSTM, GRU, Transformer)
2. Multi-user training with diverse hand shapes and skin tones
3. Word-level and sentence-level ISL recognition
4. Facial expression and body pose integration
5. Bidirectional communication (text/speech → ISL animation)
6. Mobile application (Android/iOS)
7. Hindi/Devanagari script support
8. Continuous sign language recognition (without fingerspelling)
9. Transfer learning from larger sign language datasets
10. Edge deployment on embedded devices

---

## 22. CONCLUSION

This project successfully demonstrates a real-time Indian Sign Language static alphabet recognition system using MediaPipe hand landmarks and a feedforward neural network. The system provides a complete communication interface with text formation and text-to-speech capabilities, making it a practical prototype for assistive technology.

While the system is limited to static alphabet signs, it establishes a solid foundation for future expansion to dynamic gestures, word-level recognition, and complete ISL translation.

---

## 23. REFERENCES

1. Google MediaPipe — https://developers.google.com/mediapipe
2. TensorFlow — https://www.tensorflow.org/
3. OpenCV — https://opencv.org/
4. pyttsx3 Documentation — https://pyttsx3.readthedocs.io/
5. Indian Sign Language Research Foundation — https://www.islrtc.nic.in/
6. scikit-learn — https://scikit-learn.org/
7. Census of India — Disability statistics
8. MediaPipe Hand Landmarker Guide — https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
