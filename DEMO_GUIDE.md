# 🎬 Indian Sign Language (ISL) Recognition System — Live Demonstration Guide
## 3 to 5 Minute Supervisor Demonstration Walkthrough Script

---

## 📋 Pre-Demo Preparation Checklist

Before presenting to your supervisor or review committee, ensure the following steps are verified:

- [ ] Open PowerShell or Terminal in the project directory:
  ```powershell
  cd path\to\ML_ISL_Model
  ```
- [ ] Activate the virtual environment:
  ```powershell
  venv\Scripts\activate
  ```
- [ ] Ensure your webcam is uncovered (check laptop physical privacy slider or `Fn+F10` key).
- [ ] Ensure adequate room lighting on both hands and face.
- [ ] Verify model and configuration files:
  - `models/isl_model.keras` (1.71 MB)
  - `models/hand_landmarker.task`
  - `models/labels.npy`
- [ ] Run the automated verification test suite to confirm zero errors:
  ```powershell
  python test_all_features.py
  ```
  *(Verify output: "ALL 11 APPLICATION AND GUI FEATURES VERIFIED AND PASSED! [OK]")*

---

## ⏱️ Step-by-Step Demonstration Script (5 Minutes)

---

### Step 1: Launch Application & Camera Detection (30 Seconds)

**Action:**
Run the application from terminal:
```powershell
python src/app.py
```
*(Or launch the interactive launcher `python run.py`)*

**What to Say:**
> "Good morning, Professor. This is our Indian Sign Language Recognition System. 
> Notice that upon startup, our multi-backend hardware engine in `camera_utils.py` automatically probes both DirectShow and Media Foundation backends, validates active illumination to ensure the camera lens is not covered, and streams live HD video at over 40 frames per second on this standard laptop CPU without requiring a GPU. 
> In the top header, the user can see the active camera device, and if an external webcam is plugged in, clicking 'Rescan' detects it instantly without restarting."

---

### Step 2: Demonstrate Two-Hand Tracking & Landmarks (30 Seconds)

**Action:**
1. Raise your left hand into view $\to$ observe green skeleton points drawn on the screen.
2. Raise your right hand into view $\to$ observe both hands tracked simultaneously with distinct colored skeletons.
3. Check status label: `"Status: ✋ 2 Hands Detected"`.
4. Toggle `Mirror Video` checkbox $\to$ show the feed flip horizontally for intuitive user mirroring.
5. Toggle `Show Landmarks / Skeleton` checkbox $\to$ show that skeleton lines can be toggled on/off.

**What to Say:**
> "Unlike American Sign Language, which is largely single-handed, over 75% of Indian Sign Language alphabet signs require two hands.
> Our system uses Google MediaPipe in video mode to detect up to 42 landmarks simultaneously. 
> We introduced a Deterministic Screen-x Ordering algorithm: the leftmost hand is always mapped to Hand 1 and the rightmost to Hand 2, which completely eliminates MediaPipe's erratic left-versus-right classification flipping."

---

### Step 3: Demonstrate Individual Alphabet Signs & High Accuracy (45 Seconds)

**Action:**
Perform 3–4 two-handed signs and 1–2 single-handed signs:
1. **Sign `A` (Two-handed):** Left hand palm open, right index finger touches left thumb tip $\to$ Label displays `A`, confidence $>98\%$.
2. **Sign `B` (Two-handed):** Both hands form two circles touching $\to$ Label displays `B`, confidence $>99\%$.
3. **Sign `D` (Two-handed):** Vertical index touch $\to$ Label displays `D`.
4. **Sign `X` (Two-handed):** Cross both index fingers in an 'X' $\to$ Label displays `X`.
5. **Sign `L` (Single-handed):** Thumb and index finger form an 'L' $\to$ Label displays `L`.
6. **Sign `V` (Single-handed):** Peace sign $\to$ Label displays `V`.

**What to Say:**
> "Observe the live prediction and confidence meter in the upper-right card. 
> In previous single-handed models, letters like 'X' had an 85% error rate because single-hand models saw only an ambiguous curved finger. In our two-handed relative-coordinate model, 'X' achieves 99.7% F1-score, and letters like 'D', 'N', 'M', and 'Y' are recognized with 100.0% precision. 
> Overall, our model achieves 99.72% test accuracy across all 26 classes on 12,551 completely unseen samples."

---

### Step 4: Demonstrate Auto-Confirmation & Duplicate Prevention (45 Seconds)

**Action:**
1. Point out the **Hold Speed Slider** (set to 15 frames / ~0.5s) and **Min Confidence Slider** (70%).
2. Sign `S` and hold it steady $\to$ show the green **Hold Progress bar** filling up smoothly.
3. Once full, the letter `S` is automatically appended to the text buffer with a notification: `"Auto-added: S"`.
4. Continue holding sign `S` $\to$ point out that the system **does not** continuously type `"SSSSS"`.
5. Relax your hands briefly, then sign `I`, `G`, `N` $\to$ observe `"SIGN"` forming cleanly in the text area.

**What to Say:**
> "In assistive technology, hands-free typing is essential. We engineered an auto-confirmation state engine with temporal smoothing. 
> When the user holds a sign steady for half a second, the green progress bar fills and commits the letter. 
> Notice that when I keep holding the sign, the system locks state with a 20-frame cooldown, preventing accidental repeated typing. To sign a double letter, the user simply relaxes and re-engages the posture."

---

### Step 5: Demonstrate Complete Text Editing & Undo Suite (60 Seconds)

**Action:**
1. Click `⎵ SPACE` (or press `-`) $\to$ text buffer becomes `"SIGN "`.
2. Auto-add or click `✓ ADD SIGN` to add letters to form `"SIGN TEST"`.
3. Click `⌫ DELETE LETTER` (Backspace) $\to$ text updates to `"SIGN TES"`.
4. Click `🗑️ DELETE WORD` $\to$ the word `"TES"` is deleted in one click, leaving `"SIGN "`.
5. Click `✕ CLEAR ALL` $\to$ text box is completely cleared.
6. **Click `↩ UNDO`** $\to$ the entire text `"SIGN "` is **instantly restored**!

**What to Say:**
> "Sign language users inevitably make mistakes while spelling. A common flaw in classroom projects is having only a simple clear button. 
> We implemented a full editing suite: deleting single letters, deleting whole words in one click, and most importantly, a multi-state Undo engine with up to 30 history states. When I accidentally clear all text, one click on 'UNDO' restores my work instantly."

---

### Step 6: Demonstrate Direct In-Place Text Box Typing (30 Seconds)

**Action:**
1. Click directly inside the dark text box using your mouse.
2. Use your physical keyboard to type an edit: add `" OK"` at the end so it reads `"SIGN OK"`.
3. Point out that the character and word counters in the header immediately update: `"7 characters | 2 words"`.

**What to Say:**
> "Furthermore, our text box is an interactive bidirectional Tkinter Text widget, not a static label. A hearing user or teacher can type directly into the box with a keyboard, edit words with mouse selection, and our background state manager remains perfectly synchronized."

---

### Step 7: Demonstrate Assistive Audio (TTS) & Export (45 Seconds)

**Action:**
1. Click `🔊 SPEAK SENTENCE (TTS)` (or press `S`) $\to$ the computer speaks aloud: *"SIGN OK"*.
2. Point out that the live camera feed **does not freeze** during speech.
3. Click `📋 COPY TEXT` $\to$ toast notification shows `"✓ Copied to clipboard!"`.
4. Click `💾 SAVE TO FILE` $\to$ toast notification shows `"✓ Saved to isl_history.txt"`.
5. *(Optional)* Open `isl_history.txt` to show the timestamped record: `"[2026-10-13 14:21:30] SIGN OK"`.

**What to Say:**
> "To close the communication loop between deaf signers and hearing observers, the system features offline Text-to-Speech via `pyttsx3`. 
> Crucially, our speech engine runs in a separate background daemon thread, so speaking a long sentence never stutters or freezes the live video feed. 
> We also support one-click clipboard copying and persistent timestamped logging to `isl_history.txt`."

---

### Step 8: Show the Interactive Sign Reference Guide Modal (30 Seconds)

**Action:**
1. Click `📖 SIGN GUIDE`.
2. An interactive modal window opens displaying the complete 26-letter ISL alphabet grid and best practice guidelines.
3. Click `Close Guide` to return cleanly to the main window.

**What to Say:**
> "For beginners learning sign language, clicking 'SIGN GUIDE' opens an interactive modal window showcasing all 26 supported letters and signing tips, such as keeping both hands in good lighting and distinguishing two-handed versus single-handed signs."

---

### Step 9: Conclude with the Automated Verification Suite (30 Seconds)

**Action:**
1. Close the GUI application cleanly (or press `Esc`).
2. Run the automated test script in terminal:
   ```powershell
   python test_all_features.py
   ```
3. Show the supervisor the terminal output confirming all 11 unit and integration tests passing.

**What to Say:**
> "Finally, Professor, we wanted to ensure total software reliability. Here is our automated test script, `test_all_features.py`. It runs 11 rigorous integration tests verifying camera probing, model inference, text editing buttons, undo restoration, in-place typing synchronization, auto-confirmation, checkboxes, modal windows, and file export. 
> As you can see, all 11 tests pass with 100% success. 
> This completes our demonstration, and I would be delighted to answer any questions."

---

## 💡 Quick Reference: Backup Troubleshooting Tips

| Issue During Demo | Immediate Fix |
|---|---|
| Camera appears black / not showing | Check laptop physical privacy shutter switch on bezel, or press `Fn+F10`. In GUI, select another camera from the dropdown or click `🔄 Rescan`. |
| Hands not detecting | Ensure hands are well-lit and held at chest/torso height facing the camera. |
| Letter auto-adds too fast or slow | Adjust the `Hold Speed` slider: slide right for slower (~25 frames) or left for faster (~10 frames). |
| TTS audio not speaking | Ensure computer volume is unmuted; TTS uses Windows SAPI5 voice engine via `pyttsx3`. |
| Accidental text deletion | Immediately click `↩ UNDO` or press `Ctrl + Z`. |
