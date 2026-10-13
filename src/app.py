"""
Indian Sign Language Recognition System - Main Application

Features:
- Robust multi-backend camera support (DirectShow & Media Foundation) with auto-detection
- Real-time 1-hand and 2-hand ISL alphabet recognition via MediaPipe + Deep Learning
- Comprehensive text editing options:
    * Delete Letter (Backspace)
    * Delete Word
    * Clear All Text with Undo
    * Direct in-place editing in text area
- All necessary GUI options:
    * Camera switcher & rescan dropdown
    * Mirror feed toggle
    * Hand skeleton / landmarks toggle
    * Pause / Resume recognition toggle
    * Confidence threshold & Hold duration sliders
    * Copy to Clipboard & Save to file
    * Text-to-Speech (TTS)
    * Interactive ISL Gesture Guide modal

Usage:
    python src/app.py [--camera <index>]
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
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque, Counter
from PIL import Image, ImageTk

# Camera utils
sys.path.insert(0, _script_dir)
from camera_utils import probe_all_cameras, open_best_camera


# ==================================================
# FILE PATHS
# ==================================================

MODEL_PATH = "models/isl_model.keras"
HAND_MODEL_PATH = "models/hand_landmarker.task"
LABELS_PATH = "models/labels.npy"


def check_files():
    """Check that all required model assets exist."""
    missing = []
    for f in [MODEL_PATH, HAND_MODEL_PATH, LABELS_PATH]:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        print("\nERROR: Required files missing:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease train the model or ensure files are in place.\n")
        sys.exit(1)


# ==================================================
# TEXT-TO-SPEECH ENGINE
# ==================================================

class SpeechEngine:
    """Thread-safe text-to-speech engine using pyttsx3."""

    def __init__(self):
        self.available = False
        self.speaking = False

        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            self.engine.setProperty("volume", 0.9)
            self.available = True
            print("Text-to-speech: Available")
        except Exception as e:
            print(f"Text-to-speech: Not available ({e})")
            self.engine = None

    def speak(self, text):
        """Speak text in a background thread."""
        if not self.available or not text.strip():
            return

        if self.speaking:
            return

        thread = threading.Thread(
            target=self._speak_thread,
            args=(text,),
            daemon=True
        )
        thread.start()

    def _speak_thread(self, text):
        try:
            self.speaking = True
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            self.speaking = False


# ==================================================
# HAND DETECTOR
# ==================================================

class HandDetector:
    """MediaPipe hand landmark detector supporting up to 2 hands."""

    def __init__(self, num_hands=2):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.detector = HandLandmarker.create_from_options(options)
        self.start_time = time.monotonic()
        self.last_timestamp_ms = 0

    def detect(self, rgb_frame):
        """Detect hands in frame with strictly increasing timestamp."""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        now_ms = int((time.monotonic() - self.start_time) * 1000)
        if now_ms <= self.last_timestamp_ms:
            now_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = now_ms
        return self.detector.detect_for_video(mp_image, now_ms)

    def close(self):
        self.detector.close()


# ==================================================
# LANDMARK NORMALIZERS
# ==================================================

def normalize_landmarks(hand_landmarks):
    """Normalize single hand landmarks using wrist as reference (63 features)."""
    landmarks = []
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    wrist_z = hand_landmarks[0].z

    for landmark in hand_landmarks:
        landmarks.extend([
            landmark.x - wrist_x,
            landmark.y - wrist_y,
            landmark.z - wrist_z
        ])

    landmarks = np.array(landmarks, dtype=np.float32)
    max_value = np.max(np.abs(landmarks))
    if max_value > 0:
        landmarks = landmarks / max_value
    return landmarks


def normalize_two_hands(result):
    """Normalize up to 2 detected hands into 126 features."""
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

    # 2 hands: sort by wrist x (screen-left vs screen-right)
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


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


def draw_hand(frame, hand_landmarks, line_color=(0, 255, 0)):
    """Draw hand landmarks and skeleton on frame."""
    height, width, _ = frame.shape
    points = []
    for lm in hand_landmarks:
        points.append((int(lm.x * width), int(lm.y * height)))

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], line_color, 2)

    for i, (x, y) in enumerate(points):
        color = (0, 0, 255) if i == 0 else (255, 0, 0)
        cv2.circle(frame, (x, y), 4, color, -1)


# ==================================================
# MAIN APPLICATION CLASS
# ==================================================

class ISLApp:
    """Main Indian Sign Language Recognition Application."""

    def __init__(self):
        print("Loading AI model...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.labels = np.load(LABELS_PATH, allow_pickle=True)
        print(f"Model loaded: {len(self.labels)} classes: {', '.join(self.labels)}")

        self.expected_features = self.model.input_shape[1]
        print(f"Model expects {self.expected_features} input features.")

        self.hand_detector = HandDetector(num_hands=2 if self.expected_features == 126 else 1)
        self.speech = SpeechEngine()

        # State
        self.text_buffer = ""
        self.undo_stack = []
        self._updating_display = False
        self.predicted_label = "..."
        self.confidence = 0.0
        self.hand_detected = False
        self.num_hands_detected = 0

        # Prediction smoothing & confirmation
        self.prediction_history = deque(maxlen=10)
        self.stable_count = 0
        self.last_stable_label = None
        self.cooldown_counter = 0
        self.COOLDOWN_FRAMES = 20
        self.need_change_before_repeat = False

        # Camera
        self.preferred_camera_idx = 0
        for i, a in enumerate(sys.argv):
            if a in ("--camera", "-c") and i + 1 < len(sys.argv):
                try:
                    self.preferred_camera_idx = int(sys.argv[i + 1])
                except ValueError:
                    pass

        self.camera = None
        self.active_camera_info = None
        self.probed_cameras = []
        self.running = False

        # FPS calculation
        self.prev_frame_time = time.time()
        self.fps_list = deque(maxlen=20)
        self.current_fps = 0.0

        # Create GUI
        self._create_gui()

    # ==================================================
    # GUI CREATION
    # ==================================================

    def _create_gui(self):
        """Construct the Tkinter interface with complete controls."""
        self.root = tk.Tk()
        self.root.title("Indian Sign Language Recognition System")
        self.root.configure(bg="#121826")
        self.root.resizable(False, False)

        # Bring window to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(800, lambda: self.root.attributes('-topmost', False))
        self.root.focus_force()

        # Fonts
        title_font = ("Segoe UI", 14, "bold")
        heading_font = ("Segoe UI", 11, "bold")
        label_font = ("Segoe UI", 10)
        text_font = ("Consolas", 14)
        btn_font = ("Segoe UI", 9, "bold")
        small_font = ("Segoe UI", 8)

        # Color tokens
        bg_dark = "#121826"
        card_bg = "#1a2234"
        accent_blue = "#0f3460"
        highlight_cyan = "#00d4ff"
        success_green = "#00e676"
        warning_yellow = "#ffd600"
        danger_red = "#ff4757"
        text_light = "#f1f2f6"

        # --------------------------------------------------
        # TOP HEADER
        # --------------------------------------------------
        header_frame = tk.Frame(self.root, bg=accent_blue, padx=15, pady=8)
        header_frame.pack(fill="x")

        title_lbl = tk.Label(
            header_frame,
            text="🤟 INDIAN SIGN LANGUAGE RECOGNITION SYSTEM",
            font=title_font,
            fg="white",
            bg=accent_blue
        )
        title_lbl.pack(side="left")

        # Camera selection dropdown in header
        cam_sel_frame = tk.Frame(header_frame, bg=accent_blue)
        cam_sel_frame.pack(side="right")

        tk.Label(
            cam_sel_frame, text="Camera:", font=small_font, fg="#cbd5e1", bg=accent_blue
        ).pack(side="left", padx=(0, 4))

        self.cam_combo_var = tk.StringVar()
        self.cam_combobox = ttk.Combobox(
            cam_sel_frame,
            textvariable=self.cam_combo_var,
            state="readonly",
            width=28,
            font=("Segoe UI", 8)
        )
        self.cam_combobox.pack(side="left", padx=(0, 5))
        self.cam_combobox.bind("<<ComboboxSelected>>", self._on_camera_selected)

        self.btn_rescan_cam = tk.Button(
            cam_sel_frame,
            text="🔄 Rescan",
            font=small_font,
            bg="#16213e",
            fg="white",
            activebackground="#00d4ff",
            relief="flat",
            padx=6, pady=2,
            cursor="hand2",
            command=self.rescan_cameras
        )
        self.btn_rescan_cam.pack(side="left")

        # --------------------------------------------------
        # MAIN BODY (2 COLUMNS)
        # --------------------------------------------------
        body = tk.Frame(self.root, bg=bg_dark, padx=12, pady=10)
        body.pack(fill="both", expand=True)

        # ==================================================
        # LEFT COLUMN: CAMERA & VIDEO OPTIONS
        # ==================================================
        left_col = tk.Frame(body, bg=bg_dark)
        left_col.pack(side="left", padx=(0, 10))

        # Camera viewport container
        cam_card = tk.LabelFrame(
            left_col, text="  Live Video Feed  ",
            font=heading_font, fg=highlight_cyan, bg=card_bg,
            padx=6, pady=6, bd=1, relief="solid"
        )
        cam_card.pack(fill="x")

        # Fixed 640x480 video canvas
        cam_box = tk.Frame(cam_card, width=640, height=480, bg="#050811")
        cam_box.pack_propagate(False)
        cam_box.pack()

        self.camera_label = tk.Label(
            cam_box,
            text="Initializing webcam feed...",
            font=label_font,
            bg="#050811",
            fg="#94a3b8"
        )
        self.camera_label.pack(fill="both", expand=True)

        # Video Options Toolbar underneath camera
        opts_frame = tk.Frame(cam_card, bg=card_bg, pady=5)
        opts_frame.pack(fill="x")

        self.var_mirror = tk.BooleanVar(value=True)
        cb_mirror = tk.Checkbutton(
            opts_frame, text="Mirror Video (Flip)",
            variable=self.var_mirror,
            font=small_font, fg=text_light, bg=card_bg,
            selectcolor="#0f3460", activebackground=card_bg, activeforeground=highlight_cyan
        )
        cb_mirror.pack(side="left", padx=(5, 10))

        self.var_landmarks = tk.BooleanVar(value=True)
        cb_landmarks = tk.Checkbutton(
            opts_frame, text="Show Landmarks / Skeleton",
            variable=self.var_landmarks,
            font=small_font, fg=text_light, bg=card_bg,
            selectcolor="#0f3460", activebackground=card_bg, activeforeground=highlight_cyan
        )
        cb_landmarks.pack(side="left", padx=(0, 10))

        self.var_auto_add = tk.BooleanVar(value=True)
        cb_auto = tk.Checkbutton(
            opts_frame, text="Auto-Add Held Sign",
            variable=self.var_auto_add,
            font=small_font, fg=text_light, bg=card_bg,
            selectcolor="#0f3460", activebackground=card_bg, activeforeground=highlight_cyan
        )
        cb_auto.pack(side="left", padx=(0, 10))

        self.var_paused = tk.BooleanVar(value=False)
        self.btn_pause = tk.Button(
            opts_frame,
            text="⏸ Pause Detection",
            font=small_font,
            bg="#2c3e50", fg="white",
            activebackground="#34495e", activeforeground="white",
            relief="flat", padx=8, pady=2, cursor="hand2",
            command=self.toggle_pause
        )
        self.btn_pause.pack(side="right", padx=(0, 5))

        # Sliders card for fine-tuning
        sliders_card = tk.Frame(cam_card, bg=card_bg, pady=4)
        sliders_card.pack(fill="x")

        # Confidence slider
        s_left = tk.Frame(sliders_card, bg=card_bg)
        s_left.pack(side="left", fill="x", expand=True, padx=(5, 10))

        self.conf_val_lbl = tk.Label(
            s_left, text="Min Confidence: 70%", font=small_font, fg="#94a3b8", bg=card_bg
        )
        self.conf_val_lbl.pack(anchor="w")

        self.conf_slider = tk.Scale(
            s_left, from_=50, to=95, orient="horizontal",
            bg=card_bg, fg=text_light, highlightthickness=0,
            troughcolor="#0f3460", activebackground=highlight_cyan,
            showvalue=False, command=self._on_conf_change
        )
        self.conf_slider.set(70)
        self.conf_slider.pack(fill="x")

        # Hold frames slider
        s_right = tk.Frame(sliders_card, bg=card_bg)
        s_right.pack(side="right", fill="x", expand=True, padx=(0, 5))

        self.hold_val_lbl = tk.Label(
            s_right, text="Hold Speed: 15 frames (~0.5s)", font=small_font, fg="#94a3b8", bg=card_bg
        )
        self.hold_val_lbl.pack(anchor="w")

        self.hold_slider = tk.Scale(
            s_right, from_=8, to=30, orient="horizontal",
            bg=card_bg, fg=text_light, highlightthickness=0,
            troughcolor="#0f3460", activebackground=highlight_cyan,
            showvalue=False, command=self._on_hold_change
        )
        self.hold_slider.set(15)
        self.hold_slider.pack(fill="x")

        # Camera status & resolution footer
        self.cam_info_lbl = tk.Label(
            cam_card, text="Camera: Connecting...", font=small_font, fg="#64748b", bg=card_bg
        )
        self.cam_info_lbl.pack(anchor="w", padx=5, pady=(4, 0))

        # ==================================================
        # RIGHT COLUMN: RECOGNITION & TEXT EDITING CONTROLS
        # ==================================================
        right_col = tk.Frame(body, bg=bg_dark, width=380)
        right_col.pack(side="right", fill="both", expand=True)
        right_col.pack_propagate(False)

        # --------------------------------------------------
        # CARD 1: CURRENT DETECTED SIGN
        # --------------------------------------------------
        sign_card = tk.LabelFrame(
            right_col, text="  Current Sign Detection  ",
            font=heading_font, fg=highlight_cyan, bg=card_bg,
            padx=10, pady=6, bd=1, relief="solid"
        )
        sign_card.pack(fill="x", pady=(0, 8))

        top_sign_row = tk.Frame(sign_card, bg=card_bg)
        top_sign_row.pack(fill="x")

        # Big sign label
        self.sign_label = tk.Label(
            top_sign_row,
            text="...",
            font=("Segoe UI", 42, "bold"),
            bg=card_bg,
            fg=success_green,
            width=3
        )
        self.sign_label.pack(side="left", padx=(10, 15))

        sign_meta_col = tk.Frame(top_sign_row, bg=card_bg)
        sign_meta_col.pack(side="left", fill="both", expand=True)

        self.confidence_label = tk.Label(
            sign_meta_col,
            text="Confidence: 0.0%",
            font=label_font,
            bg=card_bg,
            fg=warning_yellow
        )
        self.confidence_label.pack(anchor="w", pady=(4, 2))

        self.status_label = tk.Label(
            sign_meta_col,
            text="Status: Waiting for hand...",
            font=small_font,
            bg=card_bg,
            fg="#94a3b8"
        )
        self.status_label.pack(anchor="w", pady=(0, 2))

        # Stability hold progress bar
        p_frame = tk.Frame(sign_card, bg=card_bg, pady=2)
        p_frame.pack(fill="x")

        tk.Label(
            p_frame, text="Hold Progress:", font=("Segoe UI", 7), fg="#64748b", bg=card_bg
        ).pack(side="left", padx=(0, 5))

        self.progress_canvas = tk.Canvas(
            p_frame, height=8, bg="#0d1117", highlightthickness=0
        )
        self.progress_canvas.pack(side="left", fill="x", expand=True)

        # --------------------------------------------------
        # CARD 2: RECOGNIZED TEXT & EDITING CONTROLS
        # --------------------------------------------------
        text_card = tk.LabelFrame(
            right_col, text="  Recognized Sentence  ",
            font=heading_font, fg=highlight_cyan, bg=card_bg,
            padx=10, pady=8, bd=1, relief="solid"
        )
        text_card.pack(fill="both", expand=True)

        # Sub-header with text length counter
        text_hdr = tk.Frame(text_card, bg=card_bg)
        text_hdr.pack(fill="x", pady=(0, 4))

        self.text_stats_lbl = tk.Label(
            text_hdr,
            text="0 characters | 0 words",
            font=small_font,
            fg="#94a3b8",
            bg=card_bg
        )
        self.text_stats_lbl.pack(side="left")

        self.notif_lbl = tk.Label(
            text_hdr,
            text="",
            font=("Segoe UI", 8, "italic"),
            fg=success_green,
            bg=card_bg
        )
        self.notif_lbl.pack(side="right")

        # Interactive editable text box
        self.text_display = tk.Text(
            text_card,
            font=text_font,
            bg="#090d16",
            fg="white",
            insertbackground="white",
            height=3,
            wrap="word",
            relief="solid",
            bd=1,
            padx=8,
            pady=8
        )
        self.text_display.pack(fill="x", pady=(0, 8))
        self.text_display.bind("<KeyRelease>", self._on_text_widget_change)

        # Common button styling
        def make_btn(parent, text, cmd, bg_col, hover_col, fg_col="white"):
            b = tk.Button(
                parent, text=text, command=cmd,
                font=btn_font, bg=bg_col, fg=fg_col,
                activebackground=hover_col, activeforeground=fg_col,
                relief="flat", cursor="hand2", pady=5
            )
            return b

        # --------------------------------------------------
        # TEXT ACTION BUTTONS
        # --------------------------------------------------

        # Row 1: Add Current Sign & Add Space
        r1 = tk.Frame(text_card, bg=card_bg)
        r1.pack(fill="x", pady=2)

        self.btn_add = make_btn(r1, "✓ ADD SIGN", self.add_letter, "#1e3a8a", "#2563eb")
        self.btn_add.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_space = make_btn(r1, "⎵ SPACE", self.add_space, "#1e293b", "#334155")
        self.btn_space.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # Row 2: Delete Letter (Backspace) & Delete Word
        r2 = tk.Frame(text_card, bg=card_bg)
        r2.pack(fill="x", pady=2)

        self.btn_backspace = make_btn(r2, "⌫ DELETE LETTER", self.backspace, "#7f1d1d", "#991b1b")
        self.btn_backspace.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_del_word = make_btn(r2, "🗑️ DELETE WORD", self.delete_word, "#831843", "#9d174d")
        self.btn_del_word.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # Row 3: Clear All & Undo
        r3 = tk.Frame(text_card, bg=card_bg)
        r3.pack(fill="x", pady=2)

        self.btn_clear = make_btn(r3, "✕ CLEAR ALL", self.clear_text, "#450a0a", "#dc2626")
        self.btn_clear.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_undo = make_btn(r3, "↩ UNDO", self.undo_text, "#374151", "#4b5563")
        self.btn_undo.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # Row 4: Copy to Clipboard & Save to File
        r4 = tk.Frame(text_card, bg=card_bg)
        r4.pack(fill="x", pady=2)

        self.btn_copy = make_btn(r4, "📋 COPY TEXT", self.copy_to_clipboard, "#065f46", "#059669")
        self.btn_copy.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_save = make_btn(r4, "💾 SAVE TO FILE", self.save_to_file, "#0e7490", "#0891b2")
        self.btn_save.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # Row 5: Speak Text (Full width)
        self.btn_speak = make_btn(text_card, "🔊 SPEAK SENTENCE (TTS)", self.speak_text, "#14532d", "#16a34a")
        self.btn_speak.pack(fill="x", pady=3)

        # Row 6: Sign Reference Guide & Exit
        r6 = tk.Frame(text_card, bg=card_bg)
        r6.pack(fill="x", pady=2)

        self.btn_guide = make_btn(r6, "📖 SIGN GUIDE", self.show_sign_guide, "#312e81", "#4338ca")
        self.btn_guide.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_exit = make_btn(r6, "✕ EXIT", self.quit_app, "#3f3f46", "#52525b")
        self.btn_exit.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # --------------------------------------------------
        # FOOTER LEGEND
        # --------------------------------------------------
        footer = tk.Frame(right_col, bg=bg_dark, pady=4)
        footer.pack(fill="x")

        shortcuts_txt = "Shortcuts: Space=Add | -=Space | Backspace=Del Char | Ctrl+Del=Clear | S=Speak | P=Pause | Esc=Exit"
        tk.Label(
            footer, text=shortcuts_txt, font=("Segoe UI", 7), fg="#64748b", bg=bg_dark, wraplength=370, justify="left"
        ).pack(anchor="w")

        # --------------------------------------------------
        # GLOBAL KEYBOARD BINDINGS
        # --------------------------------------------------
        self.root.bind("<space>", lambda e: self._on_space_key(e))
        self.root.bind("<minus>", lambda e: self.add_space())
        self.root.bind("<BackSpace>", lambda e: self._on_backspace_key(e))
        self.root.bind("<Control-BackSpace>", lambda e: self.delete_word())
        self.root.bind("<Delete>", lambda e: self.clear_text())
        self.root.bind("<Control-z>", lambda e: self.undo_text())
        self.root.bind("<Control-c>", lambda e: self.copy_to_clipboard())
        self.root.bind("s", lambda e: self._on_s_key(e))
        self.root.bind("p", lambda e: self._on_p_key(e))
        self.root.bind("<Escape>", lambda e: self.quit_app())
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    # ==================================================
    # KEYBOARD & SLIDER HANDLERS
    # ==================================================

    def _on_space_key(self, event):
        # If user is actively typing in text widget, let normal space pass through
        if self.root.focus_get() == self.text_display:
            return
        if self.predicted_label not in ("...", "Unknown"):
            self.add_letter()
        else:
            self.add_space()

    def _on_backspace_key(self, event):
        if self.root.focus_get() == self.text_display:
            return
        self.backspace()

    def _on_s_key(self, event):
        if self.root.focus_get() == self.text_display:
            return
        self.speak_text()

    def _on_p_key(self, event):
        if self.root.focus_get() == self.text_display:
            return
        self.toggle_pause()

    def _on_conf_change(self, val):
        self.conf_val_lbl.config(text=f"Min Confidence: {val}%")

    def _on_hold_change(self, val):
        fps_est = max(self.current_fps, 25.0)
        secs = float(val) / fps_est
        self.hold_val_lbl.config(text=f"Hold Speed: {val} frames (~{secs:.1f}s)")

    def _on_text_widget_change(self, event=None):
        """Keep buffer in sync when user edits directly in text widget."""
        if self._updating_display:
            return
        new_text = self.text_display.get("1.0", "end-1c")
        if new_text != self.text_buffer:
            self._push_undo()
            self.text_buffer = new_text
            self._update_stats_label()

    def _push_undo(self):
        """Push current text state to undo stack."""
        if not self.undo_stack or self.undo_stack[-1] != self.text_buffer:
            self.undo_stack.append(self.text_buffer)
            if len(self.undo_stack) > 30:
                self.undo_stack.pop(0)

    def _set_notification(self, msg, duration_ms=2500):
        """Display a temporary toast notification in the UI."""
        self.notif_lbl.config(text=msg)
        self.root.after(duration_ms, lambda: self.notif_lbl.config(text=""))

    def _update_stats_label(self):
        chars = len(self.text_buffer)
        words = len(self.text_buffer.strip().split()) if self.text_buffer.strip() else 0
        self.text_stats_lbl.config(text=f"{chars} characters | {words} words")

    def _update_text_display(self):
        """Refresh text widget from internal buffer."""
        self._updating_display = True
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert("1.0", self.text_buffer)
        self.text_display.see(tk.END)
        self._updating_display = False
        self._update_stats_label()

    # ==================================================
    # BUTTON ACTIONS: TEXT & PREDICTION
    # ==================================================

    def add_letter(self):
        """Manually add current predicted sign."""
        if self.predicted_label not in ("...", "Unknown"):
            self._push_undo()
            self.text_buffer += self.predicted_label
            self._update_text_display()
            self._set_notification(f"Added letter: {self.predicted_label}")
            print(f"Added letter: {self.predicted_label} | Text: '{self.text_buffer}'")

    def add_space(self):
        """Add space word separator."""
        self._push_undo()
        self.text_buffer += " "
        self._update_text_display()
        self._set_notification("Added Space")

    def backspace(self):
        """Delete last letter / character."""
        if len(self.text_buffer) > 0:
            self._push_undo()
            removed = self.text_buffer[-1]
            self.text_buffer = self.text_buffer[:-1]
            self._update_text_display()
            self._set_notification(f"Deleted '{removed}'")
            print(f"Deleted letter | Text: '{self.text_buffer}'")

    def delete_word(self):
        """Delete the last space-separated word."""
        if not self.text_buffer:
            return
        self._push_undo()
        trimmed = self.text_buffer.rstrip()
        last_space = trimmed.rfind(" ")
        if last_space == -1:
            self.text_buffer = ""
        else:
            self.text_buffer = trimmed[:last_space + 1]
        self._update_text_display()
        self._set_notification("Deleted word")
        print(f"Deleted word | Text: '{self.text_buffer}'")

    def clear_text(self):
        """Clear all text with undo protection."""
        if not self.text_buffer:
            return
        self._push_undo()
        self.text_buffer = ""
        self._update_text_display()
        self._set_notification("Cleared all text (Press Undo to restore)")
        print("Text cleared.")

    def undo_text(self):
        """Restore text from undo stack."""
        if self.undo_stack:
            self.text_buffer = self.undo_stack.pop()
            self._update_text_display()
            self._set_notification("Undone last edit")
            print(f"Undo action | Restored: '{self.text_buffer}'")
        else:
            self._set_notification("Nothing to undo")

    def copy_to_clipboard(self):
        """Copy recognized text to system clipboard."""
        if not self.text_buffer.strip():
            self._set_notification("Nothing to copy")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.text_buffer)
        self.root.update()
        self._set_notification("✓ Copied to clipboard!")

    def save_to_file(self):
        """Save recognized sentence with timestamp to isl_history.txt."""
        if not self.text_buffer.strip():
            self._set_notification("Nothing to save")
            return
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open("isl_history.txt", "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {self.text_buffer.strip()}\n")
            self._set_notification("✓ Saved to isl_history.txt")
            print(f"Saved: '[{timestamp}] {self.text_buffer}'")
        except Exception as e:
            self._set_notification(f"Save error: {e}")

    def speak_text(self):
        """Speak recognized text aloud."""
        if not self.text_buffer.strip():
            self._set_notification("No text to speak")
            return

        if not self.speech.available:
            messagebox.showwarning(
                "TTS Unavailable",
                "Text-to-speech is not available.\nInstall pyttsx3: pip install pyttsx3"
            )
            return

        self._set_notification("🔊 Speaking...")
        self.speech.speak(self.text_buffer)

    def toggle_pause(self):
        """Pause or resume sign detection."""
        is_paused = not self.var_paused.get()
        self.var_paused.set(is_paused)
        if is_paused:
            self.btn_pause.config(text="▶ Resume Detection", bg="#15803d")
            self.status_label.config(text="Status: Detection Paused", fg="#ffd600")
            self.sign_label.config(text="PAUSED", font=("Segoe UI", 16, "bold"), fg="#ffd600")
            self._set_notification("Detection paused")
        else:
            self.btn_pause.config(text="⏸ Pause Detection", bg="#2c3e50")
            self.sign_label.config(font=("Segoe UI", 42, "bold"))
            self._set_notification("Detection resumed")

    def show_sign_guide(self):
        """Display an interactive modal showing supported alphabet gestures."""
        guide_win = tk.Toplevel(self.root)
        guide_win.title("Indian Sign Language - Gesture Reference Guide")
        guide_win.configure(bg="#121826")
        guide_win.geometry("540x520")
        guide_win.resizable(False, False)
        guide_win.lift()
        guide_win.grab_set()

        hdr = tk.Frame(guide_win, bg="#0f3460", padx=15, pady=12)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="📖 ISL Alphabet & Gesture Reference",
            font=("Segoe UI", 13, "bold"), fg="#00d4ff", bg="#0f3460"
        ).pack()
        tk.Label(
            hdr, text="Supported Indian Sign Language alphabet characters (A - Z)",
            font=("Segoe UI", 9), fg="#e0e0e0", bg="#0f3460"
        ).pack(pady=(2, 0))

        content = tk.Frame(guide_win, bg="#121826", padx=15, pady=12)
        content.pack(fill="both", expand=True)

        # Alphabet grid
        grid_frame = tk.LabelFrame(
            content, text="  Supported Letters (26 Classes)  ",
            font=("Segoe UI", 10, "bold"), fg="#00d4ff", bg="#1a2234",
            padx=10, pady=10, bd=1, relief="solid"
        )
        grid_frame.pack(fill="x", pady=(0, 10))

        cols = 7
        for idx, lbl in enumerate(self.labels):
            r = idx // cols
            c = idx % cols
            btn = tk.Label(
                grid_frame, text=lbl, font=("Segoe UI", 11, "bold"),
                bg="#0f3460", fg="#00e676", width=4, height=1, relief="ridge"
            )
            btn.grid(row=r, column=c, padx=3, pady=3)

        # Guidelines card
        tips_frame = tk.LabelFrame(
            content, text="  Signing Best Practices  ",
            font=("Segoe UI", 10, "bold"), fg="#00d4ff", bg="#1a2234",
            padx=12, pady=8, bd=1, relief="solid"
        )
        tips_frame.pack(fill="both", expand=True)

        tips = [
            "• ISL uses 2-handed signing for many letters (A, B, D, G, H, P, Q, R, S, T, W, X, Y).",
            "• Single-handed signs (C, E, I, J, L, O, U, V, Z) are also supported.",
            "• Keep both hands fully within camera view facing adequate lighting.",
            "• Hold your sign steady for ~0.5 seconds to auto-confirm into text.",
            "• If an incorrect letter is added, click 'DELETE LETTER' or 'DELETE WORD'."
        ]
        for t in tips:
            tk.Label(
                tips_frame, text=t, font=("Segoe UI", 8), fg="#cbd5e1", bg="#1a2234", anchor="w"
            ).pack(fill="x", pady=2)

        close_btn = tk.Button(
            content, text="Close Guide", font=("Segoe UI", 9, "bold"),
            bg="#2563eb", fg="white", activebackground="#3b82f6",
            relief="flat", padx=15, pady=5, cursor="hand2",
            command=guide_win.destroy
        )
        close_btn.pack(pady=(10, 0))

    # ==================================================
    # CAMERA MANAGEMENT
    # ==================================================

    def rescan_cameras(self):
        """Probe connected camera devices and populate dropdown."""
        print("Rescanning connected camera hardware...")
        self.probed_cameras = probe_all_cameras()
        menu_items = []
        best_active_item = None

        for dev in self.probed_cameras:
            label = dev["label"]
            menu_items.append(label)
            if dev["is_active"] and best_active_item is None:
                best_active_item = label

        self.cam_combobox["values"] = menu_items

        # If currently open camera is still in list, keep it selected
        if self.active_camera_info:
            curr_label = self.active_camera_info.get("label", "")
            for item in menu_items:
                if f"Camera {self.active_camera_info['index']}" in item and self.active_camera_info['backend_name'] in item:
                    self.cam_combo_var.set(item)
                    return

        # Otherwise pick best active
        if best_active_item:
            self.cam_combo_var.set(best_active_item)
        elif menu_items:
            self.cam_combo_var.set(menu_items[0])

    def _on_camera_selected(self, event=None):
        """Handler when user picks a camera from dropdown."""
        selected_label = self.cam_combo_var.get()
        for dev in self.probed_cameras:
            if dev["label"] == selected_label:
                self._switch_to_camera_device(dev)
                return

    def _switch_to_camera_device(self, dev_info):
        """Open a specific probed camera device."""
        print(f"Switching camera to: {dev_info['label']}...")
        if self.camera:
            self.camera.release()

        try:
            new_cap = cv2.VideoCapture(dev_info["index"], dev_info["backend"])
            if new_cap and new_cap.isOpened():
                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera = new_cap
                self.active_camera_info = dev_info
                self.preferred_camera_idx = dev_info["index"]
                self.cam_info_lbl.config(
                    text=f"Camera: Device {dev_info['index']} [{dev_info['backend_name']}] - {dev_info['width']}x{dev_info['height']}"
                )
                self._set_notification(f"Switched to Camera {dev_info['index']}")
                print(f"Switched successfully to {dev_info['label']}")
                return True
        except Exception as e:
            print(f"Error switching camera: {e}")

        # Fallback
        self._set_notification("Failed to switch camera, restoring auto-detected device")
        self.camera, self.active_camera_info = open_best_camera()
        return False

    def init_camera(self):
        """Initialize best working camera on startup."""
        self.rescan_cameras()
        print("Selecting best available camera...")
        self.camera, self.active_camera_info = open_best_camera(preferred_idx=self.preferred_camera_idx)
        if not self.camera:
            return False

        # Sync dropdown
        for dev in self.probed_cameras:
            if dev["index"] == self.active_camera_info["index"] and dev["backend_name"] == self.active_camera_info["backend_name"]:
                self.cam_combo_var.set(dev["label"])
                break

        self.cam_info_lbl.config(
            text=f"Camera: Device {self.active_camera_info['index']} [{self.active_camera_info['backend_name']}] - {self.active_camera_info['width']}x{self.active_camera_info['height']}"
        )
        return True

    # ==================================================
    # MAIN CAMERA PROCESSING LOOP
    # ==================================================

    def process_frame(self):
        """Main camera acquisition and AI inference frame loop."""
        if not self.running:
            return

        success, frame = self.camera.read()
        if not success or frame is None:
            self.status_label.config(text="Status: Camera read failed", fg="#ff4757")
            self.root.after(80, self.process_frame)
            return

        # FPS calculation
        now = time.time()
        fps = 1.0 / max(now - self.prev_frame_time, 0.001)
        self.prev_frame_time = now
        self.fps_list.append(fps)
        self.current_fps = sum(self.fps_list) / len(self.fps_list)

        # Mirror flip if enabled
        if self.var_mirror.get():
            frame = cv2.flip(frame, 1)

        # Check for genuine pitch black frame (cover closed)
        frame_mean = float(np.mean(frame))
        frame_std = float(np.std(frame))
        is_covered = (frame_mean < 2.0 and frame_std < 2.0)

        # Skip inference if detection paused or lens covered
        if not self.var_paused.get() and not is_covered:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result = self.hand_detector.detect(rgb_frame)

            if detection_result and detection_result.hand_landmarks:
                self.hand_detected = True
                self.num_hands_detected = len(detection_result.hand_landmarks)

                # Extract features
                if self.expected_features == 126:
                    features = normalize_two_hands(detection_result)
                    if features is not None:
                        features = features.reshape(1, 126)
                else:
                    features = normalize_landmarks(detection_result.hand_landmarks[0])
                    features = features.reshape(1, 63)

                if features is not None:
                    prediction = self.model.predict(features, verbose=0)
                    pred_idx = np.argmax(prediction[0])
                    self.confidence = float(prediction[0][pred_idx])
                    current_label = str(self.labels[pred_idx])

                    min_conf = float(self.conf_slider.get()) / 100.0
                    if self.confidence >= min_conf:
                        self.prediction_history.append(current_label)
                        self.predicted_label = Counter(self.prediction_history).most_common(1)[0][0]
                    else:
                        self.prediction_history.clear()
                        self.predicted_label = "..."

                    # Auto confirmation
                    if self.var_auto_add.get():
                        self._process_auto_confirmation()

                # Draw skeleton landmarks
                if self.var_landmarks.get():
                    for hand_i, lms in enumerate(detection_result.hand_landmarks):
                        color = (0, 255, 0) if hand_i == 0 else (255, 200, 0)
                        draw_hand(frame, lms, line_color=color)

            else:
                self.hand_detected = False
                self.num_hands_detected = 0
                self.prediction_history.clear()
                self.predicted_label = "..."
                self.confidence = 0.0
                self.stable_count = 0
                self.last_stable_label = None
                self.need_change_before_repeat = False

        # Update GUI elements
        self._update_gui_feed(frame, is_covered)

        # Schedule next frame (approx 30-60 FPS)
        if self.running:
            self.root.after(15, self.process_frame)

    def _process_auto_confirmation(self):
        """Auto-add letters when a sign is held steadily for required frame count."""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        if self.predicted_label in ("...", "Unknown"):
            return

        if self.predicted_label == self.last_stable_label:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_stable_label = self.predicted_label
            self.need_change_before_repeat = False

        target_hold = int(self.hold_slider.get())
        if self.stable_count >= target_hold:
            if not self.need_change_before_repeat:
                self._push_undo()
                self.text_buffer += self.predicted_label
                self._update_text_display()
                self._set_notification(f"Auto-added: {self.predicted_label}")
                print(f"Auto-added: {self.predicted_label} | Text: '{self.text_buffer}'")

                self.need_change_before_repeat = True
                self.stable_count = 0
                self.cooldown_counter = self.COOLDOWN_FRAMES

    def _update_gui_feed(self, frame, is_covered):
        """Render overlay graphics and update Tkinter camera canvas."""
        # Visual overlay tags on OpenCV frame
        fps_str = f"{self.current_fps:.1f} FPS"
        cv2.putText(
            frame, fps_str, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        if is_covered:
            cv2.putText(
                frame, "CAMERA SHUTTER COVERED OR PITCH BLACK",
                (30, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            self.status_label.config(text="Status: Shutter Closed / Fn+F10", fg="#ff4757")
        elif self.var_paused.get():
            cv2.putText(
                frame, "DETECTION PAUSED",
                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2
            )
        else:
            # Update labels
            if not self.var_paused.get():
                self.sign_label.config(text=self.predicted_label)
                if self.hand_detected:
                    self.sign_label.config(fg="#00e676")
                    hands_txt = f"✋ {self.num_hands_detected} Hand{'s' if self.num_hands_detected > 1 else ''} Detected"
                    self.status_label.config(text=f"Status: {hands_txt}", fg="#00e676")
                else:
                    self.sign_label.config(fg="#64748b")
                    self.status_label.config(text="Status: Show your hand", fg="#94a3b8")

                conf_percent = self.confidence * 100.0
                self.confidence_label.config(text=f"Confidence: {conf_percent:.1f}%")
                if conf_percent >= 85.0:
                    self.confidence_label.config(fg="#00e676")
                elif conf_percent >= 70.0:
                    self.confidence_label.config(fg="#ffd600")
                else:
                    self.confidence_label.config(fg="#ff4757")

        # Update hold progress canvas
        self.progress_canvas.delete("all")
        canvas_width = self.progress_canvas.winfo_width()
        target_hold = int(self.hold_slider.get())

        if canvas_width > 1 and self.stable_count > 0 and not self.need_change_before_repeat:
            prog = min(self.stable_count / max(target_hold, 1), 1.0)
            fill_w = int(canvas_width * prog)
            self.progress_canvas.create_rectangle(
                0, 0, fill_w, 8, fill="#00e676", outline=""
            )

        # Convert and blit frame to Tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def quit_app(self):
        """Gracefully terminate application."""
        print("Closing application...")
        self.running = False
        if self.camera:
            self.camera.release()
        self.hand_detector.close()
        cv2.destroyAllWindows()
        self.root.destroy()
        print("Application terminated.")
        if self.text_buffer:
            print(f"Recognized text session: '{self.text_buffer}'")

    def run(self):
        """Start the application loop."""
        print("=" * 60)
        print("  Indian Sign Language Recognition System - GUI Starting")
        print("=" * 60)

        if not self.init_camera():
            messagebox.showerror(
                "Camera Error",
                "Could not open any webcam device.\n\n"
                "Please check:\n"
                "1. Laptop camera privacy slider (bezel switch)\n"
                "2. Windows camera permissions for desktop applications\n"
                "3. Press Fn+F10 to toggle camera privacy hardware lock"
            )
            return

        self.running = True
        self.root.after(80, self.process_frame)
        self.root.mainloop()


# ==================================================
# ENTRY POINT
# ==================================================

def main():
    check_files()
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("ERROR: Pillow is required. Install: pip install pillow")
        sys.exit(1)

    app = ISLApp()
    app.run()


if __name__ == "__main__":
    main()
