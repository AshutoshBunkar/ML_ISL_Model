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
import queue
import json
import tkinter as tk
from tkinter import ttk, messagebox, font
from collections import deque, Counter
from PIL import Image, ImageTk

# Camera utils
sys.path.insert(0, _script_dir)
from camera_utils import probe_all_cameras, open_best_camera


# ==================================================
# FILE PATHS
# ==================================================

MODEL_PATH = os.path.join(_root_dir, "models", "isl_model.keras")
HAND_MODEL_PATH = os.path.join(_root_dir, "models", "hand_landmarker.task")
LABELS_PATH = os.path.join(_root_dir, "models", "labels.npy")
HISTORY_FILE = os.path.join(_root_dir, "isl_history.txt")
A11Y_CONFIG_PATH = os.path.join(_root_dir, "accessibility_settings.json")


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
    """Thread-safe text-to-speech engine using a dedicated worker thread with pyttsx3."""

    def __init__(self):
        self.available = False
        self.queue = queue.Queue()
        self.speaking = False
        self.worker = None

        try:
            import pyttsx3
            # Test that pyttsx3 can be imported
            self.available = True
            self.worker = threading.Thread(target=self._worker_loop, daemon=True, name="TTSWorker")
            self.worker.start()
            print("Text-to-speech: Available")
        except Exception as e:
            print(f"Text-to-speech: Not available ({e})")

    def _init_engine(self):
        """Helper to create and configure a pyttsx3 engine instance."""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.9)
        return engine

    def _worker_loop(self):
        """Dedicated thread loop for pyttsx3 to preserve COM apartment state."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        engine = None
        try:
            engine = self._init_engine()
        except Exception as e:
            print(f"TTS Worker init error: {e}")
            self.available = False
            return

        while True:
            try:
                item = self.queue.get()
                if item is None:
                    break
                text = str(item).strip()
                if not text:
                    self.queue.task_done()
                    continue

                self.speaking = True

                # Fix pyttsx3 driver state for repeated calls:
                # pyttsx3's SAPI5 driver leaves proxy._busy = False after runAndWait().
                # When say() is subsequently called, _pump() fires immediately and starts
                # asynchronous playback before runAndWait() can push endLoop. Then
                # startLoop() immediately pops endLoop, calling stop() and aborting audio.
                # Setting proxy._busy = True and resetting _inLoop ensures say and endLoop
                # sequence correctly on every consecutive call.
                if hasattr(engine, "proxy"):
                    engine.proxy._busy = True
                    if hasattr(engine.proxy, "_queue"):
                        engine.proxy._queue.clear()
                if hasattr(engine, "_inLoop"):
                    engine._inLoop = False

                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Speech playback error: {e}")
                try:
                    if engine is not None:
                        del engine
                except Exception:
                    pass
                try:
                    engine = self._init_engine()
                except Exception as reinit_err:
                    print(f"TTS Re-initialization error: {reinit_err}")
                    engine = None
            finally:
                self.speaking = False
                self.queue.task_done()

        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

    def speak(self, text):
        """Speak text through the dedicated TTS worker thread."""
        if not self.available or text is None:
            return
        cleaned = str(text).strip()
        if not cleaned:
            return

        self.queue.put(cleaned)

    def is_speaking(self):
        """Check if speech synthesis is currently active."""
        return self.speaking

    def stop(self):
        """Signal the TTS worker thread to terminate."""
        if self.available and self.worker and self.worker.is_alive():
            self.queue.put(None)


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
        """Construct the Tkinter interface styled with the SE_PROJECT design theme and accessibility support."""
        self.root = tk.Tk()
        self.root.title("Indian Sign Language Recognition System")
        self.root.configure(bg="#fafafa")
        self.root.resizable(True, True)
        self.root.minsize(1060, 680)

        # Accessibility State (Tkinter variables initialized after tk.Tk())
        self.a11y_dark_mode = tk.BooleanVar(value=False)
        self.a11y_invert_colors = tk.BooleanVar(value=False)
        self.a11y_large_cursor = tk.BooleanVar(value=False)
        self.a11y_high_contrast = tk.BooleanVar(value=False)
        self.a11y_text_size = tk.StringVar(value="Default")
        self._load_a11y_settings()

        # Bring window to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(800, lambda: self.root.attributes('-topmost', False))
        self.root.focus_force()

        # Dynamic Fonts for Accessibility (supports Small, Default, Large)
        try:
            display_family = "Space Grotesk" if "Space Grotesk" in font.families(self.root) else "Segoe UI"
        except Exception:
            display_family = "Segoe UI"

        self.font_title = font.Font(family=display_family, size=13, weight="bold")
        self.font_heading = font.Font(family=display_family, size=11, weight="bold")
        self.font_label = font.Font(family="Segoe UI", size=10)
        self.font_text = font.Font(family="Consolas", size=13)
        self.font_btn = font.Font(family="Segoe UI", size=9, weight="bold")
        self.font_small = font.Font(family="Segoe UI", size=8)
        self.font_sub = font.Font(family="Segoe UI", size=7, weight="bold")
        self.font_sign = font.Font(family=display_family, size=42, weight="bold")

        # Theme Accent Constants
        color_brand = "#ff6b00"
        color_brand_subtle = "#fff0e6"

        # --------------------------------------------------
        # MENUBAR (Accessibility cascade for keyboard & screen readers)
        # --------------------------------------------------
        menubar = tk.Menu(self.root)
        a11y_menu = tk.Menu(menubar, tearoff=0)
        a11y_menu.add_checkbutton(label="Dark Mode (Ctrl+D)", variable=self.a11y_dark_mode, command=self._on_a11y_change)
        a11y_menu.add_checkbutton(label="Invert Colors (Ctrl+I)", variable=self.a11y_invert_colors, command=self._on_a11y_change)
        a11y_menu.add_checkbutton(label="High Contrast Mode (Ctrl+H)", variable=self.a11y_high_contrast, command=self._on_a11y_change)
        a11y_menu.add_checkbutton(label="Large Cursor (Ctrl+L)", variable=self.a11y_large_cursor, command=self._on_cursor_change)
        a11y_menu.add_separator()
        a11y_menu.add_radiobutton(label="Text Size: Small", variable=self.a11y_text_size, value="Small", command=self._on_text_size_change)
        a11y_menu.add_radiobutton(label="Text Size: Default", variable=self.a11y_text_size, value="Default", command=self._on_text_size_change)
        a11y_menu.add_radiobutton(label="Text Size: Large", variable=self.a11y_text_size, value="Large", command=self._on_text_size_change)
        menubar.add_cascade(label="Accessibility", menu=a11y_menu)
        self.root.config(menu=menubar)

        # --------------------------------------------------
        # TOP HEADER (SE_PROJECT Navbar style)
        # --------------------------------------------------
        self.header_frame = tk.Frame(self.root, bg="#ffffff", padx=16, pady=9, bd=1, relief="solid")
        self.header_frame.pack(fill="x")

        # Brand Logo Box
        self.logo_box = tk.Label(
            self.header_frame, text="ISL", font=self.font_btn,
            bg=color_brand, fg="#ffffff", padx=6, pady=2
        )
        self.logo_box.pack(side="left", padx=(0, 8))

        self.title_lbl = tk.Label(
            self.header_frame,
            text="INDIAN SIGN LANGUAGE RECOGNITION",
            font=self.font_title,
            fg="#111827",
            bg="#ffffff"
        )
        self.title_lbl.pack(side="left")

        # Engine Badge
        self.badge_lbl = tk.Label(
            self.header_frame,
            text="Engine v1.0",
            font=self.font_small,
            fg=color_brand,
            bg=color_brand_subtle,
            padx=6, pady=2
        )
        self.badge_lbl.pack(side="left", padx=(8, 0))

        # Camera selection dropdown in header
        self.cam_sel_frame = tk.Frame(self.header_frame, bg="#ffffff")
        self.cam_sel_frame.pack(side="right")

        self.cam_sel_lbl = tk.Label(
            self.cam_sel_frame, text="Camera Source:", font=self.font_small, fg="#4b5563", bg="#ffffff"
        )
        self.cam_sel_lbl.pack(side="left", padx=(0, 4))

        self.cam_combo_var = tk.StringVar()
        self.cam_combobox = ttk.Combobox(
            self.cam_sel_frame,
            textvariable=self.cam_combo_var,
            state="readonly",
            width=26,
            font=("Segoe UI", 8)
        )
        self.cam_combobox.pack(side="left", padx=(0, 6))
        self.cam_combobox.bind("<<ComboboxSelected>>", self._on_camera_selected)

        self.btn_rescan_cam = tk.Button(
            self.cam_sel_frame,
            text="🔄 Rescan",
            font=self.font_small,
            bg="#ffffff",
            fg="#111827",
            activebackground="#f3f4f6",
            bd=1,
            relief="solid",
            padx=8, pady=3,
            cursor="hand2",
            command=self.rescan_cameras
        )
        self.btn_rescan_cam.pack(side="left")

        # --------------------------------------------------
        # ACCESSIBILITY BAR (Compact HCI Controls Toolbar)
        # --------------------------------------------------
        self.a11y_bar = tk.Frame(self.root, bg="#f9fafb", padx=14, pady=4, bd=1, relief="solid")
        self.a11y_bar.pack(fill="x")

        self.a11y_lbl = tk.Label(
            self.a11y_bar, text="♿ Accessibility:",
            font=self.font_btn, fg="#111827", bg="#f9fafb"
        )
        self.a11y_lbl.pack(side="left", padx=(0, 8))

        self.cb_dark = tk.Checkbutton(
            self.a11y_bar, text="🌙 Dark Mode",
            variable=self.a11y_dark_mode,
            font=self.font_small, fg="#111827", bg="#f9fafb",
            activebackground="#f9fafb", activeforeground=color_brand,
            cursor="hand2", command=self._on_a11y_change
        )
        self.cb_dark.pack(side="left", padx=4)

        self.cb_invert = tk.Checkbutton(
            self.a11y_bar, text="🔄 Invert Colors",
            variable=self.a11y_invert_colors,
            font=self.font_small, fg="#111827", bg="#f9fafb",
            activebackground="#f9fafb", activeforeground=color_brand,
            cursor="hand2", command=self._on_a11y_change
        )
        self.cb_invert.pack(side="left", padx=4)

        self.cb_contrast = tk.Checkbutton(
            self.a11y_bar, text="🔲 High Contrast",
            variable=self.a11y_high_contrast,
            font=self.font_small, fg="#111827", bg="#f9fafb",
            activebackground="#f9fafb", activeforeground=color_brand,
            cursor="hand2", command=self._on_a11y_change
        )
        self.cb_contrast.pack(side="left", padx=4)

        self.cb_cursor = tk.Checkbutton(
            self.a11y_bar, text="🔍 Large Cursor",
            variable=self.a11y_large_cursor,
            font=self.font_small, fg="#111827", bg="#f9fafb",
            activebackground="#f9fafb", activeforeground=color_brand,
            cursor="hand2", command=self._on_cursor_change
        )
        self.cb_cursor.pack(side="left", padx=4)

        self.a11y_sep_lbl = tk.Label(
            self.a11y_bar, text="|  Text Size:",
            font=self.font_small, fg="#4b5563", bg="#f9fafb"
        )
        self.a11y_sep_lbl.pack(side="left", padx=(8, 4))

        self.rb_small = tk.Radiobutton(
            self.a11y_bar, text="Small", value="Small",
            variable=self.a11y_text_size,
            font=self.font_small, fg="#111827", bg="#f9fafb",
            activebackground="#f9fafb", activeforeground=color_brand,
            cursor="hand2", command=self._on_text_size_change
        )
        self.rb_small.pack(side="left", padx=2)

        self.rb_default = tk.Radiobutton(
            self.a11y_bar, text="Default", value="Default",
            variable=self.a11y_text_size,
            font=self.font_small, fg="#111827", bg="#f9fafb",
            activebackground="#f9fafb", activeforeground=color_brand,
            cursor="hand2", command=self._on_text_size_change
        )
        self.rb_default.pack(side="left", padx=2)

        self.rb_large = tk.Radiobutton(
            self.a11y_bar, text="Large", value="Large",
            variable=self.a11y_text_size,
            font=self.font_small, fg="#111827", bg="#f9fafb",
            activebackground="#f9fafb", activeforeground=color_brand,
            cursor="hand2", command=self._on_text_size_change
        )
        self.rb_large.pack(side="left", padx=2)

        # --------------------------------------------------
        # MAIN BODY (2 COLUMNS)
        # --------------------------------------------------
        self.body = tk.Frame(self.root, bg="#fafafa", padx=14, pady=10)
        self.body.pack(fill="both", expand=True)

        # ==================================================
        # LEFT COLUMN: CAMERA & VIDEO OPTIONS
        # ==================================================
        self.left_col = tk.Frame(self.body, bg="#fafafa")
        self.left_col.pack(side="left", padx=(0, 12))

        # Camera viewport container
        self.cam_card = tk.LabelFrame(
            self.left_col, text="  Live Video Feed  ",
            font=self.font_heading, fg="#111827", bg="#ffffff",
            padx=8, pady=8, bd=1, relief="solid"
        )
        self.cam_card.pack(fill="x")

        # Fixed 640x480 video canvas
        self.cam_box = tk.Frame(self.cam_card, width=640, height=480, bg="#000000")
        self.cam_box.pack_propagate(False)
        self.cam_box.pack()

        self.camera_label = tk.Label(
            self.cam_box,
            text="Initializing webcam feed...",
            font=self.font_label,
            bg="#000000",
            fg="#94a3b8"
        )
        self.camera_label.pack(fill="both", expand=True)

        # Video Options Toolbar underneath camera
        self.opts_frame = tk.Frame(self.cam_card, bg="#f8f9fa", pady=6, padx=8)
        self.opts_frame.pack(fill="x", pady=(6, 0))

        self.var_mirror = tk.BooleanVar(value=True)
        self.cb_mirror = tk.Checkbutton(
            self.opts_frame, text="Mirror Video",
            variable=self.var_mirror,
            font=self.font_small, fg="#111827", bg="#f8f9fa",
            selectcolor="#ffffff", activebackground="#f8f9fa", activeforeground=color_brand
        )
        self.cb_mirror.pack(side="left", padx=(2, 10))

        self.var_landmarks = tk.BooleanVar(value=True)
        self.cb_landmarks = tk.Checkbutton(
            self.opts_frame, text="Show Skeleton",
            variable=self.var_landmarks,
            font=self.font_small, fg="#111827", bg="#f8f9fa",
            selectcolor="#ffffff", activebackground="#f8f9fa", activeforeground=color_brand
        )
        self.cb_landmarks.pack(side="left", padx=(0, 10))

        self.var_auto_add = tk.BooleanVar(value=True)
        self.cb_auto = tk.Checkbutton(
            self.opts_frame, text="Auto-Add Held Sign",
            variable=self.var_auto_add,
            font=self.font_small, fg="#111827", bg="#f8f9fa",
            selectcolor="#ffffff", activebackground="#f8f9fa", activeforeground=color_brand
        )
        self.cb_auto.pack(side="left", padx=(0, 10))

        self.var_paused = tk.BooleanVar(value=False)
        self.btn_pause = tk.Button(
            self.opts_frame,
            text="⏸ Pause",
            font=self.font_small,
            bg="#ffffff", fg="#111827",
            activebackground="#f3f4f6", activeforeground="#111827",
            bd=1, relief="solid", padx=10, pady=2, cursor="hand2",
            command=self.toggle_pause
        )
        self.btn_pause.pack(side="right", padx=(0, 2))

        # Sliders card for fine-tuning
        self.sliders_card = tk.Frame(self.cam_card, bg="#ffffff", pady=6)
        self.sliders_card.pack(fill="x")

        # Confidence slider
        self.s_left = tk.Frame(self.sliders_card, bg="#ffffff")
        self.s_left.pack(side="left", fill="x", expand=True, padx=(5, 10))

        self.conf_val_lbl = tk.Label(
            self.s_left, text="Min Confidence: 70%", font=self.font_small, fg="#4b5563", bg="#ffffff"
        )
        self.conf_val_lbl.pack(anchor="w")

        self.conf_slider = tk.Scale(
            self.s_left, from_=50, to=95, orient="horizontal",
            bg="#ffffff", fg="#111827", highlightthickness=0,
            troughcolor="#ffedd5", activebackground=color_brand,
            showvalue=False, command=self._on_conf_change
        )
        self.conf_slider.set(70)
        self.conf_slider.pack(fill="x")

        # Hold frames slider
        self.s_right = tk.Frame(self.sliders_card, bg="#ffffff")
        self.s_right.pack(side="right", fill="x", expand=True, padx=(0, 5))

        self.hold_val_lbl = tk.Label(
            self.s_right, text="Hold Speed: 15 frames (~0.5s)", font=self.font_small, fg="#4b5563", bg="#ffffff"
        )
        self.hold_val_lbl.pack(anchor="w")

        self.hold_slider = tk.Scale(
            self.s_right, from_=8, to=30, orient="horizontal",
            bg="#ffffff", fg="#111827", highlightthickness=0,
            troughcolor="#ffedd5", activebackground=color_brand,
            showvalue=False, command=self._on_hold_change
        )
        self.hold_slider.set(15)
        self.hold_slider.pack(fill="x")

        # Camera status & resolution footer
        self.cam_info_lbl = tk.Label(
            self.cam_card, text="Camera: Connecting...", font=self.font_small, fg="#6b7280", bg="#ffffff"
        )
        self.cam_info_lbl.pack(anchor="w", padx=5, pady=(4, 0))

        # ==================================================
        # RIGHT COLUMN: RECOGNITION & TEXT EDITING CONTROLS
        # ==================================================
        self.right_col = tk.Frame(self.body, bg="#fafafa", width=420)
        self.right_col.pack(side="right", fill="both", expand=True)
        self.right_col.pack_propagate(False)

        # --------------------------------------------------
        # CARD 1: CURRENT DETECTED SIGN
        # --------------------------------------------------
        self.sign_card = tk.LabelFrame(
            self.right_col, text="  Current Sign Detection  ",
            font=self.font_heading, fg="#111827", bg="#ffffff",
            padx=12, pady=8, bd=1, relief="solid"
        )
        self.sign_card.pack(fill="x", pady=(0, 10))

        self.top_sign_row = tk.Frame(self.sign_card, bg="#ffffff")
        self.top_sign_row.pack(fill="x")

        # Big sign label
        self.sign_label = tk.Label(
            self.top_sign_row,
            text="...",
            font=self.font_sign,
            bg="#ffffff",
            fg=color_brand,
            width=3
        )
        self.sign_label.pack(side="left", padx=(5, 12))

        self.sign_meta_col = tk.Frame(self.top_sign_row, bg="#ffffff")
        self.sign_meta_col.pack(side="left", fill="both", expand=True)

        self.confidence_label = tk.Label(
            self.sign_meta_col,
            text="Confidence: 0.0%",
            font=self.font_label,
            bg="#ffffff",
            fg="#111827"
        )
        self.confidence_label.pack(anchor="w", pady=(4, 2))

        self.status_label = tk.Label(
            self.sign_meta_col,
            text="Status: Waiting for hand gestures...",
            font=self.font_small,
            bg="#ffffff",
            fg="#6b7280"
        )
        self.status_label.pack(anchor="w", pady=(0, 2))

        # Stability hold progress bar
        self.p_frame = tk.Frame(self.sign_card, bg="#ffffff", pady=4)
        self.p_frame.pack(fill="x")

        self.progress_lbl = tk.Label(
            self.p_frame, text="Hold Stability Progress:", font=self.font_sub, fg="#4b5563", bg="#ffffff"
        )
        self.progress_lbl.pack(side="left", padx=(0, 6))

        self.progress_canvas = tk.Canvas(
            self.p_frame, height=8, bg="#e5e7eb", highlightthickness=0
        )
        self.progress_canvas.pack(side="left", fill="x", expand=True)

        # --------------------------------------------------
        # CARD 2: RECOGNIZED TEXT & EDITING CONTROLS
        # --------------------------------------------------
        self.text_card = tk.LabelFrame(
            self.right_col, text="  Recognized Sentence Buffer  ",
            font=self.font_heading, fg="#111827", bg="#ffffff",
            padx=12, pady=10, bd=1, relief="solid"
        )
        self.text_card.pack(fill="both", expand=True)

        # Sub-header with text length counter
        self.text_hdr = tk.Frame(self.text_card, bg="#ffffff")
        self.text_hdr.pack(fill="x", pady=(0, 6))

        self.text_stats_lbl = tk.Label(
            self.text_hdr,
            text="0 characters | 0 words",
            font=self.font_small,
            fg="#4b5563",
            bg="#ffffff"
        )
        self.text_stats_lbl.pack(side="left")

        self.notif_lbl = tk.Label(
            self.text_hdr,
            text="",
            font=self.font_small,
            fg=color_brand,
            bg="#ffffff"
        )
        self.notif_lbl.pack(side="right")

        # Interactive editable text box
        self.text_display = tk.Text(
            self.text_card,
            font=self.font_text,
            bg="#fafafa",
            fg="#111827",
            insertbackground="#000000",
            height=3,
            wrap="word",
            relief="solid",
            bd=1,
            padx=8,
            pady=8
        )
        self.text_display.pack(fill="x", pady=(0, 8))
        self.text_display.bind("<KeyRelease>", self._on_text_widget_change)

        # Helper for styled buttons
        def make_btn(parent, text, cmd, bg_col, hover_col, fg_col="white", is_bordered=False):
            b = tk.Button(
                parent, text=text, command=cmd,
                font=self.font_btn, bg=bg_col, fg=fg_col,
                activebackground=hover_col, activeforeground=fg_col,
                relief="solid" if is_bordered else "flat",
                bd=1 if is_bordered else 0,
                cursor="hand2", pady=5
            )
            return b

        # --------------------------------------------------
        # TEXT ACTION BUTTONS
        # --------------------------------------------------
        self.r1 = tk.Frame(self.text_card, bg="#ffffff")
        self.r1.pack(fill="x", pady=2)

        self.btn_add = make_btn(self.r1, "✓ ADD SIGN", self.add_letter, "#000000", "#1f2937", "#ffffff")
        self.btn_add.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_space = make_btn(self.r1, "⎵ SPACE", self.add_space, "#ffffff", "#f9fafb", "#374151", is_bordered=True)
        self.btn_space.pack(side="right", fill="x", expand=True, padx=(2, 0))

        self.r2 = tk.Frame(self.text_card, bg="#ffffff")
        self.r2.pack(fill="x", pady=2)

        self.btn_backspace = make_btn(self.r2, "⌫ BACKSPACE", self.backspace, "#ffffff", "#f9fafb", "#374151", is_bordered=True)
        self.btn_backspace.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_del_word = make_btn(self.r2, "🗑️ DELETE WORD", self.delete_word, "#fef2f2", "#fee2e2", "#dc2626", is_bordered=True)
        self.btn_del_word.pack(side="right", fill="x", expand=True, padx=(2, 0))

        self.r3 = tk.Frame(self.text_card, bg="#ffffff")
        self.r3.pack(fill="x", pady=2)

        self.btn_clear = make_btn(self.r3, "✕ CLEAR ALL", self.clear_text, "#ef4444", "#dc2626", "#ffffff")
        self.btn_clear.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_undo = make_btn(self.r3, "↩ UNDO", self.undo_text, "#ffffff", "#f9fafb", "#374151", is_bordered=True)
        self.btn_undo.pack(side="right", fill="x", expand=True, padx=(2, 0))

        self.r4 = tk.Frame(self.text_card, bg="#ffffff")
        self.r4.pack(fill="x", pady=2)

        self.btn_copy = make_btn(self.r4, "📋 COPY TEXT", self.copy_to_clipboard, "#ffffff", "#f9fafb", "#374151", is_bordered=True)
        self.btn_copy.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_save = make_btn(self.r4, "💾 SAVE TO FILE", self.save_to_file, "#ffffff", "#f9fafb", "#374151", is_bordered=True)
        self.btn_save.pack(side="right", fill="x", expand=True, padx=(2, 0))

        self.btn_speak = make_btn(self.text_card, "🔊 SPEAK SENTENCE (TTS)", self.speak_text, color_brand, "#e05e00", "#ffffff")
        self.btn_speak.pack(fill="x", pady=3)

        self.r6 = tk.Frame(self.text_card, bg="#ffffff")
        self.r6.pack(fill="x", pady=2)

        self.btn_guide = make_btn(self.r6, "📖 SIGN GUIDE", self.show_sign_guide, "#ffffff", "#f3f4f6", "#111827", is_bordered=True)
        self.btn_guide.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_exit = make_btn(self.r6, "✕ EXIT", self.quit_app, "#f3f4f6", "#e5e7eb", "#4b5563", is_bordered=True)
        self.btn_exit.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # --------------------------------------------------
        # FOOTER LEGEND
        # --------------------------------------------------
        self.footer = tk.Frame(self.right_col, bg="#fafafa", pady=4)
        self.footer.pack(fill="x")

        shortcuts_txt = "Shortcuts: Space=Add | -=Space | Backspace=Del Char | Ctrl+Del=Clear | S=Speak | P=Pause | Esc=Exit\nAccessibility: Ctrl+D=Dark | Ctrl+I=Invert | Ctrl+H=Contrast | Ctrl+L=Cursor"
        self.shortcuts_lbl = tk.Label(
            self.footer, text=shortcuts_txt, font=self.font_sub, fg="#6b7280", bg="#fafafa", wraplength=410, justify="left"
        )
        self.shortcuts_lbl.pack(anchor="w")

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

        # Accessibility Keyboard Shortcuts
        self.root.bind("<Control-d>", lambda e: self._toggle_a11y_dark())
        self.root.bind("<Control-D>", lambda e: self._toggle_a11y_dark())
        self.root.bind("<Control-i>", lambda e: self._toggle_a11y_invert())
        self.root.bind("<Control-I>", lambda e: self._toggle_a11y_invert())
        self.root.bind("<Control-h>", lambda e: self._toggle_a11y_contrast())
        self.root.bind("<Control-H>", lambda e: self._toggle_a11y_contrast())
        self.root.bind("<Control-l>", lambda e: self._toggle_a11y_cursor())
        self.root.bind("<Control-L>", lambda e: self._toggle_a11y_cursor())

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        # Apply initial accessibility states
        self.apply_text_size()
        self.apply_theme()
        self._apply_cursor()

    # ==================================================
    # ACCESSIBILITY METHODS (DARK MODE, INVERT, CONTRAST, CURSOR, TEXT SIZE)
    # ==================================================

    def _toggle_a11y_dark(self):
        self.a11y_dark_mode.set(not self.a11y_dark_mode.get())
        self._on_a11y_change()

    def _toggle_a11y_invert(self):
        self.a11y_invert_colors.set(not self.a11y_invert_colors.get())
        self._on_a11y_change()

    def _toggle_a11y_contrast(self):
        self.a11y_high_contrast.set(not self.a11y_high_contrast.get())
        self._on_a11y_change()

    def _toggle_a11y_cursor(self):
        self.a11y_large_cursor.set(not self.a11y_large_cursor.get())
        self._on_cursor_change()

    def _on_a11y_change(self):
        """Handler when theme toggles change."""
        self.apply_theme()
        self._save_a11y_settings()
        self._set_notification("Accessibility theme updated")

    def _on_cursor_change(self):
        """Handler when large cursor toggle changes."""
        self._apply_cursor()
        self._save_a11y_settings()
        msg = "Large cursor enabled" if self.a11y_large_cursor.get() else "Default cursor restored"
        self._set_notification(msg)

    def _on_text_size_change(self):
        """Handler when text size changes."""
        self.apply_text_size()
        self._save_a11y_settings()
        self._set_notification(f"Text size: {self.a11y_text_size.get()}")

    def _load_a11y_settings(self):
        """Load accessibility preferences from local JSON file."""
        try:
            if os.path.exists(A11Y_CONFIG_PATH):
                with open(A11Y_CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.a11y_dark_mode.set(bool(data.get("dark_mode", False)))
                    self.a11y_invert_colors.set(bool(data.get("invert_colors", False)))
                    self.a11y_large_cursor.set(bool(data.get("large_cursor", False)))
                    self.a11y_high_contrast.set(bool(data.get("high_contrast", False)))
                    self.a11y_text_size.set(str(data.get("text_size", "Default")))
        except Exception as e:
            print(f"Notice: Could not load accessibility settings: {e}")

    def _save_a11y_settings(self):
        """Save accessibility preferences to local JSON file."""
        try:
            data = {
                "dark_mode": self.a11y_dark_mode.get(),
                "invert_colors": self.a11y_invert_colors.get(),
                "large_cursor": self.a11y_large_cursor.get(),
                "high_contrast": self.a11y_high_contrast.get(),
                "text_size": self.a11y_text_size.get()
            }
            with open(A11Y_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Notice: Could not save accessibility settings: {e}")

    def _get_theme_palette(self):
        """Calculate color tokens and border specifications based on active accessibility states."""
        dark = self.a11y_dark_mode.get()
        invert = self.a11y_invert_colors.get()
        high_contrast = self.a11y_high_contrast.get()

        # Invert colors flips the effective lightness/darkness of the UI
        effective_dark = (dark != invert)

        if high_contrast:
            if effective_dark:
                # High Contrast Dark
                return {
                    "bg_canvas": "#000000",
                    "card_bg": "#000000",
                    "border": "#ffffff",
                    "border_width": 2,
                    "text_primary": "#ffffff",
                    "text_secondary": "#ffff00",
                    "text_muted": "#00ffff",
                    "accent": "#ffff00",
                    "accent_text": "#000000",
                    "input_bg": "#000000",
                    "input_fg": "#ffff00",
                    "btn_bg": "#000000",
                    "btn_fg": "#ffffff",
                    "btn_border": "#ffffff",
                    "btn_active_bg": "#333333",
                    "btn_primary_bg": "#ffff00",
                    "btn_primary_fg": "#000000",
                    "btn_danger_bg": "#ff0000",
                    "btn_danger_fg": "#ffffff",
                    "slider_trough": "#333333",
                    "slider_active": "#ffff00",
                }
            else:
                # High Contrast Light
                return {
                    "bg_canvas": "#ffffff",
                    "card_bg": "#ffffff",
                    "border": "#000000",
                    "border_width": 2,
                    "text_primary": "#000000",
                    "text_secondary": "#000000",
                    "text_muted": "#1a1a1a",
                    "accent": "#000000",
                    "accent_text": "#ffffff",
                    "input_bg": "#ffffff",
                    "input_fg": "#000000",
                    "btn_bg": "#ffffff",
                    "btn_fg": "#000000",
                    "btn_border": "#000000",
                    "btn_active_bg": "#cccccc",
                    "btn_primary_bg": "#000000",
                    "btn_primary_fg": "#ffffff",
                    "btn_danger_bg": "#000000",
                    "btn_danger_fg": "#ffffff",
                    "slider_trough": "#cccccc",
                    "slider_active": "#000000",
                }
        elif effective_dark:
            # Standard Dark Mode
            return {
                "bg_canvas": "#18181b",
                "card_bg": "#27272a",
                "border": "#3f3f46",
                "border_width": 1,
                "text_primary": "#f4f4f5",
                "text_secondary": "#d4d4d8",
                "text_muted": "#a1a1aa",
                "accent": "#fb923c",
                "accent_text": "#18181b",
                "input_bg": "#1e1e24",
                "input_fg": "#f4f4f5",
                "btn_bg": "#3f3f46",
                "btn_fg": "#f4f4f5",
                "btn_border": "#52525b",
                "btn_active_bg": "#52525b",
                "btn_primary_bg": "#ea580c",
                "btn_primary_fg": "#ffffff",
                "btn_danger_bg": "#dc2626",
                "btn_danger_fg": "#ffffff",
                "slider_trough": "#3f3f46",
                "slider_active": "#fb923c",
            }
        else:
            # Standard Light Mode
            return {
                "bg_canvas": "#fafafa",
                "card_bg": "#ffffff",
                "border": "#e2e8f0",
                "border_width": 1,
                "text_primary": "#111827",
                "text_secondary": "#4b5563",
                "text_muted": "#6b7280",
                "accent": "#ff6b00",
                "accent_text": "#ffffff",
                "input_bg": "#fafafa",
                "input_fg": "#111827",
                "btn_bg": "#ffffff",
                "btn_fg": "#374151",
                "btn_border": "#d1d5db",
                "btn_active_bg": "#f9fafb",
                "btn_primary_bg": "#000000",
                "btn_primary_fg": "#ffffff",
                "btn_danger_bg": "#ef4444",
                "btn_danger_fg": "#ffffff",
                "slider_trough": "#ffedd5",
                "slider_active": "#ff6b00",
            }

    def apply_text_size(self):
        """Update font sizes dynamically across the GUI for Small, Default, and Large modes."""
        size_mode = self.a11y_text_size.get()
        if size_mode == "Small":
            scales = {
                "title": 11, "heading": 10, "label": 9,
                "text": 11, "btn": 8, "small": 7, "sub": 6, "sign": 34
            }
        elif size_mode == "Large":
            scales = {
                "title": 15, "heading": 13, "label": 12,
                "text": 15, "btn": 11, "small": 10, "sub": 9, "sign": 48
            }
        else:  # Default
            scales = {
                "title": 13, "heading": 11, "label": 10,
                "text": 13, "btn": 9, "small": 8, "sub": 7, "sign": 42
            }

        self.font_title.configure(size=scales["title"])
        self.font_heading.configure(size=scales["heading"])
        self.font_label.configure(size=scales["label"])
        self.font_text.configure(size=scales["text"])
        self.font_btn.configure(size=scales["btn"])
        self.font_small.configure(size=scales["small"])
        self.font_sub.configure(size=scales["sub"])
        self.font_sign.configure(size=scales["sign"])

    def _apply_cursor(self):
        """Update mouse cursor size and visibility across application."""
        use_large = self.a11y_large_cursor.get()
        c_main = "crosshair" if use_large else ""
        c_btn = "crosshair" if use_large else "hand2"
        c_txt = "crosshair" if use_large else "xterm"

        self.root.configure(cursor=c_main)
        self._set_widget_cursors(self.root, c_main, c_btn, c_txt)

    def _set_widget_cursors(self, widget, c_main, c_btn, c_txt):
        """Recursively apply appropriate cursor to all child widgets."""
        for child in widget.winfo_children():
            try:
                if isinstance(child, (tk.Button, tk.Checkbutton, tk.Radiobutton)):
                    child.configure(cursor=c_btn)
                elif isinstance(child, (tk.Text, tk.Entry)):
                    child.configure(cursor=c_txt)
                else:
                    child.configure(cursor=c_main)
            except Exception:
                pass
            self._set_widget_cursors(child, c_main, c_btn, c_txt)

    def apply_theme(self):
        """Apply active theme (Dark Mode, Invert Colors, High Contrast) dynamically across GUI."""
        p = self._get_theme_palette()
        bw = p["border_width"]
        bcol = p["border"]

        # Window root
        self.root.configure(bg=p["bg_canvas"])

        # Header
        self.header_frame.configure(bg=p["card_bg"], highlightbackground=bcol, highlightthickness=bw)
        self.title_lbl.configure(bg=p["card_bg"], fg=p["text_primary"])
        self.cam_sel_frame.configure(bg=p["card_bg"])
        self.cam_sel_lbl.configure(bg=p["card_bg"], fg=p["text_secondary"])
        self.btn_rescan_cam.configure(
            bg=p["btn_bg"], fg=p["text_primary"],
            activebackground=p["btn_active_bg"], activeforeground=p["text_primary"],
            highlightbackground=bcol, highlightthickness=bw
        )

        # Accessibility Bar
        self.a11y_bar.configure(bg=p["card_bg"], highlightbackground=bcol, highlightthickness=bw)
        self.a11y_lbl.configure(bg=p["card_bg"], fg=p["text_primary"])
        for cb in [self.cb_dark, self.cb_invert, self.cb_contrast, self.cb_cursor]:
            cb.configure(
                bg=p["card_bg"], fg=p["text_primary"],
                selectcolor=p["input_bg"],
                activebackground=p["card_bg"], activeforeground=p["accent"]
            )
        self.a11y_sep_lbl.configure(bg=p["card_bg"], fg=p["text_muted"])
        for rb in [self.rb_small, self.rb_default, self.rb_large]:
            rb.configure(
                bg=p["card_bg"], fg=p["text_primary"],
                selectcolor=p["input_bg"],
                activebackground=p["card_bg"], activeforeground=p["accent"]
            )

        # Body & Columns
        self.body.configure(bg=p["bg_canvas"])
        self.left_col.configure(bg=p["bg_canvas"])
        self.right_col.configure(bg=p["bg_canvas"])

        # Cam Card
        self.cam_card.configure(
            bg=p["card_bg"], fg=p["text_primary"],
            highlightbackground=bcol, highlightthickness=bw
        )
        self.opts_frame.configure(bg=p["card_bg"])
        for cb in [self.cb_mirror, self.cb_landmarks, self.cb_auto]:
            cb.configure(
                bg=p["card_bg"], fg=p["text_primary"],
                selectcolor=p["input_bg"],
                activebackground=p["card_bg"], activeforeground=p["accent"]
            )
        self.btn_pause.configure(
            bg=p["btn_bg"], fg=p["text_primary"],
            activebackground=p["btn_active_bg"], activeforeground=p["text_primary"],
            highlightbackground=bcol, highlightthickness=bw
        )

        # Sliders
        self.sliders_card.configure(bg=p["card_bg"])
        self.s_left.configure(bg=p["card_bg"])
        self.s_right.configure(bg=p["card_bg"])
        self.conf_val_lbl.configure(bg=p["card_bg"], fg=p["text_secondary"])
        self.conf_slider.configure(
            bg=p["card_bg"], fg=p["text_primary"],
            troughcolor=p["slider_trough"], activebackground=p["slider_active"]
        )
        self.hold_val_lbl.configure(bg=p["card_bg"], fg=p["text_secondary"])
        self.hold_slider.configure(
            bg=p["card_bg"], fg=p["text_primary"],
            troughcolor=p["slider_trough"], activebackground=p["slider_active"]
        )
        self.cam_info_lbl.configure(bg=p["card_bg"], fg=p["text_muted"])

        # Sign Detection Card
        self.sign_card.configure(
            bg=p["card_bg"], fg=p["text_primary"],
            highlightbackground=bcol, highlightthickness=bw
        )
        self.top_sign_row.configure(bg=p["card_bg"])
        self.sign_label.configure(bg=p["card_bg"], fg=p["accent"])
        self.sign_meta_col.configure(bg=p["card_bg"])
        self.confidence_label.configure(bg=p["card_bg"], fg=p["text_primary"])
        self.status_label.configure(bg=p["card_bg"], fg=p["text_muted"])
        self.p_frame.configure(bg=p["card_bg"])
        self.progress_lbl.configure(bg=p["card_bg"], fg=p["text_secondary"])
        self.progress_canvas.configure(bg=p["slider_trough"])

        # Text Card
        self.text_card.configure(
            bg=p["card_bg"], fg=p["text_primary"],
            highlightbackground=bcol, highlightthickness=bw
        )
        self.text_hdr.configure(bg=p["card_bg"])
        self.text_stats_lbl.configure(bg=p["card_bg"], fg=p["text_secondary"])
        self.notif_lbl.configure(bg=p["card_bg"], fg=p["accent"])
        self.text_display.configure(
            bg=p["input_bg"], fg=p["input_fg"],
            insertbackground=p["text_primary"],
            highlightbackground=bcol, highlightthickness=bw
        )

        # Button Rows
        for r in [self.r1, self.r2, self.r3, self.r4, self.r6]:
            r.configure(bg=p["card_bg"])

        # Action Buttons
        if self.a11y_high_contrast.get():
            for b in [self.btn_space, self.btn_backspace, self.btn_undo, self.btn_copy, self.btn_save, self.btn_guide, self.btn_exit]:
                b.configure(bg=p["btn_bg"], fg=p["btn_fg"], activebackground=p["btn_active_bg"], activeforeground=p["btn_fg"], bd=bw, relief="solid")
            self.btn_add.configure(bg=p["btn_primary_bg"], fg=p["btn_primary_fg"], activebackground=p["btn_active_bg"], activeforeground=p["btn_primary_fg"], bd=bw, relief="solid")
            self.btn_del_word.configure(bg=p["btn_bg"], fg="#ff4444" if p["bg_canvas"] == "#000000" else "#cc0000", activebackground=p["btn_active_bg"], bd=bw, relief="solid")
            self.btn_clear.configure(bg=p["btn_danger_bg"], fg=p["btn_danger_fg"], activebackground=p["btn_active_bg"], activeforeground=p["btn_danger_fg"], bd=bw, relief="solid")
            self.btn_speak.configure(bg=p["btn_primary_bg"], fg=p["btn_primary_fg"], activebackground=p["btn_active_bg"], activeforeground=p["btn_primary_fg"], bd=bw, relief="solid")
        else:
            self.btn_add.configure(bg=p["btn_primary_bg"], fg=p["btn_primary_fg"], activebackground=p["btn_active_bg"], activeforeground=p["btn_primary_fg"], bd=0, relief="flat")
            self.btn_speak.configure(bg=p["accent"], fg=p["accent_text"], activebackground=p["btn_active_bg"], activeforeground=p["accent_text"], bd=0, relief="flat")
            self.btn_clear.configure(bg=p["btn_danger_bg"], fg=p["btn_danger_fg"], activebackground=p["btn_active_bg"], activeforeground=p["btn_danger_fg"], bd=0, relief="flat")
            is_dark_bg = p["bg_canvas"] == "#18181b"
            self.btn_del_word.configure(
                bg="#3d1f1f" if is_dark_bg else "#fef2f2",
                fg="#f87171" if is_dark_bg else "#dc2626",
                activebackground=p["btn_active_bg"],
                relief="solid", bd=1
            )
            for b in [self.btn_space, self.btn_backspace, self.btn_undo, self.btn_copy, self.btn_save, self.btn_guide, self.btn_exit]:
                b.configure(bg=p["btn_bg"], fg=p["btn_fg"], activebackground=p["btn_active_bg"], activeforeground=p["btn_fg"], relief="solid", bd=1)

        # Footer
        self.footer.configure(bg=p["bg_canvas"])
        self.shortcuts_lbl.configure(bg=p["bg_canvas"], fg=p["text_muted"])

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
        prev_id = getattr(self, "_notif_after_id", None)
        if prev_id is not None:
            try:
                self.root.after_cancel(prev_id)
            except Exception:
                pass
        self.notif_lbl.config(text=msg)
        self._notif_after_id = self.root.after(duration_ms, lambda: self.notif_lbl.config(text=""))

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
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
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

        if self.speech.is_speaking():
            self._set_notification("🔊 Queued for speech...")
        else:
            self._set_notification("🔊 Speaking...")
        self.speech.speak(self.text_buffer)

    def toggle_pause(self):
        """Pause or resume sign detection."""
        is_paused = not self.var_paused.get()
        self.var_paused.set(is_paused)
        p = self._get_theme_palette()
        if is_paused:
            self.btn_pause.config(text="▶ Resume Detection", bg="#15803d")
            self.status_label.config(text="Status: Detection Paused", fg="#ffd600")
            self.sign_label.config(text="PAUSED", font=self.font_heading, fg="#ffd600")
            self._set_notification("Detection paused")
        else:
            self.btn_pause.config(text="⏸ Pause Detection", bg=p["btn_bg"], fg=p["text_primary"])
            self.sign_label.config(font=self.font_sign, fg=p["accent"])
            self._set_notification("Detection resumed")

    def show_sign_guide(self):
        """Display an interactive modal showing supported alphabet gestures with accessibility support."""
        guide_win = tk.Toplevel(self.root)
        guide_win.title("Indian Sign Language - Gesture Reference Guide")
        p = self._get_theme_palette()
        guide_win.configure(bg=p["bg_canvas"])
        guide_win.geometry("560x540")
        guide_win.resizable(False, False)
        guide_win.lift()
        guide_win.grab_set()

        hdr = tk.Frame(guide_win, bg=p["card_bg"], padx=16, pady=12, bd=p["border_width"], relief="solid")
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="ISL Alphabet & Gesture Reference",
            font=self.font_heading,
            fg=p["text_primary"], bg=p["card_bg"]
        ).pack(anchor="w")
        tk.Label(
            hdr, text="Standardized Indian Sign Language alphabet conventions (A – Z)",
            font=self.font_small, fg=p["text_secondary"], bg=p["card_bg"]
        ).pack(anchor="w", pady=(2, 0))

        content = tk.Frame(guide_win, bg=p["bg_canvas"], padx=15, pady=12)
        content.pack(fill="both", expand=True)

        # Alphabet grid
        grid_frame = tk.LabelFrame(
            content, text="  Supported Letters (26 Classes)  ",
            font=self.font_heading, fg=p["text_primary"], bg=p["card_bg"],
            padx=10, pady=10, bd=p["border_width"], relief="solid"
        )
        grid_frame.pack(fill="x", pady=(0, 10))

        cols = 7
        for idx, lbl in enumerate(self.labels):
            r = idx // cols
            c = idx % cols
            btn = tk.Label(
                grid_frame, text=lbl, font=self.font_label,
                bg=p["input_bg"], fg=p["accent"], width=4, height=1, relief="solid", bd=1
            )
            btn.grid(row=r, column=c, padx=3, pady=3)

        # Guidelines card
        tips_frame = tk.LabelFrame(
            content, text="  Signing Best Practices  ",
            font=self.font_heading, fg=p["text_primary"], bg=p["card_bg"],
            padx=12, pady=8, bd=p["border_width"], relief="solid"
        )
        tips_frame.pack(fill="both", expand=True)

        tips = [
            "• ISL uses 2-handed signing for over 75% of letters (A, B, D, G, H, P, Q, R, S, T, W, X, Y).",
            "• Single-handed signs (C, I, L, O, U, V) are supported ambidextrously.",
            "• Keep both hands fully within camera view facing adequate lighting.",
            "• Hold your sign steady for ~0.5 seconds to auto-confirm into the text buffer.",
            "• If an incorrect letter is added, click 'DELETE LETTER' or 'DELETE WORD'."
        ]
        for t in tips:
            tk.Label(
                tips_frame, text=t, font=self.font_small, fg=p["text_secondary"], bg=p["card_bg"], anchor="w"
            ).pack(fill="x", pady=2)

        close_btn = tk.Button(
            content, text="Close Guide", font=self.font_btn,
            bg=p["btn_primary_bg"], fg=p["btn_primary_fg"], activebackground=p["btn_active_bg"],
            relief="solid" if p["border_width"] > 1 else "flat", bd=p["border_width"] if p["border_width"] > 1 else 0,
            padx=16, pady=5, cursor="hand2",
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
        """Render overlay graphics and update Tkinter camera canvas with accessibility support."""
        # Visual overlay tags on OpenCV frame
        fps_str = f"{self.current_fps:.1f} FPS"
        cv2.putText(
            frame, fps_str, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        p = self._get_theme_palette()

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
                if self.a11y_high_contrast.get():
                    if self.hand_detected:
                        self.sign_label.config(fg=p["accent"])
                        hands_txt = f"✋ {self.num_hands_detected} Hand{'s' if self.num_hands_detected > 1 else ''} Detected"
                        self.status_label.config(text=f"Status: {hands_txt}", fg=p["text_secondary"])
                    else:
                        self.sign_label.config(fg=p["text_muted"])
                        self.status_label.config(text="Status: Show your hand", fg=p["text_muted"])

                    conf_percent = self.confidence * 100.0
                    self.confidence_label.config(text=f"Confidence: {conf_percent:.1f}%", fg=p["text_primary"])
                else:
                    if self.hand_detected:
                        self.sign_label.config(fg=p["accent"])
                        hands_txt = f"✋ {self.num_hands_detected} Hand{'s' if self.num_hands_detected > 1 else ''} Detected"
                        self.status_label.config(text=f"Status: {hands_txt}", fg="#10b981")
                    else:
                        self.sign_label.config(fg=p["text_muted"])
                        self.status_label.config(text="Status: Show your hand", fg=p["text_secondary"])

                    conf_percent = self.confidence * 100.0
                    self.confidence_label.config(text=f"Confidence: {conf_percent:.1f}%")
                    if conf_percent >= 85.0:
                        self.confidence_label.config(fg="#10b981")
                    elif conf_percent >= 70.0:
                        self.confidence_label.config(fg=p["accent"])
                    else:
                        self.confidence_label.config(fg="#ef4444")

        # Update hold progress canvas
        self.progress_canvas.delete("all")
        canvas_width = self.progress_canvas.winfo_width()
        target_hold = int(self.hold_slider.get())

        if canvas_width > 1 and self.stable_count > 0 and not self.need_change_before_repeat:
            prog = min(self.stable_count / max(target_hold, 1), 1.0)
            fill_w = int(canvas_width * prog)
            bar_fill = p["accent"]
            self.progress_canvas.create_rectangle(
                0, 0, fill_w, 8, fill=bar_fill, outline=""
            )

        # Convert and blit frame to Tkinter (Invert only display copy if requested)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.a11y_invert_colors.get():
            rgb = cv2.bitwise_not(rgb)
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
        if hasattr(self, "speech"):
            self.speech.stop()
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
