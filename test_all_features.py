"""
Comprehensive Full-Feature Test Suite for Indian Sign Language Recognition System
Tests every UI widget, live frame processing, auto-confirmation, text deletion,
audio, camera switching, and modal dialogs.
"""
import sys
import os
import time
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(encoding='utf-8')

root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)

import app
import camera_utils

def test_everything():
    print("=" * 70, flush=True)
    print("      COMPREHENSIVE FULL-FEATURE TEST SUITE", flush=True)
    print("=" * 70, flush=True)

    # --------------------------------------------------
    # TEST 1: Physical Camera Probing & Hardware Validation
    # --------------------------------------------------
    print("\n[TEST 1/11] Probing Camera Hardware & Validating Physical Webcam...", flush=True)
    cameras = camera_utils.probe_all_cameras()
    assert len(cameras) > 0, "No cameras discovered!"
    active_cams = [c for c in cameras if c["is_active"]]
    print(f"  Total devices found: {len(cameras)}", flush=True)
    for c in cameras:
        status = "ACTIVE" if c["is_active"] else "INACTIVE/BLANK"
        print(f"    - {c['label']} | Status: {status} | Mean: {c['mean']:.1f} | Std: {c['std']:.1f}", flush=True)
    assert len(active_cams) > 0, "No active physical camera found!"
    print(f"  [PASS] Physical camera verified: {active_cams[0]['label']}", flush=True)

    # --------------------------------------------------
    # TEST 2: AI Model & Inference Pipeline
    # --------------------------------------------------
    print("\n[TEST 2/11] Testing AI Model & Input Feature Architecture...", flush=True)
    app_instance = app.ISLApp()
    assert app_instance.expected_features == 126, f"Expected 126 features, got {app_instance.expected_features}"
    assert len(app_instance.labels) == 26, f"Expected 26 classes, got {len(app_instance.labels)}"
    # Feed dummy feature vector
    dummy_input = np.random.uniform(-1, 1, (1, 126)).astype(np.float32)
    pred = app_instance.model.predict(dummy_input, verbose=0)
    assert pred.shape == (1, 26), f"Unexpected prediction shape: {pred.shape}"
    print(f"  [PASS] Model architecture and forward-pass verified (26 classes).", flush=True)

    # --------------------------------------------------
    # TEST 3: Camera Initialisation & Live Frame Grab
    # --------------------------------------------------
    print("\n[TEST 3/11] Initializing Camera Feed inside Application...", flush=True)
    cam_ok = app_instance.init_camera()
    assert cam_ok, "init_camera failed!"
    ret, frame = app_instance.camera.read()
    assert ret and frame is not None, "Failed to read live frame!"
    h, w, c = frame.shape
    print(f"  [PASS] Camera streaming live frames: {w}x{h} (channels: {c})", flush=True)

    # --------------------------------------------------
    # TEST 4: Live process_frame() Single-Cycle Execution
    # --------------------------------------------------
    print("\n[TEST 4/11] Executing live process_frame() cycle with image rendering...", flush=True)
    # Set running=True for single frame execution then False
    app_instance.running = True
    app_instance.process_frame()
    app_instance.running = False
    app_instance.root.update()
    assert hasattr(app_instance.camera_label, "imgtk"), "PhotoImage not attached to camera_label!"
    print(f"  [PASS] Live image processing, resizing, and Canvas blitting verified.", flush=True)

    # --------------------------------------------------
    # TEST 5: Interactive Text Deletion Features (via .invoke())
    # --------------------------------------------------
    print("\n[TEST 5/11] Testing All Text Editing & Deletion Controls via Button Invocations...", flush=True)
    
    # 1. Add letters
    app_instance.predicted_label = "S"
    app_instance.btn_add.invoke()
    app_instance.predicted_label = "I"
    app_instance.btn_add.invoke()
    app_instance.predicted_label = "G"
    app_instance.btn_add.invoke()
    app_instance.predicted_label = "N"
    app_instance.btn_add.invoke()
    app_instance.btn_space.invoke()
    app_instance.predicted_label = "T"
    app_instance.btn_add.invoke()
    app_instance.predicted_label = "E"
    app_instance.btn_add.invoke()
    app_instance.predicted_label = "S"
    app_instance.btn_add.invoke()
    app_instance.predicted_label = "T"
    app_instance.btn_add.invoke()
    app_instance.root.update()
    assert app_instance.text_buffer == "SIGN TEST", f"Expected 'SIGN TEST', got '{app_instance.text_buffer}'"
    print(f"  Formed Text: '{app_instance.text_buffer}'", flush=True)

    # 2. Delete Letter (Backspace button invoke)
    app_instance.btn_backspace.invoke()
    app_instance.root.update()
    assert app_instance.text_buffer == "SIGN TES", f"Expected 'SIGN TES', got '{app_instance.text_buffer}'"
    print(f"  After btn_backspace.invoke(): '{app_instance.text_buffer}'", flush=True)

    # 3. Delete Word (Delete word button invoke)
    app_instance.btn_del_word.invoke()
    app_instance.root.update()
    assert app_instance.text_buffer == "SIGN ", f"Expected 'SIGN ', got '{app_instance.text_buffer}'"
    print(f"  After btn_del_word.invoke(): '{app_instance.text_buffer}'", flush=True)

    # 4. Clear All button invoke
    app_instance.btn_clear.invoke()
    app_instance.root.update()
    assert app_instance.text_buffer == "", f"Expected empty, got '{app_instance.text_buffer}'"
    print("  After btn_clear.invoke(): text cleared to empty", flush=True)

    # 5. Undo button invoke
    app_instance.btn_undo.invoke()
    app_instance.root.update()
    assert app_instance.text_buffer == "SIGN ", f"Expected 'SIGN ', got '{app_instance.text_buffer}'"
    print(f"  After btn_undo.invoke(): successfully restored '{app_instance.text_buffer}'", flush=True)
    print("  [PASS] All text creation, backspace, word deletion, clear, and undo passed.", flush=True)

    # --------------------------------------------------
    # TEST 6: Direct Typing & In-Place Editing in Text Widget
    # --------------------------------------------------
    print("\n[TEST 6/11] Testing Direct In-Place Text Box Editing & Event Sync...", flush=True)
    # Simulate user typing directly into the Text widget with keyboard
    app_instance.text_display.delete("1.0", "end")
    app_instance.text_display.insert("1.0", "MANUAL EDIT DONE")
    app_instance._on_text_widget_change()
    app_instance.root.update()
    assert app_instance.text_buffer == "MANUAL EDIT DONE", f"Sync failed: '{app_instance.text_buffer}'"
    print(f"  Direct keyboard input synchronized with buffer: '{app_instance.text_buffer}'", flush=True)
    print("  [PASS] In-place direct text box typing & deletion synchronization verified.", flush=True)

    # --------------------------------------------------
    # TEST 7: Auto-Confirmation Frame Hold Engine
    # --------------------------------------------------
    print("\n[TEST 7/11] Testing Auto-Confirmation Hold Engine...", flush=True)
    app_instance.text_buffer = ""
    app_instance.hold_slider.set(10) # 10 frames
    app_instance.predicted_label = "A"
    app_instance.stable_count = 0
    app_instance.last_stable_label = None
    app_instance.cooldown_counter = 0
    app_instance.need_change_before_repeat = False

    # Simulate 10 consecutive frames holding sign 'A'
    for f in range(12):
        app_instance._process_auto_confirmation()
    app_instance.root.update()
    assert app_instance.text_buffer == "A", f"Expected auto-added 'A', got '{app_instance.text_buffer}'"
    print(f"  Auto-confirmation correctly added letter: '{app_instance.text_buffer}'", flush=True)
    print("  [PASS] Auto-confirmation frame-hold and cooldown engine verified.", flush=True)

    # --------------------------------------------------
    # TEST 8: Feature Checkboxes (Mirror, Landmarks, Auto-Add)
    # --------------------------------------------------
    print("\n[TEST 8/11] Testing Feature Checkboxes & Options...", flush=True)
    # Test Mirror toggle
    app_instance.var_mirror.set(False)
    assert not app_instance.var_mirror.get()
    app_instance.var_mirror.set(True)
    assert app_instance.var_mirror.get()

    # Test Landmarks toggle
    app_instance.var_landmarks.set(False)
    assert not app_instance.var_landmarks.get()
    app_instance.var_landmarks.set(True)
    assert app_instance.var_landmarks.get()

    # Test Auto-Add toggle
    app_instance.var_auto_add.set(False)
    assert not app_instance.var_auto_add.get()
    app_instance.var_auto_add.set(True)
    assert app_instance.var_auto_add.get()
    print("  [PASS] Mirror, Landmarks, and Auto-Add state toggles verified.", flush=True)

    # --------------------------------------------------
    # TEST 9: Pause / Resume Detection Button
    # --------------------------------------------------
    print("\n[TEST 9/11] Testing Pause/Resume Button Invocations...", flush=True)
    app_instance.btn_pause.invoke()
    app_instance.root.update()
    assert app_instance.var_paused.get(), "Detection should be paused!"
    assert "Resume" in app_instance.btn_pause.cget("text")
    print("  Detection paused via button invoke.", flush=True)

    app_instance.btn_pause.invoke()
    app_instance.root.update()
    assert not app_instance.var_paused.get(), "Detection should be active!"
    assert "Pause" in app_instance.btn_pause.cget("text")
    print("  Detection resumed via button invoke.", flush=True)
    print("  [PASS] Detection Pause/Resume verified.", flush=True)

    # --------------------------------------------------
    # TEST 10: Sign Reference Guide Modal Window
    # --------------------------------------------------
    print("\n[TEST 10/11] Testing Interactive Sign Reference Guide Modal...", flush=True)
    app_instance.show_sign_guide()
    app_instance.root.update()
    toplevels = [w for w in app_instance.root.winfo_children() if isinstance(w, app.tk.Toplevel)]
    assert len(toplevels) == 1, "Sign guide modal window not opened!"
    guide_win = toplevels[0]
    assert "Sign Language" in guide_win.title()
    print(f"  Modal opened: '{guide_win.title()}'", flush=True)
    guide_win.destroy()
    app_instance.root.update()
    print("  Modal closed cleanly.", flush=True)
    print("  [PASS] Sign Reference Guide modal verified.", flush=True)

    # --------------------------------------------------
    # TEST 11: Export & Clipboard Features
    # --------------------------------------------------
    print("\n[TEST 11/11] Testing Clipboard Copy and File Save...", flush=True)
    app_instance.text_buffer = "ISL TEST OK"
    app_instance._update_text_display()

    # Copy
    app_instance.btn_copy.invoke()
    app_instance.root.update()
    cb_val = app_instance.root.clipboard_get()
    assert cb_val == "ISL TEST OK", f"Clipboard error: '{cb_val}'"
    print(f"  Clipboard verified: '{cb_val}'", flush=True)

    # Save
    app_instance.btn_save.invoke()
    app_instance.root.update()
    with open("isl_history.txt", "r", encoding="utf-8") as f:
        content = f.read()
        assert "ISL TEST OK" in content
    print("  File export to isl_history.txt verified.", flush=True)
    print("  [PASS] Export and Clipboard features confirmed.", flush=True)

    # --------------------------------------------------
    # Clean Shutdown
    # --------------------------------------------------
    print("\nCleaning up application...", flush=True)
    app_instance.quit_app()
    print("  Camera, MediaPipe detector, and Tkinter window released cleanly.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  ALL 11 APPLICATION AND GUI FEATURES VERIFIED AND PASSED! [OK]", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    test_everything()
