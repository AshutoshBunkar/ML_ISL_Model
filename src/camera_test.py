"""
Camera Test Script for Indian Sign Language Recognition
Tests physical webcam and any connected video capture devices.
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
import time
import numpy as np

# Import our robust camera utils
sys.path.insert(0, _script_dir)
from camera_utils import probe_all_cameras, open_best_camera

def main():
    print("=" * 60)
    print("       INDIAN SIGN LANGUAGE SYSTEM - WEBCAM TEST")
    print("=" * 60)
    print("Probing available camera hardware...")

    probed_devices = probe_all_cameras()
    print(f"\nFound {len(probed_devices)} camera devices/backends:")
    for idx, d in enumerate(probed_devices):
        status_str = "ACTIVE (Normal Image)" if d["is_active"] else "INACTIVE / BLANK (Virtual or Shutter Closed)"
        print(f"  [{idx}] {d['label']} -> {status_str} (Mean={d['mean']:.1f}, Std={d['std']:.1f})")

    # Determine desired camera
    target_idx = None
    if len(sys.argv) > 1:
        try:
            target_idx = int(sys.argv[1])
            print(f"\nRequested camera index: {target_idx}")
        except ValueError:
            pass

    cap, active_info = open_best_camera(preferred_idx=target_idx)
    if not cap:
        print("\nERROR: Could not open any camera device. Check camera connections and privacy settings.")
        sys.exit(1)

    print(f"\n>>> Successfully running camera: {active_info['label']}")
    print(f"Resolution: {active_info['width']}x{active_info['height']}")
    print("\nControls:")
    print("  's' or Space = Switch to next available camera")
    print("  'q' or Esc   = Exit")
    print("=" * 60)

    curr_idx = 0
    window_name = "Camera Test - ISL Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    fps_history = []
    prev_time = time.time()

    while True:
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            pass

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        mean_brightness = float(np.mean(frame))
        std_val = float(np.std(frame))

        # Add top banner
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (20, 20, 20), -1)
        status_txt = f"{active_info['label']} | {frame.shape[1]}x{frame.shape[0]} @ {avg_fps:.1f} FPS"
        cv2.putText(frame, status_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Bottom banner with controls
        cv2.rectangle(frame, (0, frame.shape[0] - 35), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
        cv2.putText(
            frame, "Press 's' to switch camera | 'q' to quit",
            (15, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1
        )

        # Check shutter / darkness
        if mean_brightness < 2.0 and std_val < 2.0:
            cv2.putText(
                frame, "WARNING: CAMERA LENS SHUTTER MAY BE CLOSED",
                (30, frame.shape[0] // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
            )
            cv2.putText(
                frame, "Check physical privacy slider or Fn+F10",
                (30, frame.shape[0] // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # 'q' or Esc
            break
        elif key in (ord('s'), 32):  # 's' or Space to cycle camera
            if len(probed_devices) > 1:
                curr_idx = (curr_idx + 1) % len(probed_devices)
                next_dev = probed_devices[curr_idx]
                print(f"Switching to device {curr_idx}: {next_dev['label']}...")
                cap.release()
                
                new_cap = cv2.VideoCapture(next_dev["index"], next_dev["backend"])
                if new_cap and new_cap.isOpened():
                    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap = new_cap
                    active_info = next_dev
                    print(f"Successfully switched to {next_dev['label']}")
                else:
                    print("Could not switch, reverting to auto-selected best camera.")
                    cap, active_info = open_best_camera()

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test finished.")

if __name__ == "__main__":
    main()