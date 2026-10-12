"""
Camera Test Script for Indian Sign Language Recognition
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

def open_camera(idx):
    # Try default backend first (RGB webcam is index 0 default)
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"Opened camera {idx} [Default backend]: {frame.shape[1]}x{frame.shape[0]}")
            return cap
        cap.release()

    # Fallback to DirectShow
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"Opened camera {idx} [DSHOW backend]: {frame.shape[1]}x{frame.shape[0]}")
            return cap
        cap.release()

    return None

def main():
    camera_idx = 0
    if len(sys.argv) > 1:
        try:
            camera_idx = int(sys.argv[1])
        except ValueError:
            pass

    print(f"Testing camera index {camera_idx}...")
    cap = open_camera(camera_idx)
    if not cap:
        alt_idx = 1 if camera_idx == 0 else 0
        print(f"Camera {camera_idx} failed, trying camera {alt_idx}...")
        camera_idx = alt_idx
        cap = open_camera(camera_idx)

    if not cap:
        print("ERROR: Could not open any camera device. Check camera connections and privacy settings.")
        sys.exit(1)

    print("Camera running!")
    print("Controls:")
    print("  's' = Switch camera (0 <-> 1)")
    print("  'q' = Exit")

    window_name = f"Camera Test - Device {camera_idx}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

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

        # Add camera info overlay
        cv2.putText(
            frame, f"Camera Index: {camera_idx} | Press 's' to switch, 'q' to quit",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            new_idx = 1 if camera_idx == 0 else 0
            print(f"Switching to camera {new_idx}...")
            cap.release()
            cv2.destroyWindow(window_name)
            new_cap = open_camera(new_idx)
            if new_cap:
                camera_idx = new_idx
                cap = new_cap
                window_name = f"Camera Test - Device {camera_idx}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
            else:
                print(f"Could not open camera {new_idx}, returning to {camera_idx}")
                cap = open_camera(camera_idx)

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test closed.")

if __name__ == "__main__":
    main()