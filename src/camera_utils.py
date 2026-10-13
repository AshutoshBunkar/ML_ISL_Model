"""
Camera Utilities for Indian Sign Language Recognition System

Provides robust multi-backend camera discovery, active camera validation,
and seamless fallback handling across Windows DirectShow and Media Foundation (MSMF).
"""
import sys
import time
import cv2
import numpy as np

# Known backend constants
BACKENDS = [
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_MSMF, "Media Foundation"),
]

def probe_all_cameras(max_indices=3):
    """
    Probe all connected camera devices across DirectShow and MSMF backends.
    Returns a list of dicts with camera details and active validation status.
    """
    devices = []
    seen_signatures = set()

    for idx in range(max_indices):
        for backend, backend_name in BACKENDS:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                # Set standard resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                # Warm up and sample frame
                ret, frame = False, None
                for _ in range(4):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.04)

                if ret and frame is not None and frame.size > 0:
                    mean_val = float(np.mean(frame))
                    std_val = float(np.std(frame))
                    h, w = frame.shape[:2]

                    # A real active camera has variance (std > 3.0) and non-zero illumination (mean > 3.0)
                    is_active = (std_val > 3.0 and mean_val > 4.0)

                    # Distinguish physical vs dummy devices
                    if is_active:
                        label = f"Camera {idx} ({backend_name}) - Active Feed [{w}x{h}]"
                    else:
                        label = f"Camera {idx} ({backend_name}) - Blank/Cover Closed [{w}x{h}]"

                    device_info = {
                        "index": idx,
                        "backend": backend,
                        "backend_name": backend_name,
                        "width": w,
                        "height": h,
                        "mean": mean_val,
                        "std": std_val,
                        "is_active": is_active,
                        "label": label,
                    }
                    devices.append(device_info)
                cap.release()
            except Exception as e:
                pass

    return devices


def open_best_camera(preferred_idx=None, preferred_backend=None):
    """
    Open the best working camera.
    Prioritizes:
    1. Active camera matching preferred_idx (if active)
    2. Any active camera with valid image data (mean > 4, std > 3)
    3. Any opened camera if all are dark/covered
    
    Returns (cap, device_info) or (None, None).
    """
    candidates = []

    # Build priority list of (backend, idx)
    # On Windows: Index 0 DirectShow and Index 1 MSMF are the physical laptop webcam
    # We test (cv2.CAP_DSHOW, 0) and (cv2.CAP_MSMF, 1) first if no specific preference
    if preferred_idx is not None:
        if preferred_backend is not None:
            candidates.append((preferred_backend, preferred_idx))
        candidates.append((cv2.CAP_DSHOW, preferred_idx))
        candidates.append((cv2.CAP_MSMF, preferred_idx))
        candidates.append((None, preferred_idx))

    # Add standard known working configurations
    standard_configs = [
        (cv2.CAP_DSHOW, 0),
        (cv2.CAP_MSMF, 1),
        (cv2.CAP_MSMF, 0),
        (cv2.CAP_DSHOW, 1),
        (None, 0),
        (None, 1),
    ]
    for cfg in standard_configs:
        if cfg not in candidates:
            candidates.append(cfg)

    best_cap = None
    best_info = None
    best_score = -1.0

    for backend, idx in candidates:
        backend_name = "DirectShow" if backend == cv2.CAP_DSHOW else ("MSMF" if backend == cv2.CAP_MSMF else "Default")
        try:
            if backend is not None:
                cap = cv2.VideoCapture(idx, backend)
            else:
                cap = cv2.VideoCapture(idx)

            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Warm-up frames
            valid_frame = None
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    valid_frame = frame
                time.sleep(0.04)

            if valid_frame is not None:
                mean_val = float(np.mean(valid_frame))
                std_val = float(np.std(valid_frame))
                h, w = valid_frame.shape[:2]
                is_active = (std_val > 3.0 and mean_val > 4.0)

                score = (1000.0 if is_active else 0.0) + std_val + (100.0 if idx == preferred_idx else 0.0)

                info = {
                    "index": idx,
                    "backend": backend,
                    "backend_name": backend_name,
                    "width": w,
                    "height": h,
                    "mean": mean_val,
                    "std": std_val,
                    "is_active": is_active,
                    "label": f"Camera {idx} ({backend_name})",
                }

                # If this candidate is active and matches preferred, choose immediately!
                if is_active and (preferred_idx is None or idx == preferred_idx):
                    # Release any previous
                    if best_cap:
                        best_cap.release()
                    print(f"Selected active camera {idx} [{backend_name}]: {w}x{h}, brightness={mean_val:.1f}")
                    return cap, info

                if score > best_score:
                    if best_cap:
                        best_cap.release()
                    best_cap = cap
                    best_info = info
                    best_score = score
                    # Don't release cap yet since it's current best
                    continue

            cap.release()
        except Exception:
            pass

    if best_cap:
        print(f"Selected camera {best_info['index']} [{best_info['backend_name']}]: {best_info['width']}x{best_info['height']}, brightness={best_info['mean']:.1f}")
        return best_cap, best_info

    return None, None
