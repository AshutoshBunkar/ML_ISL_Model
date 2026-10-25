"""
Indian Sign Language Recognition System - GUI Launcher Popup
Styled with the SE_PROJECT SaaS Design System (#FF6B00 Brand Accent & Jet Black CTAs).
"""
import sys
import os
import subprocess
import webbrowser
import time

# Auto-redirect to venv if run with system python
root_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(root_dir, "venv", "Scripts", "python.exe")

if os.path.exists(venv_python):
    curr = os.path.normcase(os.path.abspath(sys.executable))
    target = os.path.normcase(os.path.abspath(venv_python))
    if curr != target:
        sys.exit(subprocess.run([venv_python] + sys.argv).returncode)

def launch_target(script_name, camera_idx):
    if script_name == "web_app.py":
        script_path = os.path.join(root_dir, script_name)
        print(f"\n>>> Launching ISL Recognition Web Engine at http://127.0.0.1:8000 ...\n")
        # Open browser after short delay
        def open_browser():
            time.sleep(1.2)
            webbrowser.open("http://127.0.0.1:8000")
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        proc = subprocess.run([sys.executable, script_path])
        return proc.returncode
    else:
        script_path = os.path.join(root_dir, "src", script_name)
        print(f"\n>>> Starting {script_name} with Camera {camera_idx}...\n")
        proc = subprocess.run([sys.executable, script_path, "--camera", str(camera_idx)])
        return proc.returncode

def show_gui_launcher():
    import tkinter as tk
    from tkinter import font

    root = tk.Tk()
    root.title("ISL Recognition System - Select Mode")
    root.configure(bg="#fafafa")
    root.geometry("540x620")
    root.resizable(False, False)

    # Bring to foreground
    root.lift()
    root.attributes('-topmost', True)
    root.after(800, lambda: root.attributes('-topmost', False))
    root.focus_force()

    selected_action = {"script": None, "camera": 0}

    # Theme tokens (SE_PROJECT Theme)
    bg_canvas = "#fafafa"
    card_bg = "#ffffff"
    border_col = "#e2e8f0"
    color_brand = "#ff6b00"
    color_brand_subtle = "#fff0e6"
    color_primary = "#000000"
    text_primary = "#111827"
    text_secondary = "#4b5563"
    text_muted = "#6b7280"

    # Styles
    try:
        title_font = ("Space Grotesk", 14, "bold")
        section_font = ("Space Grotesk", 10, "bold")
    except Exception:
        title_font = ("Segoe UI", 14, "bold")
        section_font = ("Segoe UI", 10, "bold")

    subtitle_font = ("Segoe UI", 9)
    btn_font = ("Segoe UI", 10, "bold")
    desc_font = ("Segoe UI", 8)

    # Header (Clean White with border)
    header = tk.Frame(root, bg="#ffffff", padx=18, pady=14, bd=1, relief="solid")
    header.pack(fill="x")

    top_row = tk.Frame(header, bg="#ffffff")
    top_row.pack(fill="x")

    logo_box = tk.Label(
        top_row, text="ISL", font=("Segoe UI", 10, "bold"),
        bg=color_brand, fg="#ffffff", padx=6, pady=2
    )
    logo_box.pack(side="left", padx=(0, 8))

    tk.Label(
        top_row, text="INDIAN SIGN LANGUAGE SYSTEM",
        font=title_font, fg=text_primary, bg="#ffffff"
    ).pack(side="left")

    badge_lbl = tk.Label(
        top_row, text="Engine v1.0",
        font=("Segoe UI", 8, "bold"),
        fg=color_brand, bg=color_brand_subtle,
        padx=6, pady=2
    )
    badge_lbl.pack(side="left", padx=(8, 0))

    tk.Label(
        header, text="Select an interface mode to launch the application",
        font=subtitle_font, fg=text_secondary, bg="#ffffff"
    ).pack(anchor="w", pady=(6, 0))

    container = tk.Frame(root, bg=bg_canvas, padx=18, pady=14)
    container.pack(fill="both", expand=True)

    # Camera selector card
    cam_card = tk.LabelFrame(
        container, text="  Camera Hardware Source  ",
        font=section_font, fg=text_primary, bg=card_bg,
        padx=14, pady=8, bd=1, relief="solid"
    )
    cam_card.pack(fill="x", pady=(0, 12))

    cam_var = tk.IntVar(value=0)

    rb0 = tk.Radiobutton(
        cam_card, text="Camera 0  (Physical Built-in HD WebCam - Recommended)",
        variable=cam_var, value=0,
        font=("Segoe UI", 9), fg=text_primary, bg=card_bg,
        selectcolor="#ffffff", activebackground=card_bg, activeforeground=color_brand
    )
    rb0.pack(anchor="w", pady=2)

    rb1 = tk.Radiobutton(
        cam_card, text="Camera 1  (Secondary / Smart Connect Camera)",
        variable=cam_var, value=1,
        font=("Segoe UI", 9), fg=text_primary, bg=card_bg,
        selectcolor="#ffffff", activebackground=card_bg, activeforeground=color_brand
    )
    rb1.pack(anchor="w", pady=2)

    def on_select(script_name):
        selected_action["script"] = script_name
        selected_action["camera"] = cam_var.get()
        root.destroy()

    def create_btn(parent, icon, title, desc, script_name, is_primary=False):
        btn_frame = tk.Frame(parent, bg=card_bg, bd=1, relief="solid")
        btn_frame.pack(fill="x", pady=5)

        inner = tk.Frame(btn_frame, bg=card_bg, padx=12, pady=10)
        inner.pack(fill="both", expand=True)

        left = tk.Frame(inner, bg=card_bg)
        left.pack(side="left", fill="both", expand=True)

        title_lbl = tk.Label(
            left, text=f"{icon}  {title}",
            font=btn_font, fg=color_brand if is_primary else text_primary, bg=card_bg
        )
        title_lbl.pack(anchor="w")

        desc_lbl = tk.Label(
            left, text=desc,
            font=desc_font, fg=text_secondary, bg=card_bg
        )
        desc_lbl.pack(anchor="w", pady=(2, 0))

        go_btn = tk.Button(
            inner, text="Launch ▶",
            font=("Segoe UI", 9, "bold"),
            bg="#000000" if is_primary else "#ffffff",
            fg="#ffffff" if is_primary else text_primary,
            activebackground="#1f2937" if is_primary else "#f3f4f6",
            activeforeground="#ffffff" if is_primary else text_primary,
            bd=1 if not is_primary else 0,
            relief="solid" if not is_primary else "flat",
            padx=12, pady=6, cursor="hand2",
            command=lambda: on_select(script_name)
        )
        go_btn.pack(side="right", padx=(10, 0))

        # Clicking anywhere on the card triggers launch
        for widget in (btn_frame, inner, left, title_lbl, desc_lbl):
            widget.bind("<Button-1>", lambda e, s=script_name: on_select(s))

    create_btn(
        container,
        "🌐", "Interactive Web UI (SE_PROJECT Theme)",
        "FastAPI + HTML5 browser studio with live camera, audio TTS & benchmarks",
        "web_app.py",
        is_primary=True
    )

    create_btn(
        container,
        "🖥️", "Desktop Application (SE_PROJECT Theme)",
        "Tkinter desktop window with live video, sentence buffer & gesture guide",
        "app.py",
        is_primary=False
    )

    create_btn(
        container,
        "⚡", "Real-Time OpenCV Predictor",
        "Lightweight high-FPS sign recognition overlay window",
        "predict.py",
        is_primary=False
    )

    create_btn(
        container,
        "📷", "Camera Hardware Diagnostic",
        "Verify your camera sensor index and preview live feed",
        "camera_test.py",
        is_primary=False
    )

    # Footer
    footer = tk.Frame(container, bg=bg_canvas)
    footer.pack(fill="x", pady=(10, 0))

    exit_btn = tk.Button(
        footer, text="Exit",
        font=("Segoe UI", 9),
        bg="#ffffff", fg="#dc2626",
        activebackground="#fef2f2", activeforeground="#dc2626",
        bd=1, relief="solid", padx=14, pady=4, cursor="hand2",
        command=root.destroy
    )
    exit_btn.pack(side="right")

    root.mainloop()

    if selected_action["script"]:
        launch_target(selected_action["script"], selected_action["camera"])

def main():
    # Direct CLI shortcuts
    args = [a.lower().strip("-") for a in sys.argv[1:]]
    cam = 0
    for i, a in enumerate(sys.argv):
        if a in ("--camera", "-c") and i + 1 < len(sys.argv):
            try:
                cam = int(sys.argv[i + 1])
            except ValueError:
                pass

    if "help" in args or "h" in args:
        print("\nIndian Sign Language Recognition System - CLI Usage:")
        print("  python run.py                   Launch graphical launcher popup")
        print("  python run.py web               Launch Web UI (FastAPI on http://127.0.0.1:8000)")
        print("  python run.py gui               Launch Desktop Tkinter GUI App")
        print("  python run.py predict           Launch OpenCV real-time predictor")
        print("  python run.py camera            Launch Camera diagnostic test")
        print("  Options:")
        print("    --camera <index>              Specify camera hardware index (e.g. --camera 0)\n")
        return

    if "web" in args or "fastapi" in args:
        launch_target("web_app.py", cam)
    elif "gui" in args or "app" in args:
        launch_target("app.py", cam)
    elif "predict" in args or "cv" in args:
        launch_target("predict.py", cam)
    elif "camera" in args or "test" in args:
        launch_target("camera_test.py", cam)
    else:
        # Launch graphical popup modal
        try:
            show_gui_launcher()
        except Exception as e:
            print(f"GUI Popup error: {e}. Falling back to terminal prompt.")
            print("\nSelect an option:")
            print("  [1] Launch Interactive Web UI Studio (SE_PROJECT Theme)")
            print("  [2] Launch Desktop Application (SE_PROJECT Theme)")
            print("  [3] Launch Real-Time OpenCV Predictor")
            print("  [4] Test Webcam Only")
            print("  [5] Exit")
            ch = input("Enter choice (1-5) [default: 1]: ").strip()
            mapping = {
                "1": "web_app.py",
                "2": "app.py",
                "3": "predict.py",
                "4": "camera_test.py"
            }
            if ch in mapping or ch == "":
                target = mapping.get(ch, "web_app.py")
                launch_target(target, cam)

if __name__ == "__main__":
    main()
