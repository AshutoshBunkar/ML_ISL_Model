"""
Indian Sign Language Recognition System - GUI Launcher Popup
"""
import sys
import os
import subprocess

# Auto-redirect to venv if run with system python
root_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(root_dir, "venv", "Scripts", "python.exe")

if os.path.exists(venv_python):
    curr = os.path.normcase(os.path.abspath(sys.executable))
    target = os.path.normcase(os.path.abspath(venv_python))
    if curr != target:
        sys.exit(subprocess.run([venv_python] + sys.argv).returncode)

def launch_target(script_name, camera_idx):
    script_path = os.path.join(root_dir, "src", script_name)
    print(f"\n>>> Starting {script_name} with Camera {camera_idx}...\n")
    proc = subprocess.run([sys.executable, script_path, "--camera", str(camera_idx)])
    return proc.returncode

def show_gui_launcher():
    import tkinter as tk
    from tkinter import font

    root = tk.Tk()
    root.title("ISL Recognition System - Select Mode")
    root.configure(bg="#1a1a2e")
    root.geometry("520x560")
    root.resizable(False, False)

    # Bring to foreground
    root.lift()
    root.attributes('-topmost', True)
    root.after(800, lambda: root.attributes('-topmost', False))
    root.focus_force()

    selected_action = {"script": None, "camera": 0}

    # Styles
    title_font = ("Segoe UI", 15, "bold")
    subtitle_font = ("Segoe UI", 9)
    btn_font = ("Segoe UI", 11, "bold")
    desc_font = ("Segoe UI", 8)
    section_font = ("Segoe UI", 10, "bold")

    # Header
    header = tk.Frame(root, bg="#0f3460", padx=15, pady=15)
    header.pack(fill="x")

    tk.Label(
        header, text="🤟 INDIAN SIGN LANGUAGE SYSTEM",
        font=title_font, fg="#00d4ff", bg="#0f3460"
    ).pack()

    tk.Label(
        header, text="Select an option to launch the application",
        font=subtitle_font, fg="#e0e0e0", bg="#0f3460"
    ).pack(pady=(4, 0))

    container = tk.Frame(root, bg="#1a1a2e", padx=20, pady=15)
    container.pack(fill="both", expand=True)

    # Camera selector card
    cam_card = tk.LabelFrame(
        container, text="  Camera Source  ",
        font=section_font, fg="#00d4ff", bg="#16213e",
        padx=15, pady=8, bd=1, relief="solid"
    )
    cam_card.pack(fill="x", pady=(0, 15))

    cam_var = tk.IntVar(value=0)

    rb0 = tk.Radiobutton(
        cam_card, text="Camera 0  (Primary Built-in RGB Webcam - Recommended)",
        variable=cam_var, value=0,
        font=("Segoe UI", 9), fg="#ffffff", bg="#16213e",
        selectcolor="#0f3460", activebackground="#16213e", activeforeground="#00d4ff"
    )
    rb0.pack(anchor="w", pady=2)

    rb1 = tk.Radiobutton(
        cam_card, text="Camera 1  (Secondary / External Sensor)",
        variable=cam_var, value=1,
        font=("Segoe UI", 9), fg="#ffffff", bg="#16213e",
        selectcolor="#0f3460", activebackground="#16213e", activeforeground="#00d4ff"
    )
    rb1.pack(anchor="w", pady=2)

    def on_select(script_name):
        selected_action["script"] = script_name
        selected_action["camera"] = cam_var.get()
        root.destroy()

    def create_btn(parent, icon, title, desc, script_name, highlight_color="#00e676"):
        btn_frame = tk.Frame(parent, bg="#16213e", bd=1, relief="solid")
        btn_frame.pack(fill="x", pady=6)

        inner = tk.Frame(btn_frame, bg="#16213e", padx=12, pady=10)
        inner.pack(fill="both", expand=True)

        left = tk.Frame(inner, bg="#16213e")
        left.pack(side="left", fill="both", expand=True)

        tk.Label(
            left, text=f"{icon}  {title}",
            font=btn_font, fg=highlight_color, bg="#16213e"
        ).pack(anchor="w")

        tk.Label(
            left, text=desc,
            font=desc_font, fg="#a0a0a0", bg="#16213e"
        ).pack(anchor="w", pady=(2, 0))

        go_btn = tk.Button(
            inner, text="Launch ▶",
            font=("Segoe UI", 9, "bold"),
            bg="#0f3460", fg="white",
            activebackground="#00d4ff", activeforeground="#1a1a2e",
            relief="flat", padx=12, pady=6, cursor="hand2",
            command=lambda: on_select(script_name)
        )
        go_btn.pack(side="right", padx=(10, 0))

        # Clicking anywhere on the card triggers launch
        for widget in (btn_frame, inner, left):
            widget.bind("<Button-1>", lambda e, s=script_name: on_select(s))

    create_btn(
        container,
        "🖥️", "Full Desktop GUI App",
        "Interactive UI, live camera, sentence builder & text-to-speech",
        "app.py",
        highlight_color="#00e676"
    )

    create_btn(
        container,
        "⚡", "Real-Time OpenCV Predictor",
        "Lightweight high-FPS sign recognition overlay window",
        "predict.py",
        highlight_color="#00d4ff"
    )

    create_btn(
        container,
        "📷", "Test Webcam Only",
        "Verify your camera sensor index and preview live feed",
        "camera_test.py",
        highlight_color="#ffd600"
    )

    # Footer
    footer = tk.Frame(container, bg="#1a1a2e")
    footer.pack(fill="x", pady=(10, 0))

    exit_btn = tk.Button(
        footer, text="Exit",
        font=("Segoe UI", 9),
        bg="#2d132c", fg="#ff7979",
        activebackground="#ff5252", activeforeground="white",
        relief="flat", padx=14, pady=4, cursor="hand2",
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

    if "gui" in args or "app" in args:
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
            print("  [1] Launch Full Desktop GUI App")
            print("  [2] Launch Real-Time OpenCV Predictor")
            print("  [3] Test Webcam Only")
            print("  [4] Exit")
            ch = input("Enter choice (1-4) [default: 1]: ").strip()
            mapping = {"1": "app.py", "2": "predict.py", "3": "camera_test.py"}
            if ch in mapping or ch == "":
                target = mapping.get(ch, "app.py")
                launch_target(target, cam)

if __name__ == "__main__":
    main()
