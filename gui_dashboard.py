"""
BioForge - GUI Dashboard
=========================
Real-time monitoring dashboard built with tkinter + matplotlib.
Shows:
  - Live EMG signal plots for all channels
  - Current gesture prediction + confidence bar
  - Servo position visualization
  - Signal quality indicators
  - Mode switching (collect / control / test)

Usage:
    python gui_dashboard.py --port COM3
    python gui_dashboard.py --model models/bioforge_model.pkl --port COM3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import argparse
import numpy as np
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from comms.arduino_link import ArduinoConnection

# Optional inference
try:
    from model.inference import GestureInferenceEngine
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NUM_CHANNELS  = 8
HISTORY_LEN   = 200   # samples to show in plot (~5 seconds at 40Hz)
UPDATE_MS     = 50    # GUI refresh rate (20Hz)

CHANNEL_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A",
    "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
]

GESTURE_EMOJIS = {
    "REST": "✋", "FIST": "✊", "PINCH": "🤏", "POINT": "☝️",
    "THUMBS_UP": "👍", "OPEN_SPREAD": "🖐️", "WRIST_FLEX": "⬇️",
    "WRIST_EXT": "⬆️", "LATERAL_GRIP": "🤙", "THREE_FINGER": "🤘",
    "UNKNOWN": "❓"
}

DARK_BG    = "#1E1E2E"
PANEL_BG   = "#2A2A3E"
ACCENT     = "#7C3AED"
TEXT_COLOR = "#CDD6F4"
GREEN      = "#A6E3A1"
YELLOW     = "#F9E2AF"
RED        = "#F38BA8"


class BioForgeDashboard:
    def __init__(self, root: tk.Tk, port: str = "COM3", model_path: str = None):
        self.root = root
        self.port = port
        self.model_path = model_path

        self.root.title("BioForge — Bionic Hand Dashboard")
        self.root.configure(bg=DARK_BG)
        self.root.geometry("1280x800")
        self.root.resizable(True, True)

        # Data
        self.emg_history = [deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
                            for _ in range(NUM_CHANNELS)]
        self.connected = False
        self.arduino = None
        self.engine = None

        # State
        self.current_gesture = "UNKNOWN"
        self.confidence = 0.0
        self.mode = tk.StringVar(value="collect")
        self.servo_angles = [90] * 15

        self._build_ui()
        self._start_update_loop()

    # ── UI CONSTRUCTION ───────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ──
        top = tk.Frame(self.root, bg=DARK_BG, pady=8)
        top.pack(fill=tk.X, padx=12)

        tk.Label(top, text="⚡ BioForge", font=("Helvetica", 18, "bold"),
                 bg=DARK_BG, fg=ACCENT).pack(side=tk.LEFT)

        tk.Label(top, text="Bionic Hand Control System", font=("Helvetica", 11),
                 bg=DARK_BG, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=10)

        # Connection controls (right side)
        conn_frame = tk.Frame(top, bg=DARK_BG)
        conn_frame.pack(side=tk.RIGHT)

        tk.Label(conn_frame, text="Port:", bg=DARK_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value=self.port)
        port_entry = tk.Entry(conn_frame, textvariable=self.port_var, width=8,
                              bg=PANEL_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
        port_entry.pack(side=tk.LEFT, padx=4)

        self.conn_btn = tk.Button(conn_frame, text="Connect", command=self._toggle_connection,
                                  bg=ACCENT, fg="white", relief=tk.FLAT, padx=12, pady=4)
        self.conn_btn.pack(side=tk.LEFT, padx=4)

        self.status_dot = tk.Label(conn_frame, text="●", font=("Helvetica", 16),
                                   bg=DARK_BG, fg=RED)
        self.status_dot.pack(side=tk.LEFT, padx=4)

        # ── Main content area ──
        content = tk.Frame(self.root, bg=DARK_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # Left panel: EMG plots
        left = tk.Frame(content, bg=DARK_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_emg_panel(left)

        # Right panel: Gesture + Servos + Controls
        right = tk.Frame(content, bg=PANEL_BG, width=320, padx=12, pady=12)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        self._build_gesture_panel(right)
        self._build_servo_panel(right)
        self._build_controls_panel(right)

    def _build_emg_panel(self, parent):
        tk.Label(parent, text="EMG Channels", font=("Helvetica", 12, "bold"),
                 bg=DARK_BG, fg=TEXT_COLOR).pack(anchor=tk.W, pady=(0, 4))

        fig = Figure(figsize=(8, 6), facecolor=DARK_BG)
        fig.subplots_adjust(hspace=0.15, left=0.08, right=0.98, top=0.97, bottom=0.05)

        self.emg_axes = []
        self.emg_lines = []

        for i in range(NUM_CHANNELS):
            ax = fig.add_subplot(NUM_CHANNELS, 1, i + 1)
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COLOR, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
            ax.set_xlim(0, HISTORY_LEN)
            ax.set_ylim(0, 512)
            ax.set_yticks([256])
            ax.set_yticklabels([f"Ch{i}"], fontsize=7, color=CHANNEL_COLORS[i])
            ax.set_xticks([])

            line, = ax.plot(np.zeros(HISTORY_LEN), color=CHANNEL_COLORS[i], linewidth=1.0)
            self.emg_axes.append(ax)
            self.emg_lines.append(line)

        self.canvas = FigureCanvasTkAgg(fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_gesture_panel(self, parent):
        tk.Label(parent, text="Current Gesture", font=("Helvetica", 11, "bold"),
                 bg=PANEL_BG, fg=TEXT_COLOR).pack(anchor=tk.W)

        # Gesture name display
        self.gesture_label = tk.Label(parent, text="✋  REST",
                                       font=("Helvetica", 22, "bold"),
                                       bg=PANEL_BG, fg=GREEN)
        self.gesture_label.pack(pady=(4, 0))

        # Confidence bar
        conf_frame = tk.Frame(parent, bg=PANEL_BG)
        conf_frame.pack(fill=tk.X, pady=4)
        tk.Label(conf_frame, text="Confidence:", bg=PANEL_BG, fg=TEXT_COLOR,
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.conf_label = tk.Label(conf_frame, text="0%", bg=PANEL_BG, fg=YELLOW,
                                    font=("Helvetica", 9, "bold"))
        self.conf_label.pack(side=tk.RIGHT)

        self.conf_bar = ttk.Progressbar(parent, length=280, mode="determinate",
                                         maximum=100)
        self.conf_bar.pack(fill=tk.X, pady=(0, 8))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

    def _build_servo_panel(self, parent):
        tk.Label(parent, text="Servo Positions", font=("Helvetica", 11, "bold"),
                 bg=PANEL_BG, fg=TEXT_COLOR).pack(anchor=tk.W)

        servo_names = [
            "Thumb Flex", "Thumb Abd", "Thumb Tip",
            "Index MCP", "Index PIP", "Index DIP",
            "Middle MCP", "Middle PIP", "Middle DIP",
            "Ring MCP", "Ring PIP", "Ring DIP",
            "Pinky MCP", "Pinky PIP", "Wrist"
        ]

        self.servo_bars = []
        servo_scroll = tk.Frame(parent, bg=PANEL_BG)
        servo_scroll.pack(fill=tk.X)

        for i, name in enumerate(servo_names):
            row = tk.Frame(servo_scroll, bg=PANEL_BG)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=name, width=12, anchor=tk.W,
                     bg=PANEL_BG, fg=TEXT_COLOR, font=("Helvetica", 8)).pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, length=120, maximum=180, mode="determinate")
            bar.pack(side=tk.LEFT, padx=4)
            bar["value"] = 90
            lbl = tk.Label(row, text="90°", width=4,
                           bg=PANEL_BG, fg=YELLOW, font=("Helvetica", 8))
            lbl.pack(side=tk.LEFT)
            self.servo_bars.append((bar, lbl))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

    def _build_controls_panel(self, parent):
        tk.Label(parent, text="Controls", font=("Helvetica", 11, "bold"),
                 bg=PANEL_BG, fg=TEXT_COLOR).pack(anchor=tk.W)

        # Mode selector
        mode_frame = tk.Frame(parent, bg=PANEL_BG)
        mode_frame.pack(fill=tk.X, pady=4)
        tk.Label(mode_frame, text="Mode:", bg=PANEL_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)

        for text, val in [("Collect", "collect"), ("Control", "control"), ("Test", "test")]:
            rb = tk.Radiobutton(mode_frame, text=text, variable=self.mode, value=val,
                                command=self._on_mode_change,
                                bg=PANEL_BG, fg=TEXT_COLOR, selectcolor=ACCENT,
                                activebackground=PANEL_BG, activeforeground=TEXT_COLOR)
            rb.pack(side=tk.LEFT, padx=4)

        # Buttons
        btn_frame = tk.Frame(parent, bg=PANEL_BG)
        btn_frame.pack(fill=tk.X, pady=4)

        btn_style = {"bg": ACCENT, "fg": "white", "relief": tk.FLAT,
                     "padx": 8, "pady": 4, "font": ("Helvetica", 9)}

        tk.Button(btn_frame, text="Reset Servos", command=self._reset_servos,
                  **btn_style).pack(side=tk.LEFT, padx=2)

        if self.model_path and HAS_MODEL:
            tk.Button(btn_frame, text="▶ Start AI", command=self._toggle_inference,
                      **btn_style).pack(side=tk.LEFT, padx=2)

        # Log area
        tk.Label(parent, text="Log", font=("Helvetica", 10, "bold"),
                 bg=PANEL_BG, fg=TEXT_COLOR).pack(anchor=tk.W, pady=(8, 2))

        self.log_text = tk.Text(parent, height=6, width=35, bg="#0D0D1A",
                                 fg=TEXT_COLOR, font=("Courier", 8), relief=tk.FLAT)
        self.log_text.pack(fill=tk.X)

    # ── CONNECTION ────────────────────────────────────────────────────────

    def _toggle_connection(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_var.get()
        self._log(f"Connecting to {port}...")

        self.arduino = ArduinoConnection(mode="usb", port=port)
        self.arduino.on_emg_data = self._on_emg_data
        self.arduino.on_status = lambda s: self._log(f"Arduino: {s}")

        def try_connect():
            if self.arduino.connect():
                self.connected = True
                self.root.after(0, self._on_connected)
            else:
                self.root.after(0, lambda: self._log(f"ERROR: Could not connect to {port}"))

        threading.Thread(target=try_connect, daemon=True).start()

    def _on_connected(self):
        self.status_dot.config(fg=GREEN)
        self.conn_btn.config(text="Disconnect")
        self._log("Connected!")
        # Start inference if model provided
        if self.model_path and HAS_MODEL:
            self._start_inference()

    def _disconnect(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        if self.arduino:
            self.arduino.disconnect()
            self.arduino = None
        self.connected = False
        self.status_dot.config(fg=RED)
        self.conn_btn.config(text="Connect")
        self._log("Disconnected.")

    # ── EMG DATA ─────────────────────────────────────────────────────────

    def _on_emg_data(self, vals: list):
        for i in range(min(len(vals), NUM_CHANNELS)):
            self.emg_history[i].append(vals[i])

    # ── INFERENCE ─────────────────────────────────────────────────────────

    def _start_inference(self):
        if not self.connected or not self.arduino:
            return
        try:
            self.engine = GestureInferenceEngine(
                model_path=self.model_path,
                arduino=self.arduino
            )
            self.engine.on_gesture_change = self._on_gesture_change
            self.engine.start()
            self._log("AI inference started.")
        except Exception as e:
            self._log(f"Model error: {e}")

    def _toggle_inference(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
            self._log("AI stopped.")
        else:
            self._start_inference()

    def _on_gesture_change(self, gid, name, conf):
        self.current_gesture = name
        self.confidence = conf
        self.servo_angles = self.engine.servo_map.get(gid, [90] * 15)

    # ── CONTROLS ──────────────────────────────────────────────────────────

    def _on_mode_change(self):
        mode_map = {"collect": 0, "control": 1, "test": 2}
        if self.arduino:
            self.arduino.set_mode(mode_map[self.mode.get()])
            self._log(f"Mode: {self.mode.get()}")

    def _reset_servos(self):
        if self.arduino:
            self.arduino.reset_servos()
            self.servo_angles = [90] * 15

    # ── GUI UPDATE LOOP ───────────────────────────────────────────────────

    def _start_update_loop(self):
        self._update()

    def _update(self):
        # Update EMG plots
        for i in range(NUM_CHANNELS):
            self.emg_lines[i].set_ydata(list(self.emg_history[i]))
        try:
            self.canvas.draw_idle()
        except:
            pass

        # Update gesture display
        emoji = GESTURE_EMOJIS.get(self.current_gesture, "❓")
        self.gesture_label.config(text=f"{emoji}  {self.current_gesture}")
        conf_pct = int(self.confidence * 100)
        self.conf_bar["value"] = conf_pct
        self.conf_label.config(text=f"{conf_pct}%")

        # Update servo bars
        for i, (bar, lbl) in enumerate(self.servo_bars):
            angle = self.servo_angles[i] if i < len(self.servo_angles) else 90
            bar["value"] = angle
            lbl.config(text=f"{angle}°")

        self.root.after(UPDATE_MS, self._update)

    # ── LOG ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)
        # Keep log from growing too large
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 200:
            self.log_text.delete("1.0", "50.0")


def main():
    parser = argparse.ArgumentParser(description="BioForge Dashboard")
    parser.add_argument("--port", type=str, default="COM3")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model .pkl")
    parser.add_argument("--sim", action="store_true",
                        help="Use hardware simulator (no Arduino needed)")
    args = parser.parse_args()

    if args.sim:
        args.port = "SIMULATOR"
        print("Running in SIMULATOR mode — make sure simulator.py is running!")

    root = tk.Tk()
    app = BioForgeDashboard(root, port=args.port, model_path=args.model)

    # Style ttk progressbars
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TProgressbar", troughcolor=PANEL_BG, background=ACCENT,
                    bordercolor=PANEL_BG, lightcolor=ACCENT, darkcolor=ACCENT)

    root.protocol("WM_DELETE_WINDOW", lambda: (app._disconnect(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
