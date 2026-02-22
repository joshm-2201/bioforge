"""
BioForge - GUI Dashboard
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tkinter as tk
import threading
import time
import argparse
from collections import deque

from comms.arduino_link import ArduinoConnection

try:
    from model.inference import GestureInferenceEngine
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

NUM_CHANNELS  = 8
HISTORY_LEN   = 100
UPDATE_MS     = 100

CHANNEL_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A",
    "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
]

GESTURE_NAMES = {
    "REST": "REST", "FIST": "FIST", "PINCH": "PINCH", "POINT": "POINT",
    "THUMBS_UP": "THUMBS UP", "OPEN_SPREAD": "OPEN", "WRIST_FLEX": "WRIST DOWN",
    "WRIST_EXT": "WRIST UP", "LATERAL_GRIP": "LATERAL", "THREE_FINGER": "THREE",
    "UNKNOWN": "UNKNOWN"
}

DARK_BG    = "#1E1E2E"
PANEL_BG   = "#2A2A3E"
ACCENT     = "#7C3AED"
TEXT_COLOR = "#CDD6F4"
GREEN      = "#A6E3A1"
YELLOW     = "#F9E2AF"
RED        = "#F38BA8"


class BioForgeDashboard:
    def __init__(self, root, port="COM3", model_path=None):
        self.root = root
        self.port = port
        self.model_path = model_path

        self.root.title("BioForge - Bionic Hand Dashboard")
        self.root.configure(bg=DARK_BG)
        self.root.geometry("1280x800")

        self.emg_history = [deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
                            for _ in range(NUM_CHANNELS)]
        self.connected = False
        self.arduino = None
        self.engine = None

        self.current_gesture = "UNKNOWN"
        self.confidence = 0.0
        self.mode = tk.StringVar(value="collect")
        self.servo_angles = [90] * 15

        self._build_ui()

        # Wait 800ms for window to fully render, then auto-connect
        self.root.after(800, self._auto_connect)

        # Start update loop after window renders
        self.root.after(900, self._start_update_loop)

    def _auto_connect(self):
        self._connect()

    def _build_ui(self):
        top = tk.Frame(self.root, bg=DARK_BG, pady=8)
        top.pack(fill=tk.X, padx=12)

        tk.Label(top, text="BioForge", font=("Helvetica", 18, "bold"),
                 bg=DARK_BG, fg=ACCENT).pack(side=tk.LEFT)
        tk.Label(top, text="Bionic Hand Control System", font=("Helvetica", 11),
                 bg=DARK_BG, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=10)

        conn_frame = tk.Frame(top, bg=DARK_BG)
        conn_frame.pack(side=tk.RIGHT)

        tk.Label(conn_frame, text="Port:", bg=DARK_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value=self.port)
        tk.Entry(conn_frame, textvariable=self.port_var, width=10,
                 bg=PANEL_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR).pack(side=tk.LEFT, padx=4)

        self.conn_btn = tk.Button(conn_frame, text="Connect",
                                  command=self._toggle_connection,
                                  bg=ACCENT, fg="white", relief=tk.FLAT, padx=12, pady=4)
        self.conn_btn.pack(side=tk.LEFT, padx=4)

        self.status_dot = tk.Label(conn_frame, text="●", font=("Helvetica", 16),
                                   bg=DARK_BG, fg=RED)
        self.status_dot.pack(side=tk.LEFT, padx=4)

        content = tk.Frame(self.root, bg=DARK_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        left = tk.Frame(content, bg=DARK_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_emg_panel(left)

        right = tk.Frame(content, bg=PANEL_BG, width=320, padx=12, pady=12)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)
        self._build_gesture_panel(right)
        self._build_servo_panel(right)
        self._build_controls_panel(right)

    def _build_emg_panel(self, parent):
        tk.Label(parent, text="EMG Channels (Live)", font=("Helvetica", 12, "bold"),
                 bg=DARK_BG, fg=TEXT_COLOR).pack(anchor=tk.W, pady=(0, 4))

        self.emg_canvases = []
        for i in range(NUM_CHANNELS):
            row = tk.Frame(parent, bg=DARK_BG)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text="Ch" + str(i), width=4, bg=DARK_BG,
                     fg=CHANNEL_COLORS[i], font=("Courier", 9, "bold")).pack(side=tk.LEFT)
            c = tk.Canvas(row, height=50, bg=PANEL_BG, highlightthickness=1,
                          highlightbackground="#444466")
            c.pack(fill=tk.X, expand=True, padx=(4, 0))
            self.emg_canvases.append(c)

    def _build_gesture_panel(self, parent):
        tk.Label(parent, text="Current Gesture", font=("Helvetica", 11, "bold"),
                 bg=PANEL_BG, fg=TEXT_COLOR).pack(anchor=tk.W)

        self.gesture_label = tk.Label(parent, text="UNKNOWN",
                                      font=("Helvetica", 24, "bold"),
                                      bg=PANEL_BG, fg=GREEN)
        self.gesture_label.pack(pady=(4, 0))

        conf_frame = tk.Frame(parent, bg=PANEL_BG)
        conf_frame.pack(fill=tk.X, pady=4)
        tk.Label(conf_frame, text="Confidence:", bg=PANEL_BG,
                 fg=TEXT_COLOR, font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.conf_label = tk.Label(conf_frame, text="0%", bg=PANEL_BG,
                                   fg=YELLOW, font=("Helvetica", 9, "bold"))
        self.conf_label.pack(side=tk.RIGHT)

        self.conf_canvas = tk.Canvas(parent, height=16, bg="#111122",
                                     highlightthickness=0)
        self.conf_canvas.pack(fill=tk.X, pady=(0, 8))

        tk.Frame(parent, bg="#444466", height=1).pack(fill=tk.X, pady=8)

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

        self.servo_labels = []
        self.servo_canvases = []

        for i, name in enumerate(servo_names):
            row = tk.Frame(parent, bg=PANEL_BG)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=name, width=11, anchor=tk.W,
                     bg=PANEL_BG, fg=TEXT_COLOR, font=("Helvetica", 8)).pack(side=tk.LEFT)
            c = tk.Canvas(row, height=12, width=120, bg="#111122",
                          highlightthickness=0)
            c.pack(side=tk.LEFT, padx=2)
            lbl = tk.Label(row, text="90", width=4, bg=PANEL_BG,
                           fg=YELLOW, font=("Helvetica", 8))
            lbl.pack(side=tk.LEFT)
            self.servo_canvases.append(c)
            self.servo_labels.append(lbl)

        tk.Frame(parent, bg="#444466", height=1).pack(fill=tk.X, pady=8)

    def _build_controls_panel(self, parent):
        tk.Label(parent, text="Controls", font=("Helvetica", 11, "bold"),
                 bg=PANEL_BG, fg=TEXT_COLOR).pack(anchor=tk.W)

        mode_frame = tk.Frame(parent, bg=PANEL_BG)
        mode_frame.pack(fill=tk.X, pady=4)
        tk.Label(mode_frame, text="Mode:", bg=PANEL_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)
        for text, val in [("Collect", "collect"), ("Control", "control"), ("Test", "test")]:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode, value=val,
                           command=self._on_mode_change, bg=PANEL_BG, fg=TEXT_COLOR,
                           selectcolor=ACCENT, activebackground=PANEL_BG,
                           activeforeground=TEXT_COLOR).pack(side=tk.LEFT, padx=2)

        btn_frame = tk.Frame(parent, bg=PANEL_BG)
        btn_frame.pack(fill=tk.X, pady=4)
        tk.Button(btn_frame, text="Reset Servos", command=self._reset_servos,
                  bg=ACCENT, fg="white", relief=tk.FLAT,
                  padx=8, pady=4, font=("Helvetica", 9)).pack(side=tk.LEFT, padx=2)

        tk.Label(parent, text="Log", font=("Helvetica", 10, "bold"),
                 bg=PANEL_BG, fg=TEXT_COLOR).pack(anchor=tk.W, pady=(8, 2))
        self.log_text = tk.Text(parent, height=5, width=35, bg="#0D0D1A",
                                fg=TEXT_COLOR, font=("Courier", 8), relief=tk.FLAT)
        self.log_text.pack(fill=tk.X)

    def _toggle_connection(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_var.get()
        self._log("Connecting to " + port + "...")
        self.arduino = ArduinoConnection(mode="usb", port=port)
        self.arduino.on_emg_data = self._on_emg_data
        self.arduino.on_status = lambda s: self._log("Arduino: " + s)

        def try_connect():
            if self.arduino.connect():
                self.connected = True
                self.root.after(500, self._on_connected)
            else:
                self.root.after(0, lambda: self._log("ERROR: Could not connect to " + port))

        threading.Thread(target=try_connect, daemon=True).start()

    def _on_connected(self):
        self.status_dot.config(fg=GREEN)
        self.conn_btn.config(text="Disconnect")
        self._log("Connected!")
        if self.model_path and HAS_MODEL:
            self.root.after(500, self._start_inference)

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

    def _on_emg_data(self, vals):
        for i in range(min(len(vals), NUM_CHANNELS)):
            self.emg_history[i].append(vals[i])

    def _start_inference(self):
        try:
            self.engine = GestureInferenceEngine(
                model_path=self.model_path,
                arduino=self.arduino
            )
            self.engine.on_gesture_change = self._on_gesture_change
            self.engine.start()
            self._log("AI inference started.")
        except Exception as e:
            self._log("Model error: " + str(e))

    def _on_gesture_change(self, gid, name, conf):
        self.current_gesture = name
        self.confidence = conf
        if self.engine:
            self.servo_angles = self.engine.servo_map.get(gid, [90] * 15)

    def _on_mode_change(self):
        mode_map = {"collect": 0, "control": 1, "test": 2}
        if self.arduino:
            self.arduino.set_mode(mode_map[self.mode.get()])
            self._log("Mode: " + self.mode.get())

    def _reset_servos(self):
        if self.arduino:
            self.arduino.reset_servos()
            self.servo_angles = [90] * 15

    def _start_update_loop(self):
        self._update()

    def _update(self):
        try:
            # Draw EMG graphs
            for i in range(NUM_CHANNELS):
                c = self.emg_canvases[i]
                w = c.winfo_width()
                h = c.winfo_height()
                c.delete("all")
                if w < 4 or h < 4:
                    continue
                data = list(self.emg_history[i])
                if len(data) < 2:
                    continue
                step = w / len(data)
                points = []
                for j, v in enumerate(data):
                    x = j * step
                    y = h - max(1, min(h - 1, (v / 512.0) * h))
                    points.extend([x, y])
                if len(points) >= 4:
                    c.create_line(points, fill=CHANNEL_COLORS[i], width=1)

            # Gesture label
            display = GESTURE_NAMES.get(self.current_gesture, self.current_gesture)
            self.gesture_label.config(text=display)
            self.conf_label.config(text=str(int(self.confidence * 100)) + "%")

            # Confidence bar
            self.conf_canvas.delete("all")
            cw = self.conf_canvas.winfo_width()
            ch = self.conf_canvas.winfo_height()
            if cw > 4 and self.confidence > 0:
                fw = int(cw * self.confidence)
                self.conf_canvas.create_rectangle(0, 0, fw, ch, fill=ACCENT, outline="")

            # Servo bars
            for i in range(len(self.servo_canvases)):
                c = self.servo_canvases[i]
                lbl = self.servo_labels[i]
                angle = self.servo_angles[i] if i < len(self.servo_angles) else 90
                lbl.config(text=str(angle))
                c.delete("all")
                cw = c.winfo_width()
                ch = c.winfo_height()
                if cw > 4:
                    fw = int((angle / 180.0) * cw)
                    if fw > 0:
                        c.create_rectangle(0, 0, fw, ch, fill=ACCENT, outline="")

        except Exception as e:
            print("UPDATE ERROR: " + str(e))

        self.root.after(UPDATE_MS, self._update)

    def _log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, "[" + timestamp + "] " + msg + "\n")
        self.log_text.see(tk.END)
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 100:
            self.log_text.delete("1.0", "20.0")


def main():
    parser = argparse.ArgumentParser(description="BioForge Dashboard")
    parser.add_argument("--port", type=str, default="COM3")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    if args.sim:
        args.port = "SIMULATOR"
        print("SIMULATOR mode - make sure simulator.py is running!")

    root = tk.Tk()
    app = BioForgeDashboard(root, port=args.port, model_path=args.model)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._disconnect(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()