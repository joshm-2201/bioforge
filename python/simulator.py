"""
BioForge - Arduino Simulator
==============================
Simulates the Arduino Mega over a virtual serial port.
Use this to test ALL Python code before hardware arrives.

Generates realistic fake EMG signals for each gesture,
responds to servo commands, and logs everything.

Usage:
    python simulator.py

Then in a separate terminal, run any of:
    python gui/gui_dashboard.py --port SIMULATOR
    python data_collection/collect_data.py --port SIMULATOR
    python model/inference.py --model models/bioforge_model.pkl --port SIMULATOR

How it works:
    Uses Python's socket to create a local TCP server on port 5001.
    The modified arduino_link.py connects to it as if it were a real serial port.
    You don't need any COM port or physical device.
"""

import socket
import threading
import time
import math
import random
import sys

# ─── SIMULATED GESTURE EMG PROFILES ────────────────────────────────────────
# Each gesture has a characteristic EMG pattern across 8 channels
# Values are base amplitudes (0-512 range, mimicking Arduino analog read)
GESTURE_EMG_PROFILES = {
    "REST":         [10,  8, 12,  9, 11,  8, 10,  7],
    "FIST":         [280, 310, 220, 190, 150, 120, 200, 180],
    "PINCH":        [180, 200,  80,  60,  40,  30, 100,  90],
    "POINT":        [100, 120, 200, 220,  40,  30,  60,  50],
    "THUMBS_UP":    [220, 180,  50,  40,  30,  20,  40,  30],
    "OPEN_SPREAD":  [150, 140, 160, 150, 160, 155, 150, 145],
    "WRIST_FLEX":   [ 50,  40,  60,  50, 200, 220, 180, 190],
    "WRIST_EXT":    [ 50,  40,  60,  50, 180, 160, 220, 200],
    "LATERAL_GRIP": [200, 180, 160, 140,  50,  40,  60,  50],
    "THREE_FINGER": [160, 180, 200, 210, 190,  40,  50,  30],
}

HOST = "127.0.0.1"
PORT = 5001
SAMPLE_RATE_HZ = 40  # matches real system


class ArduinoSimulator:
    def __init__(self):
        self.current_gesture = "REST"
        self.gesture_lock = threading.Lock()
        self.servo_angles = [90] * 15
        self.mode = 0  # 0=collect, 1=control, 2=test
        self.running = True
        self.client = None
        self.t = 0.0  # time counter for noise simulation

        # Cycle through gestures automatically in demo mode
        self.demo_mode = True
        self.gesture_list = list(GESTURE_EMG_PROFILES.keys())
        self.gesture_idx = 0

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        server.settimeout(1.0)

        print("=" * 55)
        print("  BioForge Arduino Simulator")
        print("=" * 55)
        print(f"  Listening on {HOST}:{PORT}")
        print()
        print("  To connect from another terminal, run:")
        print("    python gui/gui_dashboard.py --sim")
        print("    python data_collection/collect_data.py --sim")
        print()
        print("  Press Ctrl+C to stop")
        print("=" * 55)

        # Start gesture cycling thread
        if self.demo_mode:
            threading.Thread(target=self._cycle_gestures, daemon=True).start()

        while self.running:
            try:
                conn, addr = server.accept()
                print(f"\n[SIM] Client connected: {addr}")
                self.client = conn
                self._handle_client(conn)
                print("[SIM] Client disconnected")
                self.client = None
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break

        server.close()
        print("\n[SIM] Simulator stopped.")

    def _handle_client(self, conn: socket.socket):
        """Handle one connected client."""
        conn.sendall(b"STATUS:READY\n")

        rx_thread = threading.Thread(
            target=self._receive_loop, args=(conn,), daemon=True
        )
        rx_thread.start()

        # Main EMG sending loop
        interval = 1.0 / SAMPLE_RATE_HZ
        while self.running and self.client:
            try:
                emg_vals = self._generate_emg()
                msg = "EMG:" + ",".join(f"{v:.2f}" for v in emg_vals) + "\n"
                conn.sendall(msg.encode())
                self.t += interval
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            time.sleep(interval)

    def _receive_loop(self, conn: socket.socket):
        """Parse incoming commands from Python client."""
        buf = ""
        while self.running and self.client:
            try:
                data = conn.recv(256).decode("utf-8", errors="ignore")
                if not data:
                    break
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    self._parse_command(line.strip())
            except (socket.timeout, OSError):
                break

    def _parse_command(self, cmd: str):
        if cmd.startswith("SERVO:"):
            angles = [int(x) for x in cmd[6:].split(",") if x]
            self.servo_angles = angles
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Wrist"]
            display = " | ".join(
                f"{finger_names[i//3] if i < 15 else 'W'}:{angles[i]}°"
                for i in range(0, min(len(angles), 15), 3)
            )
            print(f"[SIM] Servos → {display}")

        elif cmd.startswith("MODE:"):
            self.mode = int(cmd[5:])
            modes = {0: "COLLECT", 1: "CONTROL", 2: "TEST"}
            print(f"[SIM] Mode set to: {modes.get(self.mode, self.mode)}")
            if self.client:
                self.client.sendall(f"STATUS:MODE_SET:{self.mode}\n".encode())

        elif cmd == "RESET":
            self.servo_angles = [90] * 15
            print("[SIM] Servos reset to 90°")
            if self.client:
                self.client.sendall(b"STATUS:RESET_DONE\n")

        elif cmd == "PING":
            if self.client:
                self.client.sendall(b"STATUS:PONG\n")

        elif cmd.startswith("GESTURE:"):
            # Allow external gesture setting for testing
            g = cmd[8:]
            if g in GESTURE_EMG_PROFILES:
                with self.gesture_lock:
                    self.current_gesture = g
                print(f"[SIM] Gesture forced to: {g}")

    def _generate_emg(self) -> list:
        """Generate realistic noisy EMG signal for current gesture."""
        with self.gesture_lock:
            profile = GESTURE_EMG_PROFILES[self.current_gesture]

        emg = []
        for i, base in enumerate(profile):
            # Gaussian noise proportional to signal strength
            noise = random.gauss(0, max(5, base * 0.15))
            # Low-frequency muscle tremor (3-12 Hz)
            tremor = base * 0.1 * math.sin(2 * math.pi * 7 * self.t + i * 0.7)
            # High-frequency EMG oscillation (50-150 Hz typical)
            hf = base * 0.2 * math.sin(2 * math.pi * 80 * self.t + i * 1.3)
            val = max(0, base + noise + tremor + hf)
            emg.append(round(val, 2))

        return emg

    def _cycle_gestures(self):
        """Automatically cycle through gestures every 4 seconds in demo mode."""
        time.sleep(2)  # wait for connection
        while self.running:
            gesture = self.gesture_list[self.gesture_idx % len(self.gesture_list)]
            with self.gesture_lock:
                self.current_gesture = gesture
            print(f"[SIM] Demo gesture: {gesture}")
            self.gesture_idx += 1
            time.sleep(4)


# ── Patch arduino_link.py to support simulator ─────────────────────────────
# When --sim flag is used, connect to localhost:5001 instead of a COM port

PATCH_CODE = '''
# ── ADD THIS to arduino_link.py connect() method ──
# Replace the USB serial block with this when using simulator:
#
# if self.mode == "usb" and self.port == "SIMULATOR":
#     # Connect to local simulator
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     sock.connect(("127.0.0.1", 5001))
#     sock.settimeout(1.0)
#     self._conn = sock
#     self.mode = "wifi"  # reuse WiFi path for socket
#
# OR just use: --wifi 127.0.0.1 --port 5001 when running other scripts
'''


if __name__ == "__main__":
    sim = ArduinoSimulator()
    try:
        sim.start()
    except KeyboardInterrupt:
        sim.running = False
        print("\nSimulator stopped.")
