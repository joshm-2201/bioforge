"""
BioForge - Serial/WiFi Communication Layer
============================================
Handles connection to Arduino either via:
  - Direct USB Serial (development/wired)
  - TCP over WiFi via ESP32 bridge (wireless/final)

Exposes a unified interface regardless of connection type.
"""

import serial
import socket
import threading
import queue
import time
import logging
from typing import Optional, Callable

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("BioForge.Comms")


class ArduinoConnection:
    """
    Unified interface for talking to the Arduino Mega.
    Use USB mode during development, WiFi mode when untethered.
    """

    def __init__(self,
                 mode: str = "usb",
                 port: str = "COM3",          # Windows: COM3, COM4 etc. Check Device Manager
                 baud: int = 115200,
                 wifi_host: str = "192.168.1.100",  # ESP32 IP address
                 wifi_port: int = 5000):

        self.mode = mode
        self.port = port
        self.baud = baud
        self.wifi_host = wifi_host
        self.wifi_port = wifi_port

        self._conn = None          # serial.Serial or socket
        self._running = False
        self._rx_thread = None
        self._tx_queue = queue.Queue()

        # Callbacks
        self.on_emg_data: Optional[Callable] = None    # called with list of floats
        self.on_status: Optional[Callable] = None       # called with string

        # Latest EMG reading (thread-safe via lock)
        self._emg_lock = threading.Lock()
        self._latest_emg = [0.0] * 8

    # ── Connect / Disconnect ──────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            if self.mode == "usb" and self.port.upper() == "SIMULATOR":
                # Connect to local hardware simulator (no Arduino needed)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(("127.0.0.1", 5001))
                sock.settimeout(1.0)
                self._conn = sock
                self.mode = "wifi"  # reuse socket path
                log.info("Connected to LOCAL SIMULATOR on 127.0.0.1:5001")
            elif self.mode == "usb":
                self._conn = serial.Serial(self.port, self.baud, timeout=1)
                log.info(f"Connected via USB: {self.port} @ {self.baud} baud")
            elif self.mode == "wifi":
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.wifi_host, self.wifi_port))
                sock.settimeout(1.0)
                self._conn = sock
                log.info(f"Connected via WiFi: {self.wifi_host}:{self.wifi_port}")
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            self._running = True
            self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._rx_thread.start()

            # Start TX thread
            self._tx_thread = threading.Thread(target=self._transmit_loop, daemon=True)
            self._tx_thread.start()

            time.sleep(0.5)
            self.send("PING")
            return True

        except Exception as e:
            log.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        self._running = False
        time.sleep(0.2)
        if self._conn:
            try:
                self._conn.close()
            except:
                pass
        log.info("Disconnected")

    # ── Send ──────────────────────────────────────────────────────────────

    def send(self, message: str):
        """Queue a message to send to Arduino."""
        self._tx_queue.put(message + "\n")

    def send_servo_angles(self, angles: list):
        """
        Send 15 servo angles to Arduino.
        angles: list of ints 0-180, length 15
        """
        angles = [int(max(0, min(180, a))) for a in angles]
        # Pad to 15 if short
        while len(angles) < 15:
            angles.append(90)
        msg = "SERVO:" + ",".join(str(a) for a in angles[:15])
        self.send(msg)

    def set_mode(self, mode: int):
        """0=collect, 1=control, 2=test"""
        self.send(f"MODE:{mode}")

    def reset_servos(self):
        self.send("RESET")

    # ── Latest EMG ────────────────────────────────────────────────────────

    def get_emg(self) -> list:
        """Returns latest EMG RMS values (thread-safe)."""
        with self._emg_lock:
            return list(self._latest_emg)

    # ── Internal Threads ──────────────────────────────────────────────────

    def _receive_loop(self):
        buffer = ""
        while self._running:
            try:
                if self.mode == "usb":
                    data = self._conn.readline().decode("utf-8", errors="ignore")
                else:
                    chunk = self._conn.recv(256).decode("utf-8", errors="ignore")
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        data = line
                        self._parse_line(data.strip())
                    continue

                self._parse_line(data.strip())

            except serial.SerialException as e:
                log.error(f"Serial error: {e}")
                break
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    log.error(f"RX error: {e}")
                break

    def _parse_line(self, line: str):
        if not line:
            return

        if line.startswith("EMG:"):
            try:
                vals = [float(x) for x in line[4:].split(",")]
                with self._emg_lock:
                    self._latest_emg = vals
                if self.on_emg_data:
                    self.on_emg_data(vals)
            except Exception as e:
                log.warning(f"EMG parse error: {e} | line: {line}")

        elif line.startswith("STATUS:"):
            msg = line[7:]
            log.info(f"Arduino: {msg}")
            if self.on_status:
                self.on_status(msg)

        elif line.startswith("TEST:"):
            log.info(f"Test mode: {line[5:]}")

    def _transmit_loop(self):
        while self._running:
            try:
                msg = self._tx_queue.get(timeout=0.1)
                raw = msg.encode("utf-8")
                if self.mode == "usb":
                    self._conn.write(raw)
                else:
                    self._conn.sendall(raw)
            except queue.Empty:
                continue
            except Exception as e:
                if self._running:
                    log.error(f"TX error: {e}")


# ── Convenience factory ───────────────────────────────────────────────────────

def connect_usb(port="COM3") -> ArduinoConnection:
    """Quick connect via USB. Change port to match your Device Manager."""
    conn = ArduinoConnection(mode="usb", port=port)
    if conn.connect():
        return conn
    raise ConnectionError(f"Could not connect to Arduino on {port}")


def connect_wifi(host="192.168.1.100") -> ArduinoConnection:
    """Quick connect via WiFi (ESP32 bridge)."""
    conn = ArduinoConnection(mode="wifi", wifi_host=host)
    if conn.connect():
        return conn
    raise ConnectionError(f"Could not connect to ESP32 at {host}")


if __name__ == "__main__":
    # Test USB connection
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    print(f"Testing connection on {port}...")

    conn = ArduinoConnection(mode="usb", port=port)

    def on_emg(vals):
        bars = " ".join(f"Ch{i}:{'█' * int(v/20):<10}" for i, v in enumerate(vals))
        print(f"\r{bars}", end="", flush=True)

    conn.on_emg_data = on_emg

    if conn.connect():
        print("Connected! Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nDisconnecting...")
            conn.disconnect()
    else:
        print("Connection failed.")
