"""
BioForge - EMG Data Collection & Labeling Tool
================================================
Records labeled EMG data for training the gesture classifier.

Usage:
    python collect_data.py --port COM3 --output data/session_001.csv

Gestures to record (can add more):
    0  = REST           (hand open, relaxed)
    1  = FIST           (close all fingers)
    2  = PINCH          (thumb + index)
    3  = POINT          (index extended)
    4  = THUMBS_UP      (thumb up, fist)
    5  = OPEN_SPREAD    (all fingers spread)
    6  = WRIST_FLEX     (wrist down)
    7  = WRIST_EXT      (wrist up)
    8  = LATERAL_GRIP   (thumb against side of index)
    9  = THREE_FINGER   (thumb + index + middle)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import csv
import time
import argparse
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from comms.arduino_link import ArduinoConnection

# ─── GESTURE DEFINITIONS ─────────────────────────────────────────────────────
GESTURES = {
    0: "REST",
    1: "FIST",
    2: "PINCH",
    3: "POINT",
    4: "THUMBS_UP",
    5: "OPEN_SPREAD",
    6: "WRIST_FLEX",
    7: "WRIST_EXT",
    8: "LATERAL_GRIP",
    9: "THREE_FINGER",
}

# ─── COLLECTION SETTINGS ─────────────────────────────────────────────────────
SAMPLES_PER_GESTURE = 200      # samples per rep (at 40Hz = 5 seconds)
REPS_PER_GESTURE    = 5        # repetitions
REST_BETWEEN_REPS   = 2.0      # seconds rest between reps
COUNTDOWN_SECONDS   = 3        # countdown before each rep
NUM_CHANNELS        = 8        # EMG channels


class DataCollector:
    def __init__(self, port: str, output_path: str):
        self.port = port
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.arduino = ArduinoConnection(mode="usb", port=port)
        self.collecting = False
        self.current_label = -1
        self.buffer = []

        # CSV file
        self.csv_file = None
        self.csv_writer = None

        # Stats
        self.total_samples = 0

    def start(self):
        print("=" * 60)
        print("  BioForge EMG Data Collector")
        print("=" * 60)
        print(f"  Output: {self.output_path}")
        print(f"  Channels: {NUM_CHANNELS}")
        print(f"  Samples/gesture: {SAMPLES_PER_GESTURE} x {REPS_PER_GESTURE} reps")
        print("=" * 60)

        # Connect Arduino
        print(f"\nConnecting to Arduino on {self.port}...")
        self.arduino.on_emg_data = self._on_emg
        if not self.arduino.connect():
            print(f"ERROR: Could not connect. Check that {self.port} is correct.")
            print("  Windows: Open Device Manager -> Ports -> find Arduino")
            return False

        print("Connected!\n")

        # Open CSV
        self._open_csv()

        # Set Arduino to collect mode
        self.arduino.set_mode(0)
        time.sleep(0.5)

        return True

    def collect_all_gestures(self, gesture_ids: list = None):
        """Collect data for specified gestures (or all by default)."""
        if gesture_ids is None:
            gesture_ids = list(GESTURES.keys())

        for gid in gesture_ids:
            self.collect_gesture(gid)
            print()

        print(f"\nCollection complete! Total samples: {self.total_samples}")
        print(f"Saved to: {self.output_path}")
        self._close_csv()
        self.arduino.disconnect()

    def collect_gesture(self, gesture_id: int):
        name = GESTURES.get(gesture_id, f"GESTURE_{gesture_id}")
        print(f"\n{'─'*50}")
        print(f"  GESTURE: {name} (label={gesture_id})")
        print(f"{'─'*50}")
        print("  Position your hand in the correct gesture.")
        input("  Press ENTER when ready...")

        for rep in range(REPS_PER_GESTURE):
            print(f"\n  Rep {rep+1}/{REPS_PER_GESTURE}")

            # Countdown
            for i in range(COUNTDOWN_SECONDS, 0, -1):
                print(f"    HOLD in {i}s...", end="\r")
                time.sleep(1)
            print(f"    RECORDING {'█' * 20}", end="\r")

            # Collect samples
            self.current_label = gesture_id
            self.collecting = True
            rep_samples = 0

            start = time.time()
            while rep_samples < SAMPLES_PER_GESTURE:
                # EMG callback fills buffer
                if len(self.buffer) > 0:
                    row = self.buffer.pop(0)
                    self.csv_writer.writerow(row)
                    self.csv_file.flush()
                    rep_samples += 1
                    self.total_samples += 1
                    pct = int(rep_samples / SAMPLES_PER_GESTURE * 20)
                    print(f"    {'█' * pct}{'░' * (20-pct)} {rep_samples}/{SAMPLES_PER_GESTURE}", end="\r")
                time.sleep(0.001)

            self.collecting = False
            print(f"    Done! ({rep_samples} samples)                    ")

            # Rest between reps
            if rep < REPS_PER_GESTURE - 1:
                print(f"    REST for {REST_BETWEEN_REPS}s...")
                time.sleep(REST_BETWEEN_REPS)

    def _on_emg(self, vals: list):
        if self.collecting and self.current_label >= 0:
            timestamp = time.time()
            # Pad to NUM_CHANNELS if needed
            padded = vals[:NUM_CHANNELS] + [0.0] * max(0, NUM_CHANNELS - len(vals))
            row = [timestamp, self.current_label] + padded
            self.buffer.append(row)

    def _open_csv(self):
        headers = ["timestamp", "label"] + [f"ch{i}" for i in range(NUM_CHANNELS)]
        self.csv_file = open(self.output_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(headers)
        print(f"CSV opened: {self.output_path}")

    def _close_csv(self):
        if self.csv_file:
            self.csv_file.close()
            print("CSV saved and closed.")


# ─── FEATURE EXTRACTION ─────────────────────────────────────────────────────

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract time-domain + Higuchi Fractal Dimension features from EMG window.
    window: shape (samples, channels)
    Returns: 1D feature vector
    """
    features = []
    for ch in range(window.shape[1]):
        sig = window[:, ch]

        # Time-domain features
        mav = np.mean(np.abs(sig))                    # Mean Absolute Value
        rms = np.sqrt(np.mean(sig**2))                # RMS
        var = np.var(sig)                             # Variance
        wl  = np.sum(np.abs(np.diff(sig)))            # Waveform Length
        zc  = np.sum(np.diff(np.sign(sig)) != 0)     # Zero Crossings
        ssc = np.sum(np.diff(np.sign(np.diff(sig))) != 0)  # Slope Sign Changes

        # Higuchi Fractal Dimension
        hfd = higuchi_fd(sig, kmax=5)

        features.extend([mav, rms, var, wl, zc, ssc, hfd])

    return np.array(features)


def higuchi_fd(x: np.ndarray, kmax: int = 5) -> float:
    """
    Compute Higuchi's Fractal Dimension.
    This is a key feature from your whiteboard - it measures signal complexity.
    Higher HFD = more complex/irregular signal (like during strong muscle contraction).
    """
    n = len(x)
    L = []
    x_arr = np.array(x)

    for k in range(1, kmax + 1):
        Lk = []
        for m in range(1, k + 1):
            # Build subsequence
            idxs = np.arange(m - 1, n, k)
            if len(idxs) < 2:
                continue
            subseq = x_arr[idxs]
            Lmk = np.sum(np.abs(np.diff(subseq))) * (n - 1) / (k * len(idxs))
            Lk.append(Lmk)
        if Lk:
            L.append(np.mean(Lk))

    if len(L) < 2:
        return 0.0

    # Slope of log(L) vs log(1/k) = fractal dimension
    ks = np.arange(1, len(L) + 1, dtype=float)
    try:
        coeffs = np.polyfit(np.log(ks), np.log(L), 1)
        return abs(coeffs[0])
    except:
        return 0.0


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioForge EMG Data Collector")
    parser.add_argument("--port", type=str, default="COM3",
                        help="Arduino serial port (e.g. COM3 on Windows)")
    parser.add_argument("--output", type=str,
                        default=f"data/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        help="Output CSV file path")
    parser.add_argument("--gestures", type=str, default="all",
                        help="Comma-separated gesture IDs to collect, or 'all'")
    parser.add_argument("--sim", action="store_true",
                        help="Use hardware simulator instead of real Arduino")
    args = parser.parse_args()

    if args.sim:
        args.port = "SIMULATOR"
        print("Running in SIMULATOR mode -- make sure simulator.py is running!
")

    collector = DataCollector(port=args.port, output_path=args.output)

    if not collector.start():
        sys.exit(1)

    if args.gestures == "all":
        gesture_ids = list(GESTURES.keys())
    else:
        gesture_ids = [int(g) for g in args.gestures.split(",")]

    print(f"\nWill collect gestures: {[GESTURES[g] for g in gesture_ids]}")
    print("Make sure EMG sensors are placed correctly on your forearm.")
    input("\nPress ENTER to begin...\n")

    collector.collect_all_gestures(gesture_ids)
