"""
BioForge - Real-Time Inference Engine
=======================================
Loads trained model and runs continuous gesture prediction
from live EMG data, sending servo commands to Arduino.

Usage:
    python inference.py --model models/bioforge_model.pkl --port COM3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pickle
import time
import logging
import argparse
import threading
from collections import deque

from comms.arduino_link import ArduinoConnection
from data_collection.collect_data import extract_features

log = logging.getLogger("BioForge.Inference")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class GestureInferenceEngine:
    """
    Continuously reads EMG, extracts features in a sliding window,
    classifies gesture, and sends servo commands.
    """

    def __init__(self,
                 model_path: str,
                 arduino: ArduinoConnection,
                 window_size: int = 40,
                 step_size: int = 5,
                 smoothing: int = 5):
        """
        Args:
            model_path:  Path to saved .pkl model bundle
            arduino:     Connected ArduinoConnection
            window_size: Samples per inference window (must match training)
            step_size:   Samples between inferences (lower = more responsive)
            smoothing:   Number of consecutive same predictions to confirm gesture
        """
        self.arduino = arduino
        self.window_size = window_size
        self.step_size = step_size
        self.smoothing = smoothing

        # Load model
        log.info(f"Loading model: {model_path}")
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.mlp          = bundle["mlp"]
        self.scaler       = bundle["scaler"]
        self.gesture_map  = bundle["gesture_map"]
        self.servo_map    = bundle["servo_map"]
        self.num_channels = bundle.get("num_channels", 8)

        log.info(f"Model loaded. Gestures: {list(self.gesture_map.values())}")

        # Rolling EMG buffer
        self._emg_buffer = deque(maxlen=window_size * 2)
        self._sample_count = 0

        # Prediction smoothing (vote over last N predictions)
        self._pred_history = deque(maxlen=smoothing)

        # State
        self.current_gesture = -1
        self.current_gesture_name = "UNKNOWN"
        self.confidence = 0.0
        self.running = False

        # Callbacks
        self.on_gesture_change = None   # called when gesture changes

        # Stats
        self.inference_count = 0
        self.last_inference_time = 0.0

    def start(self):
        """Start the inference loop."""
        self.running = True
        self.arduino.on_emg_data = self._on_emg_sample
        self.arduino.set_mode(1)  # control mode
        log.info("Inference engine started. Waiting for EMG data...")

        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()

    def stop(self):
        self.running = False
        self.arduino.set_mode(0)
        self.arduino.reset_servos()
        log.info("Inference engine stopped.")

    def _on_emg_sample(self, vals: list):
        """Called by arduino connection on each new EMG reading (40Hz)."""
        padded = vals[:self.num_channels] + [0.0] * max(0, self.num_channels - len(vals))
        self._emg_buffer.append(padded)
        self._sample_count += 1

    def _inference_loop(self):
        """Runs inference every step_size new samples."""
        last_processed = 0

        while self.running:
            # Wait until we have a full window
            if len(self._emg_buffer) < self.window_size:
                time.sleep(0.01)
                continue

            # Check if enough new samples arrived
            new_samples = self._sample_count - last_processed
            if new_samples < self.step_size:
                time.sleep(0.005)
                continue

            last_processed = self._sample_count

            # Get current window
            window = np.array(list(self._emg_buffer)[-self.window_size:])

            # Feature extraction
            try:
                feat = extract_features(window)
                feat_scaled = self.scaler.transform(feat.reshape(1, -1))
            except Exception as e:
                log.warning(f"Feature extraction error: {e}")
                continue

            # Classify
            try:
                probs = self.mlp.predict_proba(feat_scaled)[0]
                pred_class = int(np.argmax(probs))
                self.confidence = float(probs[pred_class])
            except Exception as e:
                log.warning(f"Inference error: {e}")
                continue

            # Smooth predictions
            self._pred_history.append(pred_class)
            smoothed = self._get_smoothed_gesture()

            # Update if changed
            if smoothed != self.current_gesture:
                self.current_gesture = smoothed
                self.current_gesture_name = self.gesture_map.get(smoothed, str(smoothed))
                self._send_gesture(smoothed)

                if self.on_gesture_change:
                    self.on_gesture_change(smoothed, self.current_gesture_name, self.confidence)

            self.inference_count += 1
            self.last_inference_time = time.time()

    def _get_smoothed_gesture(self) -> int:
        """Majority vote over prediction history."""
        if not self._pred_history:
            return 0
        from collections import Counter
        counts = Counter(self._pred_history)
        return counts.most_common(1)[0][0]

    def _send_gesture(self, gesture_id: int):
        """Send servo positions for this gesture to Arduino."""
        angles = self.servo_map.get(gesture_id, [90] * 15)
        self.arduino.send_servo_angles(angles)
        log.info(f"Gesture: {self.current_gesture_name} "
                 f"(conf={self.confidence:.2f}) → servos sent")

    def get_status(self) -> dict:
        return {
            "gesture": self.current_gesture,
            "gesture_name": self.current_gesture_name,
            "confidence": round(self.confidence, 3),
            "buffer_size": len(self._emg_buffer),
            "inference_count": self.inference_count,
            "emg_latest": list(self._emg_buffer[-1]) if self._emg_buffer else [],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioForge Real-Time Inference")
    parser.add_argument("--model", type=str, default="models/bioforge_model.pkl")
    parser.add_argument("--port", type=str, default="COM3")
    parser.add_argument("--wifi", type=str, default=None, help="ESP32 IP for wireless mode")
    args = parser.parse_args()

    # Connect Arduino
    if args.wifi:
        arduino = ArduinoConnection(mode="wifi", wifi_host=args.wifi)
    else:
        arduino = ArduinoConnection(mode="usb", port=args.port)

    if not arduino.connect():
        print("ERROR: Could not connect to Arduino.")
        sys.exit(1)

    # Start inference
    engine = GestureInferenceEngine(
        model_path=args.model,
        arduino=arduino,
        window_size=40,
        step_size=5,
        smoothing=5
    )

    def on_gesture(gid, name, conf):
        bar = "█" * int(conf * 20)
        print(f"  {name:<15} [{bar:<20}] {conf:.0%}")

    engine.on_gesture_change = on_gesture
    engine.start()

    print("\nRunning real-time inference. Press Ctrl+C to stop.\n")
    print(f"  {'GESTURE':<15} {'CONFIDENCE'}")
    print(f"  {'─'*15} {'─'*22}")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        engine.stop()
        arduino.disconnect()
