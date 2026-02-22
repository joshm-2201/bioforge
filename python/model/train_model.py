"""
BioForge - ML Model Training Pipeline
=======================================
Pipeline:
  1. Load labeled CSV data
  2. Extract features (MAV, RMS, HFD, etc.)
  3. Train Self-Organizing Map (SOM) for feature visualization/clustering
  4. Train MLP Neural Network for gesture classification
  5. Save trained model for real-time inference

From your whiteboard:
  - SOM (Self-Organizing Map / GSOM variant) for unsupervised clustering
  - Feature: Higuchi Fractal Dimension
  - Normalized Curve Length
  - Best Matching Unit (BMU) selection

Usage:
    python train_model.py --data data/session_001.csv
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# minisom (pip install minisom)
try:
    from minisom import MiniSom
    HAS_MINISOM = True
except ImportError:
    print("WARNING: minisom not installed. Run: pip install minisom")
    print("         SOM visualization will be skipped.")
    HAS_MINISOM = False

log = logging.getLogger("BioForge.Train")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
WINDOW_SIZE     = 40    # samples per feature window (40 samples @ 40Hz = 1 second)
WINDOW_STEP     = 10    # step between windows (75% overlap)
NUM_CHANNELS    = 8
FEATURES_PER_CH = 7     # MAV, RMS, VAR, WL, ZC, SSC, HFD

GESTURES = {
    0: "REST",        1: "FIST",         2: "PINCH",
    3: "POINT",       4: "THUMBS_UP",    5: "OPEN_SPREAD",
    6: "WRIST_FLEX",  7: "WRIST_EXT",    8: "LATERAL_GRIP",
    9: "THREE_FINGER"
}

# Servo angles for each gesture [15 servos: Thumb_F, Thumb_A, Thumb_T, I_MCP, I_PIP, I_DIP, M_MCP, M_PIP, M_DIP, R_MCP, R_PIP, R_DIP, P_MCP, P_PIP, Wrist]
GESTURE_SERVO_MAP = {
    0: [90]*15,                                                    # REST - open
    1: [45,90,30, 0,0,0, 0,0,0, 0,0,0, 0,0,90],                  # FIST
    2: [30,60,20, 30,30,30, 90,90,90, 90,90,90, 90,90,90],        # PINCH
    3: [45,90,30, 90,90,90, 0,0,0, 90,90,90, 90,90,90],           # POINT
    4: [90,45,90, 0,0,0, 0,0,0, 0,0,0, 0,0,90],                   # THUMBS_UP
    5: [90,90,90, 90,90,90, 90,90,90, 90,90,90, 90,90,90],        # OPEN_SPREAD
    6: [90]*14 + [45],                                             # WRIST_FLEX
    7: [90]*14 + [135],                                            # WRIST_EXT
    8: [60,30,60, 0,0,0, 90,90,90, 90,90,90, 90,90,90],           # LATERAL_GRIP
    9: [45,60,30, 30,30,30, 30,30,30, 90,90,90, 90,90,90],        # THREE_FINGER
}


# ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def higuchi_fd(x: np.ndarray, kmax: int = 5) -> float:
    n = len(x)
    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(1, k + 1):
            idxs = np.arange(m - 1, n, k)
            if len(idxs) < 2:
                continue
            subseq = x[idxs]
            Lmk = np.sum(np.abs(np.diff(subseq))) * (n - 1) / (k * len(idxs))
            Lk.append(Lmk)
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return 0.0
    ks = np.arange(1, len(L) + 1, dtype=float)
    try:
        coeffs = np.polyfit(np.log(ks), np.log(np.array(L) + 1e-10), 1)
        return abs(coeffs[0])
    except:
        return 0.0


def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract feature vector from one EMG window. window: (samples, channels)"""
    features = []
    for ch in range(window.shape[1]):
        sig = window[:, ch].astype(float)
        mav = np.mean(np.abs(sig))
        rms = np.sqrt(np.mean(sig**2))
        var = np.var(sig)
        wl  = np.sum(np.abs(np.diff(sig)))
        zc  = float(np.sum(np.diff(np.sign(sig)) != 0))
        ssc = float(np.sum(np.diff(np.sign(np.diff(sig))) != 0))
        hfd = higuchi_fd(sig, kmax=5)
        features.extend([mav, rms, var, wl, zc, ssc, hfd])
    return np.array(features, dtype=np.float32)


def make_windows(df: pd.DataFrame) -> tuple:
    """Slide window over time-series data and extract features."""
    channel_cols = [f"ch{i}" for i in range(NUM_CHANNELS)]
    X, y = [], []

    # Group by label so we don't create windows that cross gesture boundaries
    for label in df["label"].unique():
        chunk = df[df["label"] == label][channel_cols].values
        for start in range(0, len(chunk) - WINDOW_SIZE, WINDOW_STEP):
            window = chunk[start:start + WINDOW_SIZE]
            feat = extract_features(window)
            X.append(feat)
            y.append(int(label))

    return np.array(X), np.array(y)


# ─── SOM ─────────────────────────────────────────────────────────────────────

def train_som(X_scaled: np.ndarray, labels: np.ndarray, som_size: int = 10):
    """
    Train a Self-Organizing Map on the feature data.
    This implements the GSOM-like approach from your whiteboard.
    Returns trained SOM.
    """
    if not HAS_MINISOM:
        return None

    log.info(f"Training SOM ({som_size}x{som_size})...")
    n_features = X_scaled.shape[1]

    som = MiniSom(
        x=som_size, y=som_size,
        input_len=n_features,
        sigma=1.5,
        learning_rate=0.5,
        neighborhood_function='gaussian',
        random_seed=42
    )

    som.random_weights_init(X_scaled)
    som.train_batch(X_scaled, num_iteration=5000, verbose=False)

    log.info("SOM training complete.")

    # Compute quantization error (lower = better fit)
    qe = som.quantization_error(X_scaled)
    log.info(f"SOM quantization error: {qe:.4f}")

    return som


# ─── MLP CLASSIFIER ──────────────────────────────────────────────────────────

def train_mlp(X_train, X_test, y_train, y_test):
    """Train MLP neural network for gesture classification."""
    log.info("Training MLP classifier...")

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,           # L2 regularization
        batch_size=64,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )

    mlp.fit(X_train, y_train)

    # Evaluate
    y_pred = mlp.predict(X_test)
    acc = np.mean(y_pred == y_test)
    log.info(f"Test accuracy: {acc*100:.1f}%")

    # Detailed report
    labels_present = sorted(set(y_test))
    names = [GESTURES.get(l, str(l)) for l in labels_present]
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=names, labels=labels_present))

    return mlp, acc


# ─── SAVE / LOAD MODEL ───────────────────────────────────────────────────────

def save_model(mlp, scaler, som, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = {
        "mlp": mlp,
        "scaler": scaler,
        "som": som,
        "gesture_map": GESTURES,
        "servo_map": GESTURE_SERVO_MAP,
        "num_channels": NUM_CHANNELS,
        "window_size": WINDOW_SIZE,
        "features_per_ch": FEATURES_PER_CH,
        "trained_at": datetime.now().isoformat(),
    }

    path = out / "bioforge_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"Model saved: {path}")
    return str(path)


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── MAIN TRAINING SCRIPT ────────────────────────────────────────────────────

def main(data_path: str, output_dir: str = "models"):
    log.info(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    log.info(f"Dataset: {len(df)} samples, labels: {sorted(df['label'].unique())}")
    print("\nSamples per gesture:")
    for label, count in df.groupby("label").size().items():
        print(f"  {GESTURES.get(label, label)}: {count} samples")

    # Extract features
    log.info("Extracting features...")
    X, y = make_windows(df)
    log.info(f"Feature matrix: {X.shape} | Labels: {len(y)}")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Train SOM
    som = train_som(X_scaled, y)

    # Train MLP
    mlp, acc = train_mlp(X_train, X_test, y_train, y_test)

    # Save
    model_path = save_model(mlp, scaler, som, output_dir)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioForge Model Trainer")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--output", type=str, default="models", help="Output directory for model")
    args = parser.parse_args()
    main(args.data, args.output)
