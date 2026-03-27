import csv
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit(
        "h5py is required. Install dependencies with: pip install h5py numpy scikit-learn"
    ) from exc

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install dependencies with: pip install scikit-learn"
    ) from exc


SAMPLE_RATE_HZ = 2000
WINDOW_MS = 150
STRIDE_MS = 50
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)
STRIDE_SAMPLES = int(SAMPLE_RATE_HZ * STRIDE_MS / 1000)
CHANNEL_COUNT = 16
TRIAL_SECONDS = 5
TRIAL_SAMPLES = SAMPLE_RATE_HZ * TRIAL_SECONDS
DISPLAY_SCALE = 1e3
RANDOM_SEED = 42
MODEL_PATH = Path(__file__).with_name("emg_lda_model.pkl")

FILE_CANDIDATES = [
    Path(__file__).with_name("emg_data.hdf5"),
    Path.home() / "Downloads" / "emg_data.hdf5",
]
TRIALS_CSV_CANDIDATES = [
    Path(__file__).with_name("trials.csv"),
    Path.home() / "Downloads" / "trials.csv",
]

GRASP_NAMES = {
    1: "power",
    2: "lateral",
    3: "pointer",
    4: "tripod",
    5: "open",
    6: "rest",
}



def find_existing_path(candidates, description):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {description}.")



def load_trial_labels(csv_path):
    labels = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            trial_id_text = row.get("trial ID") or row.get("trial_id") or row.get("trial") or row.get("row_number")
            if not trial_id_text:
                continue

            try:
                trial_id = int(trial_id_text)
                grasp_code = int(row.get("grasp", "0"))
            except ValueError:
                continue

            labels[trial_id] = {
                "grasp_code": grasp_code,
                "grasp_name": GRASP_NAMES.get(grasp_code, str(grasp_code)),
                "position": row.get("target_position") or row.get("target position") or row.get("position") or "?",
                "trial_number": row.get("trial_no") or row.get("trial number") or row.get("trial_number") or "?",
                "block": row.get("block") or row.get("block number") or "?",
                "source": csv_path.name,
            }
    if not labels:
        raise RuntimeError("No usable labels were read from trials.csv.")
    return labels



def normalize_trial_length(trial):
    sample_count = trial.shape[1]
    if sample_count == TRIAL_SAMPLES:
        return trial, "kept"
    if sample_count > TRIAL_SAMPLES:
        return trial[:, :TRIAL_SAMPLES], "trimmed"

    pad_width = TRIAL_SAMPLES - sample_count
    pad_values = np.repeat(trial[:, -1:], pad_width, axis=1)
    padded = np.concatenate([trial, pad_values], axis=1)
    return padded, "padded"



def load_trials(hdf5_path):
    with h5py.File(hdf5_path, "r") as h5_file:
        trial_keys = sorted(h5_file.keys(), key=lambda key: int(key) if str(key).isdigit() else str(key))
        if not trial_keys:
            raise RuntimeError("No trial datasets found in emg_data.hdf5.")

        first_trial = np.asarray(h5_file[trial_keys[0]][()], dtype=np.float32)
        if first_trial.ndim != 2:
            raise RuntimeError("Expected each trial to be a 2D array of channels x samples.")
        if CHANNEL_COUNT not in first_trial.shape:
            raise RuntimeError("Expected one trial dimension to have 16 EMG channels.")

        channels_first = first_trial.shape[0] == CHANNEL_COUNT
        trials = []
        kept = 0
        trimmed = 0
        padded = 0

        for key in trial_keys:
            trial = np.asarray(h5_file[key][()], dtype=np.float32)
            if trial.ndim != 2:
                continue

            if channels_first:
                if trial.shape[0] != CHANNEL_COUNT:
                    continue
                arranged = trial
            else:
                if trial.shape[1] != CHANNEL_COUNT:
                    continue
                arranged = trial.T

            normalized, action = normalize_trial_length(arranged)
            if action == "kept":
                kept += 1
            elif action == "trimmed":
                trimmed += 1
            elif action == "padded":
                padded += 1

            trials.append((key, normalized * DISPLAY_SCALE))

    if not trials:
        raise RuntimeError("No valid 16-channel trials were loaded from emg_data.hdf5.")
    return trials, {"kept": kept, "trimmed": trimmed, "padded": padded}



def zero_crossings(signal):
    return np.sum((signal[:-1] * signal[1:]) < 0)



def slope_sign_changes(signal):
    diff1 = np.diff(signal)
    return np.sum((diff1[:-1] * diff1[1:]) < 0)



def extract_window_features(window):
    features = []
    for channel in window:
        mean_abs = np.mean(np.abs(channel))
        waveform_length = np.sum(np.abs(np.diff(channel)))
        zc = zero_crossings(channel)
        ssc = slope_sign_changes(channel)
        rms = np.sqrt(np.mean(channel ** 2))
        variance = np.var(channel)
        features.extend([mean_abs, waveform_length, zc, ssc, rms, variance])
    return np.asarray(features, dtype=np.float32)



def build_dataset(trials, labels):
    X = []
    y = []
    groups = []
    window_meta = []
    labeled_trials = 0

    for trial_idx, (trial_key, trial_signal) in enumerate(trials):
        try:
            trial_id = int(trial_key)
        except ValueError:
            trial_id = trial_idx

        label = labels.get(trial_id)
        if label is None:
            continue

        labeled_trials += 1
        grasp_code = label["grasp_code"]
        for start in range(0, TRIAL_SAMPLES - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            stop = start + WINDOW_SAMPLES
            window = trial_signal[:, start:stop]
            X.append(extract_window_features(window))
            y.append(grasp_code)
            groups.append(trial_id)
            window_meta.append(
                {
                    "trial_id": trial_id,
                    "window_start_sample": start,
                    "window_stop_sample": stop,
                    "grasp_name": label["grasp_name"],
                    "position": label["position"],
                    "block": label["block"],
                }
            )

    if not X:
        raise RuntimeError("No labeled windows were created. Check that the labels match the trial IDs.")

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        np.asarray(groups),
        window_meta,
        labeled_trials,
    )



def print_confusion(confusion, class_order):
    names = [GRASP_NAMES.get(class_id, str(class_id)) for class_id in class_order]
    header = "pred ->".ljust(12) + " ".join(name[:10].rjust(10) for name in names)
    print(header)
    for row_name, row_values in zip(names, confusion):
        row_text = row_name[:10].ljust(12) + " ".join(str(value).rjust(10) for value in row_values)
        print(row_text)



def majority_vote_trial_accuracy(test_idx, predictions, y, groups):
    per_trial_preds = defaultdict(list)
    per_trial_truth = {}

    for local_i, dataset_idx in enumerate(test_idx):
        trial_id = int(groups[dataset_idx])
        per_trial_preds[trial_id].append(int(predictions[local_i]))
        per_trial_truth[trial_id] = int(y[dataset_idx])

    trial_truth = []
    trial_pred = []
    for trial_id in sorted(per_trial_preds):
        voted_pred = Counter(per_trial_preds[trial_id]).most_common(1)[0][0]
        trial_pred.append(voted_pred)
        trial_truth.append(per_trial_truth[trial_id])

    accuracy = accuracy_score(trial_truth, trial_pred)
    return accuracy, trial_truth, trial_pred



def save_model(classifier, hdf5_path, trials_csv, feature_count):
    payload = {
        "model": classifier,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_ms": WINDOW_MS,
        "stride_ms": STRIDE_MS,
        "window_samples": WINDOW_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "channel_count": CHANNEL_COUNT,
        "trial_seconds": TRIAL_SECONDS,
        "display_scale": DISPLAY_SCALE,
        "feature_count": feature_count,
        "grasp_names": GRASP_NAMES,
        "hdf5_path": str(hdf5_path),
        "trials_csv": str(trials_csv),
    }
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(payload, model_file)



def main():
    hdf5_path = find_existing_path(FILE_CANDIDATES, "emg_data.hdf5")
    trials_csv = find_existing_path(TRIALS_CSV_CANDIDATES, "trials.csv")
    labels = load_trial_labels(trials_csv)
    trials, trial_stats = load_trials(hdf5_path)
    X, y, groups, window_meta, labeled_trials = build_dataset(trials, labels)

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X[train_idx], y[train_idx])
    predictions = classifier.predict(X[test_idx])

    window_accuracy = accuracy_score(y[test_idx], predictions)
    class_order = sorted(np.unique(np.concatenate([y[test_idx], predictions])))
    window_confusion = confusion_matrix(y[test_idx], predictions, labels=class_order)

    trial_accuracy, trial_truth, trial_pred = majority_vote_trial_accuracy(test_idx, predictions, y, groups)
    trial_confusion = confusion_matrix(trial_truth, trial_pred, labels=class_order)

    save_model(classifier, hdf5_path, trials_csv, X.shape[1])

    train_trials = len(set(groups[train_idx].tolist()))
    test_trials = len(set(groups[test_idx].tolist()))

    print("EMG Gesture Classifier Baseline")
    print(f"HDF5 file: {hdf5_path}")
    print(f"Label source: {trials_csv}")
    print(f"Trials loaded from HDF5: {len(trials)}")
    print(f"Trials with labels used: {labeled_trials}")
    print(f"Trial length handling: kept={trial_stats['kept']}, trimmed={trial_stats['trimmed']}, padded={trial_stats['padded']}")
    print(f"Windows created: {len(X)}")
    print(f"Window size: {WINDOW_MS} ms ({WINDOW_SAMPLES} samples)")
    print(f"Stride: {STRIDE_MS} ms ({STRIDE_SAMPLES} samples)")
    print(f"Features per window: {X.shape[1]}")
    print(f"Train trials: {train_trials}")
    print(f"Test trials: {test_trials}")
    print(f"Window-level accuracy: {window_accuracy:.4f}")
    print(f"Trial-level accuracy (majority vote): {trial_accuracy:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print()
    print("Window-level confusion matrix")
    print_confusion(window_confusion, class_order)
    print()
    print("Trial-level confusion matrix")
    print_confusion(trial_confusion, class_order)
    print()
    print("First 5 test windows")
    for local_i, dataset_idx in enumerate(test_idx[:5]):
        meta = window_meta[dataset_idx]
        true_name = GRASP_NAMES.get(int(y[dataset_idx]), str(y[dataset_idx]))
        pred_name = GRASP_NAMES.get(int(predictions[local_i]), "?")
        print(
            f"trial {meta['trial_id']} | {meta['window_start_sample']}:{meta['window_stop_sample']} | "
            f"true={true_name} | pred={pred_name} | position={meta['position']} | block={meta['block']}"
        )


if __name__ == "__main__":
    main()
