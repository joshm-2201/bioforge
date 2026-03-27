import csv
import time
from pathlib import Path
import tkinter as tk

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit(
        "h5py is required to read emg_data.hdf5. Install it with: pip install h5py numpy"
    ) from exc


WIDTH = 980
HEIGHT = 520
CHANNEL_COUNT = 16
SAMPLE_RATE_HZ = 2000
TRIAL_SECONDS = 5
TRIAL_SAMPLES = SAMPLE_RATE_HZ * TRIAL_SECONDS
UPDATE_MS = 15
DISPLAY_SCALE = 1e3
PADDING_LEFT = 90
PADDING_RIGHT = 20
PADDING_TOP = 20
PADDING_BOTTOM = 55
FILE_CANDIDATES = [
    Path(__file__).with_name("emg_data.hdf5"),
    Path.home() / "Downloads" / "emg_data.hdf5",
]
TRIALS_CSV_CANDIDATES = [
    Path(__file__).with_name("trials.csv"),
    Path.home() / "Downloads" / "trials.csv",
]
GRASP_NAMES = {
    "1": "power",
    "2": "lateral",
    "3": "pointer",
    "4": "tripod",
    "5": "open",
    "6": "rest",
}
PARTICIPANT1_DAY1_BLOCK1_SEQUENCE = [3, 1, 4, 5, 2, 5, 4, 2, 3, 1, 4, 5, 1, 2, 3, 6, 6, 6, 6, 6, 2, 3, 5, 1, 4, 1, 2, 3, 4, 5]
CONFIG_PLUS_POSITIONS = [2, 4, 5, 6, 8]


def find_existing_path(candidates, description):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {description}.")


EMG_FILE = find_existing_path(FILE_CANDIDATES, "emg_data.hdf5")
TRIALS_CSV = next((path for path in TRIALS_CSV_CANDIDATES if path.exists()), None)


def grasp_name(grasp_code):
    text = str(grasp_code).strip()
    return GRASP_NAMES.get(text, text or "?")



def format_axis_value(value):
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.3f}"
    if magnitude >= 0.01:
        return f"{value:.4f}"
    if magnitude >= 0.0001:
        return f"{value:.6f}"
    return f"{value:.2e}"



def build_participant1_fallback_labels():
    labels = {}
    for block, grasp_code in enumerate(PARTICIPANT1_DAY1_BLOCK1_SEQUENCE):
        position = CONFIG_PLUS_POSITIONS[block % len(CONFIG_PLUS_POSITIONS)]
        for trial_no in range(5):
            row_number = block * 5 + trial_no
            labels[row_number] = {
                "position": str(position),
                "gesture": grasp_name(grasp_code),
                "trial_number": str(trial_no),
                "block": str(block),
                "source": "participant1_day1_block1 fallback",
            }
    return labels



def load_trial_labels(csv_path):
    if csv_path is None:
        return build_participant1_fallback_labels()

    labels = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            trial_id_text = row.get("trial ID") or row.get("trial_id") or row.get("trial") or row.get("row_number")
            if not trial_id_text:
                continue
            try:
                trial_id = int(trial_id_text)
            except ValueError:
                continue

            raw_grasp = row.get("grasp") or "?"
            labels[trial_id] = {
                "position": row.get("target_position") or row.get("target position") or row.get("position") or "?",
                "gesture": grasp_name(raw_grasp),
                "trial_number": row.get("trial_no") or row.get("trial number") or row.get("trial_number") or "?",
                "block": row.get("block") or row.get("block number") or "?",
                "source": csv_path.name,
            }
    return labels


TRIAL_LABELS = load_trial_labels(TRIALS_CSV)


with h5py.File(EMG_FILE, "r") as h5_file:
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

        clipped = arranged[:, :TRIAL_SAMPLES] * DISPLAY_SCALE
        if clipped.shape[1] == 0:
            continue
        trials.append((key, clipped))

if not trials:
    raise RuntimeError("No valid 16-channel trials were loaded from emg_data.hdf5.")


plot_height = (HEIGHT - 80) - PADDING_TOP - PADDING_BOTTOM
plot_width = WIDTH - PADDING_LEFT - PADDING_RIGHT
left = PADDING_LEFT
right = WIDTH - PADDING_RIGHT
top = PADDING_TOP
bottom = (HEIGHT - 80) - PADDING_BOTTOM

trial_index = 0
sample_index = 0
selected_channel = 0
current_y_min = -1.0
current_y_max = 1.0
playback_active = True
loop_id = None
playback_start_time = None

root = tk.Tk()
root.title(f"EMG Trial Viewer - {EMG_FILE.name}")
root.geometry(f"{WIDTH}x{HEIGHT}")
root.resizable(False, False)

info_var = tk.StringVar()
controls = tk.Frame(root)
controls.pack(fill="x", padx=8, pady=6)

channel_var = tk.IntVar(value=1)


def current_trial_number():
    key, _ = trials[trial_index]
    try:
        return int(key)
    except ValueError:
        return trial_index


status_label = tk.Label(root, textvariable=info_var, anchor="w", padx=10)
status_label.pack(fill="x")

canvas = tk.Canvas(root, bg="black", width=WIDTH, height=HEIGHT - 80, highlightthickness=0)
canvas.pack(fill="both", expand=True)


def get_trial_label_text():
    trial_no = current_trial_number()
    label = TRIAL_LABELS.get(trial_no)
    if not label:
        return "labels unavailable"
    return (
        f"position {label['position']} | gesture {label['gesture']} | "
        f"repeat {label['trial_number']} | block {label['block']}"
    )



def label_source_text():
    label = TRIAL_LABELS.get(current_trial_number())
    if not label:
        return ""
    return label.get("source", "")



def update_status():
    elapsed = sample_index / SAMPLE_RATE_HZ
    state = "playing" if playback_active else "trial complete"
    source_text = label_source_text()
    source_suffix = f" | labels {source_text}" if source_text else ""
    info_var.set(
        f"trial {trial_index + 1}/{len(trials)} | dataset key {trials[trial_index][0]} | "
        f"channel {selected_channel + 1}/{CHANNEL_COUNT} | scale x10^3 | auto y-scale | "
        f"elapsed {elapsed:.2f}/{TRIAL_SECONDS:.2f}s | {state} | {get_trial_label_text()}{source_suffix}"
    )



def current_channel_samples():
    return trials[trial_index][1][selected_channel]



def displayed_channel_samples():
    samples = current_channel_samples()
    end_index = max(1, sample_index)
    return samples[:end_index]



def recompute_view_range(displayed_samples):
    global current_y_min, current_y_max
    window = np.asarray(displayed_samples, dtype=np.float32)
    minimum = float(window.min())
    maximum = float(window.max())
    if minimum == maximum:
        minimum -= 1.0
        maximum += 1.0
    pad = max((maximum - minimum) * 0.1, 1e-6)
    current_y_min = minimum - pad
    current_y_max = maximum + pad



def value_to_y(value):
    normalized = (value - current_y_min) / (current_y_max - current_y_min)
    return PADDING_TOP + (1.0 - normalized) * plot_height



def draw_grid():
    canvas.delete("all")
    canvas.create_rectangle(left, top, right, bottom, outline="#666666")

    for i in range(1, 10):
        x = left + (plot_width / 10) * i
        canvas.create_line(x, top, x, bottom, fill="#1f1f1f")

    for i in range(1, 8):
        y = top + (plot_height / 8) * i
        canvas.create_line(left, y, right, y, fill="#1f1f1f")

    for value in np.linspace(current_y_min, current_y_max, 5):
        y = value_to_y(float(value))
        canvas.create_line(left - 5, y, left, y, fill="white")
        canvas.create_text(left - 10, y, text=format_axis_value(float(value)), fill="white", anchor="e")

    if current_y_min <= 0.0 <= current_y_max:
        zero_y = value_to_y(0.0)
        canvas.create_line(left, zero_y, right, zero_y, fill="#4f7cff", dash=(4, 4))
        canvas.create_text(right - 6, zero_y - 8, text="0", fill="#7f9cff", anchor="e")

    for second in range(TRIAL_SECONDS + 1):
        x = left + (plot_width * second / TRIAL_SECONDS)
        canvas.create_line(x, bottom, x, bottom + 5, fill="white")
        label = f"{second}s"
        anchor = "center"
        if second == 0:
            anchor = "w"
        elif second == TRIAL_SECONDS:
            anchor = "e"
        canvas.create_text(x, bottom + 18, text=label, fill="white", anchor=anchor)

    canvas.create_text(18, top, text="EMG", fill="white", anchor="nw")
    canvas.create_text(18, top + 18, text="x10^3", fill="white", anchor="nw")

    progress_x = left + (plot_width * (sample_index / TRIAL_SAMPLES))
    canvas.create_line(progress_x, top, progress_x, bottom, fill="#ffd966")

    gesture_line = get_trial_label_text()
    if gesture_line != "labels unavailable":
        canvas.create_text(left, top - 6, text=gesture_line, fill="white", anchor="sw")

    if not playback_active:
        canvas.create_text(
            (left + right) / 2,
            top + 20,
            text="Trial complete",
            fill="#ffd966",
            font=("TkDefaultFont", 14, "bold"),
        )



def redraw_waveform():
    displayed_samples = displayed_channel_samples()
    recompute_view_range(displayed_samples)
    draw_grid()

    points = []
    for i, sample in enumerate(displayed_samples):
        x = left + (plot_width * i / (TRIAL_SAMPLES - 1))
        y = value_to_y(float(sample))
        points.extend([x, y])

    if len(points) >= 4:
        canvas.create_line(points, fill="lime", width=2, smooth=False)



def cancel_loop():
    global loop_id
    if loop_id is not None:
        root.after_cancel(loop_id)
        loop_id = None



def schedule_update():
    global loop_id
    cancel_loop()
    loop_id = root.after(UPDATE_MS, update)



def reset_view(start_playback):
    global sample_index, playback_active, playback_start_time
    sample_index = 0
    playback_active = start_playback
    playback_start_time = time.perf_counter() if start_playback else None
    update_status()
    redraw_waveform()
    if playback_active:
        schedule_update()
    else:
        cancel_loop()



def restart_current_trial():
    reset_view(start_playback=True)



def change_channel(*_args):
    global selected_channel
    selected_channel = max(0, min(CHANNEL_COUNT - 1, channel_var.get() - 1))
    restart_current_trial()



def change_trial(delta):
    global trial_index
    trial_index = (trial_index + delta) % len(trials)
    restart_current_trial()



def next_trial():
    change_trial(1)



def previous_trial():
    change_trial(-1)



def update():
    global sample_index, playback_active, loop_id
    loop_id = None

    if not playback_active:
        update_status()
        redraw_waveform()
        return

    elapsed_seconds = time.perf_counter() - playback_start_time
    target_index = min(TRIAL_SAMPLES, int(elapsed_seconds * SAMPLE_RATE_HZ))

    if target_index > sample_index:
        sample_index = target_index

    if sample_index >= TRIAL_SAMPLES:
        sample_index = TRIAL_SAMPLES
        playback_active = False

    update_status()
    redraw_waveform()
    if playback_active:
        schedule_update()



def on_close():
    cancel_loop()
    root.destroy()


prev_button = tk.Button(controls, text="Prev Trial", command=previous_trial)
prev_button.pack(side="left", padx=4)
next_button = tk.Button(controls, text="Next Trial", command=next_trial)
next_button.pack(side="left", padx=4)
replay_button = tk.Button(controls, text="Replay Trial", command=restart_current_trial)
replay_button.pack(side="left", padx=4)
channel_label = tk.Label(controls, text="Channel")
channel_label.pack(side="left", padx=(16, 4))
channel_spin = tk.Spinbox(controls, from_=1, to=CHANNEL_COUNT, textvariable=channel_var, width=5, command=change_channel)
channel_spin.pack(side="left")
channel_var.trace_add("write", change_channel)

root.protocol("WM_DELETE_WINDOW", on_close)
reset_view(start_playback=True)
root.mainloop()
