# BioForge — Bionic Hand System
## Complete Developer Guide

---

## Project Overview

BioForge is a student-built bionic hand that uses EMG (electromyography) signals from forearm muscles to control a robotic hand with up to 15 servo-driven degrees of freedom. An AI model trained on your own muscle data classifies hand gestures in real time.

---

## System Architecture

```
[Forearm Muscles]
       │
[MyoWare 2.0 Sensors x2]  ←── adhesive electrodes on skin
       │
[Arduino Mega 2560]  ←── reads analog EMG at 200Hz, drives servos
       │  Serial USB
[Raspberry Pi]  ←── runs Python AI model, GUI dashboard
       │  (optional)
[ESP32 Bridge]  ←── wireless WiFi link for untethered use
```

---

## Hardware Wiring Guide

### MyoWare 2.0 Sensor Connections
```
MyoWare 2.0 Pin  →  Arduino Mega Pin
─────────────────────────────────────
RAW              →  A0  (sensor 1)
RAW              →  A1  (sensor 2)
+                →  3.3V  (NOT 5V! MyoWare needs 3.3V)
-                →  GND
REF electrode    →  bony part of elbow (neutral reference)
```

### Servo Connections (Arduino Mega)
```
Servo            →  PWM Pin  →  Purpose
────────────────────────────────────────────
Thumb Flex       →  2        →  Thumb curl
Thumb Abduct     →  3        →  Thumb side-to-side
Thumb Tip        →  4        →  Thumb DIP joint
Index MCP        →  5        →  Index base knuckle
Index PIP        →  6        →  Index middle joint
Index DIP        →  7        →  Index fingertip
Middle MCP       →  8        →  Middle base
Middle PIP       →  9        →  Middle middle
Middle DIP       →  10       →  Middle tip
Ring MCP         →  11       →  Ring base
Ring PIP         →  12       →  Ring middle
Ring DIP         →  13       →  Ring tip
Pinky MCP        →  44       →  Pinky base
Pinky PIP        →  45       →  Pinky middle
Wrist Flex       →  46       →  Wrist up/down
```

### Servo Power
⚠️ CRITICAL: NEVER power servos from the Arduino 5V pin — it will damage the Arduino.
Use the plug-in 5V power supply (ALITOVE). Connect:
- Power supply + → breadboard + rail → all servo red wires
- Power supply GND → breadboard − rail → all servo black wires → Arduino GND
- Arduino GND → breadboard GND rail (common ground!)
- Servo signal wire (yellow/white) → Arduino PWM pin

### ESP32 Connections
```
ESP32 GPIO17 (TX2)  →  Arduino Mega pin 19 (RX1)
ESP32 GPIO16 (RX2)  →  Arduino Mega pin 18 (TX1)
ESP32 GND           →  Arduino GND
ESP32 3.3V          →  ESP32 onboard 3.3V (don't power from Mega)
```

---

## Software Setup (Windows)

### Step 1 — Install Python
Download Python 3.11 from python.org. During install, CHECK "Add Python to PATH".

### Step 2 — Install Arduino IDE
Download Arduino IDE 2.x from arduino.cc.

### Step 3 — Install Python dependencies
Open Command Prompt and run:
```bash
cd bioforge/python
pip install -r requirements.txt
```

### Step 4 — Flash Arduino firmware
1. Open `arduino/bioforge_main/bioforge_main.ino` in Arduino IDE
2. Select board: Tools → Board → Arduino Mega 2560
3. Select port: Tools → Port → COM3 (or whichever shows up)
4. Click Upload (→ button)

### Step 5 — Flash ESP32 (optional, for wireless)
1. Install ESP32 board support in Arduino IDE:
   File → Preferences → add this URL:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   Then: Tools → Board Manager → search "esp32" → install
2. Open `esp32/esp32_bridge/esp32_bridge.ino`
3. Edit WIFI_SSID and WIFI_PASS with your WiFi credentials
4. Select board: ESP32 Dev Module
5. Upload

---

## Workflow: Step-by-Step Usage

### Phase 1: Test Hardware (No AI)
Run this first to make sure servos and EMG are working:

```bash
# Test USB connection and see live EMG
cd bioforge/python
python comms/arduino_link.py --port COM3
```
You should see 8 EMG channel bars updating as you flex your arm.

Then open the GUI (no model needed yet):
```bash
python gui/gui_dashboard.py --port COM3
```

Use Mode=Test in the GUI to cycle through servo positions and verify each finger moves.

### Phase 2: Collect Training Data

```bash
python data_collection/collect_data.py --port COM3 --output data/session_001.csv
```

The tool will:
1. Tell you which gesture to make
2. Count down 3 seconds
3. Record 5 reps × 5 seconds = 25 seconds per gesture
4. Repeat for all 10 gestures (total ~5 minutes)

Tips for good data:
- Place electrodes on the thickest part of your forearm (muscle belly)
- Keep electrodes in the same position every session
- Hold each gesture firmly and consistently
- Don't move during the countdown
- Collect multiple sessions (run the script multiple times) for better accuracy

### Phase 3: Train the AI Model

```bash
python model/train_model.py --data data/session_001.csv --output models/
```

This will:
- Extract time-domain features + Higuchi Fractal Dimension from each window
- Train a Self-Organizing Map (SOM) for visualization
- Train an MLP neural network classifier
- Print accuracy and save model to `models/bioforge_model.pkl`

Target accuracy: 80%+ is good for a student project. If lower:
- Collect more data (more sessions)
- Check electrode placement
- Make sure gestures are held still during recording

### Phase 4: Run Real-Time Control

```bash
python model/inference.py --model models/bioforge_model.pkl --port COM3
```

Or launch the full dashboard with AI:
```bash
python gui/gui_dashboard.py --port COM3 --model models/bioforge_model.pkl
```

---

## File Structure

```
bioforge/
│
├── arduino/
│   └── bioforge_main/
│       └── bioforge_main.ino      ← Flash this to Arduino Mega
│
├── esp32/
│   └── esp32_bridge/
│       └── esp32_bridge.ino       ← Flash this to ESP32 (optional)
│
└── python/
    ├── requirements.txt           ← pip install -r requirements.txt
    │
    ├── comms/
    │   └── arduino_link.py        ← USB/WiFi communication layer
    │
    ├── data_collection/
    │   └── collect_data.py        ← EMG data collector + feature tools
    │
    ├── model/
    │   ├── train_model.py         ← Train SOM + MLP on your data
    │   └── inference.py           ← Real-time gesture → servo control
    │
    └── gui/
        └── gui_dashboard.py       ← Full monitoring dashboard
```

---

## Troubleshooting

**Arduino not found on COM port**
- Open Device Manager → Ports → look for "USB Serial Device" or "Arduino Mega"
- Use that COM number (e.g. COM4, COM5)

**EMG signal is flat (all zeros)**
- Check MyoWare is powered from 3.3V (not 5V)
- Check electrodes are making good skin contact — dampen skin slightly
- Check the RAW pin is connected to Arduino analog pin

**Servos jittering or not moving**
- Check power supply is connected and powered on
- Check common GND between power supply and Arduino
- One servo at a time: test with Mode=Test in GUI

**Model accuracy is low (<70%)**
- Collect more training data (3+ sessions)
- Try to keep electrode placement identical each time
- Make sure you're holding gestures firmly without moving

**GUI won't open (tkinter error)**
- tkinter is included with Python on Windows. Try re-installing Python and checking "tcl/tk"

---

## Key ML Concepts Used

**Higuchi Fractal Dimension (HFD)**
Measures the complexity of the EMG signal. During muscle contraction, the signal becomes more chaotic and irregular, so HFD increases. This was on your whiteboard and is a very effective EMG feature.

**Self-Organizing Map (SOM)**
An unsupervised neural network that arranges similar feature vectors close together on a 2D grid. Used here to visualize whether your gesture clusters are separable before training the classifier.

**MLP Classifier**
A multi-layer perceptron (feedforward neural network) with 3 hidden layers (256 → 128 → 64 neurons). Takes the feature vector as input and outputs a probability for each gesture class.

**Sliding Window**
EMG is processed in 1-second windows (40 samples at 40Hz) with 75% overlap (step=10 samples). This gives a new prediction every 250ms = ~4 predictions per second.

---

## Expanding the System

**More EMG channels**: Add ADS1115 ADC modules (4 channels each, ~$3 each). Connect via I2C (SDA=pin 20, SCL=pin 21 on Mega). Wire up to 4 modules (addresses 0x48-0x4B) = 16 extra channels.

**More gestures**: Add entries to the GESTURES dict in collect_data.py, record data, and retrain.

**Better accuracy**: Collect 10+ sessions, or try a different classifier (SVM, Random Forest) in train_model.py by swapping out MLPClassifier.

**Pressure feedback**: Add force-sensitive resistors (FSRs) to fingertips, read on analog pins A8-A12, add to feature vector.
