/*
 * BioForge Bionic Hand - Arduino Mega Firmware
 * =============================================
 * Handles:
 *   - Reading EMG signals from MyoWare 2.0 sensors (analog)
 *   - Reading additional analog channels via ADS1115 ADC modules
 *   - Driving up to 27 servo channels via PWM
 *   - Serial communication with Raspberry Pi
 *
 * Hardware:
 *   - Arduino Mega 2560
 *   - 2x MyoWare 2.0 on A0, A1
 *   - 4x ADS1115 modules via I2C (addresses 0x48-0x4B) = 16 extra channels
 *   - Servos on PWM pins 2-13, 44-46 (15 pins on Mega)
 *
 * Serial Protocol (to/from Raspberry Pi):
 *   SEND:    "EMG:<ch0>,<ch1>,...,<ch15>\n"
 *   RECEIVE: "SERVO:<s0>,<s1>,...,<s26>\n"  (angles 0-180)
 *   RECEIVE: "MODE:<0=collect|1=control|2=test>\n"
 */

#include <Wire.h>
#include <Servo.h>

// ─── CONFIG ────────────────────────────────────────────────────────────────
#define NUM_EMG_CHANNELS    8     // MyoWare x2 + 6 analog pins
#define NUM_SERVOS          15    // Mega has 15 PWM pins available
#define SERIAL_BAUD         115200
#define EMG_SAMPLE_RATE_MS  5     // 200Hz sampling
#define WINDOW_SIZE         50    // RMS window (50 samples = 250ms at 200Hz)

// EMG analog pins (A0-A7 on Mega)
const int EMG_PINS[NUM_EMG_CHANNELS] = {A0, A1, A2, A3, A4, A5, A6, A7};

// Servo PWM pins (Mega PWM: 2-13, 44, 45, 46)
const int SERVO_PINS[NUM_SERVOS] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 44, 45, 46};

// Servo name mapping (for debugging)
const char* SERVO_NAMES[NUM_SERVOS] = {
  "Thumb_Flex", "Thumb_Abd", "Thumb_Tip",
  "Index_MCP", "Index_PIP", "Index_DIP",
  "Middle_MCP", "Middle_PIP", "Middle_DIP",
  "Ring_MCP", "Ring_PIP", "Ring_DIP",
  "Pinky_MCP", "Pinky_PIP", "Wrist_Flex"
};

// ─── GLOBALS ───────────────────────────────────────────────────────────────
Servo servos[NUM_SERVOS];
int servoAngles[NUM_SERVOS];
int defaultAngles[NUM_SERVOS];

// EMG signal buffers for RMS calculation
int emgBuffer[NUM_EMG_CHANNELS][WINDOW_SIZE];
int bufferIndex = 0;

// Timing
unsigned long lastSampleTime = 0;
unsigned long lastSendTime = 0;

// Mode
enum Mode { COLLECT = 0, CONTROL = 1, TEST = 2 };
Mode currentMode = COLLECT;

// Serial parsing
String inputBuffer = "";

// ─── SETUP ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin();

  // Initialize all servos to neutral (90 degrees)
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    defaultAngles[i] = 90;
    servoAngles[i] = 90;
    servos[i].write(90);
    delay(20);
  }

  // Clear EMG buffers
  memset(emgBuffer, 0, sizeof(emgBuffer));

  Serial.println("STATUS:READY");
  delay(100);
}

// ─── MAIN LOOP ─────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // ── Sample EMG at 200Hz ──
  if (now - lastSampleTime >= EMG_SAMPLE_RATE_MS) {
    lastSampleTime = now;
    sampleEMG();
  }

  // ── Send EMG data to Pi every 25ms (40Hz) ──
  if (now - lastSendTime >= 25) {
    lastSendTime = now;
    sendEMGData();
  }

  // ── Parse incoming serial commands ──
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      parseCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }

  // ── Test mode: cycle servos ──
  if (currentMode == TEST) {
    runServoTest();
  }
}

// ─── EMG SAMPLING ──────────────────────────────────────────────────────────
void sampleEMG() {
  for (int ch = 0; ch < NUM_EMG_CHANNELS; ch++) {
    int raw = analogRead(EMG_PINS[ch]);
    emgBuffer[ch][bufferIndex] = raw;
  }
  bufferIndex = (bufferIndex + 1) % WINDOW_SIZE;
}

// Compute RMS over the sliding window for one channel
float computeRMS(int channel) {
  long sumSq = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    int val = emgBuffer[channel][i] - 512; // remove DC offset (midpoint of 0-1023)
    sumSq += (long)val * val;
  }
  return sqrt((float)sumSq / WINDOW_SIZE);
}

// ─── SEND EMG DATA ─────────────────────────────────────────────────────────
void sendEMGData() {
  Serial.print("EMG:");
  for (int ch = 0; ch < NUM_EMG_CHANNELS; ch++) {
    Serial.print(computeRMS(ch), 2);
    if (ch < NUM_EMG_CHANNELS - 1) Serial.print(",");
  }
  Serial.println();
}

// ─── PARSE INCOMING COMMANDS ───────────────────────────────────────────────
void parseCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.startsWith("SERVO:")) {
    // Format: SERVO:90,45,135,...
    String data = cmd.substring(6);
    parseServoCommand(data);

  } else if (cmd.startsWith("MODE:")) {
    int m = cmd.substring(5).toInt();
    currentMode = (Mode)m;
    Serial.print("STATUS:MODE_SET:");
    Serial.println(m);

  } else if (cmd.startsWith("RESET")) {
    resetServos();
    Serial.println("STATUS:RESET_DONE");

  } else if (cmd.startsWith("PING")) {
    Serial.println("STATUS:PONG");
  }
}

void parseServoCommand(String data) {
  int idx = 0;
  int start = 0;
  for (int i = 0; i <= data.length() && idx < NUM_SERVOS; i++) {
    if (i == data.length() || data[i] == ',') {
      int angle = data.substring(start, i).toInt();
      angle = constrain(angle, 0, 180);
      servoAngles[idx] = angle;
      servos[idx].write(angle);
      idx++;
      start = i + 1;
    }
  }
}

// ─── SERVO UTILITIES ───────────────────────────────────────────────────────
void resetServos() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    servoAngles[i] = defaultAngles[i];
    servos[i].write(defaultAngles[i]);
    delay(10);
  }
}

void setFinger(int fingerIndex, int angle) {
  // fingerIndex: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky
  // Each finger has 3 servos (MCP, PIP, DIP)
  int base = fingerIndex * 3;
  if (base + 2 < NUM_SERVOS) {
    servos[base].write(angle);
    servos[base + 1].write(angle);
    servos[base + 2].write(angle);
  }
}

// ─── TEST MODE ─────────────────────────────────────────────────────────────
bool testRunning = false;
unsigned long testTimer = 0;
int testStep = 0;

void runServoTest() {
  unsigned long now = millis();
  if (now - testTimer < 1000) return;
  testTimer = now;

  switch (testStep % 6) {
    case 0: Serial.println("TEST:Open hand"); resetServos(); break;
    case 1: Serial.println("TEST:Close fist"); for (int i=0;i<5;i++) setFinger(i,0); break;
    case 2: Serial.println("TEST:Point"); setFinger(1, 90); break;
    case 3: Serial.println("TEST:Pinch"); setFinger(0,45); setFinger(1,45); break;
    case 4: Serial.println("TEST:Wave"); for (int a=0;a<=90;a+=10){ for(int i=0;i<5;i++) setFinger(i,a); delay(50);} break;
    case 5: Serial.println("TEST:Reset"); resetServos(); break;
  }
  testStep++;
}
