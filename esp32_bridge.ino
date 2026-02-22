/*
 * BioForge - ESP32 Wireless Bridge
 * ==================================
 * Acts as a WiFi/BLE bridge between the Raspberry Pi (over WiFi)
 * and the Arduino Mega (over hardware Serial2).
 *
 * The ESP32 creates a TCP socket server.
 * The Raspberry Pi connects to it as a client.
 * All data is transparently forwarded in both directions.
 *
 * Wiring:
 *   ESP32 TX2 (GPIO17) -> Arduino Mega RX1 (pin 19)
 *   ESP32 RX2 (GPIO16) -> Arduino Mega TX1 (pin 18)
 *   Common GND
 *
 * Usage:
 *   1. Set WIFI_SSID and WIFI_PASS below
 *   2. Flash to ESP32
 *   3. Open Serial Monitor to find ESP32 IP address
 *   4. Update ARDUINO_HOST in python/comms/serial_bridge.py
 */

#include <WiFi.h>

// ─── CONFIG ────────────────────────────────────────────────────────────────
const char* WIFI_SSID = "YOUR_WIFI_NAME";   // <-- CHANGE THIS
const char* WIFI_PASS = "YOUR_WIFI_PASS";   // <-- CHANGE THIS
const int   TCP_PORT  = 5000;
const int   SERIAL_BAUD = 115200;

// ─── GLOBALS ───────────────────────────────────────────────────────────────
WiFiServer server(TCP_PORT);
WiFiClient client;

void setup() {
  Serial.begin(115200);       // USB debug
  Serial2.begin(SERIAL_BAUD); // To/from Arduino Mega

  // Connect to WiFi
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    server.begin();
    Serial.print("TCP Server started on port ");
    Serial.println(TCP_PORT);
  } else {
    Serial.println("\nWiFi FAILED - running in USB-only mode");
  }
}

void loop() {
  // Accept new client if none connected
  if (!client.connected()) {
    client = server.accept();
    if (client) {
      Serial.println("Client connected: " + client.remoteIP().toString());
    }
  }

  // Forward: Pi (WiFi) -> Arduino (Serial2)
  if (client.connected() && client.available()) {
    while (client.available()) {
      char c = client.read();
      Serial2.write(c);
    }
  }

  // Forward: Arduino (Serial2) -> Pi (WiFi)
  if (Serial2.available() && client.connected()) {
    while (Serial2.available()) {
      char c = Serial2.read();
      client.write(c);
    }
  }

  // Reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();
    delay(1000);
  }
}
