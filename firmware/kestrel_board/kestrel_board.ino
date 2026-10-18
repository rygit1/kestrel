/*
 * Kestrel - ESP32 board code
 *
 * Kestrel's Mac program sends one line per command over USB, and this board moves the two
 * gate arms and the red/yellow/green light.
 *
 * This code started as the board code of an earlier project of mine (a camera that turns to
 * follow things, on two motors called pan and tilt). The parts Kestrel never uses were removed
 * from this public copy; the pan/tilt glide code is still here because the board was built on it.
 *
 * Kestrel's commands (lowercase, starting with "k "):
 *   k hello           -> answers "kestrel ready"
 *   k gate down 030   -> gate arm 1 to 30 (flat across the road)
 *   k gate up 090     -> gate arm 1 to 90 (straight up)
 *   k gate move 075   -> any angle 0-180, for finding the up and down angles
 *   k light red       -> red on, the others off (also: yellow, green, off)
 *   G055              -> gate arm 2 to 55 (the second arm, on GPIO27)
 *
 * Wiring (ESP32 dev board):
 *   Gate arm 1 signal -> GPIO26      Gate arm 2 signal -> GPIO27
 *   Light: red GPIO25, yellow GPIO32, green GPIO12; the light's GND wire to the blue (-) strip
 *   PAN servo signal  -> GPIO13      TILT servo signal -> GPIO14
 *   Servo V+ (red)    -> EXTERNAL 5V supply (NOT the ESP32 5V pin)
 *   Servo GND (brown) -> external 5V GND, tied to ESP32 GND (COMMON GROUND)
 *   1000uF cap across the external 5V rail, close to the servos.
 * GPIO12 is a boot pin: fine for a light (the light keeps it LOW, the safe way), never a servo.
 * Servos are DS3218 270-degree: attached with a 500-2500us pulse for full travel.
 */

#include <ESP32Servo.h>

const int PAN_PIN     = 13;
const int TILT_PIN    = 14;     // GPIO14 (GPIO12 is a boot-strapping pin, do not use it)
const int GATE2_PIN   = 27;     // gate arm 2

// Kestrel: the pickup line gate and the light
const int GATE_PIN          = 26;   // gate arm 1 signal
const int LIGHT_RED_PIN     = 25;
const int LIGHT_YELLOW_PIN  = 32;
const int LIGHT_GREEN_PIN   = 12;   // a boot pin: OK for a light, never for a servo
const int GATE_UP_DEFAULT   = 115;  // where the gate starts; the Mac sends the real up and down
const int GATE_DOWN_DEFAULT = 30;   // angles it learned on Saturday, so these are only fallbacks
const int GATE2_START       = 55;   // gate arm 2 starts UP, Ryan's angle 10/12

const int PAN_MIN = 0,  PAN_MAX = 180;    // chip allows FULL range (manual jog is unrestricted)
const int TILT_MIN = 0, TILT_MAX = 180;   // safe crash-limits (tilt 60-85, pan 38-170) live in the BRAIN's auto loop only

// Motion: the brain sends a fresh target about every camera frame (50-80 ms apart). The old code
// hopped to each target at full servo speed in whole-degree steps, then WAITED for the next
// command: hop-wait-hop-wait = visible stutter. Now each new target is GLIDED to over ~GLIDE_MS
// (a hair over one camera frame) in fine sub-degree steps at a fast tick, so the head is still
// moving when the next command lands = one continuous glide at the commanded average speed.
const int   TICK_MS     = 5;                       // 200 Hz motion tick (was 15 ms / 66 Hz)
const int   GLIDE_MS    = 70;                      // spread each brain hop across about one camera frame
const int   GLIDE_TICKS = GLIDE_MS / TICK_MS;
const float PAN_TICK_MAX  = 133.0 * TICK_MS / 1000.0;  // speed ceiling per tick (133 deg/s, matches --slew-rate)
const float TILT_TICK_MAX =  66.0 * TICK_MS / 1000.0;  // tilt stays GENTLER (loose tilt joint, missing a screw)

Servo panServo, tiltServo;
Servo gateServo, gate2Servo;       // Kestrel's two gate arms

float panCur = 90, tiltCur = 72;   // boot to a tilt inside [60,85] so it never slams the cap on reset
int   panTgt = 90, tiltTgt = 72;
float panVel = 0, tiltVel = 0;     // glide speed (deg per tick), recomputed on every aim command
unsigned long lastTick = 0;
String line;

void setup() {
  Serial.begin(115200);
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);        // room for the 4th servo
  panServo.setPeriodHertz(50);     panServo.attach(PAN_PIN, 500, 2500);
  tiltServo.setPeriodHertz(50);    tiltServo.attach(TILT_PIN, 500, 2500);
  gate2Servo.setPeriodHertz(50);   gate2Servo.attach(GATE2_PIN, 500, 2500);
  panServo.write((int)panCur);
  tiltServo.write((int)tiltCur);
  gate2Servo.write(GATE2_START);     // gate arm 2 starts up
  gateServo.setPeriodHertz(50);      gateServo.attach(GATE_PIN, 500, 2500);   // Kestrel
  gateServo.write(GATE_UP_DEFAULT);                                           // Kestrel: gate starts up
  pinMode(LIGHT_RED_PIN, OUTPUT);    digitalWrite(LIGHT_RED_PIN, LOW);        // Kestrel: light starts off
  pinMode(LIGHT_YELLOW_PIN, OUTPUT); digitalWrite(LIGHT_YELLOW_PIN, LOW);
  pinMode(LIGHT_GREEN_PIN, OUTPUT);  digitalWrite(LIGHT_GREEN_PIN, LOW);
  Serial.println("kestrel ready");
}

// ---- Kestrel: the pickup line gate and the light ----

void setLight(bool red, bool yellow, bool green) {
  digitalWrite(LIGHT_RED_PIN, red ? HIGH : LOW);
  digitalWrite(LIGHT_YELLOW_PIN, yellow ? HIGH : LOW);
  digitalWrite(LIGHT_GREEN_PIN, green ? HIGH : LOW);
}

// The angle written after a command word, like the 030 in "k gate down 030".
// No number, or anything that is not a number, keeps the fallback, so a typo can never
// slam the gate to 0.
int angleAfter(const String &s, int start, int fallback) {
  String rest = s.substring(start);
  rest.trim();
  if (rest.length() == 0 || !isDigit(rest[0])) return fallback;
  return constrain(rest.toInt(), 0, 180);
}

// Handles one "k ..." line from Kestrel's Mac program and answers it. Returns true if it was
// a "k" line, so the parser below never sees it.
bool kestrelLine(const String &s) {
  if (!s.startsWith("k ")) return false;
  if (s == "k hello") {
    Serial.println("kestrel ready");
  } else if (s.startsWith("k gate down")) {
    int a = angleAfter(s, 11, GATE_DOWN_DEFAULT);
    gateServo.write(a);
    Serial.print("kestrel gate down "); Serial.println(a);
  } else if (s.startsWith("k gate up")) {
    int a = angleAfter(s, 9, GATE_UP_DEFAULT);
    gateServo.write(a);
    Serial.print("kestrel gate up "); Serial.println(a);
  } else if (s.startsWith("k gate move")) {
    int a = angleAfter(s, 11, -1);
    if (a >= 0) { gateServo.write(a); Serial.print("kestrel gate at "); Serial.println(a); }
    else Serial.println("kestrel gate move needs a number, like: k gate move 075");
  } else if (s == "k light red") {
    setLight(true, false, false);  Serial.println("kestrel light red");
  } else if (s == "k light yellow") {
    setLight(false, true, false);  Serial.println("kestrel light yellow");
  } else if (s == "k light green") {
    setLight(false, false, true);  Serial.println("kestrel light green");
  } else if (s == "k light off") {
    setLight(false, false, false); Serial.println("kestrel light off");
  } else {
    Serial.print("kestrel unknown: "); Serial.println(s);   // an unknown word does nothing
  }
  return true;
}

void parseLine(const String &s) {
  if (kestrelLine(s)) return;
  int p = s.indexOf('P');
  int t = s.indexOf('T');
  if (p >= 0 && t > p) {
    panTgt  = constrain(s.substring(p + 1, p + 4).toInt(), PAN_MIN, PAN_MAX);
    tiltTgt = constrain(s.substring(t + 1, t + 4).toInt(), TILT_MIN, TILT_MAX);
    panVel  = (panTgt  - panCur)  / (float)GLIDE_TICKS;   // pace the move to land with the next command
    tiltVel = (tiltTgt - tiltCur) / (float)GLIDE_TICKS;
  }

  int g = s.indexOf('G');   // G### = gate arm 2 to that angle
  if (g >= 0) {
    int ang = constrain(s.substring(g + 1, g + 4).toInt(), 0, 180);
    gate2Servo.write(ang);
    Serial.print("gate 2 at "); Serial.println(ang);
  }
}

void glide(float &cur, int tgt, float &vel, Servo &s, float tickMax) {
  float rem = (float)tgt - cur;
  float step = vel;
  if ((rem > 0 && step <= 0) || (rem < 0 && step >= 0))   // stale/zero pace but not there yet:
    step = (rem > 0) ? tickMax : -tickMax;                // fall back to the speed ceiling
  if (step >  tickMax) step =  tickMax;                   // hardware speed ceiling per axis
  if (step < -tickMax) step = -tickMax;
  if ((rem >= 0 && step >= rem) || (rem <= 0 && step <= rem)) { cur = tgt; vel = 0; }  // arrive, never pass
  else cur += step;
  // sub-degree writes: attach(500,2500) maps 0-180 deg over 2000 us, so 1 us = 0.09 deg.
  // plain write() rounds to whole degrees, which reads as tiny clicks at glide speeds.
  s.writeMicroseconds(500 + (int)(cur * (2000.0 / 180.0) + 0.5));
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n')                         { parseLine(line); line = ""; }
    else if (c != '\r' && line.length() < 32) line += c;
  }

  unsigned long now = millis();
  if (now - lastTick >= TICK_MS) {
    lastTick = now;
    glide(panCur, panTgt, panVel, panServo, PAN_TICK_MAX);
    glide(tiltCur, tiltTgt, tiltVel, tiltServo, TILT_TICK_MAX);
  }
}
