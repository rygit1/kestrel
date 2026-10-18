"""
KESTREL: the pickup line guard.

A webcam looks straight down at a toy road. When a toy car comes out from behind the bus,
Kestrel tracks it, asks Ryan's code where it is going, and drops a gate arm across the road
(plus a red light, a honk and a robot voice) only if the car would reach the kid.

    python kestrel_brain.py --practice     the fake road: no camera, board or speaker needed
    python kestrel_brain.py --practice-board   the fake road on screen, the REAL arms and light follow it
    python kestrel_brain.py                the real thing: the webcam plus the board
    python sims.py                         20 fake rolls and the "how fast is too fast" test

Who wrote what (the App Challenge asks): this file and the fake road are support code
written with AI help and listed on the AI-use list. The part that decides, Ryan's code, is
in ryan_code.py, and he types it himself. Nothing in this file works out where the car is
going or whether it is inside the circle. It only ever ASKS ryan_code.py.
"""

import argparse
import collections
import copy
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import ryan_code

HERE = Path(__file__).resolve().parent
SETTINGS_FILE = HERE / "settings.json"
HONK_FILE = HERE / "sounds" / "honk.wav"
CLIPS_FOLDER = HERE / "clips"                         # the slow-motion replay clips are saved here
MY_CAR_AI = HERE / "models" / "my_car.onnx"            # made by "learn my car" (key L), only if it passed its test
PRACTICE_CAR_AI = HERE / "models" / "practice_car.onnx" # the same, learned on the fake road
SQUARE = 320                                            # the AI checks a 320 by 320 square around each red thing
VIEW_W, VIEW_H = 1280, 720          # the camera picture
PANEL_W = 400                       # the command center panel beside it
SCREEN_W = VIEW_W + PANEL_W

STEPS = ["IDLE", "DETECTED", "TRACKING", "ASSESSING", "PROTECTION REQUIRED",
         "PROTECTION DEPLOYED", "ALL CLEAR", "RESET"]
LOST_SECONDS = 0.5     # the car counts as gone after this long without being seen
TRACK_AFTER = 3        # sightings in a row before it is a stable track
ASSESS_AFTER = 5       # spots in memory before Ryan's code is asked
CLEAR_HOLD = 1.0       # seconds ALL CLEAR stays up before the gate lifts

DEFAULTS = {
    "circle": None,          # [x, y, radius] in pixels, placed with key C
    "look": {"hue": 0, "hue_width": 10, "sat_min": 120, "val_min": 70},   # what the red car looks like (key K)
    "min_area": 600,         # the smallest red blob that counts as the car, in pixels
    "ignore": [],            # boxes Kestrel never looks in, [x1, y1, x2, y2] (key I)
    "px_per_cm": 12.0,       # pixels per centimeter on the road (key S)
    "gate_up": None,         # gate arm angles, found in the startup test the first time
    "gate_down": None,
    "gate2_up": None,        # the second arm (double gate, 10/12) on the old board's spare motor line
    "gate2_down": None,
    "gate_seconds": 0.30,    # how long the arm takes to come all the way down
    "voice": "Zarvox",       # the robot voice; "Fred" is a clearer one
    "replay": True,          # slow-motion replay after every roll (key P)
    "camera": 0,               # 0 = the Insta360 Link 2C (checked 10/13)
}

# colors are (blue, green, red)
WHITE, DIM, BLACK = (240, 240, 240), (150, 150, 155), (0, 0, 0)
GREEN, YELLOW, RED, CYAN, ORANGE = (80, 220, 90), (40, 210, 255), (70, 70, 255), (255, 210, 80), (40, 160, 255)
STEP_COLOR = {"IDLE": GREEN, "DETECTED": YELLOW, "TRACKING": YELLOW, "ASSESSING": YELLOW,
              "PROTECTION REQUIRED": RED, "PROTECTION DEPLOYED": RED, "ALL CLEAR": GREEN, "RESET": GREEN}
LIGHT_FOR_STEP = {"IDLE": "green", "DETECTED": "yellow", "TRACKING": "yellow", "ASSESSING": "yellow",
                  "PROTECTION REQUIRED": "red", "PROTECTION DEPLOYED": "red", "ALL CLEAR": "green", "RESET": "green"}


# ---------------------------------------------------------------- settings

def load_settings():
    s = copy.deepcopy(DEFAULTS)
    if SETTINGS_FILE.exists():
        s.update(json.loads(SETTINGS_FILE.read_text()))
    return s


def save_settings(s):
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))


def practice_settings():
    """The fake road already knows where the kid and the gate are."""
    import fake_road as fr
    s = copy.deepcopy(DEFAULTS)
    s.update(circle=[fr.px(fr.KID[0]), fr.px(fr.KID[1]), fr.px(fr.CIRCLE_RADIUS_CM)],
             px_per_cm=float(fr.PX_PER_CM), gate_up=90, gate_down=30, gate_seconds=fr.GATE_DROP_SECONDS)
    return s


# ---------------------------------------------------------------- 1. THE EYES

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def find_car(frame, look, min_area, ignore=()):
    """THE COLOR EYES: finds the red car in one picture by its color.
    Hands back (x, y), the middle of the car in pixels, or None if it is not there."""
    blobs = red_blobs(frame, look, min_area, ignore)
    return blobs[0][:2] if blobs else None


def find_motors(frame):
    """The two gate motors, found by their pink tops. Hands back their boxes, top one first,
    or fewer than two if they cannot both be seen."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = (hsv[..., i].astype(int) for i in range(3))
    # the motor tops are a bright, pale pink (hue 150 to 177, saturation about 95 to 140, brightness
    # over 230, measured 10/13 in 5 pictures); McQueen is a deeper, darker red (brightness 115 to 210)
    pink = ((hue >= 140) & (sat >= 50) & (sat <= 170) & (val >= 190)).astype(np.uint8)
    pink = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(pink)
    keep = sorted((i for i in range(1, count) if 400 <= stats[i, 4] <= 4000), key=lambda i: -stats[i, 4])
    found = [tuple(int(v) for v in stats[i, :4]) for i in keep][:2]      # the two biggest pink patches
    if len(found) == 2 and math.hypot(found[0][0] - found[1][0], found[0][1] - found[1][1]) < 100:
        return found[:1]                  # two pieces of one motor, not two motors
    return sorted(found, key=lambda b: b[1])


def auto_scene(frame, settings):
    """Kestrel sets itself up from the two motors, so a bumped board never breaks anything (10/13):
    a box over each motor so it is never mistaken for the car, and the protected bubble over the
    whole gap between them plus the crosswalk past it (Ryan's rule: any car headed through the
    gate drops the arms). Returns True if both motors were found."""
    motors = find_motors(frame)
    if len(motors) == 2 and settings.get("car_size"):
        cars = [b[3] for b in red_blobs(frame, settings["look"], settings["min_area"], [])
                if car_sized(b[3], settings["car_size"]) and strong_red(frame, b[3])]
        if any(cx <= mx + mw / 2 <= cx + cw and cy <= my + mh / 2 <= cy + ch
               for mx, my, mw, mh in motors for cx, cy, cw, ch in cars):
            return False                  # a shine on McQueen looked pink: keep the last good setup
    if len(motors) < 2:
        return False
    pad = 8
    settings["ignore"] = [[x - pad, y - pad, x + w + pad, y + h + pad] for x, y, w, h in motors]
    (ax, ay), (bx, by) = [(x + w / 2, y + h / 2) for x, y, w, h in motors]
    mid_x, mid_y = (ax + bx) / 2, (ay + by) / 2
    half = math.hypot(bx - ax, by - ay) / 2
    nx, ny = (by - ay) / (2 * half), -(bx - ax) / (2 * half)     # straight across the gate line
    ramp_x = settings.get("ramp_side_x", 0)                        # the car comes from the left
    if (nx > 0) != (mid_x > ramp_x):
        nx, ny = -nx, -ny                                          # point away from the ramp
    radius = 1.25 * half                  # wide enough that the bubble covers the whole gap at the arms
    settings["circle"] = [int(mid_x + nx * 0.75 * half), int(mid_y + ny * 0.75 * half), int(radius)]
    return True


def car_sized(box, car_size):
    """True if a red box is about the car's size (learned with K). Turned either way is fine."""
    if not car_size:
        return True
    long_side, short_side = max(box[2], box[3]), min(box[2], box[3])
    return 0.7 <= long_side / car_size[0] <= 1.4 and 0.7 <= short_side / car_size[1] <= 1.4


def strong_red(frame, box):
    """McQueen's paint is a deep red (middle saturation about 164, measured 10/13); skin under these
    lights is a paler red (about 110 to 130). Only the deep red counts when learning the car."""
    x, y, w, h = box
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0].astype(int), hsv[..., 1]
    red = sat[((hue <= 11) | (hue >= 171)) & (sat >= 90)]
    return red.size > 0 and float(np.median(red)) >= 145


def red_blobs(frame, look, min_area, ignore=()):
    """Every red patch big enough to be the car, biggest first. Each one is
    (x, y, area, (left, top, width, height)): its middle, its size, and the box around it."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = (look["hue"] - look["hue_width"]) % 180
    hi = (look["hue"] + look["hue_width"]) % 180
    s, v = look["sat_min"], look["val_min"]
    if lo <= hi:
        mask = cv2.inRange(hsv, (lo, s, v), (hi, 255, 255))
    else:   # red sits at both ends of the color wheel, so it needs two ranges
        mask = cv2.inRange(hsv, (0, s, v), (hi, 255, 255)) | cv2.inRange(hsv, (lo, s, v), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)        # wipe out specks
    for x1, y1, x2, y2 in ignore:
        mask[y1:y2, x1:x2] = 0
    n, _, stats, middles = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            box = tuple(int(v) for v in stats[i, :4])
            blobs.append((float(middles[i][0]), float(middles[i][1]), area, box))
    return sorted(blobs, key=lambda b: -b[2])


def learn_color(frame, x, y):
    """Clicking the car teaches Kestrel its color under your lights, from the WHOLE car (Ryan 10/13:
    one clicked spot could land on a sticker or a shine and teach a red that matched the wood floor).
    Returns the color and the car's box, or (None, None) if no car is under the click."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(int)
    loose = (((hue <= 15) | (hue >= 165)) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 50)).astype(np.uint8)
    loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # stickers and windows join the body
    count, labels, stats, _ = cv2.connectedComponentsWithStats(loose)
    near = labels[max(0, y - 15):y + 16, max(0, x - 15):x + 16]
    ids = [i for i in np.unique(near) if i != 0]
    if not ids:
        return None, None
    car = max(ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    if stats[car, cv2.CC_STAT_AREA] < 300:          # a speck, not a car
        return None, None
    body = hsv[(labels == car) & (loose > 0)]
    sat, val = body[:, 1], body[:, 2]
    deep = body[sat >= np.percentile(sat, 40)]          # the paint, not the shines and stickers
    h = (int(np.median((deep[:, 0].astype(int) + 90) % 180)) - 90) % 180
    look = {"hue": h, "hue_width": 10,
            "sat_min": max(90, int(np.percentile(sat, 20) * 0.85)),
            "val_min": max(40, int(np.percentile(val, 10) * 0.8))}
    bx, by, bw, bh = (int(v) for v in stats[car, :4])
    return look, (bx, by, bw, bh)


class Eyes:
    """THE EYES, working as a team.
    Color alone: finds the red car by its color, about 2 thousandths of a second a picture.
    Color + AI: once Kestrel has learned YOUR car (key L) and that AI passed its test, the color
    finds every red thing, and the AI checks a small square around it, so only your car counts.
    A red shirt or the red light can never fool it. The color still gives the car's exact middle,
    and while it follows a car the AI already checked, the AI only looks again every 3rd picture,
    which keeps Kestrel fast."""

    CHECK_EVERY = 3

    def __init__(self, settings, ai_file=None):
        self.s = settings
        self.ai, self.name = None, "color"
        self.box, self.sure, self.since = None, 0.0, 99     # the AI's last box around the car
        self.ms = 0.0                                       # how long the eyes took, in thousandths of a second
        self.start_bar, self.follow_bar = 0.75, 0.5         # how sure the AI must be (set by its own test)
        self.rejected = []                                  # red things the AI said are NOT the car: (x, y, size, pictures left)
        if ai_file is not None and Path(ai_file).exists():
            self.load(ai_file)

    def load(self, ai_file):
        try:
            from ultralytics import YOLO
            bars = Path(ai_file).with_suffix(".json")
            if bars.exists():
                b = json.loads(bars.read_text())
                self.start_bar, self.follow_bar = b["start_bar"], b["follow_bar"]
            self.ai = YOLO(str(ai_file), task="detect")
            self.ai.predict(np.zeros((SQUARE, SQUARE, 3), np.uint8), imgsz=SQUARE, verbose=False)   # warm up
            self.name = "color + AI"
            print("Eyes: color + the AI that learned your car")
        except Exception as e:
            print("The AI eyes did not load, so Kestrel uses its color eyes:", e)
            self.ai, self.name = None, "color"

    def find(self, frame):
        start = time.perf_counter()
        blobs = red_blobs(frame, self.s["look"], self.s["min_area"], self.s["ignore"])
        if self.s.get("car_size"):      # only car-sized, deep-red things: never a hand or a motor (10/13)
            blobs = [b for b in blobs if car_sized(b[3], self.s["car_size"]) and strong_red(frame, b[3])]
        if self.ai is None:
            car = blobs[0][:2] if blobs else None
        else:
            car = self.team(frame, blobs)
        self.ms = (time.perf_counter() - start) * 1000
        return car

    def team(self, frame, blobs):
        self.rejected = [(x, y, a, n - 1) for x, y, a, n in self.rejected if n > 1]
        if not blobs:
            self.box = None
            return None
        following = self.box is not None and self.since < 10
        near = [b for b in blobs if following and in_box(b, grow(self.box, 1.8))]
        if near and self.since % self.CHECK_EVERY != self.CHECK_EVERY - 1:
            return self.follow(near[0])                        # 2 of every 3 pictures: just follow the color
        best = None
        fresh = [b for b in blobs[:3] if b not in near[:1] and not self.was_rejected(b)]
        for b in near[:1] + fresh:                             # the AI checks up to 3 red things
            hit = self.check(frame, b)
            bar = self.follow_bar if b in near[:1] else self.start_bar     # hard to become the car, easy to stay it
            if hit is not None and hit[4] >= bar and (best is None or hit[4] > best[1][4]):
                best = (b, hit)
            elif b not in near[:1]:
                self.rejected.append((b[0], b[1], b[2], 15))   # not the car: skip it for half a second
        if best is not None:                                   # the one the AI is surest is the car
            b, hit = best
            self.box, self.sure, self.since = hit[:4], hit[4], 0
            return b[:2]
        if near:
            return self.follow(near[0])                        # the AI blinked: trust the color for a moment
        self.box = None
        return None

    def was_rejected(self, blob):
        """True if this red thing sits right where a rejected one was, the same size. Anything that
        moved even 3 pixels, or changed size, gets checked again, so a real car can never hide here."""
        return any(abs(blob[0] - x) <= 3 and abs(blob[1] - y) <= 3 and 0.8 * a <= blob[2] <= 1.25 * a
                   for x, y, a, _ in self.rejected)

    def follow(self, blob):
        self.since += 1
        x1, y1, x2, y2 = self.box
        dx, dy = blob[0] - (x1 + x2) / 2, blob[1] - (y1 + y2) / 2
        self.box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)        # keep the box on the car
        return blob[:2]

    def check(self, frame, blob):
        """Asks the AI: is this red thing the car? Hands back its box (x1, y1, x2, y2, sure) or None."""
        h, w = frame.shape[:2]
        x0 = int(min(max(blob[0] - SQUARE / 2, 0), w - SQUARE))
        y0 = int(min(max(blob[1] - SQUARE / 2, 0), h - SQUARE))
        r = self.ai.predict(frame[y0:y0 + SQUARE, x0:x0 + SQUARE], imgsz=SQUARE, conf=self.follow_bar, verbose=False)[0]
        best = None
        for (bx1, by1, bx2, by2), sure in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            box = (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)
            if in_box(blob, box) and (best is None or sure > best[4]):
                best = (*box, sure)
        return best


def in_box(blob, box):
    return box[0] <= blob[0] <= box[2] and box[1] <= blob[1] <= box[3]


def grow(box, times):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * times, (box[3] - box[1]) / 2 * times
    return cx - hw, cy - hh, cx + hw, cy + hh


# ---------------------------------------------------------------- the board

def find_board_port():
    ports = sorted(glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.SLAB_USBtoUART*")
                   + glob.glob("/dev/cu.wchusbserial*"))
    return ports[0] if ports else None


class RealBoard:
    """The ESP32 over the USB cable."""

    def __init__(self, port, baud=115200):
        import serial
        self.name = port
        self.link = serial.Serial(port, baud, timeout=0)
        time.sleep(2.0)                    # opening the cable restarts the board: let it wake up
        self._buf = ""

    def send(self, line, t):
        self.link.write((line + "\n").encode())

    def read(self):
        waiting = self.link.in_waiting
        if waiting:
            self._buf += self.link.read(waiting).decode(errors="replace")
        lines = self._buf.split("\n")
        self._buf = lines.pop()
        return [ln.strip() for ln in lines if ln.strip()]


class BothBoards:
    """Practice with the real board (--practice-board): the fake road still plays out on screen,
    and every gate and light line also goes to the real ESP32, so the real arms and light move
    in step with the fake car. Only the fake board's answers are read."""

    def __init__(self, fake, real):
        self.fake, self.real = fake, real
        self.name = "practice + real board"

    def send(self, line, t):
        self.real.send(line, t)
        if not line.startswith("G"):     # the second arm's line means nothing to the fake road
            self.fake.send(line, t)

    def read(self):
        self.real.read()                 # drain the real board's answers so they never pile up
        return self.fake.read()


class NoBoard:
    """No board plugged in: Kestrel prints what it would have sent instead."""
    name = "no board"

    def send(self, line, t):
        print("   (no board) would send:", line)

    def read(self):
        return []


# ---------------------------------------------------------------- 8. KESTREL'S VOICE

def make_honk(path):
    """Builds the honk the first time it is needed: two low tones together, like a bus horn."""
    rate, seconds = 44100, 0.55
    t = np.arange(int(rate * seconds)) / rate
    sound = np.zeros_like(t)
    for pitch in (330.0, 415.0):
        tone = np.sin(2 * np.pi * pitch * t)
        sound += 0.65 * tone + 0.35 * np.sign(tone)               # a little buzz makes it brassy
    fade = np.minimum(1.0, np.minimum(t / 0.015, (seconds - t) / 0.06))
    sound = (sound * fade / np.abs(sound).max() * 0.7 * 32767).astype(np.int16)
    path.parent.mkdir(exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(sound.tobytes())


class Voice:
    """KESTREL'S VOICE: the robot voice and the honk. Sound goes to the Mac's sound output,
    so picking the Bluetooth speaker in the Mac's sound menu makes the honk come from the road."""

    def __init__(self, voice="Zarvox", quiet=False):
        self.voice = voice
        self.quiet = quiet          # quiet = no sound at all (the checks use this)
        self.heard = []             # (time, what) for every sound, so the checks can count them
        self.waiting = []           # (when, words) lines to say a moment from now
        self.speaking = None

    def honk(self, t):
        self.heard.append((t, "honk"))
        if not self.quiet:
            if not HONK_FILE.exists():
                make_honk(HONK_FILE)
            subprocess.Popen(["afplay", str(HONK_FILE)])

    def say(self, words, t, after=0.0):
        self.waiting.append((t + after, words))

    def tick(self, t):
        due = [w for w in self.waiting if w[0] <= t]
        self.waiting = [w for w in self.waiting if w[0] > t]
        for _, words in due:
            self.heard.append((t, words))   # the robot voice is off (Ryan 10/13): lines are only logged, the honk still plays


# ---------------------------------------------------------------- the event log

class Log:
    """THE EVENT LOG: one line for every step, like the log in Ryan's Kestrel doc, plus a file
    per roll in logs/events/ with every spot and every answer Ryan's code gave."""

    def __init__(self, folder, clock, echo=True):
        self.folder = Path(folder)
        (self.folder / "events").mkdir(parents=True, exist_ok=True)
        self.file = self.folder / "events.log"
        self.clock = clock
        self.echo = echo
        self.lines = []

    def next_event_number(self):
        return len(list((self.folder / "events").glob("event_*.json"))) + 1

    def write(self, t, text):
        line = f"{self.clock(t)}  {text}"
        self.lines.append(line)
        with open(self.file, "a") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    def save_event(self, number, data):
        (self.folder / "events" / f"event_{number:04d}.json").write_text(json.dumps(data, indent=1))


def wall_clock(_t):
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------- 2 and 6. MEMORY AND THE STEPS

def plain(v):
    """numbers from numpy, made plain so they can be saved"""
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class Kestrel:
    """THE STEPS Kestrel goes through, its MEMORY of the car, and the calls to Ryan's code.
    Steps, from Ryan's doc: IDLE, DETECTED, TRACKING, ASSESSING, PROTECTION REQUIRED,
    PROTECTION DEPLOYED, ALL CLEAR, RESET, then back to IDLE."""

    def __init__(self, settings, board, voice, log, ryan=ryan_code, version=3,
                 warning_mode=False, auto=True, echo_board=True, eyes=None):
        self.settings = settings
        self.eyes = eyes or Eyes(settings)
        self.board, self.voice, self.log, self.ryan = board, voice, log, ryan
        self.version = version            # 2 = react when the car is in the circle, 3 = guess early
        self.warning_mode = warning_mode  # True = light and honk only, the gate stays up
        self.auto = auto                  # False = MANUAL: only the PROTECT button drops the gate
        self.echo_board = echo_board
        self.step, self.step_since = "IDLE", 0.0
        self.history = ["IDLE"]           # every step, in order, for the checks
        self.spots = []                   # THE MEMORY: (time, x, y) for every sighting this roll
        self.seen_in_a_row = 0
        self.last_seen = None
        self.car = None                   # where the car is right now, or None
        self.event = None                 # this roll's record
        self.gate_down = False
        self.light = None
        self.board_ok = False             # True only once the board said "kestrel ready"
        self.inside = None                # Ryan's part 1 answer right now
        self.answer = None                # Ryan's part 2 answer right now
        self.protected = False            # did this roll get protection?
        self.protect_time = None
        self.manual_hold = False
        self.would_drop_said = False
        self.banner = None                # ("MISSED BY 4 CM", until when)
        self.replay_request = None        # (from, to) once a roll ends
        self.ryan_status = {1: "not written yet", 2: "not written yet", 3: "not written yet"}
        self.errors_printed = set()
        self.unsent = 0

    # ---- talking to the board ----

    def hello(self, t, wait=2.5):
        """Says hello to the board. Kestrel only moves the gate or the light after the board
        answers "kestrel ready", so it can never talk to the turret's old code by mistake."""
        self.board.send("k hello", t)
        give_up = time.monotonic() + wait
        while True:
            for reply in self.board.read():
                if self.echo_board:
                    self.log.write(t, f"      board says: {reply}")
                if reply == "kestrel ready":
                    self.board_ok = True
            if self.board_ok or time.monotonic() > give_up:
                break
            time.sleep(0.05)
        self.log.write(t, "board: kestrel ready" if self.board_ok else
                       "board: NO ANSWER, so Kestrel will not move the gate or the light")
        if self.board_ok:
            self.light = None
            self.set_light("green", t)
            if self.settings["gate_up"] is not None:
                self.gate(False, t)
        return self.board_ok

    def send(self, line, t):
        if not self.board_ok:
            self.unsent += 1
            return
        self.board.send(line, t)

    def set_light(self, color, t):
        if color != self.light:
            self.light = color
            self.send(f"k light {color}", t)

    def gate(self, down, t):
        word = "down" if down else "up"
        angle = self.settings["gate_" + word]
        self.send(f"k gate {word}" if angle is None else f"k gate {word} {int(angle):03d}", t)
        angle2 = self.settings.get("gate2_" + word)
        if angle2 is not None:
            self.send(f"G{int(angle2):03d}", t)   # the second arm drops with the first
        self.gate_down = down

    # ---- the steps ----

    def go(self, step, t, detail=""):
        self.step, self.step_since = step, t
        self.history.append(step)
        self.log.write(t, f"{step:<20} {detail}".rstrip())
        if self.event is not None:
            self.event["steps"].append([round(t, 3), step, detail])

    def circle(self):
        return self.settings["circle"]

    def see(self, frame, t):
        """ONE MOMENT: find the car, remember it, ask Ryan's code, move through the steps."""
        for reply in self.board.read():
            if self.echo_board:
                self.log.write(t, f"      board says: {reply}")
        self.voice.tick(t)
        found = self.eyes.find(frame)
        self.car = found
        if found:
            self.last_seen = t
            self.seen_in_a_row += 1
            if self.step not in ("IDLE", "ALL CLEAR", "RESET"):
                self.spots.append((t, found[0], found[1]))
        else:
            self.seen_in_a_row = 0
        gone = self.last_seen is None or t - self.last_seen > LOST_SECONDS

        self.inside = None
        if found and self.circle():
            self.inside = self.ask_part1(found)        # turns the circle red when Ryan's code says inside

        step = self.step
        if step == "IDLE":
            if found:
                self.start_event(t, found)
        elif step == "DETECTED":
            if self.seen_in_a_row >= TRACK_AFTER:
                self.go("TRACKING", t, "stable track")
                self.voice.say("Car detected.", t)
            elif gone:
                self.go("IDLE", t, "only a blip")
                self.event, self.spots = None, []
                self.set_light("green", t)
        elif step == "TRACKING":
            if len(self.spots) >= ASSESS_AFTER:
                self.go("ASSESSING", t, "asking Ryan's part " + ("1" if self.version == 2 else "2"))
            elif gone:
                self.all_clear(t)
        elif step == "ASSESSING":
            if found:
                self.decide(t, found)
            if self.step == "ASSESSING" and gone:
                self.all_clear(t)
        elif step == "PROTECTION REQUIRED":
            if t - self.step_since >= self.settings["gate_seconds"]:
                self.go("PROTECTION DEPLOYED", t, "warning given" if self.warning_mode else "gate is down")
        elif step == "PROTECTION DEPLOYED":
            if gone and not self.manual_hold:
                self.all_clear(t)
        elif step in ("ALL CLEAR", "RESET") and found:
            # a new car while the last one is still being cleared: never make it wait (Ryan's
            # practice run, Oct 6: a car rolled right away was only noticed past the kid)
            if self.gate_down:
                self.gate(False, t)
            self.manual_hold = False
            self.go("IDLE", t, "a new car is already coming")
            self.event = None
            self.start_event(t, found)
        elif step == "ALL CLEAR":
            if t - self.step_since >= CLEAR_HOLD:
                self.reset(t)
        elif step == "RESET":
            if t - self.step_since >= self.settings["gate_seconds"]:
                self.go("IDLE", t)
                self.event = None

    def start_event(self, t, found=None, why=""):
        self.event = {"event": self.log.next_event_number(), "version": self.version,
                      "warning_mode": self.warning_mode, "auto": self.auto,
                      "circle": plain(self.circle()), "px_per_cm": self.settings["px_per_cm"],
                      "steps": [], "decisions": []}
        self.spots = [(t, found[0], found[1])] if found else []
        self.protected, self.protect_time, self.answer = False, None, None
        self.would_drop_said = False
        self.banner = None    # a new car wipes the last roll's MISSED BY sign (Oct 14: it showed over a closed gate)
        self.log.write(t, f"EVENT {self.event['event']:04d}")
        self.go("DETECTED" if found else "ASSESSING", t,
                f"car seen at ({found[0]:.0f}, {found[1]:.0f})" if found else why)
        self.set_light("yellow", t)

    def decide(self, t, found):
        """Asks Ryan's code, and acts on its answer. The deciding is all his."""
        if self.circle() is None:
            if not getattr(self, "no_circle_said", False):
                self.log.write(t, "   no circle yet: press C and click the kids first. Kestrel only watches until then.")
                self.no_circle_said = True
            return
        if self.version == 2:
            drop, why = self.inside, "the car is inside the circle (Ryan's part 1)"
        else:
            self.answer = self.ask_part2()
            drop = None if self.answer is None else self.answer["drop_gate"]
            why = "Ryan's guesser says it will reach the kid (part 2)"
        a = self.answer or {}
        self.event["decisions"].append(plain([round(t, 3), drop, a.get("speed"), a.get("seconds_left"), a.get("path_end")]))
        if not drop:
            return
        if self.auto:
            self.protect(t, why)
        elif not self.would_drop_said:
            self.would_drop_said = True
            self.log.write(t, "   MANUAL mode: Ryan's code says drop the gate now. Press SPACE to do it.")

    def protect(self, t, why):
        self.protected, self.protect_time = True, t
        left = (self.answer or {}).get("seconds_left")
        self.go("PROTECTION REQUIRED", t, why + (f", {float(left):.1f} s to impact" if left is not None else ""))
        if self.warning_mode:
            self.log.write(t, "   warning mode: no gate, just the red light and the honk")
            self.voice.say("Danger.", t, after=0.45)
        else:
            self.gate(True, t)
            self.voice.say("Danger. Gate down.", t, after=0.45)
        self.set_light("red", t)
        self.voice.honk(t)

    def all_clear(self, t):
        self.go("ALL CLEAR", t, "car gone")
        words = "All clear."
        if not self.protected and self.circle() and self.spots:
            gap = self.ask_part3()
            if gap is not None and gap > 0:
                cm = float(gap) / self.settings["px_per_cm"]
                amount = f"{cm:.1f}" if cm < 1 else f"{cm:.0f}"
                self.banner = (f"MISSED BY {amount} CM", t + 4.0)
                self.event["missed_by_cm"] = round(cm, 1)
                self.log.write(t, f"   how close it came (Ryan's part 3): missed by {amount} cm")
                words += f" Missed by {amount} centimeters."
        self.voice.say(words, t)
        self.set_light("green", t)
        self.finish_event(t)

    def reset(self, t):
        self.go("RESET", t, "gate up" if self.gate_down else "")
        if self.gate_down:
            self.gate(False, t)
        self.set_light("green", t)
        self.manual_hold = False

    def finish_event(self, t):
        ev = self.event
        ev["spots"] = [[round(a, 3), round(b, 1), round(c, 1)] for a, b, c in self.spots]
        ev["gate_dropped"] = self.protected and not self.warning_mode
        self.log.save_event(ev["event"], ev)
        self.replay_request = self.replay_window()

    def replay_window(self):
        """Which seconds the instant replay shows: around the gate drop, or around the moment
        the car went past the kid (when its x crossed the circle's x)."""
        if not self.spots:
            return None
        anchor = self.protect_time
        if anchor is None and self.circle():
            kid_x = self.circle()[0]
            first_side = self.spots[0][1] < kid_x
            for st, sx, _ in self.spots:
                if (sx < kid_x) != first_side:
                    anchor = st
                    break
        if anchor is None:
            anchor = self.spots[-1][0] - 0.8
        return max(anchor - 1.2, self.spots[0][0] - 0.2), anchor + self.settings["gate_seconds"] + 0.6

    # ---- buttons ----

    def press_protect(self, t):
        """The PROTECT button from Ryan's doc (SPACE): drops the gate now, or lifts it if it is down."""
        if self.gate_down or self.step in ("PROTECTION REQUIRED", "PROTECTION DEPLOYED"):
            self.log.write(t, "PROTECT button: lifting")
            if self.event is not None and self.step not in ("ALL CLEAR", "RESET"):
                self.finish_event(t)
            self.reset(t)
            return
        if self.event is None or self.step in ("ALL CLEAR", "RESET", "IDLE"):
            self.start_event(t, None, "PROTECT button")
        self.manual_hold = True
        self.protect(t, "PROTECT button pressed")

    def press_reset(self, t):
        if self.event is not None and self.step not in ("ALL CLEAR", "RESET", "IDLE"):
            self.finish_event(t)
        self.reset(t)

    # ---- asking Ryan's code ----

    def ask(self, part, func, *args):
        try:
            answer = func(*args)
        except Exception as e:
            here = traceback.extract_tb(e.__traceback__)[-1]
            msg = f"{type(e).__name__}: {e} (ryan_code.py line {here.lineno})"
            self.ryan_status[part] = f"error on line {here.lineno}"
            if msg not in self.errors_printed:
                self.errors_printed.add(msg)
                print(f"\n*** Ryan's part {part} hit an error. Kestrel keeps the gate up and keeps going. ***")
                traceback.print_exception(type(e), e, e.__traceback__)
            return None
        if answer is not None:
            self.ryan_status[part] = "ready"
        return answer

    def ask_part1(self, found):
        answer = self.ask(1, self.ryan.is_inside_circle, (found[0], found[1]), tuple(self.circle()))
        if answer is None:
            return None
        if not isinstance(answer, (bool, np.bool_)):
            self.ryan_status[1] = "needs True or False"
            print(f"Part 1 handed back {answer!r}. Kestrel needs True or False.")
            return None
        return bool(answer)

    def ask_part2(self):
        answer = self.ask(2, self.ryan.guess_and_decide, list(self.spots), tuple(self.circle()),
                          self.settings["gate_seconds"])
        if answer is None:
            return None
        if isinstance(answer, (bool, np.bool_)):
            return {"drop_gate": bool(answer)}
        if isinstance(answer, dict) and isinstance(answer.get("drop_gate"), (bool, np.bool_)):
            return dict(answer, drop_gate=bool(answer["drop_gate"]))
        self.ryan_status[2] = "needs True or False"
        print(f"Part 2 handed back {answer!r}. Kestrel needs True, False, or a dictionary with drop_gate.")
        return None

    def ask_part3(self):
        answer = self.ask(3, self.ryan.how_close, list(self.spots), tuple(self.circle()))
        if answer is None:
            return None
        if not isinstance(answer, (int, float, np.integer, np.floating)) or isinstance(answer, bool):
            self.ryan_status[3] = "needs a number"
            print(f"Part 3 handed back {answer!r}. Kestrel needs a number of pixels.")
            return None
        return float(answer)


# ---------------------------------------------------------------- 3 (fun). THE INSTANT REPLAY

class Replay:
    """Keeps the last 20 seconds of the screen, and after each roll plays the key moment back
    at quarter speed, with the car's whole path drawn on it."""
    SLOW = 4

    def __init__(self, keep=20.0, scale=0.75):
        self.frames = collections.deque()
        self.keep, self.scale = keep, scale
        self.clip, self.started, self.path = None, None, []

    def add(self, t, screen):
        small = cv2.resize(screen, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.frames.append((t, jpg))
        while self.frames and self.frames[0][0] < t - self.keep:
            self.frames.popleft()

    def start(self, window, now, path):
        clip = [f for f in self.frames if window[0] <= f[0] <= window[1]]
        if len(clip) >= 2:
            self.clip, self.started, self.path = clip, now, path

    def save(self, start, end, folder):
        """Saves the screen from start to end (real speed) as a video file, so every roll is kept
        (Ryan 10/13: "all my projects will go to waste since we dont take videos")."""
        frames = [f for f in self.frames if start <= f[0] <= end]
        if len(frames) < 10:
            return
        def write():
            folder.mkdir(parents=True, exist_ok=True)
            first = cv2.imdecode(frames[0][1], cv2.IMREAD_COLOR)
            fps = (len(frames) - 1) / max(frames[-1][0] - frames[0][0], 0.01)
            path = folder / (datetime.now().strftime("roll_%Y-%m-%d_%H%M%S") + ".mp4")
            out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
            for _, jpg in frames:
                out.write(cv2.imdecode(jpg, cv2.IMREAD_COLOR))
            out.release()
        threading.Thread(target=write, daemon=True).start()

    def playing(self):
        return self.clip is not None

    def stop(self):
        self.clip = None

    def length(self):
        """How long the replay takes to watch, in seconds."""
        return 0.0 if self.clip is None else (self.clip[-1][0] - self.clip[0][0]) * self.SLOW

    def frame(self, now):
        """The replay picture to show right now, or None once it is over."""
        if self.clip is None:
            return None
        clip_time = self.clip[0][0] + (now - self.started) / self.SLOW
        if clip_time > self.clip[-1][0]:
            self.clip = None
            return None
        pick = self.clip[0][1]
        for ft, jpg in self.clip:
            if ft > clip_time:
                break
            pick = jpg
        img = cv2.resize(cv2.imdecode(pick, cv2.IMREAD_COLOR), (SCREEN_W, VIEW_H))
        pts = np.array([[int(x), int(y)] for _, x, y in self.path], np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, CYAN, 2, cv2.LINE_AA)
        shade = img.copy()
        cv2.rectangle(shade, (0, 0), (VIEW_W, 64), BLACK, -1)
        cv2.addWeighted(shade, 0.55, img, 0.45, 0, img)
        cv2.circle(img, (34, 32), 11, RED, -1)
        label(img, f"INSTANT REPLAY   1/{self.SLOW} SPEED", (56, 44), 1.1, WHITE)
        return img


# ---------------------------------------------------------------- 7. THE SCREEN

def label(img, text, org, scale, color, thick=2):
    """Text with a dark outline, so it reads over any picture and in a video."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def label_centered(img, text, y, scale, color, thick=2, x0=0, width=VIEW_W):
    w = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)[0][0]
    label(img, text, (x0 + (width - w) // 2, y), scale, color, thick)


def draw_screen(frame, k, now, practice=False, note=None):
    """The whole screen: the camera picture with Kestrel's marks on it, plus the panel."""
    screen = np.zeros((VIEW_H, SCREEN_W, 3), np.uint8)
    view = frame.copy()
    for x1, y1, x2, y2 in k.settings["ignore"]:
        cv2.rectangle(view, (x1, y1), (x2, y2), DIM, 1)

    danger = k.step in ("PROTECTION REQUIRED", "PROTECTION DEPLOYED")
    c = k.circle()
    if c:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        hot = danger or k.inside is True
        if danger:
            fill = view.copy()
            cv2.circle(fill, (cx, cy), r, RED, -1)
            cv2.addWeighted(fill, 0.28, view, 0.72, 0, view)
        cv2.circle(view, (cx, cy), r, RED if hot else GREEN, 3, cv2.LINE_AA)
        label(view, "PROTECTED ZONE", (cx - r, cy - r - 10), 0.55, RED if hot else GREEN, 1)

    trail = [(int(x), int(y)) for _, x, y in k.spots[-90:]]
    if len(trail) >= 2:
        cv2.polylines(view, [np.array(trail, np.int32)], False, CYAN, 2, cv2.LINE_AA)
    for p in trail[-12:]:
        cv2.circle(view, p, 3, CYAN, -1, cv2.LINE_AA)

    a = k.answer or {}
    if k.car is not None and k.eyes.box is not None:
        x1, y1, x2, y2 = (int(v) for v in k.eyes.box)
        cv2.rectangle(view, (x1, y1), (x2, y2), ORANGE, 2)
        label(view, f"MY CAR {k.eyes.sure * 100:.0f}%", (x1, y2 + 22), 0.55, ORANGE, 1)
    if k.car is not None:
        x, y = int(k.car[0]), int(k.car[1])
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):          # tracking brackets
            corner = (x + sx * 46, y + sy * 30)
            cv2.line(view, corner, (corner[0] - sx * 16, corner[1]), WHITE, 2)
            cv2.line(view, corner, (corner[0], corner[1] - sy * 12), WHITE, 2)
        label(view, "TRACK 001", (x - 46, y - 38), 0.55, WHITE, 1)
        end = a.get("path_end")
        if end is not None:
            try:
                cv2.arrowedLine(view, (x, y), (int(end[0]), int(end[1])), RED if a.get("drop_gate") else ORANGE,
                                3, cv2.LINE_AA, tipLength=0.04)
            except (TypeError, ValueError, IndexError):
                pass

    left = a.get("seconds_left")
    if left is not None and c and k.step in ("ASSESSING", "PROTECTION REQUIRED"):
        label(view, f"IMPACT {float(left):.1f} s", (int(c[0]) - 70, int(c[1]) + int(c[2]) + 38), 0.9, RED, 2)

    if danger:
        shade = view.copy()
        cv2.rectangle(shade, (0, 0), (VIEW_W, 70), (20, 20, 150), -1)
        cv2.addWeighted(shade, 0.75, view, 0.25, 0, view)
        title = "WARNING: CAR COMING" if k.warning_mode else k.step
        label_centered(view, title, 50, 1.4, WHITE, 3)
    if k.banner and now < k.banner[1] and k.step in ("ALL CLEAR", "RESET"):   # only while its own roll ends (Oct 14, Ryan)
        label_centered(view, k.banner[0], 130, 2.0, YELLOW, 4)

    cv2.circle(view, (26, VIEW_H - 24), 8, RED if not practice else YELLOW, -1)
    fps = getattr(k, "fps", 0.0)
    label(view, "PRACTICE ROAD" if practice else f"LIVE  {fps:.0f} pictures a second",
        (42, VIEW_H - 15), 0.65, WHITE, 1)
    if note:
        label_centered(view, note, VIEW_H - 60, 0.85, YELLOW, 2)

    screen[:, :VIEW_W] = view
    draw_panel(screen, k, practice)
    return screen


def panel_line(img, label, value, y, color=WHITE):
    x0 = VIEW_W + 20
    cv2.putText(img, label, (x0, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, DIM, 1, cv2.LINE_AA)
    cv2.putText(img, value, (x0 + 175, y), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)


def draw_panel(img, k, practice):
    x0 = VIEW_W + 20
    img[:, VIEW_W:] = (24, 20, 18)
    cv2.line(img, (VIEW_W, 0), (VIEW_W, VIEW_H), (90, 80, 70), 2)
    cv2.putText(img, "KESTREL", (x0, 52), cv2.FONT_HERSHEY_DUPLEX, 1.5, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "PICKUP LINE GUARD", (x0 + 2, 78), cv2.FONT_HERSHEY_DUPLEX, 0.5, DIM, 1, cv2.LINE_AA)
    chips = ["AUTO" if k.auto else "MANUAL", "V3 GUESS EARLY" if k.version == 3 else "V2 REACT"]
    if k.warning_mode:
        chips.append("WARNING MODE")
    cx = x0
    for chip in chips:
        w = cv2.getTextSize(chip, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)[0][0] + 14
        cv2.rectangle(img, (cx, 92), (cx + w, 114), ORANGE if chip == "WARNING MODE" else (90, 80, 70), -1)
        cv2.putText(img, chip, (cx + 7, 108), cv2.FONT_HERSHEY_DUPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        cx += w + 8

    cv2.putText(img, "STEP", (x0, 150), cv2.FONT_HERSHEY_DUPLEX, 0.5, DIM, 1, cv2.LINE_AA)
    step_scale = 1.05 if len(k.step) <= 12 else 0.78
    cv2.putText(img, k.step, (x0, 186), cv2.FONT_HERSHEY_DUPLEX, step_scale, STEP_COLOR[k.step], 2, cv2.LINE_AA)

    a = k.answer or {}
    ppc = k.settings["px_per_cm"]
    speed = a.get("speed")
    left = a.get("seconds_left")
    decision = "--" if not a else ("DROP GATE" if a.get("drop_gate") else "WAIT")
    event = f"EVENT {k.event['event']:04d}" if k.event else "EVENT --"
    panel_line(img, "TRACK 001" if k.car is not None else "NO TRACK", event, 226, WHITE)
    panel_line(img, "SPEED", "--" if speed is None else f"{float(speed) / ppc:.0f} cm/s", 256)
    panel_line(img, "DECISION", decision, 286, RED if decision == "DROP GATE" else WHITE)
    panel_line(img, "IMPACT IN", "--" if left is None else f"{float(left):.1f} s", 316)

    gate = "OFF (warning)" if k.warning_mode else ("DOWN" if k.gate_down else "UP")
    panel_line(img, "GATE", gate, 362, RED if k.gate_down else WHITE)
    cv2.putText(img, "LIGHT", (x0, 398), cv2.FONT_HERSHEY_DUPLEX, 0.55, DIM, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0 + 175, 380), (x0 + 265, 408), (10, 10, 10), -1)
    for i, (name, on) in enumerate((("red", RED), ("yellow", YELLOW), ("green", GREEN))):
        cv2.circle(img, (x0 + 192 + i * 28, 394), 10, on if k.light == name else (55, 55, 55), -1, cv2.LINE_AA)
    board = "practice board" if practice else ("ready" if k.board_ok else "NOT READY")
    panel_line(img, "BOARD", board, 432, GREEN if k.board_ok else ORANGE)
    panel_line(img, "EYES", f"{k.eyes.name}  {k.eyes.ms:.0f} ms", 456, GREEN if k.eyes.ai else WHITE)

    cv2.putText(img, "RYAN'S CODE", (x0, 488), cv2.FONT_HERSHEY_DUPLEX, 0.55, DIM, 1, cv2.LINE_AA)
    names = {1: "1 inside circle", 2: "2 guess + decide", 3: "3 how close"}
    for i, part in enumerate((1, 2, 3)):
        status = k.ryan_status[part] if k.circle() else "place the circle: C"
        color = GREEN if status == "ready" else (RED if status.startswith("error") else ORANGE)
        cv2.putText(img, names[part], (x0, 512 + i * 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, status[:20], (x0 + 160, 512 + i * 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

    keys = ["SPACE protect   R reset   W warning",
            "2 / 3 version   A auto/manual   P replay",
            "C circle   K car color   S scale   I ignore",
            "L learn my car   T startup test   Q quit"]
    if practice:
        keys.insert(0, "H at kid  M to miss  F fast  D red brick")
    for i, line in enumerate(keys):
        cv2.putText(img, line, (x0, 604 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, DIM, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- 1 (fun). THE STARTUP TEST

ANY = "any"
GATE_TIP = ["The gate did not move right. Check 3 things:",
            "1. its signal wire is in the empty hole right after gate 2's wire (the leg labeled 26)",
            "2. its red wire is on the red (+) strip    3. its brown wire is on the blue (-) strip"]
SOUND_TIP = ["Click the speaker icon at the top right of the Mac's screen,",
             "pick your Bluetooth speaker, and turn the volume up."]
LIGHT_LEGS = {"red": "R wire beside the leg labeled 25", "yellow": "Y wire beside the leg labeled 32",
              "green": "G wire beside the leg labeled 12"}


class Ask:
    def __init__(self, text, keys):
        self.text, self.keys = text, keys


class Pause:
    def __init__(self, seconds):
        self.seconds = seconds


class Watch:
    """Wait until something is true (like the camera seeing the car), or a key skips it."""
    def __init__(self, text, keys, until, timeout):
        self.text, self.keys, self.until, self.timeout = text, keys, until, timeout


class StartupTest:
    """FUN FEATURE 1: Kestrel checks itself when it starts, like a jet before takeoff. It moves
    the gate and flashes the light by itself and asks Y or N, so the wiring gets checked with a
    key press instead of a photo. Anything that fails says the one thing to try."""

    def __init__(self, k, practice=False, road=None):
        self.k, self.practice, self.road = k, practice, road
        self.rows = []              # [name, state, note]; state is "...", "OK", "FIX" or "SKIP"
        self.tip = None
        self.done = False
        self.waiting, self.since = None, 0.0
        self.now, self.car = 0.0, None
        self.gen = self.checks()

    def update(self, now, key, frame):
        """Called every moment while the test runs. key is the key just pressed, or None."""
        self.now = now
        k = self.k
        k.voice.tick(now)
        for reply in k.board.read():
            if k.echo_board:
                k.log.write(now, f"      board says: {reply}")
        self.car = find_car(frame, k.settings["look"], k.settings["min_area"], k.settings["ignore"])
        if self.done:
            return
        if key == "esc":
            self.rows.append(["SKIPPED", "SKIP", "the rest was skipped"])
            self.done = True
            return
        w, value = self.waiting, None
        if isinstance(w, Pause):
            if now - self.since < w.seconds:
                return
        elif isinstance(w, Ask):
            if key is None or (w.keys != ANY and key not in w.keys):
                return
            value = key
        elif isinstance(w, Watch):
            if w.until():
                value = "seen"
            elif key is not None and key in w.keys:
                value = key
            elif now - self.since > w.timeout:
                value = "timeout"
            else:
                return
        try:
            self.waiting = self.gen.send(value)
            self.since = now
        except StopIteration:
            self.done = True

    def row(self, name, note):
        self.rows.append([name, "...", note])
        return self.rows[-1]

    def checks(self):
        k = self.k
        k.voice.say("Kestrel online.", self.now)
        yield Pause(0.6)

        r = self.row("BOARD", "saying hello")
        if k.hello(self.now, wait=0 if self.practice else 2.5):
            r[1:] = ["OK", "kestrel ready"]
        else:
            r[1:] = ["FIX", "the board did not answer"]
            self.tip = ["The board did not say kestrel ready. Is its USB cable in?",
                        "If it still has the turret's code, put the Kestrel code on first (README)."]
            yield Ask("Press any key to keep going without the board", ANY)
            self.tip = None

        if k.board_ok and not self.practice and (k.settings["gate_up"] is None or k.settings["gate_down"] is None):
            r = self.row("GATE ANGLES", "finding up and down")
            for word, where, angle in (("up", "UP (about halfway is enough)", 90), ("down", "FLAT across the road", 30)):
                while True:
                    k.board.send(f"k gate move {angle:03d}", self.now)
                    key = yield Ask(f"Press < or > to move the arm {where}. Enter when it is right.", "<>,.\r")
                    if key in "<,":
                        angle = max(0, angle - 3)
                    elif key in ">.":
                        angle = min(180, angle + 3)
                    else:
                        break
                k.settings["gate_" + word] = angle
            save_settings(k.settings)
            r[1:] = ["OK", f"up {k.settings['gate_up']}, down {k.settings['gate_down']}"]

        if k.board_ok:
            r = self.row("GATE", "dropping it")
            while True:
                k.gate(True, self.now)
                yield Pause(1.0)
                key = yield Ask("Did the gate arm come DOWN across the road?   Y or N", "yn")
                if key == "y":
                    k.gate(False, self.now)
                    yield Pause(0.8)
                    key = yield Ask("Did it go back UP?   Y or N", "yn")
                    if key == "y":
                        r[1:] = ["OK", "down and up"]
                        break
                k.gate(False, self.now)
                r[1:] = ["FIX", "did not move right"]
                self.tip = GATE_TIP
                key = yield Ask("Fix it, then press any key to try again (S skips)", ANY)
                self.tip = None
                if key == "s":
                    r[1:] = ["SKIP", "skipped"]
                    break

            r = self.row("LIGHT", "red, yellow, green")
            wrong = []
            for color in ("red", "yellow", "green"):
                k.set_light(color, self.now)
                key = yield Ask("Which light is on?   R, Y, G, or N for none", "rygn")
                if key != color[0]:
                    wrong.append((color, {"r": "red", "y": "yellow", "g": "green", "n": "nothing"}[key]))
            k.set_light("green", self.now)
            if not wrong:
                r[1:] = ["OK", "all three"]
            else:
                r[1:] = ["FIX", f"{len(wrong)} of 3 wrong"]
                self.tip = [f"When Kestrel lit {want.upper()}, you saw {saw.upper()}." for want, saw in wrong]
                self.tip.append("Check: " + ", ".join(LIGHT_LEGS[want] for want, _ in wrong)
                                + ", and the GND wire on the blue strip.")
                yield Ask("Press any key to keep going", ANY)
                self.tip = None

        r = self.row("SOUND", "honk and voice")
        k.voice.honk(self.now)
        k.voice.say("Testing.", self.now, after=0.8)
        yield Pause(1.8)
        key = yield Ask("Did you hear the honk and the voice?   Y or N", "yn")
        if key == "y":
            r[1:] = ["OK", "loud and clear"]
        else:
            r[1:] = ["FIX", "no sound"]
            self.tip = SOUND_TIP
            yield Ask("Press any key to keep going", ANY)
            self.tip = None

        r = self.row("CAMERA", "looking for the red car")
        if self.practice and self.road is not None:
            self.road.parked = True
        got = yield Watch("Put the red car on the road where the camera can see it   (S skips)", "s",
                          until=lambda: self.car is not None, timeout=30)
        if self.practice and self.road is not None:
            self.road.parked = False
        if got == "seen":
            r[1:] = ["OK", "car found"]
        else:
            r[1:] = ["FIX", "no car found"]
            self.tip = ["Kestrel could not find the red car. After the test, press K",
                        "and click the middle of the car to teach Kestrel its color."]
            yield Ask("Press any key to keep going", ANY)
            self.tip = None

        r = self.row("CIRCLE", "")
        r[1:] = ["OK", "placed"] if k.circle() else ["FIX", "after the test, press C and click the kid"]

        good = all(row[1] in ("OK", "SKIP") for row in self.rows)
        if good:
            self.row("ALL SYSTEMS GO", "")[1] = "OK"
        k.voice.say("All systems go." if good else "Check the red ones.", self.now)
        yield Pause(1.6)

    def draw(self, screen):
        """The checklist on top of the camera picture."""
        box = screen.copy()
        cv2.rectangle(box, (240, 60), (VIEW_W - 240, VIEW_H - 60), (18, 16, 14), -1)
        cv2.addWeighted(box, 0.88, screen, 0.12, 0, screen)
        label_centered(screen, "KESTREL ONLINE", 120, 1.6, WHITE, 3)
        label_centered(screen, "SYSTEMS CHECK", 158, 0.8, DIM, 1)
        for i, (name, state, note) in enumerate(self.rows[-8:]):
            y = 210 + i * 42
            color = {"OK": GREEN, "FIX": RED, "SKIP": DIM}.get(state, YELLOW)
            cv2.circle(screen, (300, y - 9), 13, color, -1, cv2.LINE_AA)
            if state == "OK":
                cv2.line(screen, (293, y - 9), (299, y - 3), BLACK, 3, cv2.LINE_AA)
                cv2.line(screen, (299, y - 3), (309, y - 16), BLACK, 3, cv2.LINE_AA)
            elif state == "FIX":
                cv2.line(screen, (294, y - 15), (306, y - 3), BLACK, 3, cv2.LINE_AA)
                cv2.line(screen, (306, y - 15), (294, y - 3), BLACK, 3, cv2.LINE_AA)
            label(screen, name, (330, y), 0.85, WHITE, 2)
            label(screen, note[:34], (600, y), 0.65, color if state != "..." else DIM, 1)
        ask = getattr(self.waiting, "text", None)
        if self.tip:
            for i, line in enumerate(self.tip):
                label_centered(screen, line, VIEW_H - 170 + i * 30, 0.62, ORANGE, 1)
        if ask:
            label_centered(screen, ask, VIEW_H - 84, 0.8, YELLOW, 2)
        label_centered(screen, "ESC skips the test", VIEW_H - 40, 0.5, DIM, 1)


# ---------------------------------------------------------------- the clicking tools

class Tools:
    """Placing the circle (C), teaching the car's color (K), the scale (S), ignore boxes (I)."""

    def __init__(self, k, save):
        self.k, self.save = k, save
        self.mode, self.points, self.note = None, [], None
        self.click = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and x < VIEW_W:
            self.click = (x, y)

    def start(self, mode):
        self.mode, self.points, self.click = mode, [], None
        self.note = {"circle": "Click the KID.",
                     "color": "Click the middle of the red car.",
                     "scale": "Lay the ruler along the road. Click its 0 mark, then its 30 cm mark.",
                     "ignore": "Click two corners of a box Kestrel should ignore (X clears them all)."}[mode]

    def update(self, frame, key):
        """Handles a click or a key for the tool that is on. Returns True while a tool is on."""
        if self.mode is None:
            return False
        s = self.k.settings
        click, self.click = self.click, None
        if key == "esc":
            self.mode, self.note = None, None
            return False
        if self.mode == "circle":
            if click and not self.points:
                self.points = [click]
                s["circle"] = [click[0], click[1], 60]
                self.note = "Press [ or ] to grow the circle until its edge touches the gate arm. Enter saves."
            elif self.points and key in ("[", "]"):
                s["circle"][2] = max(10, s["circle"][2] + (6 if key == "]" else -6))
            elif self.points and key == "\r":
                self.finish("circle saved")
        elif self.mode == "color" and click:
            look, box = learn_color(frame, click[0], click[1])
            if look is None:
                self.note = "No red car under that click. Click the middle of the car."
            else:
                s["look"] = look
                s["car_size"] = [max(box[2], box[3]), min(box[2], box[3])]   # so "learn my car" skips hands
                self.finish(f"car color learned from the whole car (hue {look['hue']})")
        elif self.mode == "scale" and click:
            self.points.append(click)
            if len(self.points) == 2:
                gap = abs(self.points[1][0] - self.points[0][0])        # the ruler lies along the road
                if gap > 30:
                    s["px_per_cm"] = round(gap / 30.0, 2)
                    s["min_area"] = int(7 * 3 * s["px_per_cm"] ** 2 * 0.2)   # a fifth of a 7 by 3 cm car
                    self.finish(f"scale saved: {s['px_per_cm']} pixels per cm")
                else:
                    self.points = []
                    self.note = "Those clicks were too close. Click the 0 mark, then the 30 cm mark."
        elif self.mode == "ignore":
            if key == "x":
                s["ignore"] = []
                self.finish("ignore boxes cleared")
            elif click:
                self.points.append(click)
                if len(self.points) == 2:
                    (x1, y1), (x2, y2) = self.points
                    s["ignore"].append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                    self.finish("ignore box saved")
        return self.mode is not None

    def finish(self, message):
        self.save()
        self.mode, self.points = None, []
        self.note = message
        print("   " + message)


# ---------------------------------------------------------------- running it

def open_camera(index):
    for backend in (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            # the Insta360 only gives 60 pictures a second at 1080p (measured 10/13: 60.0 at 1080p,
            # 30 at 720p), so ask for 1080p and shrink each picture to VIEW_W x VIEW_H in Camera.run
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 60)
            return cap
        cap.release()
    return None


class Camera:
    """Reads the webcam on its own, all the time, and keeps only the NEWEST picture. Otherwise
    pictures can wait in line, and Kestrel would be deciding about where the car WAS."""

    def __init__(self, index):
        self.cap = open_camera(index)
        self.ok = self.cap is not None
        self.lock = threading.Lock()
        self.frame, self.stamp, self.number, self.fps = None, 0.0, 0, 0.0
        if self.ok:
            self.cap.set(cv2.CAP_PROP_FPS, 60)          # asks for 60 pictures a second; it may say no
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def close(self):
        """Lets go of the webcam cleanly, so quitting never crashes."""
        if self.cap is None:
            return
        self.ok = False
        if getattr(self, "thread", None):
            self.thread.join(timeout=1.0)
        self.cap.release()

    def run(self):
        count, since = 0, time.monotonic()
        misses = 0
        while self.ok:
            ok, frame = self.cap.read()
            if not ok:
                # one missed picture is normal for this webcam, mostly right after it opens
                # (Oct 13 test night); only give up after 2 seconds of nothing
                misses += 1
                if misses > 120:
                    self.ok = False
                    break
                time.sleep(1 / 60)
                continue
            misses = 0
            if frame.shape[1] != VIEW_W:
                frame = cv2.resize(frame, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
            now = time.monotonic()
            with self.lock:
                self.frame, self.stamp, self.number = frame, now, self.number + 1
            count += 1
            if now - since >= 2.0:
                self.fps, count, since = count / (now - since), 0, now

    def newest(self, seen):
        """Waits (up to a fifth of a second) for a picture newer than number `seen`."""
        give_up = time.monotonic() + 0.2
        while time.monotonic() < give_up:
            with self.lock:
                if self.number > seen:
                    return self.frame, self.stamp, self.number
            time.sleep(0.001)
        return None, None, seen


class Learning:
    """KEY L, Kestrel learns your car: 1) it collects pictures while the car rolls, 2) it trains
    the AI on the laptop, 3) it tests the AI on pictures it never saw. Only an AI that passes the
    test becomes part of Kestrel's eyes (the bar is in learn_car.py)."""

    def __init__(self, practice):
        import learn_car
        self.lc = learn_car
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.folder = HERE / "learn" / (("practice-" if practice else "") + stamp)
        self.collector = learn_car.Collector(self.folder)
        self.final = PRACTICE_CAR_AI if practice else MY_CAR_AI
        self.stage, self.proc, self.result, self.started = "collect", None, None, None

    def feed(self, frame, settings):
        if self.stage == "collect":
            blobs = red_blobs(frame, settings["look"], settings["min_area"], settings["ignore"])
            cars = [b[3] for b in blobs if car_sized(b[3], settings.get("car_size")) and strong_red(frame, b[3])]
            if cars:
                self.collector.add(frame, cars[0])
            elif not any(strong_red(frame, b[3]) for b in blobs):
                self.collector.add(frame, None)     # no deep red anywhere: an empty road picture
            # else deep red that is not car-sized (a hand holding the car): skip the picture
            if self.collector.done():
                self.stage = "ready"

    def start_training(self):
        self.out = open(self.folder / "training.log", "w")
        # caffeinate -d -i keeps the screen and the Mac awake while it trains: a dark screen made the
        # Mac slow the training down, up to 16 minutes for a round that takes 8 seconds (measured Oct 6)
        self.proc = subprocess.Popen(["caffeinate", "-d", "-i", sys.executable, str(HERE / "learn_car.py"), "learn",
                                      str(self.folder), str(self.final)], stdout=self.out, stderr=subprocess.STDOUT,
                                     cwd=str(HERE))
        self.stage, self.started = "training", time.monotonic()

    def poll(self):
        """True once, at the moment training finishes."""
        if self.stage == "training" and self.proc.poll() is not None:
            self.out.close()
            path = self.folder / "result.json"
            self.result = json.loads(path.read_text()) if path.exists() else None
            self.stage = "done"
            return True
        return False

    def rounds(self):
        csv = self.folder / "runs" / "my_car" / "results.csv"
        return max(0, len(csv.read_text().splitlines()) - 1) if csv.exists() else 0

    def note(self):
        c = self.collector
        if self.stage == "collect":
            return (f"LEARNING YOUR CAR: {c.cars} of {self.lc.WANT_CAR} car pictures, "
                    f"{c.empties} of {self.lc.WANT_EMPTY} road pictures. Keep rolling, both aims.")
        if self.stage == "ready":
            return "Got them all. Plug in the charger, then press Enter to train the AI (about 5 minutes)."
        if self.stage == "training":
            return f"TRAINING THE AI: round {self.rounds()} of 40   ({(time.monotonic() - self.started) / 60:.1f} min)"
        r = self.result
        if r is None:
            return f"Training stopped with an error. See {self.folder.name}/training.log"
        verdict = "PASSED: the AI is now part of Kestrel's eyes" if r["passed"] else "NOT GOOD ENOUGH YET, color eyes stay"
        return f"{verdict}. Found the car {r['found']} of {r['cars']}, false cars {r['false_cars']} of {r['empties']}."


def find_cameras():
    """Saves one picture from each camera number to logs/, to see which one is the webcam."""
    (HERE / "logs").mkdir(exist_ok=True)
    for i in range(5):
        cap = open_camera(i)
        frame = None
        if cap is not None:
            for _ in range(8):
                ok, frame = cap.read()
            cap.release()
        if frame is not None:
            path = HERE / "logs" / f"camera_{i}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"camera {i}: {frame.shape[1]}x{frame.shape[0]}  saved {path}")
        else:
            print(f"camera {i}: nothing")


KEY_NAMES = {27: "esc", 13: "\r", 10: "\r", 32: " "}


def read_key(wait_ms):
    code = cv2.waitKey(wait_ms)
    if code < 0:
        return None
    code &= 0xFF
    return KEY_NAMES.get(code, chr(code).lower() if 32 <= code < 127 else None)


def main():
    ap = argparse.ArgumentParser(description="Kestrel, the pickup line guard")
    ap.add_argument("--practice", action="store_true", help="the fake road: no camera or board needed")
    ap.add_argument("--practice-board", action="store_true",
                    help="the fake road on screen, but the real gate arms and light move with it")
    ap.add_argument("--camera", type=int, default=None, help="which camera number is the webcam")
    ap.add_argument("--find-camera", action="store_true", help="save a picture from each camera number")
    ap.add_argument("--serial", default=None, help="the board's port (found by itself if left out)")
    ap.add_argument("--no-board", action="store_true", help="run without the board")
    ap.add_argument("--version", type=int, choices=(2, 3), default=3, help="2 = react, 3 = guess early")
    ap.add_argument("--warning", action="store_true", help="start in warning mode (no gate)")
    ap.add_argument("--skip-test", action="store_true", help="skip the startup test")
    ap.add_argument("--quiet", action="store_true", help="no sound")
    args = ap.parse_args()
    if args.find_camera:
        return find_cameras()

    road = None
    camera = None
    if args.practice_board:
        args.practice = True
    if args.practice:
        import fake_road
        road = fake_road.FakeRoad(seed=int(time.time()))
        board = fake_road.FakeBoard(road)
        settings = practice_settings()
        if args.practice_board:
            port = args.serial or find_board_port()
            if not port:
                raise SystemExit("No board found: plug the ESP32's USB cable into the laptop.")
            real = load_settings()
            for key in ("gate_up", "gate_down", "gate2_up", "gate2_down"):
                settings[key] = real.get(key)    # the real arms' own angles, never the fake road's
            board = BothBoards(board, RealBoard(port))
        log = Log(HERE / "logs" / "practice", wall_clock)
        save = lambda: None                      # practice changes are not saved
    else:
        settings = load_settings()
        if args.camera is not None:
            settings["camera"] = args.camera
        camera = Camera(settings["camera"])
        if not camera.ok:
            raise SystemExit(f"Camera {settings['camera']} did not open. Try: python kestrel_brain.py --find-camera")
        port = None if args.no_board else (args.serial or find_board_port())
        board = RealBoard(port) if port else NoBoard()
        log = Log(HERE / "logs", wall_clock)
        save = lambda: save_settings(settings)

    # keeps the screen and the Mac awake while Kestrel is open, and stops by itself when Kestrel quits
    subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())])
    voice = Voice(settings["voice"], quiet=args.quiet)
    eyes = Eyes(settings, PRACTICE_CAR_AI if args.practice else MY_CAR_AI)
    k = Kestrel(settings, board, voice, log, version=args.version, warning_mode=args.warning,
                echo_board=not args.practice, eyes=eyes)
    learning = None
    tools = Tools(k, save)
    replay = Replay()
    test = None if args.skip_test else StartupTest(k, args.practice, road)
    start = time.monotonic()
    if args.skip_test:
        k.hello(0.0, wait=0 if args.practice else 2.5)

    cv2.namedWindow("Kestrel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kestrel", 1400, 600)
    cv2.setMouseCallback("Kestrel", tools.on_mouse)
    last_live, key, frame_time, seen = -1.0, None, 1 / 30, 0
    print("Kestrel is running. Keys are on the right side of the screen. Q quits.")

    frame_count = 0
    while True:
        tick = time.monotonic()
        now = tick - start
        if road is not None:
            road.update(now)
            frame = road.frame(now)
            if learning is not None and learning.stage == "collect" and road.roll_over(now):
                road.start_roll(road.rng.choice(["hit", "miss"]), road.rng.uniform(0.4, 0.9), now)   # practice rolls itself
        else:
            frame, stamp, seen = camera.newest(seen)
            if frame is None:
                if not camera.ok:
                    print("The camera stopped sending pictures.")
                    break
                continue
            now = stamp - start                      # when this picture arrived, not when we got to it
            k.fps = camera.fps
            if frame.shape[:2] != (VIEW_H, VIEW_W):
                frame = cv2.resize(frame, (VIEW_W, VIEW_H))
        if learning is not None:
            learning.feed(frame, settings)
            if learning.poll():
                r = learning.result
                log.write(now, "learn my car: " + learning.note())
                if r and r.get("passed"):
                    k.eyes.load(learning.final)

        if test is not None and not test.done:
            test.update(now, key, frame)
            screen = draw_screen(frame, k, now, args.practice)
            test.draw(screen)
            if test.done:
                test = None
        else:
            if key is not None and replay.playing():
                replay.stop()
            if not tools.update(frame, key) and key is not None:
                if key == "q":
                    break
                elif key == " ":
                    k.press_protect(now)
                elif key == "r":
                    k.press_reset(now)
                elif key == "w":
                    if k.gate_down:
                        k.press_reset(now)
                    k.warning_mode = not k.warning_mode
                    log.write(now, "warning mode " + ("ON: light and honk only" if k.warning_mode else "OFF: the gate is back"))
                elif key == "a":
                    k.auto = not k.auto
                    log.write(now, "AUTO: Kestrel drops the gate itself" if k.auto else "MANUAL: only SPACE drops the gate")
                elif key in ("2", "3"):
                    k.version = int(key)
                    log.write(now, "version 2: react when the car is in the circle" if k.version == 2
                              else "version 3: guess early")
                elif key == "p":
                    settings["replay"] = not settings["replay"]
                    save()
                elif key == "t":
                    test = StartupTest(k, args.practice, road)
                elif key == "l" and (learning is None or learning.stage == "done"):
                    learning = Learning(args.practice)
                    log.write(now, "learn my car: collecting pictures in " + learning.folder.name)
                elif key == "\r" and learning is not None and learning.stage == "ready":
                    learning.start_training()
                    log.write(now, "learn my car: training started")
                elif key in ("c", "k", "s", "i"):
                    tools.start({"c": "circle", "k": "color", "s": "scale", "i": "ignore"}[key])
                elif road is not None and key == "d":
                    road.decoy = not road.decoy
                    log.write(now, "practice: a red brick " + ("dropped on the sidewalk. Can it fool Kestrel?" if road.decoy else "picked up"))
                elif road is not None and key in ("h", "m", "f") and road.roll_over(now):
                    kind = "miss" if key == "m" else "hit"
                    speed = 1.1 if key == "f" else road.rng.uniform(0.45, 0.75)
                    road.start_roll(kind, speed, now)
                    log.write(now, f"practice: rolling a car {'at the kid' if kind == 'hit' else 'to just miss'}, {speed:.2f} m/s")
            if road is None and settings.get("auto_scene", True) and k.step == "IDLE" and frame_count % 30 == 0:
                auto_scene(frame, settings)      # twice a second, only while no car is on the road
            frame_count += 1
            k.see(frame, now)
            if k.step != "IDLE" and k.step not in ("ALL CLEAR", "RESET") and replay.playing():
                replay.stop()                              # a new car: back to live right away
            note = tools.note if tools.mode else None
            if learning is not None and not tools.mode and (learning.stage != "done" or now % 20 < 8):
                note = learning.note()
            elif not k.circle() and not tools.mode:
                note = "No protected circle yet: press C and click the kid"
            screen = draw_screen(frame, k, now, args.practice, note)
            replay.add(now, screen)
            if k.replay_request is not None:
                pts = [(x, y) for _, x, y in k.spots]
                moved = max((math.hypot(x - pts[0][0], y - pts[0][1]) for x, y in pts), default=0)
                if moved >= 15 * settings["px_per_cm"]:                         # a real roll, not a tap
                    if settings["replay"]:
                        replay.start(k.replay_request, now, list(k.spots))
                    if not args.practice:
                        replay.save(k.replay_request[0] - 1.5, now, CLIPS_FOLDER)
                k.replay_request = None
            if replay.playing():
                shown = replay.frame(now)
                if shown is not None:
                    if k.banner and now < k.banner[1]:
                        label_centered(shown, k.banner[0], 130, 2.0, YELLOW, 4)
                    screen = shown

        if now - last_live >= 1.0:                # what Claude looks at instead of a photo
            last_live = now
            live = HERE / "logs" / "live.jpg"
            cv2.imwrite(str(live) + ".tmp.jpg", cv2.resize(screen, None, fx=0.75, fy=0.75))
            os.replace(str(live) + ".tmp.jpg", live)

        cv2.imshow("Kestrel", screen)
        spent = time.monotonic() - tick
        key = read_key(max(1, int((frame_time - spent) * 1000)) if road is not None else 1)

    if k.gate_down:
        k.gate(False, time.monotonic() - start)
    k.set_light("off", time.monotonic() - start)
    if camera is not None:
        camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
