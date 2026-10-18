"""
The fake road: a pretend webcam looking straight down at a toy road, plus a pretend board.

Practice mode and the simulations use it, so Kestrel can be tested before any real part is
plugged in. Nothing in this file is part of Kestrel itself. Kestrel only ever sees the
pictures this file draws, exactly the way it will see the real webcam's pictures.

Everything is measured in centimeters on the poster board and drawn at 12 pixels per cm,
about what the real webcam sees from 2 feet up. The road runs left to right, the car comes
off the ramp on the left, passes the bus, and reaches the crosswalk where the kid is.
"""

import random

import cv2
import numpy as np

PX_PER_CM = 12
W, H = 1280, 720                         # same picture size as the real webcam

# The scene, in cm from the top-left corner of the picture
FAR_CURB_Y, CENTER_Y, NEAR_CURB_Y = 15.0, 30.0, 45.0
BUS = (28.0, 31.5, 58.0, 43.5)           # left, top, right, bottom (the bus sits at the near curb)
CROSSWALK = (62.0, 70.0)                 # left and right edge of the crosswalk
KID = (66.0, 27.0)                       # the LEGO kid, stepping out in front of the bus
GATE_X = 60.0                            # where the gate arm comes down across the road
GATE_POST = (60.0, 47.0)
LIGHT_SPOT = (55.0, 50.0)
DECOY = (83.0, 49.0, 89.0, 55.0)         # a red toy brick on the sidewalk, bigger than the car (key D)
CAR_LONG, CAR_WIDE = 7.0, 3.0            # a Hot Wheels car is about 7 cm by 3 cm
CIRCLE_RADIUS_CM = KID[0] - GATE_X       # the protected circle reaches the gate: 6 cm

# Each roll starts at the ramp's end and is aimed at a spot. "hit" rolls are aimed at the kid,
# "miss" rolls at a spot 10 cm away from the kid, so they pass about 4 cm outside the 6 cm
# circle. The labels come from how each roll is set up, like the tape marks on the real board.
ROLLS = {
    "hit": {"start": (8.0, 26.0), "aim": KID},
    "miss": {"start": (8.0, 23.0), "aim": (66.0, 17.0)},
}

CAMERA_LATE = 0.10        # seconds: a real webcam shows each moment a little late
GATE_DROP_SECONDS = 0.30  # how long the fake gate arm takes to come all the way down
SLOW_DOWN = 12.0          # cm per second, every second: how fast a rolling car loses speed
PICKUP_AFTER = 1.0        # seconds a stopped car sits there before "Ryan" picks it up


def px(cm):
    """Centimeters on the board to pixels in the picture."""
    return int(round(cm * PX_PER_CM))


class Roll:
    """One toy car rolling down the road. This is the fake world moving its toy, the way
    gravity and the ramp move the real one. Kestrel never sees any of these numbers."""

    def __init__(self, kind, speed_mps, start_time, rng):
        setup = ROLLS[kind]
        self.kind = kind
        self.t0 = start_time
        self.speed = speed_mps * 100.0                      # cm per second along the road
        wobble = rng.uniform(-1.0, 1.0)                     # the ramp is never aimed perfectly
        (x0, y0), (xa, ya) = setup["start"], setup["aim"]
        self.x0, self.y0 = x0, y0
        self.drift = (ya + wobble - y0) / (xa - x0)          # sideways cm per cm along the road
        self.stop_time = start_time + self.speed / SLOW_DOWN  # when it would run out of speed
        self.blocked_at = None    # world time the gate arm stopped it, if it did
        self.gone_at = None       # world time it was picked up or left the picture

    def x_at(self, t):
        """How far along the road the car's middle is at world time t, in cm."""
        if t < self.t0:
            return None
        if self.blocked_at is not None and t >= self.blocked_at:
            t = self.blocked_at
        tt = min(t, self.stop_time) - self.t0
        return self.x0 + self.speed * tt - 0.5 * SLOW_DOWN * tt * tt

    def where(self, t):
        """(x, y) of the car's middle in cm at world time t, or None when there is no car."""
        if t < self.t0 or (self.gone_at is not None and t >= self.gone_at):
            return None
        x = self.x_at(t)
        return x, self.y0 + (x - self.x0) * self.drift

    def front_reaches_gate(self):
        """World time the car's front would reach the gate line, or None if it stops first."""
        need = (GATE_X - CAR_LONG / 2) - self.x0           # cm the middle travels until the front is there
        lo, hi = self.t0, self.stop_time
        if self.x_at(hi) < self.x0 + need:
            return None
        for _ in range(40):                                # halve the time window 40 times
            mid = (lo + hi) / 2
            if self.x_at(mid) < self.x0 + need:
                lo = mid
            else:
                hi = mid
        return hi


_NOISE = []


def camera_noise():
    """A few frames of camera fuzz, made once and reused (making them is the slow part)."""
    if not _NOISE:
        rng = np.random.default_rng(0)
        _NOISE.extend(rng.normal(0, 4, (H, W, 3)).astype(np.int16) for _ in range(6))
    return _NOISE


class FakeRoad:
    """The practice world: draws what the webcam would see, moves the fake gate arm and light."""

    def __init__(self, seed=0, noisy=True):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.noisy = noisy
        self.roll = None
        self.parked = False            # a car parked at the ramp, for the startup test's camera check
        self.decoy = False             # a red brick on the sidewalk, to try to fool Kestrel's eyes
        self.gate_moves = []           # (world time, "down" or "up") in the order the board got them
        self.gate_down_at = None       # world time the arm finished coming down, for the scoring
        self.light = "off"
        self.kid_hit = False
        self.base = self._draw_still_parts()
        self.noise = camera_noise()

    # ---- what the pretend board does ----

    def gate_command(self, down, t):
        self.gate_moves.append((t, "down" if down else "up"))
        if down and self.roll is not None:
            done = t + GATE_DROP_SECONDS
            if self.gate_down_at is None:
                self.gate_down_at = done
            reach = self.roll.front_reaches_gate()
            if reach is not None and done <= reach and self.roll.blocked_at is None:
                self.roll.blocked_at = reach          # the arm is down before the car gets there

    def gate_angle(self, t):
        """How far down the arm is at world time t: 0 = straight up, 90 = flat across the road."""
        angle, last_t, going_down = 0.0, None, False
        for when, move in self.gate_moves:
            if when > t:
                break
            if last_t is not None:
                angle = self._swing(angle, going_down, when - last_t)
            last_t, going_down = when, move == "down"
        if last_t is not None:
            angle = self._swing(angle, going_down, t - last_t)
        return angle

    @staticmethod
    def _swing(angle, going_down, seconds):
        step = 90.0 * seconds / GATE_DROP_SECONDS
        return min(90.0, angle + step) if going_down else max(0.0, angle - step)

    # ---- rolls ----

    def start_roll(self, kind, speed_mps, t):
        self.parked = False
        self.kid_hit = False
        self.gate_down_at = None
        self.roll = Roll(kind, speed_mps, t, self.rng)

    def update(self, t):
        """Moves the world to time t: a stopped car gets picked up, a hit roll that gets past
        the gate knocks the kid over (by its label, the way the real tape marks work)."""
        r = self.roll
        if r is None or r.gone_at is not None:
            return
        spot = r.where(t)
        if spot is None:
            return
        if spot[0] - CAR_LONG / 2 > W / PX_PER_CM:
            r.gone_at = t                                     # rolled out of the picture
        stopped = r.blocked_at if r.blocked_at is not None else r.stop_time
        if t >= stopped + PICKUP_AFTER:
            r.gone_at = t                                     # "Ryan" picks the car up
        if r.kind == "hit" and r.blocked_at is None and spot[0] >= KID[0]:
            self.kid_hit = True

    def roll_over(self, t):
        return self.roll is None or (self.roll.gone_at is not None and t >= self.roll.gone_at)

    def car_spot_px(self, t):
        """For the checks only: where the car's middle really is in the picture at time t
        (what the camera shows, so it includes the camera being late). Kestrel never calls this."""
        if self.roll is None:
            return None
        spot = self.roll.where(t - CAMERA_LATE)
        return None if spot is None else (spot[0] * PX_PER_CM, spot[1] * PX_PER_CM)

    # ---- the picture ----

    def frame(self, t):
        """What the webcam shows at time t (the world a little earlier, since cameras are late)."""
        seen = t - CAMERA_LATE
        img = self.base.copy()
        self._draw_ramp(img)
        self._draw_kid(img)
        self._draw_light(img)
        if self.decoy:
            self._draw_decoy(img)
        if self.roll is not None:
            spot = self.roll.where(seen)
            if spot is not None:
                self._draw_car(img, spot, self.roll.drift)
        elif self.parked:
            self._draw_car(img, ROLLS["hit"]["start"], 0.0)
        self._draw_gate(img, self.gate_angle(seen))
        if not self.noisy:
            return img
        dx, dy = self.rng.randint(-1, 1), self.rng.randint(-1, 1)   # the camera shakes a little
        img = np.roll(img, (dy, dx), axis=(0, 1))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        noise = self.noise[self.rng.randrange(len(self.noise))]
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _draw_still_parts(self):
        img = np.full((H, W, 3), (214, 216, 218), np.uint8)                       # sidewalks
        cv2.rectangle(img, (0, px(FAR_CURB_Y)), (W, px(NEAR_CURB_Y)), (74, 76, 80), -1)   # road
        for y in (FAR_CURB_Y, NEAR_CURB_Y):
            cv2.line(img, (0, px(y)), (W, px(y)), (240, 240, 240), px(0.4))
        x = 0.0
        while x < W / PX_PER_CM:                                                    # center line
            cv2.line(img, (px(x), px(CENTER_Y)), (px(x + 3.5), px(CENTER_Y)), (40, 200, 235), px(0.35))
            x += 6.0
        y = FAR_CURB_Y + 0.8
        while y < NEAR_CURB_Y - 1.5:                                               # crosswalk
            cv2.rectangle(img, (px(CROSSWALK[0]), px(y)), (px(CROSSWALK[1]), px(y + 1.8)), (238, 238, 238), -1)
            y += 3.4
        l, t, r, b = BUS                                                            # the bus, from above
        cv2.rectangle(img, (px(l), px(t)), (px(r), px(b)), (20, 185, 245), -1)
        cv2.rectangle(img, (px(l), px(t)), (px(r), px(b)), (10, 120, 170), 2)
        cv2.rectangle(img, (px(r - 2.2), px(t + 0.8)), (px(r - 0.6), px(b - 0.8)), (60, 50, 40), -1)
        cv2.putText(img, "SCHOOL BUS", (px(l + 6), px(t + 7.5)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
        gx, gy = GATE_POST                                                          # gate post
        cv2.rectangle(img, (px(gx - 1.3), px(gy - 1.3)), (px(gx + 1.3), px(gy + 1.3)), (120, 120, 125), -1)
        cv2.rectangle(img, (px(gx - 1.3), px(gy - 1.3)), (px(gx + 1.3), px(gy + 1.3)), (60, 60, 60), 2)
        paper = self.np_rng.normal(0, 2.5, (H, W, 1)).astype(np.int16)              # paper texture
        return np.clip(img.astype(np.int16) + paper, 0, 255).astype(np.uint8)

    def _draw_ramp(self, img):
        kind = self.roll.kind if self.roll is not None else "hit"
        (x0, y0), (xa, ya) = ROLLS[kind]["start"], ROLLS[kind]["aim"]
        slope = (ya - y0) / (xa - x0)
        back = -6.0
        p1, p2 = (px(back), px(y0 + (back - x0) * slope)), (px(x0 - 3.0), px(y0 - 3.0 * slope))
        cv2.line(img, p1, p2, (110, 170, 215), px(3.2))                             # a wooden ruler
        for i in range(0, 12, 2):
            tx = back + i
            cv2.circle(img, (px(tx), px(y0 + (tx - x0) * slope - 1.2)), 2, (40, 60, 80), -1)

    def _draw_kid(self, img):
        kx, ky = KID
        if self.kid_hit:                                                             # knocked over
            cv2.rectangle(img, (px(kx + 1.0), px(ky - 1.2)), (px(kx + 3.4), px(ky + 0.2)), (200, 90, 30), -1)
            cv2.circle(img, (px(kx + 4.2), px(ky - 0.5)), px(0.75), (30, 210, 250), -1)
            cv2.putText(img, "BONK", (px(kx - 1), px(ky - 3)), cv2.FONT_HERSHEY_DUPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
            return
        cv2.rectangle(img, (px(kx - 1.2), px(ky - 0.65)), (px(kx + 1.2), px(ky + 0.65)), (200, 90, 30), -1)
        cv2.circle(img, (px(kx), px(ky)), px(0.75), (30, 210, 250), -1)
        cv2.circle(img, (px(kx), px(ky)), px(0.75), (20, 20, 20), 1)

    def _draw_light(self, img):
        lx, ly = LIGHT_SPOT
        cv2.rectangle(img, (px(lx - 0.8), px(ly - 2.1)), (px(lx + 0.8), px(ly + 2.1)), (25, 25, 25), -1)
        lamps = [("red", -1.25, (70, 70, 255), (45, 45, 80)), ("yellow", 0.0, (60, 215, 255), (45, 80, 90)),
                 ("green", 1.25, (90, 250, 90), (45, 80, 45))]
        for name, dy, on, off in lamps:
            cv2.circle(img, (px(lx), px(ly + dy)), px(0.5), on if self.light == name else off, -1)

    def _draw_decoy(self, img):
        l, t, r, b = DECOY
        cv2.rectangle(img, (px(l), px(t)), (px(r), px(b)), (36, 30, 200), -1)          # same red as the car
        cv2.rectangle(img, (px(l), px(t)), (px(r), px(b)), (20, 20, 120), 2)
        for sx in (l + 1.5, r - 1.5):                                                     # its bumps on top
            for sy in (t + 1.5, b - 1.5):
                cv2.circle(img, (px(sx), px(sy)), px(0.9), (50, 45, 225), -1)
                cv2.circle(img, (px(sx), px(sy)), px(0.9), (20, 20, 120), 1)

    def _draw_gate(self, img, angle):
        gx, gy = GATE_POST
        length = 33.0 * np.sin(np.radians(angle))          # an arm pointing up looks short from above
        if length < 0.6:
            cv2.circle(img, (px(gx), px(gy)), px(0.7), (30, 30, 30), -1)
            return
        seg = 0.0
        while seg < length:                                  # black and white stripes
            end = min(length, seg + 3.0)
            color = (25, 25, 25) if int(seg / 3.0) % 2 == 0 else (245, 245, 245)
            cv2.rectangle(img, (px(gx - 0.6), px(gy - end)), (px(gx + 0.6), px(gy - seg)), color, -1)
            seg = end
        cv2.rectangle(img, (px(gx - 0.6), px(gy - length)), (px(gx + 0.6), px(gy)), (40, 40, 40), 1)

    def _draw_car(self, img, spot, drift):
        x, y = spot
        along = np.array([1.0, drift])
        along = along / np.linalg.norm(along)               # which way the toy car's nose points
        side = np.array([-along[1], along[0]])
        def corner(a, s):
            p = np.array([x, y]) + along * a + side * s
            return [px(p[0]), px(p[1])]
        body = np.array([corner(a, s) for a, s in ((3.5, 1.5), (3.5, -1.5), (-3.5, -1.5), (-3.5, 1.5))], np.int32)
        for a, s in ((2.3, 1.6), (2.3, -1.6), (-2.3, 1.6), (-2.3, -1.6)):     # wheels peeking out
            cv2.circle(img, tuple(corner(a, s)), px(0.55), (20, 20, 20), -1)
        cv2.fillConvexPoly(img, body, (38, 32, 205), cv2.LINE_AA)               # red body
        windshield = np.array([corner(a, s) for a, s in ((1.8, 1.1), (1.8, -1.1), (0.9, -1.1), (0.9, 1.1))], np.int32)
        back_glass = np.array([corner(a, s) for a, s in ((-1.4, 1.1), (-1.4, -1.1), (-2.2, -1.1), (-2.2, 1.1))], np.int32)
        cv2.fillConvexPoly(img, windshield, (70, 45, 30), cv2.LINE_AA)
        cv2.fillConvexPoly(img, back_glass, (70, 45, 30), cv2.LINE_AA)


class FakeBoard:
    """Pretends to be the ESP32: takes the same lines Kestrel sends the real board, moves the
    fake gate and light, and answers the same way the real Kestrel board code does."""

    def __init__(self, road, answers_hello=True):
        self.road = road
        self.answers_hello = answers_hello
        self.sent = []          # (time, line) for every line Kestrel sent, for the checks
        self.replies = []
        self.name = "practice board"

    def send(self, line, t):
        self.sent.append((t, line))
        if line == "k hello":
            if self.answers_hello:
                self.replies.append("kestrel ready")
        elif line.startswith("k gate down"):
            self.road.gate_command(True, t)
            self.replies.append(line.replace("k ", "kestrel ", 1))
        elif line.startswith("k gate up"):
            self.road.gate_command(False, t)
            self.replies.append(line.replace("k ", "kestrel ", 1))
        elif line.startswith("k gate move"):
            self.replies.append("kestrel gate at " + line.split()[-1])
        elif line.startswith("k light "):
            self.road.light = line.split()[-1]
            self.replies.append(line.replace("k ", "kestrel ", 1))
        else:
            self.replies.append("kestrel unknown: " + line)

    def read(self):
        out, self.replies = self.replies, []
        return out
