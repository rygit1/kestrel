"""
KESTREL LEARNS YOUR CAR: the AI half of Kestrel's eyes.

Why Kestrel trains its own AI (measured Oct 6, 2026): ready-made AI models were tested on 20
real toy car photos shrunk to the size Kestrel's camera sees a car from 2 feet up. The best one
(YOLO26s) found 13 of the 20, and only 1 of the 5 taken from straight above. So Kestrel learns
YOUR car instead:

  1. COLLECT: while you roll the car, Kestrel saves small square pictures around it. The color
     eyes already know where the red car is, so they mark it in every picture by themselves.
     It also saves squares of the road with no car, and into some of them it paints fake red
     shapes in your car's exact red, marked "not a car". Without those, the AI learns that
     anything red is your car (that really happened on Oct 6: it called a red brick a car).
     With them, it has to learn the car's shape: its windows, wheels and outline.
  2. TRAIN: a small YOLO26 model trains on those pictures, right on the laptop.
  3. TEST: the trained AI looks at pictures it never trained on, and Kestrel prints how many
     it got right. That number decides whether the AI is good enough to use.

After that, Kestrel's eyes work as a team: the color finds every red thing fast, and the AI
checks a square around each one, so only YOUR car counts and a red shirt or a red light never
fools it.

    Kestrel runs this by itself when you press L. By hand:
    python learn_car.py learn <folder> models/my_car.onnx
"""

import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SQUARE = 320                      # every picture is a 320 by 320 square around the car
WANT_CAR, WANT_EMPTY = 240, 90    # how many pictures to collect


class Collector:
    """Saves the training pictures while the car rolls. Kestrel hands it each camera picture and
    where the color eyes saw the car (its box), and it keeps one square every few pictures."""

    def __init__(self, folder, seed=0):
        self.folder = Path(folder)
        for part in ("train", "val"):
            (self.folder / "images" / part).mkdir(parents=True, exist_ok=True)
            (self.folder / "labels" / part).mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.cars = len(list((self.folder / "labels").glob("*/car_*.txt")))
        self.empties = len(list((self.folder / "images").glob("*/empty_*.jpg")))
        self.tick = 0
        self.red = (40, 35, 200)          # the car's red, measured from the car itself as it rolls

    def done(self):
        return self.cars >= WANT_CAR and self.empties >= WANT_EMPTY

    def add(self, frame, box):
        """box is the car's (x, y, w, h) in the picture from the color eyes, or None if no car."""
        self.tick += 1
        h, w = frame.shape[:2]
        if box is not None and self.cars < WANT_CAR and self.tick % 2 == 0:
            bx, by, bw, bh = box
            self.red = self._car_red(frame, box) or self.red
            # put the car somewhere inside the square, not always in the middle
            x0 = int(np.clip(bx + bw / 2 - SQUARE / 2 + self.rng.randint(-80, 80), 0, w - SQUARE))
            y0 = int(np.clip(by + bh / 2 - SQUARE / 2 + self.rng.randint(-80, 80), 0, h - SQUARE))
            x1, y1 = max(bx, x0), max(by, y0)
            x2, y2 = min(bx + bw, x0 + SQUARE), min(by + bh, y0 + SQUARE)
            if x2 - x1 >= bw * 0.8 and y2 - y1 >= bh * 0.8:          # most of the car is in the square
                label = f"0 {((x1 + x2) / 2 - x0) / SQUARE:.5f} {((y1 + y2) / 2 - y0) / SQUARE:.5f} " \
                        f"{(x2 - x1) / SQUARE:.5f} {(y2 - y1) / SQUARE:.5f}\n"
                square = frame[y0:y0 + SQUARE, x0:x0 + SQUARE].copy()
                if self.rng.random() < 0.3:                          # a fake red shape beside the real car
                    self._paint_decoy(square, avoid=(x1 - x0, y1 - y0, x2 - x0, y2 - y0))
                self._save(square, f"car_{self.cars:04d}", label)
                self.cars += 1
        elif box is None and self.empties < WANT_EMPTY and self.tick % 5 == 0:
            x0, y0 = self.rng.randint(0, w - SQUARE), self.rng.randint(0, h - SQUARE)
            square = frame[y0:y0 + SQUARE, x0:x0 + SQUARE].copy()
            if self.rng.random() < 0.6:                              # red, but not a car
                for _ in range(self.rng.randint(1, 2)):
                    self._paint_decoy(square)
            self._save(square, f"empty_{self.empties:04d}", "")
            self.empties += 1

    @staticmethod
    def _car_red(frame, box):
        """The average color of the reddest part of the car, so the fake shapes match it."""
        bx, by, bw, bh = box
        patch = frame[by:by + bh, bx:bx + bw].reshape(-1, 3).astype(int)
        if len(patch) < 20:
            return None
        redness = patch[:, 2] - np.maximum(patch[:, 0], patch[:, 1])
        top = patch[redness >= np.percentile(redness, 60)]
        return tuple(int(v) for v in top.mean(axis=0))

    def _paint_decoy(self, square, avoid=None):
        """Paints one random red shape: a box, a blob, a stripe or a bent line. Never a car shape."""
        r = self.rng
        for _ in range(20):
            cx, cy = r.randint(20, SQUARE - 20), r.randint(20, SQUARE - 20)
            size = r.randint(18, 70)
            if avoid is None or not (avoid[0] - size - 10 < cx < avoid[2] + size + 10 and
                                     avoid[1] - size - 10 < cy < avoid[3] + size + 10):
                break
        else:
            return
        shade = tuple(int(np.clip(c * r.uniform(0.8, 1.15), 0, 255)) for c in self.red)
        kind = r.choice(["box", "blob", "stripe", "line"])
        if kind == "box":
            rect = ((cx, cy), (size, size * r.uniform(0.4, 1.0)), r.uniform(0, 180))
            cv2.fillConvexPoly(square, cv2.boxPoints(rect).astype(np.int32), shade, cv2.LINE_AA)
        elif kind == "blob":
            cv2.ellipse(square, (cx, cy), (size // 2, int(size * r.uniform(0.3, 0.6))), r.uniform(0, 180),
                        0, 360, shade, -1, cv2.LINE_AA)
        elif kind == "stripe":
            cv2.line(square, (cx - size, cy), (cx + size, cy + r.randint(-20, 20)), shade, r.randint(6, 14), cv2.LINE_AA)
        else:
            pts = np.array([[cx + r.randint(-size, size), cy + r.randint(-size, size)] for _ in range(4)], np.int32)
            cv2.polylines(square, [pts], False, shade, r.randint(5, 10), cv2.LINE_AA)

    def _save(self, square, name, label):
        part = "val" if self.rng.random() < 0.2 else "train"       # 1 in 5 is kept for the test
        cv2.imwrite(str(self.folder / "images" / part / f"{name}.jpg"), square)
        (self.folder / "labels" / part / f"{name}.txt").write_text(label)


def train(folder, epochs=40, device=None):
    """Trains the small YOLO26 on the collected squares. Hands back the trained model's file
    (inside the folder). It only becomes Kestrel's eyes after it passes the test."""
    from ultralytics import YOLO
    import torch
    folder = Path(folder).resolve()
    data = folder / "data.yaml"
    data.write_text(f"path: {folder}\ntrain: images/train\nval: images/val\nnames:\n  0: my car\n")
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    start = time.time()
    model = YOLO(str(HERE / "models" / "yolo26n.pt"))
    model.train(data=str(data), epochs=epochs, imgsz=SQUARE, batch=16, device=device, patience=10,
                project=str(folder / "runs"), name="my_car", exist_ok=True, workers=0, plots=False,
                verbose=False, cache=True, seed=0, deterministic=False)
    best = folder / "runs" / "my_car" / "weights" / "best.pt"
    onnx = Path(YOLO(str(best)).export(format="onnx", imgsz=SQUARE, simplify=True))
    minutes = (time.time() - start) / 60
    print(f"TRAINED in {minutes:.1f} minutes on {device}")
    return onnx, minutes, device


FOLLOW_BAR = 0.5    # how sure the AI must be to keep following the car it already accepted


def test(folder, model_path):
    """How sure the trained AI is about each test square it never trained on. From that it sets
    the START BAR: how sure it must be before a NEW red thing counts as the car. The bar is the
    surety it has about the real car in 95 of every 100 test pictures, minus a little, so a red
    thing has to look as much like your car as your car does. Hands back the counts."""
    from ultralytics import YOLO
    model = YOLO(str(model_path), task="detect")
    folder = Path(folder)
    car_sure, empty_sure, times = [], [], []
    for img in sorted((folder / "images" / "val").glob("*.jpg")):
        has_car = (folder / "labels" / "val" / (img.stem + ".txt")).read_text().strip() != ""
        t = time.perf_counter()
        r = model.predict(cv2.imread(str(img)), imgsz=SQUARE, conf=0.05, verbose=False)[0]
        times.append((time.perf_counter() - t) * 1000)
        sure = max(r.boxes.conf.tolist(), default=0.0)
        (car_sure if has_car else empty_sure).append(sure)
    car_sure.sort()
    p5 = car_sure[int(len(car_sure) * 0.05)] if car_sure else 0.0
    start_bar = round(min(0.92, max(0.6, p5 - 0.05)), 2)
    found = sum(c >= FOLLOW_BAR for c in car_sure)
    false = sum(e >= start_bar for e in empty_sure)
    ms = sorted(times)[len(times) // 2] if times else 0.0
    print(f"TEST: found the car in {found} of {len(car_sure)} test pictures, start bar {start_bar:.2f}, "
          f"saw a car that was not there in {false} of {len(empty_sure)} empty ones, {ms:.1f} ms per check")
    return {"found": found, "cars": len(car_sure), "false_cars": false, "empties": len(empty_sure),
            "start_bar": start_bar, "follow_bar": FOLLOW_BAR, "ms": round(ms, 1),
            "car_sure_lowest": round(car_sure[0], 2) if car_sure else None,
            "empty_sure_highest": round(max(empty_sure), 2) if empty_sure else None}


def passed(result):
    """The bar: finds the car in at least 95 of every 100 test pictures, and never sees a car in
    an empty one. Below that, Kestrel keeps using its color eyes."""
    return result["cars"] > 0 and result["found"] >= 0.95 * result["cars"] and result["false_cars"] == 0


def keep(result, onnx, final_model):
    """Makes a passed AI Kestrel's eyes: the model, plus a small file with its bars beside it."""
    final_model = Path(final_model)
    final_model.parent.mkdir(exist_ok=True)
    shutil.copy(onnx, final_model)
    final_model.with_suffix(".json").write_text(json.dumps({"start_bar": result["start_bar"],
                                                            "follow_bar": result["follow_bar"]}))


def learn(folder, final_model, epochs=40):
    """Train, test, and only if the test passes, make it Kestrel's AI eyes. Writes result.json."""
    onnx, minutes, device = train(folder, epochs)
    result = test(folder, onnx)
    result.update(minutes=round(minutes, 1), device=device, epochs=epochs, passed=passed(result))
    if result["passed"]:
        keep(result, onnx, final_model)
        result["model"] = str(final_model)
        print(f"PASSED: this AI is now Kestrel's eyes ({final_model})")
    else:
        print("NOT GOOD ENOUGH YET: Kestrel keeps its color eyes. Collect more pictures (key L) and try again.")
    (Path(folder) / "result.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] not in ("learn", "retest"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "learn":
        learn(sys.argv[2], sys.argv[3])
    else:                                   # test an already trained run again, and keep it if it passes
        onnx = Path(sys.argv[2]) / "runs" / "my_car" / "weights" / "best.onnx"
        r = test(sys.argv[2], onnx)
        if passed(r):
            keep(r, onnx, sys.argv[3])
            print("PASSED and kept:", sys.argv[3])
