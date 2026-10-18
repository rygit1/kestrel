"""Gate panel: drag a slider, or use the arrow keys, and the gate motor turns. Q quits.

LEFT / RIGHT arrows = gate 1, UP / DOWN arrows = gate 2, 5 degrees a press.
SPACE = drop or lift BOTH arms at once, to the angles saved in settings.json.
R / Y / G = turn that light on, X = light off.

  .venv/bin/python gate_panel.py

Gate 1 = the motor on pin 26 (Kestrel's "k gate move"), gate 2 = the motor on pin 27
(the old board's spare motor line, "G###"). Written by Claude, listed in the AI-use list.
"""
import glob
import json
from pathlib import Path
import time

import cv2
import numpy as np
import serial

port = (glob.glob("/dev/cu.usbserial-*") or [None])[0]
if not port:
    raise SystemExit("No chip found: plug its USB cable into the laptop.")
link = serial.Serial(port, 115200, timeout=0)
time.sleep(2)

WIN = "Kestrel gate panel"
cv2.namedWindow(WIN)
cv2.createTrackbar("Gate 1", WIN, 90, 180, lambda v: None)
cv2.createTrackbar("Gate 2", WIN, 90, 180, lambda v: None)
sent = {"Gate 1": None, "Gate 2": None}
reply = ""
saved = json.loads((Path(__file__).parent / "settings.json").read_text())
down = False

while True:
    for name in sent:
        a = cv2.getTrackbarPos(name, WIN)
        if a != sent[name]:
            line = f"k gate move {a:03d}\n" if name == "Gate 1" else f"G{a:03d}\n"
            link.write(line.encode())
            sent[name] = a
    got = link.read(200).decode(errors="ignore").strip()
    if got:
        reply = got.splitlines()[-1]

    view = np.full((220, 760, 3), 30, np.uint8)
    cv2.putText(view, f"Gate 1: {sent['Gate 1']}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.putText(view, f"Gate 2: {sent['Gate 2']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.putText(view, f"chip says: {reply}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(view, "SPACE = both arms   ARROWS = one arm   R/Y/G/X = light   Q = quit", (20, 207),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.imshow(WIN, view)
    k = cv2.waitKeyEx(30)
    if k in (ord("q"), ord("Q")):
        break
    light = {ord("r"): "red", ord("y"): "yellow", ord("g"): "green", ord("x"): "off"}.get(k)
    if light:
        link.write(f"k light {light}\n".encode())
        continue
    if k == ord(" "):
        down = not down
        word = "down" if down else "up"
        cv2.setTrackbarPos("Gate 1", WIN, saved["gate_" + word])
        cv2.setTrackbarPos("Gate 2", WIN, saved["gate2_" + word])
        continue
    step = {63234: ("Gate 1", -5), 63235: ("Gate 1", 5), 63232: ("Gate 2", 5), 63233: ("Gate 2", -5)}.get(k)
    if k != -1 and not step:
        reply = f"key {k} is not a control"   # shows the real number if an arrow reads differently
    if step:
        name, d = step
        cv2.setTrackbarPos(name, WIN, max(0, min(180, cv2.getTrackbarPos(name, WIN) + d)))

link.close()
cv2.destroyAllWindows()
