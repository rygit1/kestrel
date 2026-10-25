"""Reads a stretch of the street clip once and saves, for every frame (30 a second): how that frame
lines up with the reference frame, and every vehicle each car finder saw (box, how sure, kind)."""
import sys, time, pickle, subprocess, cv2, numpy as np
from ultralytics import YOLO
from register import Register, SCALE, S

VIDEO = '/Users/ryanyousefi/Desktop/important/Projects/Kestrel/drone/street-test-2026-10-24.mp4'
FPS = 30000 / 1001
W, H = 1920, 1080
VEHICLES = [1, 2, 3, 5, 7]            # bicycle, car, motorcycle, bus, truck


def frames(start, end):
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-hwaccel', 'videotoolbox', '-ss', str(start), '-t', str(end - start),
           '-i', VIDEO, '-vf', f'fps={FPS},scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        yield start + i / FPS, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        i += 1
    p.wait()


def main(start, end, out, specs):
    reg = Register(cv2.imread(S + 'ref_5.png'))
    models = [(name, sz, YOLO(name)) for name, sz in specs]
    rec = {"start": start, "end": end, "t": [], "H": [], "inliers": [], "dets": {f"{n}@{s}": [] for n, s, _ in models}}
    t0 = time.time()
    for k, (t, im) in enumerate(frames(start, end)):
        small = cv2.cvtColor(cv2.resize(im, (int(3840 * SCALE), int(2160 * SCALE)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        Hm, n = reg.to_ref(small)
        rec["t"].append(t); rec["H"].append(Hm); rec["inliers"].append(n)
        for name, sz, m in models:
            r = m.predict(im, imgsz=sz, device='mps', verbose=False, classes=VEHICLES, conf=0.05)[0]
            b = r.boxes
            rec["dets"][f"{name}@{sz}"].append(np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()].astype(np.float32))
        if k % 300 == 0:
            print(f"read {t - start:.0f} s of {end - start:.0f} s ({time.time() - t0:.0f} s so far), lined-up spots {n}", flush=True)
            pickle.dump(rec, open(out, 'wb'))
    pickle.dump(rec, open(out, 'wb'))
    print("done", len(rec["t"]), "frames", f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    start, end, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    specs = [(s.split('@')[0], int(s.split('@')[1])) for s in sys.argv[4].split(',')]
    main(start, end, out, specs)
