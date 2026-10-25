"""PRIVATE clip for Ryan only (never posted: it shows the real street): the pretend kid's circle on
the real video, each followed car's path, and a red WARN tag the moment Ryan's code warned."""
import pickle, json, subprocess, sys, cv2, numpy as np
from pathlib import Path
import street_kestrel as K
from ground import to_ground
from register import full
from cache_dets import frames, FPS

HERE = Path(__file__).resolve().parent
OUT = "/Users/ryanyousefi/Desktop/important/Projects/Kestrel/drone/PRIVATE-pretend-kid-test.mp4"


def ground_to_ref_px(ab):
    g = K.G
    H = g["K"] @ np.c_[g["e1"], g["e2"], g["n"] * K.HEIGHT_M]
    p = (H @ np.c_[np.atleast_2d(ab), np.ones(len(np.atleast_2d(ab)))].T).T
    return p[:, :2] / p[:, 2:]


def main(start, end):
    fz = json.loads((HERE / "FROZEN.json").read_text())
    ex = pickle.load(open(HERE / "exam_tracks.pkl", "rb"))
    rec = pickle.load(open(HERE / "cache" / "exam_240_606.pkl", "rb"))
    rows = {r["id"]: r for r in ex["kid_rows"]["kid1_toward_camera"]}
    rings = []
    for name, (cx, cy, r) in fz["kids"].items():
        rings.append((name, ground_to_ref_px(np.c_[cx + r * np.cos(np.linspace(0, 6.3, 60)), cy + r * np.sin(np.linspace(0, 6.3, 60))])))
    w = cv2.VideoWriter("/private/tmp/claude-501/-Users-ryanyousefi/199ee794-4390-4e10-baef-111d9fbc38f2/scratchpad/street/clip_raw.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
    ts = np.array(rec["t"])
    for t, im in frames(start, end):
        k = int(np.argmin(abs(ts - t)))
        Hm = rec["H"][k]
        if Hm is None:
            continue
        inv = np.linalg.inv(full(Hm))                    # reference 4K -> this frame 4K
        def to_frame(px):
            p = (inv @ np.c_[px, np.ones(len(px))].T).T
            return (p[:, :2] / p[:, 2:] / 2).astype(np.int32)
        img = im.copy()
        for name, ring in rings:
            col = (0, 220, 255) if name.startswith("kid1") else (200, 200, 200)
            cv2.polylines(img, [to_frame(ring)], True, col, 3)
            cv2.putText(img, "PRETEND KID 1" if name.startswith("kid1") else "PRETEND KID 2 (blind side)", tuple(to_frame(ring)[15] + [10, -10]), 0, 0.8, col, 2)
        for tid, raw in ex["tracks"]:
            past = [p for p in raw if t - 3 <= p[0] <= t]
            if len(past) < 2 or past[-1][0] < t - 0.1:      # only cars still on screen
                continue
            row = rows.get(tid)
            warned = row and row["warn"] is not None and row["warn"] <= t
            color = (0, 0, 255) if warned else (0, 255, 0)
            pts = to_frame(ground_to_ref_px(np.array([[p[1], p[2]] for p in past])))
            cv2.polylines(img, [pts], False, color, 3)
            if warned and row["enter"] is not None:
                cv2.putText(img, f"WARNED {row['enter'] - row['warn']:.1f} s AHEAD", tuple(pts[-1] + [12, -12]), 0, 0.8, color, 2)
            elif warned:
                cv2.putText(img, "WARNED (car missed the kid)", tuple(pts[-1] + [12, -12]), 0, 0.8, color, 2)
        cv2.putText(img, f"PRIVATE TEST CLIP, NOT FOR POSTING   {int(t // 60)}:{t % 60:05.2f}", (20, 40), 0, 1.0, (255, 255, 255), 2)
        w.write(img)
    w.release()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", "/private/tmp/claude-501/-Users-ryanyousefi/199ee794-4390-4e10-baef-111d9fbc38f2/scratchpad/street/clip_raw.mp4",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]))
