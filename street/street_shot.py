"""The real-traffic shot for the video (v7): pretend kid 1 only, drawn as a kid standing in the
crosswalk with its circle, each followed car's path (red once Kestrel warned), plates blurred."""
import pickle, json, subprocess, sys, cv2, numpy as np
from pathlib import Path
import street_kestrel as K
from register import full
from cache_dets import frames, FPS
from draw_clip import ground_to_ref_px

HERE = Path(__file__).resolve().parent
OUT = "/Users/ryanyousefi/kestrel-edit-work/footage/street-shot-v7.mp4"
TMP = "/private/tmp/claude-501/-Users-ryanyousefi/199ee794-4390-4e10-baef-111d9fbc38f2/scratchpad/street/shot_raw.mp4"
KID_H = 1.3                                   # a pretend kid about 1.3 m tall


def above_ground_ref_px(ab, height):
    """Reference pixel of a point `height` meters above the road at road spot ab."""
    g = K.G
    X = ab[0] * g["e1"] + ab[1] * g["e2"] + (K.HEIGHT_M - height) * g["n"]
    p = g["K"] @ X
    return p[:2] / p[2]


def draw_kid(img, foot, head):
    """A simple flat kid figure (clearly a drawing) from foot point to head point."""
    foot, head = np.array(foot, float), np.array(head, float)
    up = head - foot
    L = np.linalg.norm(up)
    u = up / L
    side = np.array([-u[1], u[0]])
    w = L * 0.16
    col, edge = (0, 200, 255), (0, 0, 0)
    hip = foot + up * 0.45
    neck = foot + up * 0.78
    head_c = foot + up * 0.88
    def P(x): return tuple(np.int32(np.round(x)))
    for a, b in [(foot + side * w * 0.5, hip), (foot - side * w * 0.5, hip),          # legs
                 (neck, hip), (neck + side * w * 1.1 - up * 0.2, neck), (neck - side * w * 1.1 - up * 0.2, neck)]:
        cv2.line(img, P(a), P(b), edge, max(3, int(L * 0.09)))
        cv2.line(img, P(a), P(b), col, max(2, int(L * 0.06)))
    cv2.circle(img, P(head_c), max(4, int(L * 0.11)), edge, -1)
    cv2.circle(img, P(head_c), max(3, int(L * 0.09)), col, -1)


def main(start, end):
    fz = json.loads((HERE / "FROZEN.json").read_text())
    ex = pickle.load(open(HERE / "exam_tracks.pkl", "rb"))
    rec = pickle.load(open(HERE / "cache" / "exam_240_606.pkl", "rb"))
    dets = rec["dets"][fz["settings"]["model"]]
    rows = {r["id"]: r for r in ex["kid_rows"]["kid1_toward_camera"]}
    cx, cy, r = fz["kids"]["kid1_toward_camera"]
    ring = ground_to_ref_px(np.c_[cx + r * np.cos(np.linspace(0, 6.3, 60)), cy + r * np.sin(np.linspace(0, 6.3, 60))])
    foot_ref = ground_to_ref_px(np.array([[cx, cy]]))[0]
    head_ref = above_ground_ref_px((cx, cy), KID_H)
    w = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
    ts = np.array(rec["t"])
    for t, im in frames(start, end):
        k = int(np.argmin(abs(ts - t)))
        inv = np.linalg.inv(full(rec["H"][k]))
        def to_frame(px):
            px = np.atleast_2d(px)
            p = (inv @ np.c_[px, np.ones(len(px))].T).T
            return (p[:, :2] / p[:, 2:] / 2).astype(np.int32)
        img = im.copy()
        for x1, y1, x2, y2, c, cl in dets[k]:                       # blur the plate area of every vehicle
            if c < 0.2:
                continue
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1 + (y2 - y1) * 0.45)), int(min(1919, x2)), int(min(1079, y2))
            if x2 - x1 > 4 and y2 - y1 > 4:
                img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (0, 0), 7)
        warned_now = False
        for tid, raw in ex["tracks"]:
            past = [p for p in raw if t - 2 <= p[0] <= t]
            if len(past) < 2 or past[-1][0] < t - 0.1:
                continue
            row = rows.get(tid)
            if row is None or row["enter"] is None:
                continue                                              # only cars headed through kid 1's spot
            warned = row["warn"] is not None and row["warn"] <= t
            warned_now |= warned and t <= row["enter"] + 0.3
            pts = to_frame(ground_to_ref_px(np.array([[p[1], p[2]] for p in past])))
            cv2.polylines(img, [pts], False, (0, 0, 255) if warned else (0, 255, 0), 4)
            if warned and t <= row["enter"] + 0.3:                # how early Kestrel warned this car
                tag = f"{row['enter'] - row['warn']:.1f} s EARLY"
                at = tuple(pts[-1] + [-200, -40])
                cv2.putText(img, tag, at, 0, 1.0, (0, 0, 0), 6)
                cv2.putText(img, tag, at, 0, 1.0, (255, 255, 255), 2)
        ring_px = to_frame(ring)
        cv2.polylines(img, [ring_px], True, (0, 0, 255) if warned_now else (0, 200, 255), 4)
        draw_kid(img, to_frame(foot_ref)[0], to_frame(head_ref)[0])
        lab = tuple(ring_px[15] + [14, -14])
        cv2.putText(img, "PRETEND KID", lab, 0, 0.9, (0, 0, 0), 5)
        cv2.putText(img, "PRETEND KID", lab, 0, 0.9, (0, 200, 255), 2)
        if warned_now:
            cv2.putText(img, "WARNING", (lab[0], lab[1] - 36), 0, 1.1, (0, 0, 0), 6)
            cv2.putText(img, "WARNING", (lab[0], lab[1] - 36), 0, 1.1, (0, 0, 255), 3)
        w.write(img)
    w.release()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", TMP, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", OUT], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]))
