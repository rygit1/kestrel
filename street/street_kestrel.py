"""Kestrel on a real street: follows every vehicle the car finder sees on the flat road map (in
meters), and asks Ryan's own decision code (ryan_code.py, unchanged) about each one.

Works from a saved run (cache_dets.py): every frame's line-up with the reference view and every
box each car finder saw. Nothing here looks at the hand counts."""
import sys, pickle, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ryan_code
from ground import build, to_ground
from register import full

HEIGHT_M = 10.3                       # the drone's own height reading at the reference frame
G = build()
CROSSWALK_A = 2.25 * HEIGHT_M         # middle of the crosswalk bars, meters along the road
ROAD_B = (0.15 * HEIGHT_M, 1.95 * HEIGHT_M)
ROAD_MID_B = sum(ROAD_B) / 2

DEFAULTS = dict(model="yolo26m.pt@1280", stride=1, point=0.0, conf_hi=0.4, conf_lo=0.15,
                gate_m=4.0, lost_s=0.5, track_after=3, nms_iou=0.6, smooth=1)


def nms(d, iou):
    """Class-blind: one box per vehicle even if the finder called it both car and truck."""
    if len(d) == 0:
        return d
    d = d[np.argsort(-d[:, 4])]
    keep = []
    for i in range(len(d)):
        x1, y1, x2, y2 = d[i, :4]
        ok = True
        for j in keep:
            X1, Y1, X2, Y2 = d[j, :4]
            iw, ih = max(0, min(x2, X2) - max(x1, X1)), max(0, min(y2, Y2) - max(y1, Y1))
            inter = iw * ih
            if inter / ((x2 - x1) * (y2 - y1) + (X2 - X1) * (Y2 - Y1) - inter + 1e-9) > iou:
                ok = False
                break
        if ok:
            keep.append(i)
    return d[keep]


def ground_points(rec, s):
    """Per kept frame: (t, [(a, b, conf)]) with a, b in meters on the road."""
    out = []
    dets = rec["dets"][s["model"]]
    Hs = rec["H"]
    for k in range(0, len(rec["t"]), s["stride"]):
        H = Hs[k]
        if H is None:                                    # could not line up: borrow a neighbor's
            near = [Hs[j] for j in (k - 1, k + 1, k - 2, k + 2) if 0 <= j < len(Hs) and Hs[j] is not None]
            if not near:
                continue
            H = near[0]
        d = nms(dets[k][dets[k][:, 5] != 1], s["nms_iou"]) if len(dets[k]) else dets[k]   # no bicycles
        d = d[d[:, 4] >= s["conf_lo"]] if len(d) else d
        if len(d) == 0:
            out.append((rec["t"][k], np.zeros((0, 3))))
            continue
        x = (d[:, 0] + d[:, 2]) / 2
        y = d[:, 3] - s["point"] * (d[:, 3] - d[:, 1])
        px = np.c_[x, y] * 2.0                           # 1920 -> 4K
        ref = (full(H) @ np.c_[px, np.ones(len(px))].T).T
        ref = ref[:, :2] / ref[:, 2:]
        ab = to_ground(G, ref, HEIGHT_M)
        out.append((rec["t"][k], np.c_[ab, d[:, 4]]))
    return out


class Track:
    n = 0

    def __init__(self, t, a, b):
        Track.n += 1
        self.id, self.spots, self.hits = Track.n, [(t, a, b)], 1

    def guess(self, t):
        t1, a1, b1 = self.spots[-1]
        if len(self.spots) < 3:
            return a1, b1
        t0, a0, b0 = self.spots[-min(5, len(self.spots))]
        dt = t1 - t0
        return (a1 + (a1 - a0) / dt * (t - t1), b1 + (b1 - b0) / dt * (t - t1)) if dt > 0 else (a1, b1)


def follow(points, s):
    """Follows vehicles from frame to frame. Sure sightings first, then less sure ones may only
    continue a vehicle already being followed. Returns confirmed tracks (TRACK_AFTER sightings)."""
    Track.n = 0
    live, done = [], []
    for t, P in points:
        for tr in [tr for tr in live if t - tr.spots[-1][0] > s["lost_s"]]:
            live.remove(tr)
            done.append(tr)
        free = list(range(len(P)))
        for stage_hi in (True, False):
            cand = [i for i in free if (P[i, 2] >= s["conf_hi"]) == stage_hi]
            pairs = []
            for tr in live:
                if tr.spots[-1][0] == t:
                    continue
                ga, gb = tr.guess(t)
                for i in cand:
                    dist = np.hypot(P[i, 0] - ga, P[i, 1] - gb)
                    if dist < s["gate_m"]:
                        pairs.append((dist, tr, i))
            pairs.sort(key=lambda x: x[0])
            used_t, used_i = set(), set()
            for dist, tr, i in pairs:
                if tr.id in used_t or i in used_i:
                    continue
                used_t.add(tr.id); used_i.add(i)
                tr.spots.append((t, P[i, 0], P[i, 1])); tr.hits += 1
            free = [i for i in free if i not in used_i]
        for i in free:
            if P[i, 2] >= s["conf_hi"]:
                live.append(Track(t, P[i, 0], P[i, 1]))
    done += live
    kept = [tr for tr in done if tr.hits >= s["track_after"]]
    for tr in kept:
        tr.raw = tr.spots
        n = s["smooth"]
        if n > 1:
            sp = np.array(tr.raw)
            tr.spots = [tuple(sp[max(0, i - n + 1):i + 1].mean(0)) for i in range(len(sp))]
            tr.now = [sp[i][0] for i in range(len(sp))]      # the moment each spot was ready
        else:
            tr.now = [p[0] for p in tr.spots]
    return kept


def run(rec, **over):
    s = dict(DEFAULTS, **over)
    return follow(ground_points(rec, s), s), s


# ------------------------------------------------------------------ scoring

def crossings(tracks):
    """When and which way each followed vehicle crossed the middle of the crosswalk."""
    out = []
    for tr in tracks:
        sp = tr.spots
        for (t0, a0, b0), (t1, a1, b1) in zip(sp, sp[1:]):
            if (a0 - CROSSWALK_A) * (a1 - CROSSWALK_A) <= 0 and a0 != a1:
                f = (CROSSWALK_A - a0) / (a1 - a0)
                b = b0 + f * (b1 - b0)
                if ROAD_B[0] - 1 <= b <= ROAD_B[1] + 1:
                    out.append(dict(t=t0 + f * (t1 - t0), dir="up" if a1 > a0 else "down", b=b, id=tr.id))
                    break
    return out


def match_counts(hand, mine, window=1.5, span=None):
    """Matches each hand-counted car to a followed car crossing the same way within 1.5 s. A car right at
    the edge of a stretch may match a crossing just outside it; extras only count inside the stretch."""
    used, found = set(), []
    for h in sorted(hand, key=lambda c: c["time"]):
        best = None
        for j, m in enumerate(mine):
            if j in used or m["dir"] != h["direction"]:
                continue
            d = abs(m["t"] - h["time"])
            if d <= window and (best is None or d < best[0]):
                best = (d, j)
        if best:
            used.add(best[1])
        found.append(best is not None)
    inside = [j for j, m in enumerate(mine) if span is None or span[0] <= m["t"] < span[1]]
    return sum(found), len(hand), sum(1 for j in inside if j not in used)


def at(spots, t):
    """Where the followed path was at time t (between two spots no more than 0.25 s apart)."""
    for (t0, a0, b0), (t1, a1, b1) in zip(spots, spots[1:]):
        if t0 <= t <= t1:
            if t1 - t0 > 0.25:
                return None
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            return a0 + f * (a1 - a0), b0 + f * (b1 - b0)
    return None


FAR = (1e6, 1e6, 1.0)          # a circle no car reaches, so Ryan's code draws its full 2-second path


def one_second_guesses(tracks, zone_m=25.0, t_range=None):
    """Ryan's guess_and_decide, asked at every spot with 5 spots: the point 1.0 s along its path
    vs where the car really was 1.0 s later."""
    errs = []
    for tr in tracks:
        sp = tr.spots
        for i in range(4, len(sp)):
            t, a, b = sp[i]
            now = tr.now[i]
            if t_range and not (t_range[0] <= now < t_range[1]):
                continue
            if np.hypot(a - CROSSWALK_A, b - ROAD_MID_B) > zone_m:
                continue
            real = at(tr.raw, now + 1.0)
            if real is None:
                continue
            ans = ryan_code.guess_and_decide(sp[:i + 1], FAR, 0.3)
            pe = ans["path_end"]
            ahead = now + 1.0 - t                       # 1 s after the decision moment
            g1 = (a + (pe[0] - a) / 2 * ahead, b + (pe[1] - b) / 2 * ahead)
            errs.append(float(np.hypot(g1[0] - real[0], g1[1] - real[1])))
    e = np.array(errs)
    return dict(n=len(e), within_1m=float((e <= 1.0).mean()) if len(e) else None,
                median_m=float(np.median(e)) if len(e) else None)


def entry_time(spots, circle):
    """First moment the followed path is inside the circle (checks between spots too)."""
    cx, cy, r = circle
    for (t0, a0, b0), (t1, a1, b1) in zip(spots, spots[1:]):
        if np.hypot(a0 - cx, b0 - cy) < r:
            return t0
        da, db = a1 - a0, b1 - b0
        A = da * da + db * db
        if A == 0:
            continue
        B = 2 * (da * (a0 - cx) + db * (b0 - cy))
        C = (a0 - cx) ** 2 + (b0 - cy) ** 2 - r * r
        disc = B * B - 4 * A * C
        if disc >= 0:
            f = (-B - disc ** 0.5) / (2 * A)
            if 0 <= f <= 1:
                return t0 + f * (t1 - t0)
    t, a, b = spots[-1]
    return t if np.hypot(a - cx, b - cy) < r else None


def closest(spots, circle):
    cx, cy, _ = circle
    return min(np.hypot(a - cx, b - cy) for _, a, b in spots)


def pretend_kid(tracks, circle, gate_seconds=0.3, t_range=None, lane_m=3.6):
    """For every followed car: did Kestrel (Ryan's code, unchanged) warn at least 1 s before the car
    REALLY entered the kid's circle? And did it stay quiet for cars that passed close but missed?"""
    rows = []
    for tr in tracks:
        sp = tr.spots
        if t_range and not (t_range[0] <= sp[0][0] < t_range[1]):
            continue
        warn = None
        for i in range(4, len(sp)):
            ans = ryan_code.guess_and_decide(sp[:i + 1], tuple(circle), gate_seconds)
            if ans["drop_gate"]:
                warn = sp[i][0]
                break
        enter = entry_time(sp, circle)
        rows.append(dict(id=tr.id, start=sp[0][0], warn=warn, enter=enter, closest=closest(sp, circle)))
    went = [r for r in rows if r["enter"] is not None]
    saved = [r for r in went if r["warn"] is not None and r["enter"] - r["warn"] >= 1.0]
    late = [r for r in went if r["warn"] is not None and r["enter"] - r["warn"] < 1.0]
    missed = [r for r in went if r["warn"] is None]
    near = [r for r in rows if r["enter"] is None and r["closest"] - circle[2] <= lane_m]
    far = [r for r in rows if r["enter"] is None and r["closest"] - circle[2] > lane_m]
    return dict(went_through=len(went), saved=len(saved), warned_late=len(late), never_warned=len(missed),
                near_misses=len(near), near_left_alone=sum(r["warn"] is None for r in near),
                far_cars=len(far), far_left_alone=sum(r["warn"] is None for r in far),
                leads=[round(r["enter"] - r["warn"], 2) for r in went if r["warn"] is not None], rows=rows)


if __name__ == "__main__":
    rec = pickle.load(open(sys.argv[1], "rb"))
    tracks, s = run(rec)
    print(len(rec["t"]), "frames,", len(tracks), "followed vehicles")
    print(one_second_guesses(tracks))


def one_second_guesses_moving(tracks, min_speed=1.0, zone_m=25.0):
    """The same 1-second check, but only at moments the car was really moving (at least 1 m/s over
    the last half second), since a parked or waiting car is an easy guess."""
    errs = []
    for tr in tracks:
        sp, raw = tr.spots, np.array(tr.raw)
        for i in range(4, len(sp)):
            t, a, b = sp[i]
            now = tr.now[i]
            if np.hypot(a - CROSSWALK_A, b - ROAD_MID_B) > zone_m:
                continue
            j = np.searchsorted(raw[:, 0], now - 0.5)
            if raw[i, 0] - raw[j, 0] <= 0 or np.hypot(*(raw[i, 1:] - raw[j, 1:])) / (raw[i, 0] - raw[j, 0]) < min_speed:
                continue
            real = at(tr.raw, now + 1.0)
            if real is None:
                continue
            ans = ryan_code.guess_and_decide(sp[:i + 1], FAR, 0.3)
            pe = ans["path_end"]
            ahead = now + 1.0 - t
            g1 = (a + (pe[0] - a) / 2 * ahead, b + (pe[1] - b) / 2 * ahead)
            errs.append(float(np.hypot(g1[0] - real[0], g1[1] - real[1])))
    e = np.array(errs)
    return dict(n=len(e), within_1m=float((e <= 1.0).mean()) if len(e) else None,
                median_m=float(np.median(e)) if len(e) else None,
                within_2m=float((e <= 2.0).mean()) if len(e) else None)
