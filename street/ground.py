"""The ruler: turns a spot in the fixed reference picture (4K pixels) into meters on the road.
Built from the crosswalk bars: they are parallel on the road, so the lines through them meet at
one point in the picture (the road's direction); the lines through their ends meet at another
(the direction across the road). Those two directions are at right angles on the real road,
which tells the camera's zoom, and from there every pixel maps onto the flat road."""
import json, numpy as np

S = '/private/tmp/claude-501/-Users-ryanyousefi/199ee794-4390-4e10-baef-111d9fbc38f2/scratchpad/street/'
GOOD = [1, 2, 4, 5, 7, 8, 9, 10, 11]          # the 9 whole bars of the main crosswalk


def meet(lines):
    """Least-squares meeting point of lines given as (point, point)."""
    A, b = [], []
    for p, q in lines:
        n = np.array([q[1] - p[1], p[0] - q[0]], float)
        n /= np.linalg.norm(n)
        A.append(n); b.append(n @ np.array(p, float))
    return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]


def fit_line(pts):
    pts = np.array(pts, float)
    c = pts.mean(0)
    d = np.linalg.svd(pts - c)[2][0]
    return c, c + d * 1000


def build(lane_m=None, lane_units=None):
    bars = [json.load(open(S + 'bars.json'))[i] for i in GOOD]
    v1 = meet([(b["far"], b["near"]) for b in bars])
    v2 = meet([fit_line([b["far"] for b in bars]), fit_line([b["near"] for b in bars])])
    c = np.array([1920.0, 1080.0])
    f2 = -np.dot(v1 - c, v2 - c)
    f = np.sqrt(f2)
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    d1 = Ki @ np.r_[v1, 1]; d1 /= np.linalg.norm(d1)
    d2 = Ki @ np.r_[v2, 1]; d2 /= np.linalg.norm(d2)
    n = np.cross(d1, d2); n /= np.linalg.norm(n)
    if n[1] < 0: n = -n                      # ground is "below": camera y points down the picture
    e1 = d1
    e2 = np.cross(n, e1)
    return dict(v1=v1, v2=v2, f=f, K=K, Ki=Ki, n=n, e1=e1, e2=e2)


def to_ground(g, px, h=1.0):
    """px: (N,2) reference pixels -> (N,2) road coords in units where the camera is h above the road."""
    px = np.atleast_2d(np.asarray(px, float))
    r = (g["Ki"] @ np.c_[px, np.ones(len(px))].T).T
    t = h / (r @ g["n"])
    X = r * t[:, None]
    return np.c_[X @ g["e1"], X @ g["e2"]]


if __name__ == "__main__":
    g = build()
    print("road direction point", g["v1"].round(1), "across point", g["v2"].round(1), "zoom f", round(g["f"], 1))
    bars = [json.load(open(S + 'bars.json'))[i] for i in GOOD]
    far = to_ground(g, [b["far"] for b in bars]); near = to_ground(g, [b["near"] for b in bars])
    print("bar lengths (camera-height units):", np.linalg.norm(near - far, axis=1).round(3))
    mid = (far + near) / 2
    print("bar centers across:", mid[:, 1].round(3), "spacing:", np.diff(mid[:, 1]).round(3))
    print("bar centers along:", mid[:, 0].round(3))
