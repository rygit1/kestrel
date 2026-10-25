"""Lines every frame back up with the reference frame (minute 0), so the drone drifting or turning
does not look like the cars moving. Uses spots on the road itself (paint, cracks, curbs), which lie
flat, and ignores trees and hedges, which stick up and would shift differently."""
import cv2, numpy as np
from ground import build, to_ground, S

SCALE = 0.25                     # register on a 960 x 540 copy of the 4K picture


def road_mask(g, shape):
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    px = np.c_[xs.ravel(), ys.ravel()] / SCALE
    ab = to_ground(g, px)
    a, b = ab[:, 0], ab[:, 1]
    ok = ((b > 0.15) & (b < 1.95) & (a > 0.6) & (a < 7)) | ((b > 0.15) & (b < 3.2) & (a > 0.6) & (a < 2.45))
    return (ok.reshape(h, w) * 255).astype(np.uint8)


class Register:
    def __init__(self, ref_bgr):
        self.g = build()
        small = cv2.resize(ref_bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.mask = road_mask(self.g, gray.shape)
        self.sift = cv2.SIFT_create(nfeatures=3000)
        self.kp, self.des = self.sift.detectAndCompute(gray, self.mask)
        self.match = cv2.BFMatcher(cv2.NORM_L2)

    def to_ref(self, bgr_small_gray):
        """Homography from THIS frame's 960x540 pixels to the reference's 960x540 pixels."""
        kp, des = self.sift.detectAndCompute(bgr_small_gray, None)
        if des is None or len(kp) < 20:
            return None, 0
        m = self.match.knnMatch(des, self.des, k=2)
        good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
        if len(good) < 20:
            return None, len(good)
        src = np.float32([kp[a.queryIdx].pt for a in good])
        dst = np.float32([self.kp[a.trainIdx].pt for a in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0, maxIters=4000, confidence=0.999)
        return H, int(inl.sum()) if inl is not None else 0


def full(H):
    """Turn a 960x540 homography into a 4K one."""
    Sm = np.diag([SCALE, SCALE, 1.0])
    return np.linalg.inv(Sm) @ H @ Sm


if __name__ == "__main__":
    import time
    ref = cv2.imread(S + 'ref_5.png')
    R = Register(ref)
    print("reference road spots:", len(R.kp))
    for t in (120, 200, 235):
        im = cv2.imread(S + f'ref_{t}.png')
        gray = cv2.cvtColor(cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        t0 = time.time(); H, n = R.to_ref(gray); dt = time.time() - t0
        Hf = full(H)
        warped = cv2.warpPerspective(im, Hf, (3840, 2160))
        # check: crosswalk bar ends should land where they are in the reference
        blend = cv2.addWeighted(ref, 0.5, warped, 0.5, 0)
        cv2.imwrite(S + f'reg_{t}.jpg', cv2.resize(blend[900:1700, 1200:3400], (1100, 400)))
        print(t, "s: matched road spots", n, f"{dt*1000:.0f} ms", "corner shift px:", np.round(cv2.perspectiveTransform(np.float32([[[0, 0]], [[3839, 2159]]]), Hf).ravel() - [0, 0, 3839, 2159]))
