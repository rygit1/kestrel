"""Picture sheets for counting cars BY EYE: 4 frames a second, each lined up with the reference
view and cropped to the crosswalk, 20 frames per sheet with the clip time stamped big on each."""
import sys, os, cv2, numpy as np
from register import Register, SCALE, full, S
from cache_dets import frames

X0, Y0, X1, Y1 = 1250, 420, 3500, 1750          # crop in reference 4K pixels
TW = 480
TH = int(TW * (Y1 - Y0) / (X1 - X0))


def make(start, end, folder):
    os.makedirs(folder, exist_ok=True)
    reg = Register(cv2.imread(S + 'ref_5.png'))
    tiles, k, sheet = [], 0, 0
    for i, (t, im) in enumerate(frames(start, end)):
        if i and int(i * 4 / 29.97) == int((i - 1) * 4 / 29.97):
            continue                                   # keep 4 frames a second, evenly spaced
        big = cv2.resize(im, (3840, 2160))
        small = cv2.cvtColor(cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        Hm, n = reg.to_ref(small)
        warped = cv2.warpPerspective(big, full(Hm), (3840, 2160)) if Hm is not None else big
        tile = cv2.resize(warped[Y0:Y1, X0:X1], (TW, TH), interpolation=cv2.INTER_AREA)
        m, s = divmod(t, 60)
        cv2.rectangle(tile, (0, 0), (150, 34), (0, 0, 0), -1)
        cv2.putText(tile, f"{int(m)}:{s:05.2f}", (5, 26), 0, 0.85, (255, 255, 255), 2)
        tiles.append(tile)
        if len(tiles) == 20:
            write(tiles, folder, sheet, start); tiles, sheet = [], sheet + 1
    if tiles:
        write(tiles, folder, sheet, start)


def write(tiles, folder, sheet, start):
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack([cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles[r:r + 4]]) for r in range(0, len(tiles), 4)]
    cv2.imwrite(f"{folder}/sheet_{sheet + 1}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    make(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
