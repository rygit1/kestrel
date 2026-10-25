"""Scores one set of settings on the TUNING minutes only (0 to 4)."""
import json, pickle, sys, numpy as np
import street_kestrel as K

HAND = json.load(open('tune_hand_counts.json'))
SPANS = {'tune_A': (30, 60), 'tune_B': (105, 135), 'tune_C': (180, 210)}


def score(rec, **over):
    tracks, s = K.run(rec, **over)
    cr = K.crossings(tracks)
    tmax = rec["t"][-1]
    f = n = extra = 0
    per = {}
    for k, (a, b) in SPANS.items():
        if b > tmax:
            continue
        mine = [c for c in cr if a - 1.5 <= c["t"] < b + 1.5]
        ff, nn, ee = K.match_counts(HAND[k], mine, span=(a, b))
        per[k] = f"{ff}/{nn}+{ee}"
        f += ff; n += nn; extra += ee
    g = K.one_second_guesses(tracks)
    return dict(found=f"{f}/{n}", extra=extra, per=per, within_1m=round(g["within_1m"], 3), median_m=round(g["median_m"], 2), n_guess=g["n"], tracks=len(tracks)), tracks


if __name__ == "__main__":
    rec = pickle.load(open('cache/tune_0_240.pkl', 'rb'))
    print(score(rec)[0])
