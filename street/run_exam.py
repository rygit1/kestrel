"""THE EXAM: minutes 4 to 10, run once, after FROZEN.json is written.
Refuses to run if the code or the settings changed since they were frozen, or if it already ran."""
import hashlib, json, pickle, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "FROZEN.json"
RESULT = HERE / "EXAM-RESULT.json"
CODE = ["street_kestrel.py", "ground.py", "register.py", "cache_dets.py", "run_exam.py", "../ryan_code.py"]
SEALED = "/private/tmp/claude-501/-Users-ryanyousefi/199ee794-4390-4e10-baef-111d9fbc38f2/scratchpad/street/handcount/exam_hand_counts.json"
EXAM = (240.0, 606.5)
STRETCHES = {"exam_A": (270, 300), "exam_B": (375, 405), "exam_C": (465, 495), "exam_D": (555, 585)}


def fingerprint():
    return {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE}


def freeze(settings, kids, notes):
    if FROZEN.exists():
        sys.exit("already frozen")
    FROZEN.write_text(json.dumps(dict(settings=settings, kids=kids, notes=notes, code=fingerprint()), indent=1))
    print("frozen:", FROZEN)


def main():
    if RESULT.exists():
        sys.exit("The exam already ran once. It does not run again.")
    fz = json.loads(FROZEN.read_text())
    if fz["code"] != fingerprint():
        sys.exit("Code changed after freezing. Not running.")
    import street_kestrel as K
    s = fz["settings"]
    parts = [pickle.load(open(HERE / "cache" / f, "rb")) for f in ("exam_p0.pkl", "exam_p1.pkl")]
    keep = [i for i, t in enumerate(parts[0]["t"]) if t < parts[1]["t"][0] - 0.01]
    m = s["model"]
    rec = {"t": [parts[0]["t"][i] for i in keep] + parts[1]["t"], "H": [parts[0]["H"][i] for i in keep] + parts[1]["H"],
           "dets": {m: [parts[0]["dets"][m][i] for i in keep] + parts[1]["dets"][m]}}
    assert rec["t"][0] < 240.1 and rec["t"][-1] > 606.0, "exam minutes not fully read"
    assert max(np.diff(rec["t"])) < 0.05, "a gap in the exam frames"
    pickle.dump(rec, open(HERE / "cache" / "exam_240_606.pkl", "wb"))
    tracks, _ = K.run(rec, **s)
    hand = {x["stretch"]: x["cars"] for x in json.load(open(SEALED))}
    cr = K.crossings(tracks)
    found = {}
    tf = tn = te = 0
    for k, (a, b) in STRETCHES.items():
        f, n, e = K.match_counts(hand[k], [c for c in cr if a - 1.5 <= c["t"] < b + 1.5], span=(a, b))
        found[k] = dict(found=f, hand_counted=n, extra=e)
        tf += f; tn += n; te += e
    guesses_all = K.one_second_guesses(tracks)
    kids = {name: K.pretend_kid(tracks, tuple(c), gate_seconds=0.3) for name, c in fz["kids"].items()}
    out = dict(when_frozen=FROZEN.stat().st_mtime, cars_found=dict(total_found=tf, total_hand_counted=tn, total_extra=te, by_stretch=found),
               one_second_guess=guesses_all, one_second_guess_moving=K.one_second_guesses_moving(tracks),
               pretend_kids={n: {k: v for k, v in r.items() if k != "rows"} for n, r in kids.items()}, followed_vehicles=len(tracks),
               frames=len(rec["t"]), lined_up_frames=int(sum(h is not None for h in rec["H"])))
    RESULT.write_text(json.dumps(out, indent=1))
    pickle.dump(dict(tracks=[(tr.id, tr.raw) for tr in tracks], kid_rows={n: r["rows"] for n, r in kids.items()}), open(HERE / "exam_tracks.pkl", "wb"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
