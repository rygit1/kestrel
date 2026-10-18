"""
The simulations: Kestrel on the fake road, before anything real is plugged in.

    python sims.py                20 fake rolls (do the eyes and the steps work, and how does
                                  Ryan's code score), then "how fast is too fast"
    python sims.py --selftest     the checks that Kestrel's own parts work, each with its count
    python sims.py --picture P    saves one screen from the middle of a roll to the file P

The "stand-ins" below are NOT Ryan's code and never guess anything. They only say "drop" at
a set moment, or when the car passes a fixed line across the road (a tripwire), so the checks
can test Kestrel's own parts: the steps, the gate, the light, the voice, the replay.
"""

import argparse
import ast
import random
import sys
import tempfile
import types
from pathlib import Path

import fake_road as fr
import kestrel_brain as kb
import ryan_code

FPS = 30
HIT_STEPS = ["IDLE", "DETECTED", "TRACKING", "ASSESSING", "PROTECTION REQUIRED",
             "PROTECTION DEPLOYED", "ALL CLEAR", "RESET", "IDLE"]
MISS_STEPS = ["IDLE", "DETECTED", "TRACKING", "ASSESSING", "ALL CLEAR", "RESET", "IDLE"]


# ---------------------------------------------------------------- stand-ins (not Ryan's code)

def stand_in(guess=None, inside=None, close=None):
    """A pretend ryan_code. Any part left out acts like it is not written yet."""
    return types.SimpleNamespace(
        is_inside_circle=inside or (lambda car, circle: None),
        guess_and_decide=guess or (lambda spots, circle, gate_seconds: None),
        how_close=close or (lambda spots, circle: None))


def drop_at(moment):
    """Says drop once the clock reaches a set moment."""
    return lambda spots, circle, gate_seconds: spots[-1][0] >= moment


def never_drop():
    return lambda spots, circle, gate_seconds: False


def tripwire(x_px):
    """Says drop once the car's middle passes a fixed line across the road. A tripwire, not a guess."""
    return lambda spots, circle, gate_seconds: spots[-1][1] >= x_px


def crashes():
    def guess(spots, circle, gate_seconds):
        return 1 / 0
    return guess


# ---------------------------------------------------------------- one fake roll

def run_roll(kind, speed, ryan=None, version=3, warning=False, auto=True, seed=1,
             answers_hello=True, press_at=None, with_replay=False, picture=None, ai=None, decoy=False):
    """Rolls one fake car with Kestrel watching, start to finish, and reports what happened."""
    road = fr.FakeRoad(seed=seed)
    board = fr.FakeBoard(road, answers_hello=answers_hello)
    voice = kb.Voice(quiet=True)
    log = kb.Log(tempfile.mkdtemp(prefix="kestrel-sim-"), clock=lambda t: f"{t:7.3f}s", echo=False)
    settings = kb.practice_settings()
    k = kb.Kestrel(settings, board, voice, log, ryan=ryan or ryan_code, version=version,
                   warning_mode=warning, auto=auto, echo_board=False, eyes=kb.Eyes(settings, ai))
    road.decoy = decoy
    k.hello(0.0, wait=0)
    road.start_roll(kind, speed, 0.5)
    replay = kb.Replay() if with_replay else None
    replay_info = None
    found = real_found = should = decoy_picks = 0
    misses_cm, eye_ms = [], []
    t, end, pressed = 0.0, None, False
    while t < 20.0:
        road.update(t)
        frame = road.frame(t)
        k.see(frame, t)
        eye_ms.append(k.eyes.ms)
        if k.car is not None and fr.px(fr.DECOY[0]) <= k.car[0] <= fr.px(fr.DECOY[2]) \
                and fr.px(fr.DECOY[1]) <= k.car[1] <= fr.px(fr.DECOY[3]):
            decoy_picks += 1
        if press_at is not None and t >= press_at and not pressed:
            pressed = True
            k.press_protect(t)
        truth = road.car_spot_px(t)
        if truth is not None and fr.px(fr.CAR_LONG / 2) <= truth[0] <= fr.W - fr.px(fr.CAR_LONG / 2):
            should += 1                      # the car is all the way in the picture
            if k.car is not None:
                found += 1
                off = max(abs(k.car[0] - truth[0]), abs(k.car[1] - truth[1])) / fr.PX_PER_CM
                misses_cm.append(off)
                real_found += off <= 1.5                 # it is the real car, not something else
        if picture is not None and k.step == "PROTECTION REQUIRED" and not Path(picture).exists():
            kb.cv2.imwrite(str(picture), kb.draw_screen(frame, k, t, practice=True))
        if replay is not None:
            replay.add(t, kb.draw_screen(frame, k, t, practice=True))
            if k.replay_request is not None and replay_info is None:
                window = k.replay_request
                replay.start(window, t, list(k.spots))
                span = replay.clip[-1][0] - replay.clip[0][0] if replay.clip else 0.0
                pictures = len(replay.clip or [])
                shown, now = 0, t
                while replay.frame(now) is not None:
                    shown += 1
                    now += 1.0 / FPS
                replay_info = {"window": window, "span": span, "played": shown / FPS,
                               "frames": pictures}
        if end is None and road.roll_over(t) and k.step == "IDLE":
            end = t
        if end is not None and t > end + 0.2:
            break
        t += 1.0 / FPS
    sent = [line for _, line in board.sent]
    return {
        "history": k.history, "sent": sent, "log": log.lines,
        "gate_downs": sum(line.startswith("k gate down") for line in sent),
        "red_lights": sum(line == "k light red" for line in sent),
        "lights": [line for line in sent if line.startswith("k light")],
        "honks": sum(w == "honk" for _, w in voice.heard),
        "voice": [w for _, w in voice.heard if w != "honk"],
        "saved": road.roll.blocked_at is not None,
        "reaches_gate": road.roll.front_reaches_gate() is not None,     # False = it stopped on its own first
        "gate_dropped": road.gate_down_at is not None,
        "kid_hit": road.kid_hit,
        "found": found, "real_found": real_found, "should": should, "misses_cm": misses_cm,
        "status": dict(k.ryan_status), "banner": k.banner, "protect_time": k.protect_time,
        "replay": replay_info, "unsent": k.unsent, "decoy_picks": decoy_picks,
        "eye_ms": sorted(eye_ms)[len(eye_ms) // 2] if eye_ms else 0.0,
    }


def back_to_back():
    """Two rolls, the second one let go while Kestrel still says "all clear" about the first, the
    way Ryan did in his practice run on Oct 6. Hands back where Kestrel first saw each car, in cm."""
    road = fr.FakeRoad(seed=11)
    board = fr.FakeBoard(road)
    log = kb.Log(tempfile.mkdtemp(prefix="kestrel-sim-"), clock=lambda t: f"{t:7.3f}s", echo=False)
    settings = kb.practice_settings()
    k = kb.Kestrel(settings, board, kb.Voice(quiet=True), log, ryan=stand_in(guess=never_drop()),
                   echo_board=False, eyes=kb.Eyes(settings))
    k.hello(0.0, wait=0)
    road.start_roll("hit", 0.6, 0.5)
    firsts, second_started, t = [], False, 0.0
    while t < 20.0:
        road.update(t)
        k.see(road.frame(t), t)
        if k.step == "DETECTED" and (not firsts or k.history.count("DETECTED") > len(firsts)):
            firsts.append(k.spots[0][1] / fr.PX_PER_CM)
        if not second_started and k.step == "ALL CLEAR":
            road.start_roll("hit", 0.6, t)          # the next car, let go during "all clear", like Ryan did
            second_started = True
        if second_started and len(firsts) >= 2 and road.roll_over(t) and k.step == "IDLE":
            break
        t += 1.0 / FPS
    return firsts


# ---------------------------------------------------------------- sim 1: 20 fake rolls

def twenty_rolls(ryan=None, quiet=False):
    rng = random.Random(7)
    plan = ["hit"] * 10 + ["miss"] * 10
    rng.shuffle(plan)
    rolls = []
    for i, kind in enumerate(plan):
        speed = rng.uniform(0.45, 0.75)
        r = run_roll(kind, speed, ryan=ryan, seed=100 + i)
        r["kind"], r["speed"] = kind, speed
        rolls.append(r)
        if not quiet:
            outcome = ("SAVED" if r["saved"] else "kid hit") if kind == "hit" else \
                      ("false alarm" if r["gate_dropped"] else "left alone")
            print(f"  roll {i + 1:2d}  aimed {'at the kid' if kind == 'hit' else 'to miss   '}  "
                  f"{speed:.2f} m/s   {outcome}")
    return rolls


def report_rolls(rolls):
    found = sum(r["found"] for r in rolls)
    should = sum(r["should"] for r in rolls)
    misses = [m for r in rolls for m in r["misses_cm"]]
    in_order = sum(r["history"] in (HIT_STEPS, MISS_STEPS) for r in rolls)
    hits = [r for r in rolls if r["kind"] == "hit"]
    near = [r for r in rolls if r["kind"] == "miss"]
    saved = sum(r["saved"] for r in hits)
    left_alone = sum(not r["gate_dropped"] for r in near)
    print(f"\n  EYES:   found the car in {found} of {should} pictures ({100 * found / max(1, should):.1f}%), "
          f"off by {sum(misses) / max(1, len(misses)):.2f} cm on average")
    print(f"  STEPS:  {in_order} of {len(rolls)} rolls went through the steps in order")
    status = rolls[0]["status"][2]
    print(f"  YOUR CODE (part 2: {status}):")
    print(f"     kids saved          {saved} of {len(hits)}   (the bar is 9 of 10)")
    print(f"     near misses left alone {left_alone} of {len(near)}   (the bar is 9 of 10)")
    if status == "not written yet":
        print("     Your part 2 is empty, so Kestrel never drops the gate. That is the safe way to wait.")
    return found, should, misses, in_order, saved, left_alone


# ---------------------------------------------------------------- sim 2: how fast is too fast

def how_fast(drop_seconds=None):
    """Races the gate against fake cars. Kestrel says drop when the car's middle passes a line
    across the road (a tripwire), at different distances before the gate. ok = the arm was all
    the way down before the car got there; LATE = it was not; stops = the car ran out of speed
    and never reached the gate at all."""
    keep = fr.GATE_DROP_SECONDS
    if drop_seconds is not None:
        fr.GATE_DROP_SECONDS = drop_seconds
    speeds = [0.3, 0.5, 0.7, 0.9, 1.1]
    early = [0, 10, 20, 30, 40]
    print(f"\n  HOW FAST IS TOO FAST: the arm takes {fr.GATE_DROP_SECONDS:.2f} s to come down, the camera is "
          f"{fr.CAMERA_LATE:.2f} s late, the car leaves the ramp {fr.GATE_X - fr.ROLLS['hit']['start'][0]:.0f} cm before the gate")
    print("  Kestrel says drop when the car's middle passes a line this far before the gate:")
    print("                " + "".join(f"{'at gate' if d == 0 else f'{d} cm':>9}" for d in early))
    best = {}
    for v in speeds:
        row = []
        for d in early:
            r = run_roll("hit", v, ryan=stand_in(guess=tripwire(fr.px(fr.GATE_X - d))), seed=5)
            row.append("stops" if not r["reaches_gate"] else ("ok" if r["saved"] else "LATE"))
            if r["saved"]:
                best[d] = max(best.get(d, 0), v)
        print(f"  car {v:.1f} m/s  " + "".join(f"{x:>9}" for x in row))
    fr.GATE_DROP_SECONDS = keep
    return best


def speed_report():
    full = how_fast()
    half = how_fast(drop_seconds=0.15)
    print("\n  What it means:")
    print("   - Saying drop at the gate itself, the way version 2 waits for the car to be in the circle,")
    print("     is LATE at every speed. That is why version 3 has to guess early.")
    print("   - Kestrel watches a car for 5 pictures (about a sixth of a second) before it asks your code,")
    print("     so a fast car is already well off the ramp when the first answer can come.")
    for d in (30, 40):
        if d in full or d in half:
            print(f"   - Deciding {d} cm early beats cars up to {full.get(d, 0):.1f} m/s with the arm straight up, "
                  f"and up to {half.get(d, 0):.1f} m/s with the arm only halfway up.")
    if 30 in full:
        v = full[30]
        top = v * v / (2 * 9.81 * 0.7) * 100       # rough: a toy car keeps about 70% of the ramp's push
        print(f"   - To keep the car under {v:.1f} m/s, start with the ramp's top about {top:.0f} cm up")
        print("     (a rough start; Sunday's real rolls decide).")


# ---------------------------------------------------------------- the self-test

def ryan_code_is_empty():
    """True if each of Ryan's three functions holds only its plain-words description."""
    tree = ast.parse(Path(ryan_code.__file__).read_text())
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        for stmt in fn.body:
            is_words = isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant) \
                and isinstance(stmt.value.value, str)
            if not (is_words or isinstance(stmt, ast.Pass)):
                return False
    return True


def run_startup_test(keys):
    road = fr.FakeRoad(seed=3)
    board = fr.FakeBoard(road)
    voice = kb.Voice(quiet=True)
    log = kb.Log(tempfile.mkdtemp(prefix="kestrel-sim-"), clock=lambda t: f"{t:7.3f}s", echo=False)
    k = kb.Kestrel(kb.practice_settings(), board, voice, log, echo_board=False)
    test = kb.StartupTest(k, practice=True, road=road)
    keys, tips, t = list(keys), [], 0.0
    while not test.done and t < 90:
        key = None
        if isinstance(test.waiting, kb.Ask) and keys and t - test.since > 0.3:
            key = keys.pop(0)
        test.update(t, key, road.frame(t))
        if test.tip and test.tip not in tips:
            tips.append(list(test.tip))
        t += 1.0 / FPS
    return test, tips, board, keys


def selftest():
    results = []

    def check(name, ok, count):
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}   [{count}]")

    print("KESTREL SELF-TEST (on the fake road)\n")
    hit = run_roll("hit", 0.6, ryan=stand_in(guess=drop_at(0.9)), with_replay=True)
    check("hit roll: the steps go in order", hit["history"] == HIT_STEPS, " > ".join(hit["history"]))
    check("hit roll: gate-down sent exactly once", hit["gate_downs"] == 1, f"{hit['gate_downs']} sent")
    check("hit roll: red light sent exactly once", hit["red_lights"] == 1, f"{hit['red_lights']} sent")
    check("hit roll: the light goes green, yellow, red, green",
          hit["lights"] == ["k light green", "k light yellow", "k light red", "k light green"], ", ".join(hit["lights"]))
    check("hit roll: exactly one honk", hit["honks"] == 1, f"{hit['honks']} honks")
    want = ["Car detected.", "Danger. Gate down.", "All clear."]
    check("hit roll: the voice says each line once, in order", hit["voice"] == want, " / ".join(hit["voice"]))
    check("hit roll: the arm was down before the car got there", hit["saved"], "car stopped at the arm" if hit["saved"] else "car got past")
    step_lines = sum(any(f"  {s}" in line for s in kb.STEPS) for line in hit["log"])
    check("hit roll: one log line for every step", step_lines == len(hit["history"]) - 1,
          f"{step_lines} step lines for {len(hit['history']) - 1} steps")

    miss = run_roll("miss", 0.6, ryan=stand_in(guess=never_drop()))
    check("miss roll: the steps go in order and skip PROTECTION", miss["history"] == MISS_STEPS, " > ".join(miss["history"]))
    check("miss roll: zero gate-down, zero red light, zero honks",
          miss["gate_downs"] == 0 and miss["red_lights"] == 0 and miss["honks"] == 0,
          f"{miss['gate_downs']} gate, {miss['red_lights']} red, {miss['honks']} honks")

    warn = run_roll("hit", 0.6, ryan=stand_in(guess=drop_at(0.9)), warning=True)
    check("warning mode: red light once, one honk, zero gate-down",
          warn["red_lights"] == 1 and warn["honks"] == 1 and warn["gate_downs"] == 0,
          f"{warn['red_lights']} red, {warn['honks']} honks, {warn['gate_downs']} gate")
    check("warning mode: the voice never says gate down", not any("Gate down" in w for w in warn["voice"]),
          " / ".join(warn["voice"]))

    empty = run_roll("hit", 0.6, ryan=stand_in())
    check("a part not written yet: zero gate-down", empty["gate_downs"] == 0, f"{empty['gate_downs']} sent")
    check("a part not written yet: the screen says not written yet for all 3 parts",
          all(v == "not written yet" for v in empty["status"].values()), str(empty["status"]))
    real_hit, real_miss = run_roll("hit", 0.6), run_roll("miss", 0.6)
    check("Ryan's real code (10/13): a car headed at the kid is stopped by the arm, a car that misses rolls through",
          real_hit["saved"] and real_miss["gate_downs"] == 0,
          f"hit {'stopped' if real_hit['saved'] else 'got past'}, miss {real_miss['gate_downs']} gate-downs")

    crash = run_roll("hit", 0.6, ryan=stand_in(guess=crashes()))
    check("a part that crashes: zero gate-down, and Kestrel keeps running to the end",
          crash["gate_downs"] == 0 and crash["history"][-1] == "IDLE", f"{crash['gate_downs']} sent, ended in {crash['history'][-1]}")
    check("a part that crashes: the screen names the line", crash["status"][2].startswith("error on line"), crash["status"][2])

    near = run_roll("miss", 0.6, ryan=stand_in(guess=never_drop(), close=lambda spots, circle: 48.0))
    check("how close it came: MISSED BY 4 CM on the screen", near["banner"] is not None and near["banner"][0] == "MISSED BY 4 CM",
          near["banner"][0] if near["banner"] else "no banner")
    check("how close it came: the voice says the same number", "All clear. Missed by 4 centimeters." in near["voice"],
          " / ".join(near["voice"]))
    saved = run_roll("hit", 0.6, ryan=stand_in(guess=drop_at(0.9), close=lambda spots, circle: 48.0))
    check("how close it came: no banner when the gate stopped the car", saved["banner"] is None, str(saved["banner"]))
    check("how close it came: no banner while part 3 is empty", miss["banner"] is None, str(miss["banner"]))

    rp = hit["replay"]
    ok = rp is not None and rp["window"][0] <= hit["protect_time"] <= rp["window"][1]
    check("replay: the clip covers the moment the gate dropped", ok, f"{rp['window'][0]:.2f} to {rp['window'][1]:.2f} s" if rp else "no replay")
    ok = rp is not None and rp["frames"] >= 30 and abs(rp["played"] - 4 * rp["span"]) <= 0.1
    check("replay: plays at quarter speed", ok,
          f"{rp['span']:.2f} s of roll, {rp['frames']} pictures, plays in {rp['played']:.2f} s" if rp else "no replay")

    v2 = run_roll("hit", 0.6, ryan=stand_in(inside=lambda car, circle: car[0] >= 700), version=2)
    check("version 2: gate-down once when part 1 says inside", v2["gate_downs"] == 1, f"{v2['gate_downs']} sent")

    manual = run_roll("hit", 0.6, ryan=stand_in(guess=drop_at(0.9)), auto=False)
    check("MANUAL: Kestrel never drops the gate by itself", manual["gate_downs"] == 0, f"{manual['gate_downs']} sent")
    check("MANUAL: the log tells Ryan to press SPACE", any("MANUAL mode" in line for line in manual["log"]), "log line found")
    button = run_roll("hit", 0.6, ryan=stand_in(guess=never_drop()), press_at=1.0)
    check("the PROTECT button (SPACE) drops the gate once", button["gate_downs"] == 1, f"{button['gate_downs']} sent")

    deaf = run_roll("hit", 0.6, ryan=stand_in(guess=drop_at(0.9)), answers_hello=False)
    moved = [line for line in deaf["sent"] if line != "k hello"]
    check("a board that never says kestrel ready gets zero gate or light lines", not moved,
          f"{len(moved)} sent, {deaf['unsent']} held back")

    firsts = back_to_back()
    check("a car rolled right after the last one is seen from the ramp, not halfway down the road",
          len(firsts) == 2 and all(x <= 15 for x in firsts), ", ".join(f"first seen at {x:.0f} cm" for x in firsts))

    every = [line for r in (hit, miss, warn, empty, crash, near, saved, v2, manual, button) for line in r["sent"]]
    caps = [line for line in every if line != line.lower()]
    check("every line sent to the board is lowercase, so the old board's uppercase commands can never run by mistake",
          not caps, f"{len(every)} lines, {len(caps)} with capitals")

    test, tips, board, left = run_startup_test(["y", "y", "r", "y", "g", "y"])
    rows = {row[0]: row[1] for row in test.rows}
    check("startup test: every check OK on a good setup", all(v == "OK" for v in rows.values()) and "ALL SYSTEMS GO" in rows,
          ", ".join(f"{n} {s}" for n, s in rows.items()))
    test, tips, board, left = run_startup_test(["n", "s", "r", "y", "g", "y"])
    rows = {row[0]: row[1] for row in test.rows}
    check("startup test: a gate that does not move gets the wiring tip", rows.get("GATE") == "SKIP" and
          any("labeled 26" in line for tip in tips for line in tip), f"GATE {rows.get('GATE')}, tip names the leg labeled 26")
    test, tips, board, left = run_startup_test(["y", "y", "y", "y", "g", "x", "y"])
    rows = {row[0]: row[1] for row in test.rows}
    check("startup test: swapped light wires get named", rows.get("LIGHT") == "FIX" and
          any("labeled 25" in line for tip in tips for line in tip), f"LIGHT {rows.get('LIGHT')}, tip names the leg labeled 25")

    print("\n  20 fake rolls with Ryan's real code, for the eyes:")
    rolls = twenty_rolls(quiet=True)
    found, should, misses, in_order, _, _ = report_rolls(rolls)
    check("eyes: found the car in at least 95% of the pictures it was fully in", found >= 0.95 * should,
          f"{found} of {should}")
    check("eyes: off by under 1 cm on average", sum(misses) / len(misses) < 1.0, f"{sum(misses) / len(misses):.2f} cm")
    check("steps: all 20 rolls went through the steps in order", in_order == 20, f"{in_order} of 20")

    ai = kb.PRACTICE_CAR_AI
    print("\n  The AI eyes (the one that learned the fake car with key L) against a red brick decoy:")
    check("there is a practice AI that passed its own test (press L in practice mode to make one)",
          ai.exists(), str(ai.name) if ai.exists() else "missing")
    if ai.exists():
        alone = run_roll("hit", 0.6, ryan=stand_in(guess=never_drop()), decoy=True)
        team = run_roll("hit", 0.6, ryan=stand_in(guess=never_drop()), decoy=True, ai=ai)
        check("the decoy really fools the color eyes alone (or this test would prove nothing)",
              alone["decoy_picks"] > 0, f"color alone picked the brick in {alone['decoy_picks']} pictures")
        check("color + AI never picks the brick", team["decoy_picks"] == 0, f"{team['decoy_picks']} pictures")
        check("color + AI finds the real car in at least 95% of the pictures it is fully in",
              team["real_found"] >= 0.95 * team["should"], f"{team['real_found']} of {team['should']}")
        check("color + AI stays fast: under 15 thousandths of a second a picture (half the time)",
              team["eye_ms"] < 15, f"{team['eye_ms']:.1f} ms, color alone {alone['eye_ms']:.1f} ms")

    print(f"\n{sum(results)} of {len(results)} checks passed")
    return all(results)


def smoke(real_camera=False):
    """Runs the real program (kestrel_brain.main) with no window on the screen: the window calls are
    swapped for pretend ones and the keys are typed by a script. Proves the main loop starts, answers
    every key and quits cleanly. With real_camera=True it uses a pretend webcam, to test the camera
    reader that always hands over the newest picture."""
    import time as clock
    folder = Path(tempfile.mkdtemp(prefix="kestrel-smoke-"))
    kb.HERE, kb.SETTINGS_FILE = folder, folder / "settings.json"
    shown, calls = [], {"n": 0}
    plan = {20: "h", 200: "m", 380: "d", 440: "d", 450: "2", 455: "3", 460: "w", 470: "w", 475: "a", 480: "a",
            490: " ", 540: " ", 600: "p", 605: "p", 610: "l", 700: "q"}
    def wait_key(ms):
        calls["n"] += 1
        clock.sleep(max(ms, 1) / 1000)
        key = plan.get(calls["n"])
        return -1 if key is None else ord(key)
    kb.cv2.imshow = lambda name, img: shown.append(img.shape)
    kb.cv2.waitKey = wait_key
    for name in ("namedWindow", "resizeWindow", "setMouseCallback", "destroyAllWindows"):
        setattr(kb.cv2, name, lambda *a, **k: None)
    args = ["kestrel_brain.py", "--quiet", "--skip-test"]
    if real_camera:
        road = fr.FakeRoad(seed=9)

        class PretendWebcam:
            def __init__(self):
                self.t0 = clock.monotonic()
            def read(self):
                clock.sleep(1 / 30)
                t = clock.monotonic() - self.t0
                road.update(t)
                return True, road.frame(t)
            def set(self, *a):
                return True
        kb.open_camera = lambda index: PretendWebcam()
        args.append("--no-board")
        plan = {30: "q"}
    else:
        args.append("--practice")
    sys.argv = args
    kb.main()
    logs = list(folder.rglob("events.log"))
    text = logs[0].read_text() if logs else ""
    return {"pictures_shown": len(shown), "sizes": sorted(set(shown)), "keys_typed": len(plan),
            "loop_calls": calls["n"], "log": text, "live_picture": (folder / "logs" / "live.jpg").exists()}


def main():
    ap = argparse.ArgumentParser(description="Kestrel on the fake road")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="run the real program with no window and typed keys")
    ap.add_argument("--picture", default=None, help="save one screen from the middle of a roll")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.smoke:
        practice = smoke()
        want = ["rolling a car at the kid", "rolling a car to just miss", "red brick dropped", "version 2",
                "version 3", "warning mode ON", "warning mode OFF", "MANUAL", "AUTO", "PROTECT button",
                "learn my car: collecting"]
        missing = [w for w in want if w not in practice["log"]]
        print(f"PRACTICE: {practice['pictures_shown']} screens shown, sizes {practice['sizes']}, "
              f"live picture written: {practice['live_picture']}, key results missing from the log: {missing or 'none'}")
        real = smoke(real_camera=True)
        print(f"PRETEND WEBCAM: {real['pictures_shown']} screens shown, sizes {real['sizes']}, quit cleanly")
        ok = not missing and practice["live_picture"] and practice["pictures_shown"] > 600 and real["pictures_shown"] > 20
        print("SMOKE TEST", "PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    if args.picture:
        Path(args.picture).unlink(missing_ok=True)
        show = stand_in(guess=lambda spots, circle, gate_seconds: {   # a stand-in with fixed numbers,
            "drop_gate": spots[-1][0] >= 0.9, "speed": 600.0,           # only to see where things sit
            "path_end": (circle[0], circle[1]), "seconds_left": 0.5})   # on the screen
        run_roll("hit", 0.6, ryan=show, picture=args.picture)
        print("saved", args.picture)
        return
    print("20 FAKE ROLLS, using your code in ryan_code.py:")
    report_rolls(twenty_rolls())
    speed_report()


if __name__ == "__main__":
    main()
