# Kestrel street test: the rules, written BEFORE the exam (Sat Oct 24 2026)

The clip: drone/street-test-2026-10-24.mp4, filmed by Ryan Sat Oct 24 at 11:27 AM (DJI Neo 2, 4K, 59.94 frames a second, 10 min 6 s).

## The split
- Tuning: minutes 0 to 4 (0:00 to 4:00). Everything is tuned here and only here.
- Exam: minutes 4 to 10 (4:00 to 10:06). Run ONCE, after the settings are frozen. Nothing changes after the exam score is seen.

## The moving view
The drone drifted and turned a little. Every frame is lined up with one fixed reference frame (0:05) using spots on the flat road (paint, cracks, curbs), never trees or hedges. So the WHOLE clip is used, moving view included, not just the steady parts.

## The ruler
The crosswalk bars are parallel on the real road. The lines through them, and through their ends, give the road's two directions, and from those the flat top-down map. Meters come from the drone's own height reading in its flight log at the reference frame (10.3 m). Check: the lanes then come out about 3.2 to 3.7 m wide, a normal city lane (the brief said about 3.6 m).

## Score 1: cars found
People count by eye every motor vehicle that drives across the main crosswalk (the white bars across the road in front of the camera) in 30-second stretches, from picture sheets (4 frames a second, lined up, cropped to the crosswalk). Each stretch is counted by 2 helpers working alone; where they disagree, a third look settles it. They never see Kestrel's output.
- Tuning stretches: 0:30 to 1:00, 1:45 to 2:15, 3:00 to 3:30.
- Exam stretches: 4:30 to 5:00, 6:15 to 6:45, 7:45 to 8:15, 9:15 to 9:45.
Kestrel "found" a counted car if one of its followed cars crosses the middle of the crosswalk in the same direction within 1.5 seconds of the hand-counted time. Extra followed cars with no hand-counted match are reported too.

## Score 2: the 1-second path guess
For every followed car, at every spot where Ryan's code has its 5 spots, Ryan's guess_and_decide (ryan_code.py, unchanged) is asked for the path; the point 1.0 s along that path is compared with where the car really was 1.0 s later (from the followed path). Only spots within 25 m of the middle of the crosswalk count. Reported: the share within 1 meter, and the middle (median) miss in meters.

## Score 3: the pretend kid
- One pretend kid stands in the crosswalk: a protected circle of radius 1.5 m (a car's middle closer than that would hit a kid), centered on the middle of the crosswalk in the lane that had the MOST cars crossing in minutes 0 to 4 (picked from the tuning minutes only).
- For each followed car, Kestrel asks Ryan's guess_and_decide at every spot, with gate_seconds 0.3 (the setting from the LEGO board) and Ryan's own 1.0 s margin, exactly as on the board.
- Answer key: the car REALLY went through the kid's spot if its followed path entered the circle.
- SAVED: the car really went through, and Kestrel warned at least 1.0 s before the car really entered the circle.
- WARNED LATE: it warned, but less than 1.0 s ahead. Counted as NOT saved.
- MISSED: it never warned. Counted as NOT saved.
- LEFT ALONE: the car did NOT go through the spot, but passed within one lane (3.6 m) of the kid, and Kestrel never warned. Cars that passed farther away are reported separately, since staying quiet for them is easy.
- False alarm: a car that did not go through, and Kestrel warned anyway.

## What gets frozen before the exam
The car finder, its settings, the follower settings, the spot rate, the ruler and the kid's circle. Their exact values and a fingerprint of the code go in FROZEN.json before the exam runs.

## Log
- Sat Oct 24 15:32: CAR FINDER LOCKED = yolo26m.pt at 1280 (chosen on minutes 0 to 1:30 only: found 7 of 7 hand-counted cars vs 6 of 7 for the two smaller ones, smallest typical miss). Exam picture-reading (boxes only, no scoring) started now to save time; no exam score is computed until FROZEN.json is written.
- Sat Oct 24 15:33: KID RULE AMENDED BEFORE THE EXAM (found on tuning minute 0 to 1:30): cars heading toward the bend come from BEHIND the drone and first appear about 1.2 s before the crosswalk, so no guess can warn 1 s ahead for them. Cars coming down from the bend are in view for 60+ m. So TWO pretend kids are scored, both chosen from minutes 0 to 4 only: KID 1 (the main one) in the busiest lane of traffic coming toward the camera from the bend, and KID 2 (the blind side) in the busiest lane of traffic coming from behind the drone. Both are reported, side by side. Same radius (1.5 m), same gate_seconds (0.3, from the LEGO board), Ryan's code unchanged.
- Sat Oct 24 15:58: RYAN CHECKED the hand count for exam stretch 4:30 to 5:00 himself on the picture sheets: all 21 vehicles confirmed (his words: "all 21 check out").
- Sat Oct 24 16:18: SCORING EDGE RULE (fixed on tuning stretch C, before the exam): a hand-counted car at the very edge of a stretch may match a Kestrel crossing up to 1.5 s outside the stretch (a car at 2:59.85 was counted by hand at 3:00.00); extra Kestrel cars still only count inside the stretch. Applies to every stretch.
- Sat Oct 24 16:21: FROZEN. Settings, both kid spots and the code fingerprint are in FROZEN.json. Nothing below this line changes any setting.
- Sat Oct 24 16:22: CAMERA REACH (measured on tuning minutes, for Answer 4): cars coming toward the camera were in view a median 4.55 s before reaching kid 1 (first seen a median 131 m away); blind-side cars only 1.26 s before kid 2 (first seen 11.6 m away, median speed 13.3 m/s). Kestrel needs about 0.6 s of watching before its guess settles, so a 1 s warning needs about 1.6 s in view: about 20 m at 13 m/s, 4 to 5 car lengths.
- Sat Oct 24 16:57: EXAM SCORED ONCE (EXAM-RESULT.json): cars found 59 of 63 hand-counted (3 extra; Ryan's stretch 19 of 21); kid 1 warned for 24 of 24 cars through its spot, all 24 at least 1.0 s early (22 between 1.2 and 3.2 s; 2 cars that waited at the light were warned early and stayed warned), stayed quiet for 14 of 31 near misses; kid 2 (blind side) warned for 37 of 37, 1 at least 1.0 s early; 1-second guess on moving cars within 1 m 29.0%, within 2 m 61.5%; 10,984 exam pictures, all lined up. Nothing changed after this.
- Sat Oct 24 17:31: Ryan: v7 SHIP (edit/kestrel-video-v7.mp4, 2:58.60, -14.0 LUFS). Answer 1 uses his '19 of 20' wording.
