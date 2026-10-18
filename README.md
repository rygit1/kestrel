# Kestrel

by Ryan Yousefi

My entry for the 2026 Congressional App Challenge.

## What Kestrel is

Kestrel is a camera that watches a toy road from above. When a toy car comes down the road, Kestrel guesses where the car is going. If the car is headed into a protected circle of LEGO kids, Kestrel acts before the car gets there: it turns a red light on for the driver, honks for the kid, and drops a gate across the road. If the car is going to miss the kids, Kestrel leaves it alone. That second half matters, because it shows Kestrel is guessing ahead and not just reacting to anything that moves.

The camera sits on a Mac. The Mac runs the Python program, and a small ESP32 board moves the two gate arms and the light.

## What each file does

- `kestrel_brain.py`: the main program. It finds the car, remembers where it has been, asks my code what to do, moves the gate and light, honks, and draws the screen.
- `ryan_code.py`: my code. It decides if the car is in the circle, guesses where the car is going, decides when to drop the gate, and works out how close a missing car came.
- `learn_car.py`: teaches Kestrel what my exact toy car looks like, then tests that it learned it before using it.
- `fake_road.py`: a pretend road on the screen, with a pretend car and board, so you can try Kestrel with no camera or board.
- `sims.py`: runs 20 pretend rolls to score my code, checks how fast is too fast, and has a self-test that checks every part.
- `gate_panel.py`: two sliders that turn each gate arm by hand, for testing.
- `firmware/kestrel_board/kestrel_board.ino`: the code on the ESP32 board. It started as the board code for an earlier project of mine, and Kestrel adds the gate and light lines to it. The parts of the old project Kestrel never uses are removed from this public copy.
- `models/my_car.onnx` and `models/my_car.json`: the trained car finder for my real toy car, and how sure it has to be before it counts something as the car.
- `models/practice_car.onnx` and `models/practice_car.json`: the same thing, trained on the pretend road.
- `sounds/honk.wav`: the honk. If it is missing, Kestrel makes a new one by itself.
- `settings.json`: the numbers for my own setup, like where the circle sits in the camera picture and how far each gate arm turns. On a new setup you place the circle again with the C key.
- `requirements.txt`: the Python add-ons Kestrel needs.

## What I typed myself

`ryan_code.py` is mine. I typed it line by line. It has the parts that make Kestrel's decisions:

1. Is the car inside the circle right now?
2. The path guess: where is the car going, and will it reach the circle?
3. The drop-or-wait decision: drop the gate now, or wait until the last safe moment.
4. How close a car that missed came to the circle.

Everything else in Kestrel only asks `ryan_code.py`. None of the other files work out where the car is going or whether it is in the circle.

## AI help

Below is the AI-use list from my App Challenge entry, with the dates taken out. Claude is Anthropic's AI.

- Sentry turret camera tracking and motor code: written by Claude, reused in Kestrel.
- Written by Claude (Anthropic's AI) in `~/kestrel`, all of it support code:
  - `kestrel_brain.py`: the eyes (finding the red car by color, then teaming with the car AI), the memory of the car's spots, the steps from Ryan's doc (IDLE to RESET) with the event log, the command-center screen, the voice (robot voice and the honk sound), the slow-motion replay, the startup self-test that checks the wiring, the click tools (circle, car color, scale, ignore boxes), and the camera reader that always uses the newest picture.
  - `learn_car.py`: "learn my car". It collects pictures of the car that the color eyes mark by themselves, trains a small YOLO26 AI model on the laptop, tests it on pictures it never saw, and only uses it if it passes.
  - `fake_road.py` and `sims.py`: the practice road, the simulations ("how fast is too fast", scoring Ryan's code on 20 fake rolls) and the self-test.
  - `firmware/kestrel_board/kestrel_board.ino`: the gate and light lines added to the old board code.
  - By Claude: robot voice turned off at Ryan's ask, a fix so Kestrel waits for the circle instead of crashing, and a check so "learn my car" only saves deep-red, car-sized pictures (it had been saving Ryan's hand as the car), the whole-car color learner, and the auto setup that finds the two motors and places the protected bubble by itself (Ryan's idea and his rule: any car headed through the gate drops the arms).
  - `README.md`, and the plain-words guide text inside `ryan_code.py` (what goes in, what comes out, the steps in words).
  - Outside tools Kestrel uses: OpenCV (pictures), Ultralytics YOLO26 (an AI object detector, trained on Ryan's own car), PyTorch and ONNX Runtime (run the AI), pyserial (talks to the board), the Mac's built-in voice.
- Written by Claude: `gate_panel.py` (two sliders that turn each gate motor by hand, for testing), and the double gate (Ryan's idea: two arms from both sides) runs its second arm on a spare motor line of the old board. Also by Claude: `kestrel_brain.py` sends the second arm its own up and down angles so both arms move together, and the board code starts gate 2 at its up spot. Ryan found all four angles himself with the panel. Claude also added light keys to the panel and `--practice-board` in `kestrel_brain.py` (the fake road on screen while the real arms and light move with it, for rehearsing before the webcam). By Claude: the camera code now asks the Insta360 for 1080p at 60 pictures a second and shrinks each picture to Kestrel's size, and lets go of the camera cleanly on quit. Also `~/phone-cam` (a camera tool for build photos, not part of Kestrel).
- Ryan writes each line of `ryan_code.py` himself in the chat; Claude pastes his lines into the file, fixes small typos and spacing, and explains the Python ideas before each line. Claude also tidies spacing only.
- Ryan's code, at the start: the three parts in `ryan_code.py` (part 1, is the car inside the circle; part 2, guess the path and decide; part 3, how close a missing car came) held only Claude's plain-words guide text and no code. Ryan types all three.
- Ideas: the pickup line story, the warning mode (light and honk where a gate cannot go) and the near-miss test are Ryan's. The robot voice, the slow-motion replay and the "MISSED BY 4 CM" screen were Claude's suggestions, and Ryan picked them. The startup self-test and "learn my car" were Claude's answers to Ryan asking for a build at least 5 times faster than the turret's and "the best model for detection".
- By Claude, for the research (not part of Kestrel's code): the research write-up, the crash maps (built with Claude from LAPD's public crash reports; credited "Map by Ryan Yousefi" because the idea and the direction are Ryan's, and disclosed here because the contest asks that "all AI usage is fully disclosed in the submission materials"), the scoring of about 240 past entry videos, the stand-in judges, the email drafts and the to-do list itself. If the map appears in the video, say in the tools line that Claude helped build it.
- By Claude at Ryan's ask: both crash maps redrawn on a satellite photo (USGS The National Map, US government imagery), with the worst corners, the deaths and the neighborhoods named on the map.
- By Claude at Ryan's idea ("what if we make the graphs a website like our world in data"): the District 32 Crash Atlas website and the two Our World in Data style graphs. The look, the direction and every pick are Ryan's; the code is Claude's, disclosed here.
- The cover picture and the Kestrel falcon logo: made by ChatGPT's picture maker from Ryan's photos of Kestrel, with a prompt Claude wrote at Ryan's ask. It is an AI-made picture, never a photo, and it redraws some details (neater wires, LEGO plates for the crosswalk stripes, clear stands under the motors). The words were Claude's pick and Ryan's yes, the falcon was Claude's idea, and Ryan picked the picture and had his name made bigger. Claude trimmed it to 600x800 and traced the logo into clean files.
- Ryan's own code, typed by Ryan line by line in the chat (Claude explained each Python idea first, then pasted his lines, fixing only the typo "distnances" and spacing): part 3 `how_close`, all 7 lines (a list of every distance, the straight-line distance the same way as part 1, the smallest one minus the radius; his hand check gave 10 and his code gave 10.0), and the part 2 change that makes the gate wait for the last safe moment (`margin` and the new `drop` line using `gate_seconds`). He picked the margin himself: 0.2 ("in case the car speeds up"), then 0.4 after Claude timed the arm's spare time on the fake road, then 0.6, 0.8 and 1.0 after watching real rolls ("this still feels like its too late"). Fake road before and after: 10 of 10 kids saved, 10 of 10 near misses left alone, both times.
- By Claude, support code in `kestrel_brain.py`: the camera reader no longer quits after one missed picture (it waits up to 2 seconds), and the MISSED BY sign no longer carries into the next roll (wiped when a new car shows up, and drawn only while its own roll ends). Kestrel's self-test: 41 of 41 after both.
- By Claude, for the test (not Kestrel's code): matching every roll in Ryan's two phone takes to Kestrel's own log, the reaction times, and the tally.
- By Claude, for the video (not Kestrel's code), at Ryan's ask: the narration lines, built from Ryan's own words where they exist (his near miss, his code explanation, his margin reasons); Ryan says them in his own voice. The intro line is read by an ElevenLabs AI voice.
- By Claude, for the video (not Kestrel's code): the green-wall test (Ryan cut out of his own clip by the free open-source Robust Video Matting AI, his color matched to the room) and the background pictures, made by Higgsfield's GPT Image 2.5 from Claude's written description (8 pictures; Ryan picked #1, a distribution center). It is an AI-made place, never Kestrel's real building.
- The video's first edit, by Claude at Ryan's ask (not Kestrel's code). Every AI piece is also tagged on screen while it shows:
  - AI studio shots of Ryan's real Kestrel ("AI studio shot of my real Kestrel"): Higgsfield's GPT Image 2.5 made studio pictures from 4 of Ryan's photos (a white pedestal, a board-level angle on the LEGO kids, a close-up of a servo and gate arm, a top-down view), and Kling 3.0 (on Higgsfield) turned them into moving shots, including the 360 spin. The AI redraws some details (neater wires); it is never Kestrel working.
  - AI reenactments ("AI reenactment"), photoreal, in the style of documentary reenactments: from the bus driver's seat a kid steps off, a car in the side mirror, a car rushing past the bus, the driver's hand on the horn; and a crossing signal, a roadside speed camera, a student at a busy curb. Pictures by GPT Image 2.5, moving shots and their sound by Kling 3.0, both on Higgsfield. Nobody is hit. They act out Ryan's story; they are not his real day.
  - The three concept shots ("CONCEPT (AI)"): the warning box by a school bus door, on a streetlight pole, and on a post with a pickup-line gate arm. GPT Image 2.5 and Kling 3.0 on Higgsfield. Ideas for later, never real installs.
  - The intro voice ("AI voice"): ElevenLabs, voice "Jessica" on the Eleven v4 model (Ryan's pick), reading "Introducing Kestrel, an autonomous crossing guard."
  - The music bed and the sound effects (whooshes, ticks, a low hit, street sound): made by ElevenLabs (AI). Kestrel's honk is Kestrel's own sound file.
  - The edit itself: built in code by Claude (Remotion), with Ryan's voice cleaned, trimmed between sentences and played at 1.07x speed to fit 3 minutes (pitch kept); captions from a speech-to-text model (Whisper), checked by hand.
- By Claude, for the proof round before the third video (not Kestrel's code): every caption and on-screen word pulled by script and traced to its proof, with helper sessions that traced and then attacked every verdict; the ready-made AI test on Ryan's own 63 photos (YOLO26n, YOLO26s and YOLOE-26s, the same settings as the first test); the real gate timing and the camera's delay read off Ryan's phone video, with 11 blind helper readings; the stand-in judges' no-facts-sheet option; one of Ryan's voice lines cut the same way as his other lines; and the fixes in the test labels, the entry notes, the narration and the judges' facts sheet.
- By Claude at Ryan's ask (not Kestrel's code): four short narration lines rebuilt from his own earlier lines so they say only what was measured: the LA part of the district, "neither one" warns a kid, his model picks out the car, and the street box as a plan. Ryan said them in his own voice and his own way.
- By Claude at Ryan's ask (not Kestrel's code): the narration line about why his speed code looks five spots back, written from Ryan's own three answers in the chat ("its an average", "its like error", "11 pixel because its higher % even if the unit is the same"), after Claude asked him questions about his own line 74. Ryan says it in his own voice.
- By Claude, the video's third edit (not Kestrel's code), at Ryan's ask ("make 3 versions, run those all on judges and then bring me the highest scoring"): three versions built in code (Remotion), each one under 3 minutes, with every new word traced to its proof first. Claude picked what to cut for time (Ryan approved the set-up): his "I started at 0.2 seconds" sentence, his "the same box could go on poles" sentence with its two concept shots and the map, his "Speed is calculated by how far the object moved divided by the time" sentence, and in some versions the AI intro, his laptop-screen sentence or his "warning matters more than the gate" sentence. Also by Claude: the timing of Ryan's kid's-eye roll against Kestrel's own log; the "light turns red" marker drawn over that shot because the light faces the car (Ryan's idea: "a red light shake or something substituted"); the "LA part of" words added to his crash chart in the chart's own font; the dashed "drawing: a car that turns" in the what's-next shot (labeled a drawing on screen); the timing card; and the wording of the who-built-what card. The stand-in judges are Claude helper sessions. No new AI pictures, videos or voices were made: the AI pieces are the same Higgsfield reenactments and bus concept (and, only in the version with the intro, the AI video made from his photos and the ElevenLabs intro voice), the ElevenLabs music and sound effects, and Whisper's captions.
- The version picked as the third video: it has NO AI intro voice and no AI video made from his photos. Its AI pieces are the Higgsfield reenactments (his near miss, and the street scenes), one Higgsfield concept (the box on a school bus), the ElevenLabs music and sound effects, and Whisper's captions; the edit is by Claude. Kestrel's honk in the film is Kestrel's own sound file.
- By Claude at Ryan's ask, the video's fourth version (not Kestrel's code): the wording of his results line "9 of the 10 misses, it stayed quiet. The 10th, it lost the car.", suggested by Claude from the test's own numbers and said by Ryan in his own voice; the six-line script he re-recorded so his voice sounds the same all the way through (his own earlier lines, word for word, printed back for him to read); cutting the new takes, cleaning them so the room sound matches his first recording, and placing the line in the edit. The stand-in judges are Claude helper sessions. No new AI pictures, videos or voices: the fourth version has the same AI pieces as the third.

This README was also written by Claude, from my notes.

## The test

I rolled the real toy car 20 times on the real road, using my code. Before counting, I set the rule: the first 10 rolls aimed at the kids and the first 10 aimed to miss, and every failure stays in the count.

- Kids saved: 10 of 10. Every time, the gate came down and stopped the car.
- Misses left alone: 9 of 10. Kestrel kept the gate up and showed how close the car came.
- The 10th miss does not count as a win. Kestrel saw the car once, lost it while it was moving, and only found it again after it stopped. The gate stayed up, but only because Kestrel lost the car, not because it guessed right.

## How to run it

You need a Mac with Python 3.12 (Kestrel uses the Mac's own tools to play the honk and keep the screen awake). To try it with no camera and no board, use the pretend road.

1. Download this folder and open a terminal inside it.
2. Make a clean space for Kestrel's add-ons: `python3 -m venv .venv`
3. Put the add-ons in: `.venv/bin/pip install -r requirements.txt` (this takes a few minutes, because the AI part is big).
4. Try the pretend road: `.venv/bin/python kestrel_brain.py --practice`
   On the pretend road, press H to roll a car at the kids, M to roll one that just misses, F for a fast one, and Q to quit.
5. Score my code on 20 pretend rolls: `.venv/bin/python sims.py`
6. Check that every part works: `.venv/bin/python sims.py --selftest`

With the real setup (a webcam looking down at the road and the ESP32 board plugged in), run `.venv/bin/python kestrel_brain.py`. If it picks the wrong camera, run `.venv/bin/python kestrel_brain.py --find-camera` to save one picture from each camera, then add `--camera 1` (or whichever number is right). Press C to place the protected circle, and SPACE drops or lifts the gate by hand.
