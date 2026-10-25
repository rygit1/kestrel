# How I used AI in Kestrel

by Ryan Yousefi

Claude is Anthropic's AI. This is every way AI was used in my Kestrel entry, by Claude and by other AI tools.

## Kestrel's code

**My code, `ryan_code.py`.** I typed its working lines myself: part 1 (is the car inside the circle), part 2 (guess the car's path and decide to drop the gate or wait) and part 3 (how close a missed car came). Claude wrote the plain-words guide text inside that file (what goes in, what comes out, the steps in words), and at first the three parts held only that guide text and no code. I typed each line in the chat. Claude explained each Python idea before the line, then pasted my lines into the file, fixing only small typos (like "distnances") and spacing.

- Part 3, `how_close`, is all 7 of my lines: a list of every distance, the straight-line distance worked out the same way as part 1, and the smallest one minus the circle's size. My hand check gave 10 and my code gave 10.0.
- My change to part 2 makes the gate wait for the last safe moment (the `margin` and the new `drop` line that uses `gate_seconds`). I picked the margin myself: 0.2 seconds ("in case the car speeds up"), then 0.4 after Claude timed the arm's spare time on the practice road, then 0.6, 0.8 and 1.0 after watching real rolls ("this still feels like its too late"). On the practice road it saved 10 of 10 kids and left 10 of 10 near misses alone, both before and after.

**Everything else in the code was written by Claude:**

- `kestrel_brain.py`: the eyes (finding the red car by its color, then teaming up with the car-finding AI), the memory of where the car has been, the steps from my plan with the event log, the command-center screen, the voice (a robot voice and the honk sound), the slow-motion replay, the startup self-test that checks the wiring, the click tools (placing the circle, picking the car's color, setting the scale, boxes to ignore), and the camera reader that always uses the newest picture. Later fixes by Claude: the robot voice turned off at my ask; Kestrel waits for the circle to be placed instead of crashing; the camera reader waits up to 2 seconds instead of quitting after one missed picture; the "MISSED BY" sign is wiped when a new car shows up so it does not carry into the next roll; a learner for the whole car's color; and the auto setup that finds the two motors and places the protected zone by itself (my idea and my rule: any car headed through the gate drops the arms). The self-test passed 41 of 41 after these fixes.
- `learn_car.py` ("learn my car"): it collects pictures of my car that the color eyes mark by themselves, trains a small YOLO26 AI model on the laptop, tests it on pictures it never saw, and only uses it if it passes. Claude also added a check so it only saves deep-red, car-sized pictures, because it had been saving my hand as the car.
- `fake_road.py` and `sims.py`: the practice road on screen, the tests ("how fast is too fast", and scoring my code on 20 fake rolls) and the self-test.
- The README for this code page.

**Whose ideas.** The pickup line story, the warning mode (a light and a honk where a gate cannot go), the near-miss test, the double gate and the auto setup rule are mine. The robot voice, the slow-motion replay and the "MISSED BY 4 CM" screen were Claude's suggestions, and I picked them. The startup self-test and "learn my car" were Claude's answers when I asked for a build at least 5 times faster than my last one and "the best model for detection".

## The build and wiring

- My sentry turret from summer 2026 had camera tracking and motor code written by Claude, and Kestrel reuses it.
- `kestrel_board.ino`, the board's code: Claude added the gate and light lines to the turret's board code (no turret line changed). For the double gate (my idea: two arms, one from each side), Claude ran the second arm on the turret's old trigger line and made the board start gate 2 at its up spot.
- `gate_panel.py`, by Claude: two sliders that turn each gate motor by hand, for testing, plus keys for the light. I found all four gate angles myself with it. Claude then made `kestrel_brain.py` send the second arm its own up and down angles so both arms move together.
- Also by Claude in `kestrel_brain.py`: a practice mode that runs the practice road on screen while the real arms and light move with it, for rehearsing before the webcam; and camera code that asks my Insta360 camera for 1080p at 60 pictures a second, shrinks each picture to Kestrel's size, and lets go of the camera cleanly on quit.
- Claude walked me through the wires one at a time.
- Claude also wrote a small camera tool I used to take build photos. It is not part of Kestrel.

## Checking the test results

All of this is by Claude and is not Kestrel's code.

- Matching every roll in my two phone videos of the test to Kestrel's own log, working out the reaction times, and counting up the results.
- Timing my kid's-eye test roll against Kestrel's log.
- Reading the real gate timing and the camera's delay off my phone video, checked by 11 separate Claude helper readings.
- Testing ready-made AI car finders (YOLO26n, YOLO26s and YOLOE-26s) on 63 of my own photos, to see if a ready-made one could find my car.
- Checking every caption and on-screen word in the video against its proof, with extra Claude helper sessions that checked each answer and then tried to prove it wrong. Where something did not hold up, Claude fixed the notes, the narration and the judges' facts sheet.

## Research and planning

All by Claude, not part of Kestrel's code:

- A research write-up on what wins this contest, including scoring about 240 past entry videos.
- The stand-in judges (see the video section).
- Drafts of emails and my to-do list.

## The crash map and website

- The crash maps were built with Claude from LAPD's public crash reports. They say "Map by Ryan Yousefi" because the idea and the direction are mine, and Claude's help is disclosed here. At my ask, Claude redrew both maps on a satellite photo (USGS The National Map, US government pictures), named the worst corners, the deaths and the neighborhoods on the map, and fixed a wrong crash count on one map (81 crashes, not 65).
- The District 32 Crash Atlas website and its two graphs in the style of Our World in Data were my idea ("what if we make the graphs a website like our world in data"). The look, the direction and every pick are mine; the code is Claude's.

## The video

Every AI piece is tagged on screen while it shows. The video is not Kestrel's code.

**Script and voice.**

- Claude wrote the narration lines from my own words where they existed (my near miss, my code explanation, my reasons for the margin). I say them in my own voice.
- Claude rebuilt four short lines from my earlier lines so they say only what was measured: the LA part of the district, that "neither one" warns a kid, that my model picks out the car, and the street box as a plan. I said them in my own voice and my own way.
- The line about why my speed code looks five spots back: Claude asked me questions about that line of my code and wrote the narration from my three answers ("its an average", "its like error", "11 pixel because its higher % even if the unit is the same"). I say it in my own voice.
- My results line, "9 of the 10 misses, it stayed quiet. The 10th, it lost the car.", was worded by Claude from the test's own numbers. I said it in my own voice.
- My line "The red light is for the driver, the honk is for the kid." was Claude's suggestion, and I said it my own way, in my own voice. It replaced my line "A kid won't look at a screen."
- I re-recorded my own line "Neither one warns a kid in the moment." to give it more life.
- When I re-recorded six lines so my voice sounds the same all the way through, Claude printed my own earlier lines back to me, word for word, to read.
- The AI intro voice: ElevenLabs, voice "Jessica" on the Eleven v4 model (my pick), reading "Introducing Kestrel, an autonomous crossing guard." It was only in early drafts and is not in the later versions.

**Pictures and clips.** These were made on Higgsfield: pictures by GPT Image 2.5 from Claude's written descriptions, turned into moving shots by Kling 3.0.

- AI reenactments, labeled "AI reenactment": photoreal, in the style of documentary reenactments. From the bus driver's seat, a kid steps off, a car shows in the side mirror, a car rushes past the bus, the driver's hand hits the horn; also a crossing signal, a roadside speed camera, and a student at a busy curb. Kling 3.0 also made their sound. Nobody is hit. They act out my story; they are not my real day.
- Concept shots, labeled "CONCEPT (AI)": the warning box by a school bus door, on a streetlight pole, and on a post with a pickup-line gate arm. They are ideas for later, never real installs. Only the school bus one stayed in the later versions. At my ask it was remade twice: first a picture of the whole bus in true size (the old one was out of proportion) turned into a slow-turn shot, then the same picture redrawn with a bigger box and turned into a shot that pushes in to the box. It is labeled from its first frame. The red ring pointing at the box is drawn by the edit, not the AI.
- AI studio shots of my real Kestrel, labeled "AI studio shot of my real Kestrel": GPT Image 2.5 made studio pictures from 4 of my photos (a white stand, a low angle on the LEGO kids, a close-up of a motor and gate arm, a top-down view), and Kling 3.0 turned them into moving shots, including a 360 spin. The AI redraws some details (like neater wires), and it never shows Kestrel working. These were only in early drafts and are not in the later versions.
- A green-screen test: the free, open-source Robust Video Matting AI cut me out of my own clip, and my color was matched to the room. The backgrounds were 8 pictures made by GPT Image 2.5 from Claude's description; I picked one, a distribution center. It is an AI-made place, never Kestrel's real building.

**Music and sound.**

- The music and the sound effects (whooshes, ticks, a low hit, street sound) were made by ElevenLabs, an AI. Kestrel's honk in the video is Kestrel's own sound file.
- Claude cleaned my voice recordings, trimmed the gaps between sentences, and played them at 1.07 times speed to fit 3 minutes (same pitch). When I re-recorded lines in a different room, Claude cleaned the new takes so the room sound matches my first recording.
- Every one of my voice recordings (the voice memos and my on-camera sound) went through DeepFilterNet, a free, open-source AI that removes background noise from speech. It does not change or make words, and Claude had every line read back by speech-to-text to prove the words stayed the same. Then each spoken piece was set to the same loudness.

**Captions.** The captions came from Whisper, an AI that turns speech into text, and were checked by hand.

**Editing.** Claude built the whole edit in code (with Remotion, a tool that makes videos from code), and cut, cleaned and placed every new take of mine.

- At my ask ("make 3 versions, run those all on judges and then bring me the highest scoring"), Claude built three versions, each under 3 minutes. Claude picked which of my sentences to cut for time, and I approved the plan.
- Also by Claude: a "light turns red" marker drawn over my kid's-eye shot, because the real light faces the car (my idea: "a red light shake or something substituted"); the words "LA part of" added to my crash chart in the chart's own font; a dashed drawing of a car that turns, labeled as a drawing on screen; the timing card built from the measurements; and the wording of the card that says who built what.
- Later fixes by Claude: holding the timing card 2 seconds longer, moving the speed camera reenactment so the whole box shows, re-cutting one of my real test rolls so its light turns red on the word "red", and moving the "CONCEPT (AI)" label so it does not cover the box.

**The stand-in judges.** To score each version of the video before I picked one, Claude ran stand-in judges, which are Claude helper sessions, some with a facts sheet about Kestrel and some without one.

**ChatGPT's review.** I also asked ChatGPT to review an early draft of the video. Claude checked its notes against the real files before I decided what to use.

## The street test (Oct 24 2026)

- What I did: passed the FAA's TRUST drone safety test, picked the crosswalk from its crash history, flew my DJI Neo 2 and filmed 10 minutes of real traffic, and checked one 30-second stretch of the car count by hand (21 of 21 cars).
- Written by Claude: all of the street test code in `street/`: lining up every frame of the drifting drone video with the road, turning the picture into a flat map in meters (from the crosswalk stripes and the drone's own height reading), finding and following real cars with a ready-made car finder (Ultralytics YOLO26, trained by Ultralytics on everyday photos, not retrained by me), the pretend kid, and the scoring. Claude also picked the settings, on the first 4 minutes only, and locked them before the last 6 minutes were graded.
- Claude helpers counted the cars by eye from picture sheets to make the answer key (2 helpers per stretch, working alone).
- My decision code (`ryan_code.py`) ran unchanged. It makes every warning decision in the street test.
- The video (v7): Claude drew the street shot (the pretend kid figure, each car's path, the blurred plates, the "x.x s EARLY" tags and the 24 of 24 card) and edited it in, in place of my timers and speed cameras line. The new line about the street test is my own voice. Claude also made the short street test GIF on my code page from that shot.

## The cover picture and logo

- The cover picture and the Kestrel falcon logo were made by ChatGPT's picture maker from my photos of Kestrel, with a prompt Claude wrote at my ask.
- It is an AI-made picture, never a photo, and it redraws some details: neater wires, LEGO plates for the crosswalk stripes, and clear stands under the motors.
- The words on it were Claude's pick and my yes. The falcon was Claude's idea. I picked the picture and had my name made bigger.
- Claude trimmed it to 600 by 800 and traced the logo into clean files.

## The written answers and this page

- The six written answers in my entry: Claude asked me interview questions, edited my spoken answers into writing, added a few lines that I checked and approved, and supplied the street test numbers from its test code. The ideas are mine.
- This code page's README and this list were written by Claude from my notes.

## Outside tools Kestrel uses

- OpenCV: works with the camera pictures.
- Ultralytics YOLO26: an AI that finds objects in pictures, trained on my own car.
- PyTorch and ONNX Runtime: run the AI.
- pyserial: lets the laptop talk to the board.
- The Mac's built-in voice.

## What I did myself

My own work is listed in the README.
