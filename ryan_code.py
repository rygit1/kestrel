"""
Ryan's code: the brain of Kestrel's pickup line guard.

Everything else in Kestrel (the eyes, the memory, the steps, the screen, the voice, the fake
road) is support code written with AI help and listed on the App Challenge AI-use list.
The three parts in THIS file are Ryan's to type himself.

Until a part is typed, it hands back nothing (None), and Kestrel plays it safe: it never
drops the gate on a part that is not written yet, and its screen says so.

How spots on the screen are counted: x is pixels from the left edge, y is pixels down from
the top edge. So a bigger y is lower on the screen.

Try each part on the fake road before the real one: python sims.py
"""


def is_inside_circle(car, circle):
    """
    PART 1 (Sun Oct 11): is the car inside the protected circle right now?
    Kestrel uses it in version 2 (react when the car is in the circle), and to turn the
    circle red on the screen whenever the car is inside it.

    What comes in:
        car     where the car is right now, as (x, y) in pixels.
        circle  the protected circle around the kid, as (x, y, radius) in pixels:
                the x and y of its center, and how far its edge is from its center.

    What must come out:
        True if the car is inside the circle, False if it is not.

    The steps, in words:
        1. How far is the car from the circle's center, left to right? And up and down?
        2. Turn those two into one straight-line distance.
        3. Is that distance smaller than the radius?
    """
    dx = car[0] - circle[0]
    dy = car[1] - circle[1]
    distance = (dx**2 + dy**2) **0.5
    return distance < circle[2]


def guess_and_decide(spots, circle, gate_seconds):
    """
    PART 2 (Oct 16 to 17): guess where the car is going, and decide whether to drop the gate.
    Kestrel uses it in version 3 (guess early), many times a second while a car is moving.

    What comes in:
        spots         every spot the car has been at so far, oldest first. Each spot is
                      (t, x, y): t is the time in seconds, x and y are pixels like in part 1.
                      The last spot is where the car is now.
        circle        the protected circle, (x, y, radius) in pixels, like in part 1.
                      Its edge reaches the gate arm, so reaching the circle means reaching
                      the gate.
        gate_seconds  how many seconds the gate arm takes to come all the way down.

    What must come out:
        True to drop the gate now, False to wait.

        Extra, once True and False work: hand back a dictionary instead, with "drop_gate"
        (True or False) plus any of these, and Kestrel shows them on the screen:
            "speed"         how fast the car is going, in pixels per second
            "path_end"      (x, y) where your guessed path ends, so Kestrel draws the line
            "seconds_left"  how long until the car reaches the circle, for the countdown

    The steps, in words:
        1. From the last few spots, work out the car's speed and which way it is going.
        2. Work out where its path goes.
        3. Does that path cross the circle?
        4. If it does, how long until the car gets there?
        5. Decide: drop the gate now, or wait? (Think about gate_seconds.)
    """
    now = spots[-1]
    before = spots[-5]
    vx = (now[1] - before[1]) / (now[0] - before[0])
    vy = (now[2] -before[2])  / (now[0] - before [0])
    seconds_left = None
    for step in range(1, 101):
        ahead = step * 0.02
        future = (now[1] + vx * ahead, now[2] + vy * ahead)
        if is_inside_circle(future, circle):
            seconds_left = ahead
            break
    margin = 1.0
    drop = seconds_left is not None and seconds_left <= gate_seconds + margin
    return {"drop_gate": drop, "speed": (vx**2 + vy**2) ** 0.5, "path_end": future, "seconds_left": seconds_left}


def how_close(spots, circle):
    """
    PART 3, if there is time (Oct 17): how close did a car that missed come to the kid?
    Kestrel asks this after a car goes by without the gate dropping, then shows
    "MISSED BY 4 CM" on the screen and says it out loud.

    What comes in:
        spots   every spot the car was at during that roll, (t, x, y), like in part 2.
        circle  the protected circle, (x, y, radius) in pixels.

    What must come out:
        The smallest gap there ever was between the car and the circle's edge, in pixels.
        Kestrel turns pixels into centimeters for the screen.

    The steps, in words:
        1. For each spot, how far was the car from the circle's center? (The same way as
           part 1.)
        2. Keep the smallest one.
        3. Take away the radius, so it is the gap to the circle's edge, not to its center.
    """
    distances = []
    for spot in spots:
        dx = spot[1] - circle[0]
        dy = spot[2] - circle[1]
        distance = (dx**2 + dy**2) ** 0.5
        distances.append(distance)
    return min(distances) - circle[2]
