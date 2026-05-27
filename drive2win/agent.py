"""Recovery agent — wraps the MLP policy with stuck detection.

When the car's speed stays below a threshold for too long, it executes
a reverse-and-turn recovery maneuver before handing back to the MLP.

Usage:
    python 03_benchmark.py --tag v2_agent --weights nav_v2.npz --module drive2win.agent
"""
from __future__ import annotations
import numpy as np
from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action

STUCK_SPEED      = 0.2   # m/s — below this counts as stuck (not just slow cornering)
STUCK_FRAMES     = 30    # frames at 20 Hz = 1.5 s before recovery kicks in
RECOVER_FRAMES   = 14    # frames to reverse (0.7 s)


def make_policy(weights_path: str):
    w = nn_mod.load(weights_path)

    stuck_count  = [0]
    recover_left = [0]
    recover_steer = [0.0]
    rng = np.random.default_rng()

    def policy(state):
        sensors = state.get("sensors", {})
        speed   = sensors.get("speed", 0.0)

        # still in recovery manoeuvre
        if recover_left[0] > 0:
            recover_left[0] -= 1
            return clip_action(np.array([-1.0, recover_steer[0]]))

        # check if stuck
        if speed < STUCK_SPEED:
            stuck_count[0] += 1
        else:
            stuck_count[0] = 0

        if stuck_count[0] >= STUCK_FRAMES:
            stuck_count[0]  = 0
            recover_left[0] = RECOVER_FRAMES
            # alternate steering direction each recovery to avoid looping
            recover_steer[0] = float(rng.choice([-1.0, 1.0]))
            return clip_action(np.array([-1.0, recover_steer[0]]))

        # normal MLP policy
        x = sensors_to_input(sensors)
        return clip_action(nn_mod.forward(x, w))

    return policy
