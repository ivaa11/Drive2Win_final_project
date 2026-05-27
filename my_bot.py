"""Tournament bot — MLP policy (nav_v16.npz) with stuck-detection recovery.

Solo test:
    1. Open https://ml.ferit.tech/?room=YOUR-NAME in a browser tab
    2. py my_bot.py --room YOUR-NAME --name YOUR-NAME

Presentation day (instructor announces room name):
    py my_bot.py --room final2026 --name YOUR-NAME
"""
import argparse
import numpy as np
from game_client import RoomBot
from drive2win import nn as nn_mod
from drive2win.normalize import sensors_to_input, clip_action

WEIGHTS = "nav_v2.npz"

STUCK_SPEED   = 0.2   # m/s — below this counts as stuck
STUCK_FRAMES  = 30    # frames at 20 Hz = 1.5 s before recovery kicks in
RECOVER_FRAMES = 14   # frames to reverse (0.7 s)


def make_controller(weights_path: str):
    w = nn_mod.load(weights_path)
    recover_left  = [0]
    recover_steer = [0.0]
    rng = np.random.default_rng()

    def controller(obs):
        speed = obs["speed"]
        rays  = obs["rays"]
        nav   = obs["navigation"]
        heading_error = nav["heading_error"]

        # Recovery: reverse out when surrounded by walls/terrain edges
        if recover_left[0] > 0:
            recover_left[0] -= 1
            return -0.8, recover_steer[0]

        if max(rays) < 3.0:
            recover_left[0] = RECOVER_FRAMES
            recover_steer[0] = float(rng.choice([-1.0, 1.0]))
            return -0.8, recover_steer[0]

        # Pass clean rays to MLP — the RoomBot obstacle grid doesn't match
        # the game's physics raycast, so using computed rays breaks steering.
        # heading_error (computed from world_map checkpoints) IS correct.
        sensors = {
            "speed": speed,
            "heading_error": heading_error,
            "checkpoint_distance": nav["distance"],
            "rays": [50.0] * 8,
            "ground_friction": obs["ground_friction"],
        }
        x = sensors_to_input(sensors)
        _, steering = clip_action(nn_mod.forward(x, w))

        # Throttle: slow down when facing away from checkpoint or obstacle ahead
        front = rays[0]
        if front < 8.0:
            throttle = 0.3
        elif abs(heading_error) > 1.2:
            throttle = 0.5
        else:
            throttle = 0.75

        return throttle, steering

    return controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="YOUR-NAME")
    ap.add_argument("--name", default="YOUR-NAME")
    ap.add_argument("--weights", default=WEIGHTS)
    args = ap.parse_args()

    print(f"Loading weights: {args.weights}")
    controller = make_controller(args.weights)

    print(f"Joining room '{args.room}' as '{args.name}' ...")
    bot = RoomBot("https://ml.ferit.tech", room=args.room, name=args.name)
    standings = bot.run(controller, hz=20.0)
    print("\nFinal standings:")
    for r in standings:
        print(f"  #{r.get('rank')} {r.get('name')}  checkpoints={r.get('total_checkpoints')}")


if __name__ == "__main__":
    main()
