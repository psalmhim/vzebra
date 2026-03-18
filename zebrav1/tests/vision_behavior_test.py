import os
import sys

import numpy as np
import torch

# Ensure the project root is on sys.path so `zebrav1` can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.brain.zebrafish_snn import ZebrafishSNN
from zebrav1.brain.device_util import get_device
from zebrav1.world.world_env import WorldEnv

def run_test():
    device = get_device()
    world = WorldEnv(xmin=-200, xmax=200, ymin=-150, ymax=150, n_food=0)
    model = ZebrafishSNN(device=device)

    headings = [
        ("Left_far", (-60, -20)),
        ("Left_mid", (-40, -5)),
        ("Left_near", (-20, 0)),
        ("Front", (40, 0)),
        ("Right_far", (60, 20)),
        ("Right_mid", (40, 5)),
        ("Right_near", (20, 0))
    ]

    pos = np.array([0., 0.])
    heading = 0.0

    passed = 0

    for name, food in headings:
        world.foods = [tuple(food)]
        model.reset()
        out = model.forward(pos, heading, world)
        turn = float(out["motor"][0,:100].sigmoid().mean() - out["motor"][0,100:].sigmoid().mean())

        print(f"{name:12s} food={food}  turn={turn:+.3f}")

        if ("Left" in name and turn < 0) or ("Right" in name and turn > 0) or ("Front" in name and abs(turn) < 0.05):
            passed += 1

    print("=== Evaluation ===")
    print(f"Passed {passed}/7")

if __name__ == "__main__":
    run_test()
