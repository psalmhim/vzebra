import torch
import numpy as np

from zebra_v55.brain.zebrafish_snn_5k import ZebrafishSNN_5k
from zebra_v55.brain.biological_wiring import (
    apply_topographic_wiring,
    apply_biological_directionality,
    decode_brain_output
)

class DummyWorld:
    def __init__(self, food):
        self.food = np.array(food, dtype=float)

    def sample_direction(self, origin, dx, dy, max_dist=200):
        fx, fy = self.food - origin
        food_vec = np.array([fx, fy])
        food_vec /= (np.linalg.norm(food_vec) + 1e-6)

        ray = np.array([dx, dy])
        ray /= (np.linalg.norm(ray) + 1e-6)

        cos_sim = float(np.dot(food_vec, ray))

        return float(cos_sim > 0.8)


def simulate(model, food_pos):
    pos = np.array([0.0, 0.0])
    heading = 0.0

    world = DummyWorld(food_pos)
    out = model.step_with_retina(position=pos,
                                 heading=heading,
                                 world=world,
                                 T=1)
    decoded = decode_brain_output(out)
    return decoded["turn_force"]


def main():
    device = "cpu"
    model = ZebrafishSNN_5k(device=device)

    apply_topographic_wiring(model)
    apply_biological_directionality(model)

    tests = [
        ("Left_far",   (-60, -20), -1),
        ("Left_mid",   (-40, -5),  -1),
        ("Left_near",  (-20, 0),   -1),
        ("Front",      (40, 0),     0),
        ("Right_far",  (60, 20),   +1),
        ("Right_mid",  (40, 5),    +1),
        ("Right_near", (20, 0),    +1),
    ]

    passed = 0
    for name, food, target_dir in tests:
        out_dir = simulate(model, food)
        print(f"{name:12} food={food}   turn={out_dir:+.3f}")

        if target_dir == -1 and out_dir < -0.05: passed += 1
        if target_dir == +1 and out_dir > +0.05: passed += 1
        if target_dir == 0  and abs(out_dir) < 0.05: passed += 1

    print()
    print("=== Evaluation ===")
    print(f"Passed {passed}/7")
    if passed == 7:
        print("RESULT: PERFECT")
    else:
        print("RESULT: PARTIAL")


if __name__ == "__main__":
    main()
