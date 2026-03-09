"""
Step 1: Basic Vision Pursuit
Make v60 SNN track a moving object with eye movements.

The object sweeps laterally (along y-axis) at a fixed forward distance,
crossing between left and right eye hemifields. The retinal sum asymmetry
(right eye - left eye) drives eye position with exponential smoothing.

Run: python -m zebra_v60.tests.step1_vision_pursuit
Output: plots/v60_step1_vision_pursuit.png
"""
import os
import sys
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebra_v60.brain.zebrafish_snn_v60 import ZebrafishSNN_v60
from zebra_v60.brain.device_util import get_device
from zebra_v60.world.world_env import WorldEnv


def make_object_trajectory(T=200, amplitude=60.0, period=100, forward_dist=80.0):
    """Object sweeps laterally at a fixed forward distance from the fish.
    Fish faces +x, object moves along y-axis.
    """
    t = np.arange(T)
    x = np.full(T, forward_dist)
    y = amplitude * np.sin(2 * np.pi * t / period)
    return x, y


def compute_retinal_turn(out):
    """Compute turn signal from bilateral retinal asymmetry.
    Right eye sum > left eye sum → object is to the right → positive turn.
    """
    retR_sum = float(out["retR"].sum())
    retL_sum = float(out["retL"].sum())
    total = retR_sum + retL_sum + 1e-8
    return (retR_sum - retL_sum) / total


class TurnSmoother:
    """Exponential moving average for turn signal to reduce jitter."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed = 0.0

    def step(self, raw_turn):
        self.smoothed = (1 - self.alpha) * self.smoothed + self.alpha * raw_turn
        return self.smoothed


def run_step1(T=200, k_eye=1.5, amplitude=60.0):
    print("=" * 60)
    print("Step 1: Basic Vision Pursuit")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN_v60(device=device)
    model.reset()

    world = WorldEnv(xmin=-500, xmax=500, ymin=-500, ymax=500, n_food=0)
    obj_x, obj_y = make_object_trajectory(T, amplitude=amplitude)

    fish_pos = np.array([0.0, 0.0])
    heading = 0.0
    eye_offset = 0.0
    smoother = TurnSmoother(alpha=0.3)

    F_hist = []
    eye_hist = []
    obj_hist = []
    turn_hist = []

    for t in range(T):
        world.foods = [(obj_x[t], obj_y[t])]
        effective_heading = heading + eye_offset

        with torch.no_grad():
            out = model.forward(fish_pos, effective_heading, world)

        raw_turn = compute_retinal_turn(out)
        turn = smoother.step(raw_turn)

        eye_offset += k_eye * turn
        eye_offset *= 0.95

        F = model.compute_free_energy()

        F_hist.append(F)
        eye_hist.append(eye_offset)
        obj_hist.append(obj_y[t] / amplitude)
        turn_hist.append(turn)

        if t % 50 == 0:
            print(f"  t={t:4d}  F={F:.4f}  eye={eye_offset:+.3f}  "
                  f"obj_y={obj_y[t]:+.1f}  turn={turn:+.4f}  "
                  f"retL={float(out['retL'].sum()):.1f}  "
                  f"retR={float(out['retR'].sum()):.1f}")

    corr = np.corrcoef(eye_hist, obj_hist)[0, 1]
    print(f"\nTracking correlation: {corr:.3f}")
    print(f"System stable: {not any(math.isnan(f) for f in F_hist)}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(F_hist, color="steelblue")
    axes[0].set_ylabel("Free Energy (F)")
    axes[0].set_title("Step 1: Basic Vision Pursuit")

    axes[1].plot(eye_hist, label="Eye offset (rad)", color="coral")
    axes[1].plot(obj_hist, label="Object lateral pos (norm)", color="gray", alpha=0.7)
    axes[1].set_ylabel("Position")
    axes[1].legend()

    axes[2].plot(turn_hist, color="green", alpha=0.7)
    axes[2].set_ylabel("Turn signal (smoothed)")
    axes[2].set_xlabel("Time step")

    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v60_step1_vision_pursuit.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved: {save_path}")

    return corr


if __name__ == "__main__":
    run_step1()
