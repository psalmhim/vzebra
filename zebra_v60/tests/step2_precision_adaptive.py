"""
Step 2: Precision-Adaptive Vision
Precision tracks prediction uncertainty — increases during stable
tracking, drops on novelty (direction reversal).

Run: python -m zebra_v60.tests.step2_precision_adaptive
Output: plots/v60_step2_precision_adaptive.png
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
from zebra_v60.tests.step1_vision_pursuit import (
    make_object_trajectory, compute_retinal_turn, TurnSmoother,
)


def run_step2(T=250, k_eye=1.5, amplitude=60.0):
    print("=" * 60)
    print("Step 2: Precision-Adaptive Vision")
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

    # History
    F_hist = []
    eye_hist = []
    obj_hist = []
    pi_OT_hist = []
    pi_PC_hist = []

    prev_oF = None

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

        # Precision update: temporal prediction error at OT
        oF = out["oF"]
        if prev_oF is not None:
            error_OT = oF - prev_oF
            model.prec_OT.update_precision(error_OT)
            error_PC = torch.tensor([[F]])
            model.prec_PC.update_precision(error_PC)
        prev_oF = oF.clone()

        F_hist.append(F)
        eye_hist.append(eye_offset)
        obj_hist.append(obj_y[t] / amplitude)
        pi_OT_hist.append(out["pi_OT"])
        pi_PC_hist.append(out["pi_PC"])

        if t % 50 == 0:
            print(f"  t={t:4d}  F={F:.4f}  eye={eye_offset:+.3f}  "
                  f"pi_OT={out['pi_OT']:.3f}  pi_PC={out['pi_PC']:.3f}")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(F_hist, color="steelblue")
    axes[0].set_ylabel("Free Energy (F)")
    axes[0].set_title("Step 2: Precision-Adaptive Vision")

    axes[1].plot(eye_hist, label="Eye offset", color="coral")
    axes[1].plot(obj_hist, label="Object (norm)", color="gray", alpha=0.7)
    axes[1].set_ylabel("Position")
    axes[1].legend()

    axes[2].plot(pi_OT_hist, label="Precision OT", color="purple")
    axes[2].set_ylabel("Precision OT")
    axes[2].legend()

    axes[3].plot(pi_PC_hist, label="Precision PC", color="teal")
    axes[3].set_ylabel("Precision PC")
    axes[3].set_xlabel("Time step")
    axes[3].legend()

    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v60_step2_precision_adaptive.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved: {save_path}")


if __name__ == "__main__":
    run_step2()
