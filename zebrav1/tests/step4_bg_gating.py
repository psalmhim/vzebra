"""
Step 4: Basal Ganglia Motor Gating
BG gating for saccade oscillation and dopamine-guided exploration.

Run: python -m zebrav1.tests.step4_bg_gating
Output: plots/v1_step4_bg_gating.png
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

from zebrav1.brain.zebrafish_snn import ZebrafishSNN
from zebrav1.brain.dopamine import DopamineSystem
from zebrav1.brain.basal_ganglia import BasalGanglia
from zebrav1.brain.device_util import get_device
from zebrav1.world.world_env import WorldEnv
from zebrav1.tests.step1_vision_pursuit import (
    make_object_trajectory, compute_retinal_turn, TurnSmoother,
)


def run_step4(T=300, k_eye=1.5, amplitude=60.0):
    print("=" * 60)
    print("Step 4: Basal Ganglia Motor Gating")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN(device=device)
    model.reset()
    dopa_sys = DopamineSystem()
    bg = BasalGanglia(mode="alternating")

    world = WorldEnv(xmin=-500, xmax=500, ymin=-500, ymax=500, n_food=0)
    obj_x, obj_y = make_object_trajectory(T, amplitude=amplitude, period=120)

    fish_pos = np.array([0.0, 0.0])
    heading = 0.0
    eye_offset = 0.0
    smoother = TurnSmoother(alpha=0.3)

    # History
    F_hist = []
    eye_hist = []
    obj_hist = []
    rpe_hist = []
    dopa_hist = []
    bg_gate_hist = []
    valL_hist = []
    valR_hist = []
    pi_OT_hist = []

    prev_oF = None

    for t in range(T):
        world.foods = [(obj_x[t], obj_y[t])]
        effective_heading = heading + eye_offset

        with torch.no_grad():
            out = model.forward(fish_pos, effective_heading, world)

        raw_turn = compute_retinal_turn(out)
        turn = smoother.step(raw_turn)
        F = model.compute_free_energy()

        # Dopamine
        oL_mean = float(out["oL"].abs().mean())
        oR_mean = float(out["oR"].abs().mean())
        dopa, rpe, valL, valR = dopa_sys.step(F, oL_mean, oR_mean)

        # Basal ganglia gating
        bg_gate = bg.step(valL, valR, dopa, rpe)

        # Motor output modulated by BG gate
        modulated_turn = turn + 0.15 * bg_gate
        eye_offset += k_eye * modulated_turn
        eye_offset *= 0.95

        # Precision update
        oF = out["oF"]
        if prev_oF is not None:
            error_OT = oF - prev_oF
            model.prec_OT.update_precision(error_OT)
            model.prec_PC.update_precision(torch.tensor([[F]]))
            with torch.no_grad():
                model.prec_OT.gamma.data += 0.01 * (dopa - 0.5)
                model.prec_PC.gamma.data += 0.01 * (dopa - 0.5)
        prev_oF = oF.clone()

        F_hist.append(F)
        eye_hist.append(eye_offset)
        obj_hist.append(obj_y[t] / amplitude)
        rpe_hist.append(rpe)
        dopa_hist.append(dopa)
        bg_gate_hist.append(bg_gate)
        valL_hist.append(valL)
        valR_hist.append(valR)
        pi_OT_hist.append(out["pi_OT"])

        if t % 50 == 0:
            print(f"  t={t:4d}  F={F:.4f}  RPE={rpe:+.3f}  DA={dopa:.3f}  "
                  f"BG={bg_gate:+.3f}  eye={eye_offset:+.3f}")

    # Plot — 5 panels
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(F_hist, color="steelblue")
    axes[0].set_ylabel("Free Energy (F)")
    axes[0].set_title("Step 4: Basal Ganglia Motor Gating")

    axes[1].plot(valL_hist, label="Valence L", color="blue", alpha=0.7)
    axes[1].plot(valR_hist, label="Valence R", color="red", alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Valence L/R")
    axes[1].legend()

    axes[2].plot(rpe_hist, label="RPE", color="red", alpha=0.6)
    axes[2].plot(dopa_hist, label="Dopamine", color="orange")
    axes[2].axhline(0.5, color="black", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("RPE / Dopamine")
    axes[2].legend()

    axes[3].plot(bg_gate_hist, label="BG Gate", color="darkgreen")
    axes[3].plot(eye_hist, label="Eye offset", color="coral", alpha=0.7)
    axes[3].plot(obj_hist, label="Object (norm)", color="gray", alpha=0.5)
    axes[3].set_ylabel("BG Gate / Eye")
    axes[3].legend()

    axes[4].plot(pi_OT_hist, label="Precision OT", color="purple")
    axes[4].set_ylabel("Precision")
    axes[4].set_xlabel("Time step")
    axes[4].legend()

    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v1_step4_bg_gating.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved: {save_path}")


if __name__ == "__main__":
    run_step4()
