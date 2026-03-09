"""
Step 9: Genomic Foraging — Pre-trained vs Random Weights
Compare closed-loop foraging using SNN MOTOR OUTPUT for turn control:
  (a) random (untrained) weights
  (b) genomic pre-trained weights from Step 8

The key difference from earlier versions: the turn signal comes from the
SNN motor output (mot layer), not raw retinal asymmetry. This is why
genomic pre-training matters — it initializes the motor pathway to produce
correct turn signals from visual input.

Run: python -m zebra_v60.tests.step9_genomic_foraging
Output: plots/v60_step9_genomic_foraging.png
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
from zebra_v60.brain.dopamine_v60 import DopamineSystem_v60
from zebra_v60.brain.basal_ganglia_v60 import BasalGanglia_v60
from zebra_v60.brain.optic_tectum_v60 import OpticTectum_v60
from zebra_v60.brain.thalamus_v60 import ThalamusRelay_v60
from zebra_v60.brain.device_util import get_device
from zebra_v60.world.world_env import WorldEnv
from zebra_v60.tests.step1_vision_pursuit import compute_retinal_turn, TurnSmoother


def compute_motor_turn(out):
    """Decode SNN motor output into turn signal.
    Left motor (0:100) → left turn, Right motor (100:200) → right turn.
    Positive = turn right, Negative = turn left.
    """
    motor = out["motor"]
    left_act = motor[0, :100].sigmoid().mean()
    right_act = motor[0, 100:].sigmoid().mean()
    return float(right_act - left_act)


def run_foraging(model, label, T=600, swim_speed=1.5, turn_gain=0.15,
                 use_motor=False, saccade_period=8, saccade_steps=3, seed=42):
    """Run closed-loop foraging.

    If use_motor=True, periodically resets SNN and reads fresh motor output
    (saccadic processing). Otherwise uses pure retinal turn.
    """
    np.random.seed(seed)
    model.reset()
    dopa_sys = DopamineSystem_v60()
    bg = BasalGanglia_v60(mode="exploratory")
    ot = OpticTectum_v60()
    thal = ThalamusRelay_v60()
    smoother = TurnSmoother(alpha=0.35)

    world = WorldEnv(xmin=-500, xmax=500, ymin=-500, ymax=500, n_food=0)

    # Place food ahead and around the fish
    for _ in range(12):
        angle = np.random.uniform(-math.pi / 3, math.pi / 3)
        dist = np.random.uniform(30, 150)
        world.foods.append((dist * math.cos(angle), dist * math.sin(angle)))

    fish_x, fish_y = 0.0, 0.0
    heading = 0.0

    pos_x, pos_y = [], []
    heading_hist = []
    eaten_times = []
    F_hist = []
    dopa_hist = []
    total_eaten = 0
    prev_oF = None
    motor_bias = 0.0  # Saccadic motor heading bias

    for t in range(T):
        fish_pos = np.array([fish_x, fish_y])
        effective_heading = heading + ot.eye_pos * 0.25

        with torch.no_grad():
            out = model.forward(fish_pos, effective_heading, world)

        # Turn signal: retinal is always the base
        raw_turn = compute_retinal_turn(out)

        # Saccadic motor readout: periodically reset, read fresh motor, restore state
        if use_motor and t % saccade_period == 0:
            # Save current SNN state
            saved_state = {
                name: layer.v.clone()
                for name, layer in [("OT_L", model.OT_L), ("OT_R", model.OT_R),
                                     ("OT_F", model.OT_F), ("PT_L", model.PT_L),
                                     ("PC_per", model.PC_per), ("PC_int", model.PC_int)]
            }
            model.reset()
            with torch.no_grad():
                for _ in range(saccade_steps):
                    saccade_out = model.forward(fish_pos, heading, world)
                motor_turn = compute_motor_turn(saccade_out)
                motor_bias = 0.3 * motor_turn
            # Restore SNN state
            model.OT_L.v = saved_state["OT_L"]
            model.OT_R.v = saved_state["OT_R"]
            model.OT_F.v = saved_state["OT_F"]
            model.PT_L.v = saved_state["PT_L"]
            model.PC_per.v = saved_state["PC_per"]
            model.PC_int.v = saved_state["PC_int"]

        raw_turn += motor_bias
        turn = smoother.step(raw_turn)
        F_visual = model.compute_free_energy()

        F_audio = 0.1 * abs(math.sin(0.05 * t))
        cms = thal.step(F_visual, F_audio)
        dopa_sys.beta = thal.modulate_dopamine_gain()

        oL_mean = float(out["oL"].abs().mean())
        oR_mean = float(out["oR"].abs().mean())
        eaten = world.try_eat(fish_x, fish_y)
        dopa, rpe, valL, valR = dopa_sys.step(F_visual, oL_mean, oR_mean,
                                               eaten=eaten)

        valL_eff = valL - 0.1 * turn
        valR_eff = valR + 0.1 * turn

        bg.noise = thal.modulate_bg_exploration()
        bg_gate = bg.step(valL_eff, valR_eff, dopa, rpe)
        eye_pos = ot.step(valL_eff, valR_eff, F_visual, bg_gate, dopa)

        oF = out["oF"]
        if prev_oF is not None:
            error_OT = oF - prev_oF
            model.prec_OT.update_precision(error_OT)
            model.prec_PC.update_precision(torch.tensor([[F_visual]]))
            with torch.no_grad():
                model.prec_OT.gamma.data += 0.008 * (dopa - 0.5)
                model.prec_PC.gamma.data += 0.008 * (dopa - 0.5)
        prev_oF = oF.clone()

        # Turn + BG exploration + eye position
        turn_command = turn_gain * turn + 0.03 * bg_gate + 0.02 * eye_pos
        heading += turn_command

        speed = swim_speed * (0.8 + 0.4 * dopa)
        fish_x += speed * math.cos(heading)
        fish_y += speed * math.sin(heading)

        # Soft boundary avoidance
        margin = 80
        if fish_x > world.xmax - margin:
            heading += 0.05 * (fish_x - (world.xmax - margin)) / margin
        if fish_x < world.xmin + margin:
            heading -= 0.05 * ((world.xmin + margin) - fish_x) / margin
        if fish_y > world.ymax - margin:
            heading -= 0.05 * (fish_y - (world.ymax - margin)) / margin
        if fish_y < world.ymin + margin:
            heading += 0.05 * ((world.ymin + margin) - fish_y) / margin

        fish_x = max(world.xmin + 5, min(world.xmax - 5, fish_x))
        fish_y = max(world.ymin + 5, min(world.ymax - 5, fish_y))
        heading = math.atan2(math.sin(heading), math.cos(heading))

        # Eat food
        if eaten > 0:
            total_eaten += eaten
            eaten_times.append(t)
            for _ in range(eaten):
                angle = heading + np.random.uniform(-math.pi / 2, math.pi / 2)
                dist = np.random.uniform(50, 150)
                fx = fish_x + dist * math.cos(angle)
                fy = fish_y + dist * math.sin(angle)
                fx = max(world.xmin + 30, min(world.xmax - 30, fx))
                fy = max(world.ymin + 30, min(world.ymax - 30, fy))
                world.foods.append((fx, fy))

        pos_x.append(fish_x)
        pos_y.append(fish_y)
        heading_hist.append(heading)
        F_hist.append(F_visual)
        dopa_hist.append(dopa)

    dist_traveled = sum(
        math.sqrt((pos_x[i] - pos_x[i - 1]) ** 2 +
                   (pos_y[i] - pos_y[i - 1]) ** 2)
        for i in range(1, T)
    )

    print(f"\n  [{label}] Food eaten: {total_eaten}  "
          f"Distance: {dist_traveled:.0f}  "
          f"Efficiency: {total_eaten / (dist_traveled / 1000):.2f} food/km")

    return {
        "pos_x": pos_x, "pos_y": pos_y,
        "heading": heading_hist, "F": F_hist, "dopa": dopa_hist,
        "eaten_times": eaten_times, "total_eaten": total_eaten,
        "dist_traveled": dist_traveled,
        "world": world,
    }


def run_step9():
    print("=" * 60)
    print("Step 9: Genomic Foraging Comparison")
    print("=" * 60)

    device = get_device()

    # (a) Random weights — pure retinal control (no useful motor pathway)
    print("\n--- Running with RANDOM weights (retinal only) ---")
    model_random = ZebrafishSNN_v60(device=device)
    res_random = run_foraging(model_random, "Random", use_motor=False)

    # (b) Genomic pre-trained weights — retinal + saccadic motor readout
    print("\n--- Running with GENOMIC pre-trained weights (saccadic motor) ---")
    model_genomic = ZebrafishSNN_v60(device=device)
    weights_path = os.path.join(PROJECT_ROOT, "zebra_v60", "weights", "genomic_v60.pt")
    if os.path.exists(weights_path):
        model_genomic.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
        print(f"  Loaded weights from {weights_path}")
    else:
        print(f"  WARNING: No pre-trained weights found at {weights_path}")
        print("  Run step8_genomic_pretraining.py first!")
        return
    res_genomic = run_foraging(model_genomic, "Genomic", use_motor=True)

    # === COMPARISON PLOT: 3x2 panels ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, (res, title) in enumerate([(res_random, "Random Weights"),
                                         (res_genomic, "Genomic Pre-trained")]):
        # Row 0: Trajectory
        ax = axes[0, col]
        ax.plot(res["pos_x"], res["pos_y"], color="coral", alpha=0.6, linewidth=0.8)
        ax.plot(res["pos_x"][0], res["pos_y"][0], "go", markersize=10, label="Start")
        ax.plot(res["pos_x"][-1], res["pos_y"][-1], "rs", markersize=10, label="End")
        for i, et in enumerate(res["eaten_times"]):
            ax.plot(res["pos_x"][et], res["pos_y"][et], "y*", markersize=12,
                    zorder=6, label="Eat" if i == 0 else None)
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_title(f"{title}\n(eaten: {res['total_eaten']})")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)

        # Row 1: Free Energy
        ax = axes[1, col]
        ax.plot(res["F"], color="steelblue", alpha=0.7)
        ax.set_ylabel("Free Energy")
        ax.set_xlabel("Time step")
        for et in res["eaten_times"]:
            ax.axvline(et, color="gold", linewidth=1, alpha=0.5)

    # Column 2: Comparison bar chart
    ax = axes[0, 2]
    names = ["Random", "Genomic"]
    eaten = [res_random["total_eaten"], res_genomic["total_eaten"]]
    colors = ["lightcoral", "mediumseagreen"]
    bars = ax.bar(names, eaten, color=colors, edgecolor="black")
    ax.set_ylabel("Food Eaten")
    ax.set_title("Foraging Performance")
    for bar, val in zip(bars, eaten):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")

    # Efficiency comparison
    ax = axes[1, 2]
    eff_random = res_random["total_eaten"] / (res_random["dist_traveled"] / 1000 + 0.01)
    eff_genomic = res_genomic["total_eaten"] / (res_genomic["dist_traveled"] / 1000 + 0.01)
    bars = ax.bar(names, [eff_random, eff_genomic], color=colors, edgecolor="black")
    ax.set_ylabel("Efficiency (food/km)")
    ax.set_title("Foraging Efficiency")
    for bar, val in zip(bars, [eff_random, eff_genomic]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Step 9: Genomic Pre-training vs Random Weights (Motor-driven)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v60_step9_genomic_foraging.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")


if __name__ == "__main__":
    run_step9()
