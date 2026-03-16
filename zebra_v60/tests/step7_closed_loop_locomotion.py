"""
Step 7: Closed-Loop Locomotion
The fish swims through the world, turning toward prey based on
retinal asymmetry + dopamine + BG gating + OT dynamics.
Full perception → action → world → perception loop.

Run: python -m zebra_v60.tests.step7_closed_loop_locomotion
Output: plots/v60_step7_closed_loop_locomotion.png
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


def run_step7(T=500, swim_speed=1.5, turn_gain=0.15):
    print("=" * 60)
    print("Step 7: Closed-Loop Locomotion")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN_v60(device=device)
    model.reset()
    dopa_sys = DopamineSystem_v60()
    bg = BasalGanglia_v60(mode="exploratory")
    ot = OpticTectum_v60()
    thal = ThalamusRelay_v60()
    smoother = TurnSmoother(alpha=0.35)

    # Large world so boundaries don't dominate
    world = WorldEnv(xmin=-500, xmax=500, ymin=-500, ymax=500, n_food=0)

    # Place food in accessible positions ahead and around the fish
    np.random.seed(42)
    for _ in range(12):
        angle = np.random.uniform(-math.pi / 3, math.pi / 3)  # mostly ahead
        dist = np.random.uniform(30, 150)
        world.foods.append((dist * math.cos(angle), dist * math.sin(angle)))

    fish_x, fish_y = 0.0, 0.0
    heading = 0.0

    # History
    F_hist = []
    dopa_hist = []
    rpe_hist = []
    bg_gate_hist = []
    eye_hist = []
    pi_OT_hist = []
    pi_PC_hist = []
    pos_x_hist = []
    pos_y_hist = []
    heading_hist = []
    eaten_times = []
    cms_hist = []
    retL_hist = []
    retR_hist = []

    prev_oF = None
    total_eaten = 0

    for t in range(T):
        fish_pos = np.array([fish_x, fish_y])
        effective_heading = heading + ot.eye_pos * 0.25

        with torch.no_grad():
            out = model.forward(fish_pos, effective_heading, world)

        raw_turn = compute_retinal_turn(out)
        turn = smoother.step(raw_turn)
        retL_sum = float(out["retL"].sum())
        retR_sum = float(out["retR"].sum())

        F_visual = model.compute_free_energy()
        F_audio = 0.1 * abs(math.sin(0.05 * t))

        cms = thal.step(F_visual, F_audio)

        dopa_sys.beta = thal.modulate_dopamine_gain()
        oL_mean = float(out["oL"].abs().mean())
        oR_mean = float(out["oR"].abs().mean())
        dopa, rpe, valL, valR = dopa_sys.step(F_visual, oL_mean, oR_mean)

        valL_eff = valL - 0.1 * turn
        valR_eff = valR + 0.1 * turn

        bg.noise = thal.modulate_bg_exploration()
        bg_gate = bg.step(valL_eff, valR_eff, dopa, rpe)

        eye_pos = ot.step(valL_eff, valR_eff, F_visual, bg_gate, dopa)

        # Precision
        oF = out["oF"]
        if prev_oF is not None:
            error_OT = oF - prev_oF
            model.prec_OT.update_precision(error_OT)
            model.prec_PC.update_precision(torch.tensor([[F_visual]]))
            with torch.no_grad():
                model.prec_OT.gamma.data += 0.008 * (dopa - 0.5)
                model.prec_PC.gamma.data += 0.008 * (dopa - 0.5)
        prev_oF = oF.clone()

        # === LOCOMOTION ===
        # Retinal asymmetry is the primary turn drive
        # BG adds exploration, OT adds smooth pursuit
        turn_command = turn_gain * turn + 0.03 * bg_gate + 0.02 * eye_pos
        heading += turn_command

        # Dopamine-modulated speed
        speed = swim_speed * (0.8 + 0.4 * dopa)
        fish_x += speed * math.cos(heading)
        fish_y += speed * math.sin(heading)

        # Soft boundary avoidance (gradual turn away from walls)
        margin = 80
        if fish_x > world.xmax - margin:
            heading += 0.05 * (fish_x - (world.xmax - margin)) / margin
        if fish_x < world.xmin + margin:
            heading -= 0.05 * ((world.xmin + margin) - fish_x) / margin
        if fish_y > world.ymax - margin:
            heading -= 0.05 * (fish_y - (world.ymax - margin)) / margin
        if fish_y < world.ymin + margin:
            heading += 0.05 * ((world.ymin + margin) - fish_y) / margin

        # Hard clamp as safety
        fish_x = max(world.xmin + 5, min(world.xmax - 5, fish_x))
        fish_y = max(world.ymin + 5, min(world.ymax - 5, fish_y))

        # Normalize heading to [-pi, pi]
        heading = math.atan2(math.sin(heading), math.cos(heading))

        # Try to eat food
        eaten, _ = world.try_eat(fish_x, fish_y)
        if eaten > 0:
            total_eaten += eaten
            eaten_times.append(t)
            # Respawn food ahead of fish
            for _ in range(eaten):
                angle = heading + np.random.uniform(-math.pi / 2, math.pi / 2)
                dist = np.random.uniform(50, 150)
                fx = fish_x + dist * math.cos(angle)
                fy = fish_y + dist * math.sin(angle)
                fx = max(world.xmin + 30, min(world.xmax - 30, fx))
                fy = max(world.ymin + 30, min(world.ymax - 30, fy))
                world.foods.append((fx, fy))

        # Record
        F_hist.append(F_visual)
        dopa_hist.append(dopa)
        rpe_hist.append(rpe)
        bg_gate_hist.append(bg_gate)
        eye_hist.append(eye_pos)
        pi_OT_hist.append(out["pi_OT"])
        pi_PC_hist.append(out["pi_PC"])
        pos_x_hist.append(fish_x)
        pos_y_hist.append(fish_y)
        heading_hist.append(heading)
        cms_hist.append(cms)
        retL_hist.append(retL_sum)
        retR_hist.append(retR_sum)

        if t % 50 == 0:
            print(f"  t={t:4d}  pos=({fish_x:+6.1f},{fish_y:+6.1f})  "
                  f"head={math.degrees(heading):+6.1f}°  F={F_visual:.3f}  "
                  f"DA={dopa:.3f}  BG={bg_gate:+.3f}  eaten={total_eaten}")

    dist_traveled = sum(
        math.sqrt((pos_x_hist[i] - pos_x_hist[i - 1]) ** 2 +
                   (pos_y_hist[i] - pos_y_hist[i - 1]) ** 2)
        for i in range(1, T)
    )
    print(f"\nTotal food eaten: {total_eaten}")
    print(f"Distance traveled: {dist_traveled:.1f}")
    if eaten_times:
        print(f"Food capture times: {eaten_times}")

    # === PLOT: 6-panel figure ===
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Panel 1: Trajectory
    ax = axes[0, 0]
    ax.plot(pos_x_hist, pos_y_hist, color="coral", alpha=0.6, linewidth=0.8)
    ax.plot(pos_x_hist[0], pos_y_hist[0], "go", markersize=10, label="Start", zorder=5)
    ax.plot(pos_x_hist[-1], pos_y_hist[-1], "rs", markersize=10, label="End", zorder=5)
    for i, et in enumerate(eaten_times):
        ax.plot(pos_x_hist[et], pos_y_hist[et], "y*", markersize=12,
                zorder=6, label="Eat" if i == 0 else None)
    for food in world.foods:
        fx, fy = (food["x"], food["y"]) if isinstance(food, dict) else (food[0], food[1])
        ax.plot(fx, fy, "g^", markersize=6, alpha=0.4)
    ax.set_xlim(world.xmin, world.xmax)
    ax.set_ylim(world.ymin, world.ymax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Fish Trajectory")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # Panel 2: Free Energy + CMS
    ax = axes[0, 1]
    ax.plot(F_hist, color="steelblue", label="F visual", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(cms_hist, color="magenta", alpha=0.5, linestyle="--", label="CMS")
    ax.set_ylabel("Free Energy")
    ax2.set_ylabel("CMS", color="magenta")
    ax.set_title("Free Energy & CMS")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    for et in eaten_times:
        ax.axvline(et, color="gold", linewidth=1, alpha=0.5)

    # Panel 3: Dopamine / RPE
    ax = axes[1, 0]
    ax.plot(rpe_hist, label="RPE", color="red", alpha=0.5)
    ax.plot(dopa_hist, label="Dopamine", color="orange")
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("RPE / Dopamine")
    ax.set_title("Dopamine & RPE")
    ax.legend(fontsize=7)
    for et in eaten_times:
        ax.axvline(et, color="gold", linewidth=1, alpha=0.5)

    # Panel 4: BG + Eye + Retinal asymmetry
    ax = axes[1, 1]
    ax.plot(bg_gate_hist, label="BG Gate", color="darkgreen", alpha=0.6)
    ax.plot(eye_hist, label="Eye (OT)", color="coral", alpha=0.6)
    ax.set_ylabel("BG / Eye")
    ax.set_title("BG Gate & Eye Dynamics")
    ax.legend(fontsize=7)

    # Panel 5: Precision
    ax = axes[2, 0]
    ax.plot(pi_OT_hist, label="Precision OT", color="purple")
    ax.plot(pi_PC_hist, label="Precision PC", color="teal")
    ax.set_ylabel("Precision")
    ax.set_xlabel("Time step")
    ax.set_title("Precision Adaptation")
    ax.legend(fontsize=7)

    # Panel 6: Heading + retinal signals
    ax = axes[2, 1]
    ax.plot(np.degrees(heading_hist), color="navy", alpha=0.7, label="Heading")
    ax.set_ylabel("Heading (°)")
    ax.set_xlabel("Time step")
    ax.set_title(f"Heading (food eaten: {total_eaten})")
    ax.legend(fontsize=7)
    for et in eaten_times:
        ax.axvline(et, color="gold", linewidth=1, alpha=0.5)

    fig.suptitle("Step 7: Closed-Loop Locomotion", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v60_step7_closed_loop_locomotion.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


if __name__ == "__main__":
    run_step7()
