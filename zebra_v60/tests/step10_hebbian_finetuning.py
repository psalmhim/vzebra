"""
Step 10: Hebbian Fine-tuning (Post-natal Learning)
Starting from genomic pre-trained weights, apply RPE-gated Hebbian
plasticity during closed-loop foraging. This mimics post-natal learning
where the larva refines its prey capture circuits through experience.

Uses SNN motor output for turn control (same as step 9).
Hebbian rule: dW = eta * RPE * dopa * outer(pre, post)

Run: python -m zebra_v60.tests.step10_hebbian_finetuning
Output: plots/v60_step10_hebbian_finetuning.png
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
    """Decode SNN motor output into turn signal."""
    motor = out["motor"]
    left_act = motor[0, :100].sigmoid().mean()
    right_act = motor[0, 100:].sigmoid().mean()
    return float(right_act - left_act)


def saccadic_motor_readout(model, fish_pos, heading, world, n_steps=3):
    """Reset SNN, run n_steps, read motor direction, restore state."""
    saved_state = {
        name: layer.v.clone()
        for name, layer in [("OT_L", model.OT_L), ("OT_R", model.OT_R),
                             ("OT_F", model.OT_F), ("PT_L", model.PT_L),
                             ("PC_per", model.PC_per), ("PC_int", model.PC_int)]
    }
    model.reset()
    forage_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                                device=model.device)
    with torch.no_grad():
        for _ in range(n_steps):
            saccade_out = model.forward(fish_pos, heading, world,
                                        goal_probs=forage_goal)
        motor_turn = compute_motor_turn(saccade_out)
    # Restore state
    model.OT_L.v = saved_state["OT_L"]
    model.OT_R.v = saved_state["OT_R"]
    model.OT_F.v_s = saved_state["OT_F"]
    model.PT_L.v_s = saved_state["PT_L"]
    model.PC_per.v_s = saved_state["PC_per"]
    model.PC_int.v_s = saved_state["PC_int"]
    return motor_turn


class HebbianPlasticity:
    """RPE-gated Hebbian plasticity for motor pathway weights.

    Updates weights of TwoComp layers and linear output heads using:
        dW = eta * RPE * dopamine_mod * outer(pre, post)

    Only updates downstream of the frozen topographic map (OT_L, OT_R).
    """

    def __init__(self, eta=0.0005, decay=0.9999):
        self.eta = eta
        self.decay = decay

    def update(self, model, rpe, dopa):
        """Apply Hebbian update to trainable pathway weights."""
        dopa_mod = max(0.1, dopa)
        lr = self.eta * rpe * dopa_mod

        with torch.no_grad():
            # OT_F: fused → OT features
            if hasattr(model, '_last_fused') and hasattr(model, '_last_oF'):
                pre = model._last_fused
                post = model._last_oF
                dW = lr * (pre.t() @ post)
                model.OT_F.W_FF.data += dW
                model.OT_F.W_FF.data *= self.decay

            # Motor output head: PC_int → motor
            intent = model.PC_int.v_s
            mot_v = model.mot.v_s
            dW_mot = lr * (intent.t() @ mot_v.sigmoid())
            model.mot.W_FF.data += dW_mot
            model.mot.W_FF.data *= self.decay


def run_foraging_with_hebbian(model, label, hebbian=None,
                               T=800, swim_speed=1.5, turn_gain=0.15,
                               use_motor=False, saccade_period=8, seed=42):
    """Run closed-loop foraging with optional saccadic motor and Hebbian."""
    np.random.seed(seed)
    model.reset()
    dopa_sys = DopamineSystem_v60()
    bg = BasalGanglia_v60(mode="exploratory")
    ot = OpticTectum_v60()
    thal = ThalamusRelay_v60()
    smoother = TurnSmoother(alpha=0.35)

    world = WorldEnv(xmin=-500, xmax=500, ymin=-500, ymax=500, n_food=0)

    for _ in range(15):
        angle = np.random.uniform(-math.pi / 3, math.pi / 3)
        dist = np.random.uniform(30, 150)
        world.foods.append((dist * math.cos(angle), dist * math.sin(angle)))

    fish_x, fish_y = 0.0, 0.0
    heading = 0.0

    pos_x, pos_y = [], []
    eaten_times = []
    F_hist = []
    dopa_hist = []
    rpe_hist = []
    cumulative_eaten = []
    total_eaten = 0
    prev_oF = None
    motor_bias = 0.0
    forage_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                                device=model.device)

    for t in range(T):
        fish_pos = np.array([fish_x, fish_y])
        effective_heading = heading + ot.eye_pos * 0.25

        with torch.no_grad():
            out = model.forward(fish_pos, effective_heading, world,
                                goal_probs=forage_goal)

        # Retinal turn as base
        raw_turn = compute_retinal_turn(out)

        # Saccadic motor readout (preserves SNN state)
        if use_motor and t % saccade_period == 0:
            motor_turn = saccadic_motor_readout(model, fish_pos, heading, world)
            motor_bias = 0.3 * motor_turn

        raw_turn += motor_bias
        turn = smoother.step(raw_turn)
        F_visual = model.compute_free_energy()

        F_audio = 0.1 * abs(math.sin(0.05 * t))
        cms = thal.step(F_visual, F_audio)
        dopa_sys.beta = thal.modulate_dopamine_gain()

        oL_mean = float(out["oL"].abs().mean())
        oR_mean = float(out["oR"].abs().mean())

        # Check eating BEFORE dopamine update
        eaten, _ = world.try_eat(fish_x, fish_y)
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

        # === HEBBIAN UPDATE ===
        if hebbian is not None:
            hebbian.update(model, rpe, dopa)

        # Locomotion
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
        F_hist.append(F_visual)
        dopa_hist.append(dopa)
        rpe_hist.append(rpe)
        cumulative_eaten.append(total_eaten)

    dist_traveled = sum(
        math.sqrt((pos_x[i] - pos_x[i - 1]) ** 2 +
                   (pos_y[i] - pos_y[i - 1]) ** 2)
        for i in range(1, T)
    )

    print(f"  [{label}] Food eaten: {total_eaten}  "
          f"Distance: {dist_traveled:.0f}  "
          f"Efficiency: {total_eaten / (dist_traveled / 1000 + 0.01):.1f} food/km")

    return {
        "pos_x": pos_x, "pos_y": pos_y,
        "F": F_hist, "dopa": dopa_hist, "rpe": rpe_hist,
        "eaten_times": eaten_times, "total_eaten": total_eaten,
        "cumulative_eaten": cumulative_eaten,
        "dist_traveled": dist_traveled,
        "world": world,
    }


def run_step10():
    print("=" * 60)
    print("Step 10: Hebbian Fine-tuning (Post-natal Learning)")
    print("=" * 60)

    device = get_device()
    weights_path = os.path.join(PROJECT_ROOT, "zebra_v60", "weights", "genomic_v60.pt")
    if not os.path.exists(weights_path):
        print(f"ERROR: No pre-trained weights at {weights_path}")
        print("Run step8_genomic_pretraining.py first!")
        return

    # (a) Genomic-only (no Hebbian) — with saccadic motor
    print("\n--- Genomic Pre-trained (no Hebbian) ---")
    model_a = ZebrafishSNN_v60(device=device)
    model_a.load_saveable_state(
        torch.load(weights_path, weights_only=True, map_location=device))
    res_a = run_foraging_with_hebbian(model_a, "Genomic-only", hebbian=None,
                                       use_motor=True)

    # (b) Genomic + Hebbian — saccadic motor with learning
    print("\n--- Genomic + Hebbian Fine-tuning ---")
    model_b = ZebrafishSNN_v60(device=device)
    model_b.load_saveable_state(
        torch.load(weights_path, weights_only=True, map_location=device))
    hebb = HebbianPlasticity(eta=0.0005, decay=0.9999)
    res_b = run_foraging_with_hebbian(model_b, "Genomic+Hebbian", hebbian=hebb,
                                       use_motor=True)

    # (c) Random + Hebbian (tabula rasa) — pure retinal (no useful motor)
    print("\n--- Random + Hebbian (tabula rasa) ---")
    model_c = ZebrafishSNN_v60(device=device)
    hebb_c = HebbianPlasticity(eta=0.0005, decay=0.9999)
    res_c = run_foraging_with_hebbian(model_c, "Random+Hebbian", hebbian=hebb_c,
                                       use_motor=False)

    # === PLOT: 3x2 panels ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    conditions = [
        (res_a, "Genomic Only", "steelblue"),
        (res_b, "Genomic + Hebbian", "mediumseagreen"),
        (res_c, "Random + Hebbian", "coral"),
    ]

    # Row 0: Trajectories
    for col, (res, title, color) in enumerate(conditions):
        ax = axes[0, col]
        ax.plot(res["pos_x"], res["pos_y"], color=color, alpha=0.6, linewidth=0.8)
        ax.plot(res["pos_x"][0], res["pos_y"][0], "go", markersize=10)
        ax.plot(res["pos_x"][-1], res["pos_y"][-1], "rs", markersize=10)
        for i, et in enumerate(res["eaten_times"]):
            ax.plot(res["pos_x"][et], res["pos_y"][et], "y*", markersize=10,
                    zorder=6, label="Eat" if i == 0 else None)
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_title(f"{title}\n(eaten: {res['total_eaten']})")
        ax.set_aspect("equal")

    # Row 1, Col 0: Cumulative food eaten comparison
    ax = axes[1, 0]
    for res, title, color in conditions:
        ax.plot(res["cumulative_eaten"], color=color, label=title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative Food Eaten")
    ax.set_title("Learning Curve")
    ax.legend(fontsize=8)

    # Row 1, Col 1: RPE over time (Hebbian model)
    ax = axes[1, 1]
    ax.plot(res_b["rpe"], color="red", alpha=0.5, label="RPE (Genomic+Hebb)")
    ax.plot(res_b["dopa"], color="orange", alpha=0.7, label="Dopamine")
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time step")
    ax.set_ylabel("RPE / Dopamine")
    ax.set_title("Dopamine & RPE (Genomic+Hebbian)")
    ax.legend(fontsize=8)
    for et in res_b["eaten_times"]:
        ax.axvline(et, color="gold", linewidth=1, alpha=0.3)

    # Row 1, Col 2: Bar chart comparison
    ax = axes[1, 2]
    names = ["Genomic\nOnly", "Genomic\n+Hebbian", "Random\n+Hebbian"]
    eaten = [res_a["total_eaten"], res_b["total_eaten"], res_c["total_eaten"]]
    colors = ["steelblue", "mediumseagreen", "coral"]
    bars = ax.bar(names, eaten, color=colors, edgecolor="black")
    ax.set_ylabel("Food Eaten")
    ax.set_title("Total Performance")
    for bar, val in zip(bars, eaten):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Step 10: Hebbian Fine-tuning (Motor-driven Foraging)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v60_step10_hebbian_finetuning.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # Save Hebbian-refined weights
    weights_dir = os.path.join(PROJECT_ROOT, "zebra_v60", "weights")
    hebb_path = os.path.join(weights_dir, "genomic_hebbian_v60.pt")
    torch.save(model_b.state_dict(), hebb_path)
    print(f"Hebbian-refined weights saved: {hebb_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Genomic Only:     {res_a['total_eaten']} food  "
          f"({res_a['dist_traveled']:.0f} dist)")
    print(f"  Genomic+Hebbian:  {res_b['total_eaten']} food  "
          f"({res_b['dist_traveled']:.0f} dist)")
    print(f"  Random+Hebbian:   {res_c['total_eaten']} food  "
          f"({res_c['dist_traveled']:.0f} dist)")
    print("=" * 60)


if __name__ == "__main__":
    run_step10()
