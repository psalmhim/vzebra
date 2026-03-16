"""
Step 12: Classification-Guided Behavior
The fish uses its trained classifier (step 11) to identify what it sees
and selects behavior accordingly:
  - Food:        approach (turn toward, speed up)
  - Enemy:       flee (turn away, burst speed)
  - Colleague:   school (maintain moderate distance)
  - Nothing/Env: explore (BG-driven saccadic search)

The classifier output modulates the gain of approach/avoidance and
the dopamine/BG system parameters. This creates emergent adaptive
behavior from the combination of:
  1. Retinal turn signal (hardwired direction)
  2. Classification (learned identity)
  3. Dopamine + BG (motivation + exploration)

Run: python -m zebra_v60.tests.step12_classification_guided_behavior
Output: plots/v60_step12_classification_guided_behavior.png
"""
import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F
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

# Class indices (must match step 11)
CLASS_NOTHING = 0
CLASS_FOOD = 1
CLASS_ENEMY = 2
CLASS_COLLEAGUE = 3
CLASS_ENVIRONMENT = 4
CLASS_NAMES = ["nothing", "food", "enemy", "colleague", "environment"]


def classification_to_behavior(cls_probs):
    """Convert classifier probabilities to behavioral modulation.

    Key design: nothing/environment → normal approach (gain=1.0, baseline).
    Only food/enemy/colleague actively modify behavior.

    Returns:
        approach_gain: positive = approach, negative = flee
        speed_mod: multiplier for swim speed
        explore_mod: multiplier for BG exploration
    """
    p_food = cls_probs[CLASS_FOOD]
    p_enemy = cls_probs[CLASS_ENEMY]
    p_colleague = cls_probs[CLASS_COLLEAGUE]
    p_nothing = cls_probs[CLASS_NOTHING] + cls_probs[CLASS_ENVIRONMENT]

    # Baseline approach: always 1.0 (retinal turn works by default)
    # Food boosts approach slightly. Enemy only flees at high confidence.
    # This avoids the problem where a weak enemy signal cancels food approach.
    enemy_flee = -1.5 * max(0, p_enemy - 0.6)  # only flee if >60% enemy
    approach_gain = 1.0 + 0.3 * p_food + enemy_flee + 0.15 * p_colleague

    # Speed: enemy triggers burst only at high confidence
    speed_mod = 1.0 + 0.2 * p_food + 0.6 * max(0, p_enemy - 0.5)

    # Exploration: high when nothing around, low when entity detected
    explore_mod = 1.0 + 0.3 * p_nothing - 0.3 * (p_food + p_enemy)

    return approach_gain, speed_mod, max(0.2, explore_mod)


def run_step12(T=800, swim_speed=1.5, base_turn_gain=0.15):
    print("=" * 60)
    print("Step 12: Classification-Guided Behavior")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN_v60(device=device)

    # Load classifier weights
    cls_path = os.path.join(PROJECT_ROOT, "zebra_v60", "weights", "classifier_v60.pt")
    if os.path.exists(cls_path):
        model.load_state_dict(
            torch.load(cls_path, weights_only=True, map_location=device))
        print(f"  Loaded classifier weights from {cls_path}")
    else:
        print(f"  WARNING: No classifier weights at {cls_path}")
        print("  Run step11_object_classification.py first!")
        return

    model.reset()
    model.eval()

    dopa_sys = DopamineSystem_v60()
    bg = BasalGanglia_v60(mode="exploratory")
    ot = OpticTectum_v60()
    thal = ThalamusRelay_v60()
    smoother = TurnSmoother(alpha=0.35)

    # Rich world with all entity types
    world = WorldEnv(xmin=-500, xmax=500, ymin=-500, ymax=500, n_food=0)

    np.random.seed(42)

    # Scatter food, enemies, colleagues
    for _ in range(10):
        a = np.random.uniform(-math.pi, math.pi)
        d = np.random.uniform(40, 200)
        world.foods.append((d * math.cos(a), d * math.sin(a)))

    for _ in range(4):
        a = np.random.uniform(-math.pi, math.pi)
        d = np.random.uniform(80, 250)
        world.enemies.append((d * math.cos(a), d * math.sin(a)))

    for _ in range(5):
        a = np.random.uniform(-math.pi, math.pi)
        d = np.random.uniform(60, 200)
        world.colleagues.append((d * math.cos(a), d * math.sin(a)))

    fish_x, fish_y = 0.0, 0.0
    heading = 0.0

    # History
    pos_x, pos_y = [], []
    heading_hist = []
    eaten_times = []
    flee_times = []
    cls_hist = []
    approach_hist = []
    speed_hist = []
    F_hist = []
    dopa_hist = []
    total_eaten = 0
    total_fled = 0
    prev_oF = None

    for t in range(T):
        fish_pos = np.array([fish_x, fish_y])
        effective_heading = heading + ot.eye_pos * 0.25

        with torch.no_grad():
            out = model.forward(fish_pos, effective_heading, world)

        # Classification → behavioral modulation
        cls_logits = out["cls"]  # [1, 5]
        cls_probs = F.softmax(cls_logits, dim=1)[0].cpu().numpy()
        predicted_class = int(cls_logits.argmax(dim=1).item())
        approach_gain, speed_mod, explore_mod = classification_to_behavior(cls_probs)

        # Retinal turn (hardwired direction signal)
        raw_turn = compute_retinal_turn(out)

        # Classification modulates turn: approach_gain > 0 → approach,
        # approach_gain < 0 → flee (invert turn direction)
        modulated_turn = raw_turn * approach_gain
        turn = smoother.step(modulated_turn)

        # Free energy and neuromodulation
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

        # Classification modulates exploration
        bg.noise = thal.modulate_bg_exploration() * explore_mod
        bg_gate = bg.step(valL_eff, valR_eff, dopa, rpe)
        eye_pos = ot.step(valL_eff, valR_eff, F_visual, bg_gate, dopa)

        # Precision update
        oF = out["oF"]
        if prev_oF is not None:
            error_OT = oF - prev_oF
            model.prec_OT.update_precision(error_OT)
            model.prec_PC.update_precision(torch.tensor([[F_visual]],
                                                         device=device))
            with torch.no_grad():
                model.prec_OT.gamma.data += 0.008 * (dopa - 0.5)
                model.prec_PC.gamma.data += 0.008 * (dopa - 0.5)
        prev_oF = oF.clone()

        # === LOCOMOTION ===
        turn_command = base_turn_gain * turn + 0.03 * bg_gate + 0.02 * eye_pos
        heading += turn_command

        # Speed: classification modulates (enemy = burst, food = moderate)
        speed = swim_speed * speed_mod * (0.8 + 0.4 * dopa)
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
                a = heading + np.random.uniform(-math.pi / 2, math.pi / 2)
                d = np.random.uniform(60, 180)
                fx = fish_x + d * math.cos(a)
                fy = fish_y + d * math.sin(a)
                fx = max(world.xmin + 30, min(world.xmax - 30, fx))
                fy = max(world.ymin + 30, min(world.ymax - 30, fy))
                world.foods.append((fx, fy))

        # Check enemy proximity → flee event
        enemy_dist = world.check_enemy_proximity(fish_x, fish_y, danger_radius=35)
        if enemy_dist is not None:
            total_fled += 1
            if len(flee_times) == 0 or t - flee_times[-1] > 10:
                flee_times.append(t)

        # Move enemies slowly (patrol behavior)
        if t % 5 == 0:
            new_enemies = []
            for (ex, ey) in world.enemies:
                ex += np.random.uniform(-3, 3)
                ey += np.random.uniform(-3, 3)
                ex = max(world.xmin + 20, min(world.xmax - 20, ex))
                ey = max(world.ymin + 20, min(world.ymax - 20, ey))
                new_enemies.append((ex, ey))
            world.enemies = new_enemies

        # Move colleagues (schooling drift toward fish)
        if t % 3 == 0:
            new_colleagues = []
            for (cx, cy) in world.colleagues:
                dx = fish_x - cx
                dy = fish_y - cy
                dist = math.sqrt(dx * dx + dy * dy) + 1e-8
                # Drift toward fish but keep distance
                if dist > 100:
                    cx += 2.0 * dx / dist
                    cy += 2.0 * dy / dist
                elif dist < 40:
                    cx -= 1.0 * dx / dist
                    cy -= 1.0 * dy / dist
                cx += np.random.uniform(-2, 2)
                cy += np.random.uniform(-2, 2)
                cx = max(world.xmin + 20, min(world.xmax - 20, cx))
                cy = max(world.ymin + 20, min(world.ymax - 20, cy))
                new_colleagues.append((cx, cy))
            world.colleagues = new_colleagues

        pos_x.append(fish_x)
        pos_y.append(fish_y)
        heading_hist.append(heading)
        cls_hist.append(predicted_class)
        approach_hist.append(approach_gain)
        speed_hist.append(speed)
        F_hist.append(F_visual)
        dopa_hist.append(dopa)

        if t % 100 == 0:
            cls_name = CLASS_NAMES[predicted_class]
            print(f"  t={t:4d}  pos=({fish_x:+6.1f},{fish_y:+6.1f})  "
                  f"cls={cls_name:10s}  approach={approach_gain:+.2f}  "
                  f"speed={speed:.2f}  eaten={total_eaten}  fled={total_fled}")

    # === SUMMARY ===
    dist_traveled = sum(
        math.sqrt((pos_x[i] - pos_x[i - 1]) ** 2 +
                   (pos_y[i] - pos_y[i - 1]) ** 2)
        for i in range(1, T)
    )

    # Classification distribution
    cls_counts = [cls_hist.count(i) for i in range(5)]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Food eaten:      {total_eaten}")
    print(f"  Flee events:     {len(flee_times)}")
    print(f"  Distance:        {dist_traveled:.0f}")
    print(f"  Efficiency:      {total_eaten / (dist_traveled / 1000 + 0.01):.1f} food/km")
    print(f"  Classification distribution:")
    for i, name in enumerate(CLASS_NAMES):
        pct = 100 * cls_counts[i] / T
        print(f"    {name:12s}: {cls_counts[i]:4d} ({pct:.1f}%)")
    print(f"{'='*60}")

    # === PLOT: 3x2 panels ===
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # Panel 1: Trajectory with entity markers
    ax = axes[0, 0]
    # Color trajectory by classification
    for i in range(1, len(pos_x)):
        c = {CLASS_NOTHING: "gray", CLASS_FOOD: "green", CLASS_ENEMY: "red",
             CLASS_COLLEAGUE: "blue", CLASS_ENVIRONMENT: "brown"}[cls_hist[i]]
        ax.plot([pos_x[i - 1], pos_x[i]], [pos_y[i - 1], pos_y[i]],
                color=c, alpha=0.5, linewidth=1)
    ax.plot(pos_x[0], pos_y[0], "ko", markersize=10, label="Start", zorder=6)
    ax.plot(pos_x[-1], pos_y[-1], "ks", markersize=10, label="End", zorder=6)
    # Mark food/enemy/colleague positions
    for food in world.foods:
        fx, fy = (food["x"], food["y"]) if isinstance(food, dict) else (food[0], food[1])
        ax.plot(fx, fy, "g^", markersize=6, alpha=0.4)
    for (ex, ey) in world.enemies:
        ax.plot(ex, ey, "rv", markersize=8, alpha=0.6)
    for (cx, cy) in world.colleagues:
        ax.plot(cx, cy, "bs", markersize=6, alpha=0.4)
    for i, et in enumerate(eaten_times):
        ax.plot(pos_x[et], pos_y[et], "y*", markersize=14,
                zorder=7, label="Eat" if i == 0 else None)
    for i, ft in enumerate(flee_times):
        ax.plot(pos_x[ft], pos_y[ft], "rx", markersize=10,
                zorder=7, label="Flee" if i == 0 else None)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_title(f"Trajectory (eaten:{total_eaten}, fled:{len(flee_times)})")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")

    # Panel 2: Classification over time (stacked area)
    ax = axes[0, 1]
    cls_onehot = np.zeros((T, 5))
    for i, c in enumerate(cls_hist):
        cls_onehot[i, c] = 1
    # Smooth for visibility
    window = 20
    cls_smooth = np.zeros((T, 5))
    for i in range(T):
        start = max(0, i - window)
        cls_smooth[i] = cls_onehot[start:i + 1].mean(axis=0)
    colors = ["gray", "green", "red", "blue", "brown"]
    ax.stackplot(range(T), cls_smooth.T, labels=CLASS_NAMES,
                 colors=colors, alpha=0.7)
    ax.set_ylabel("Classification (smoothed)")
    ax.set_xlabel("Time step")
    ax.set_title("What the Fish Sees")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1)

    # Panel 3: Approach/flee gain over time
    ax = axes[1, 0]
    ax.plot(approach_hist, color="darkgreen", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.fill_between(range(T), approach_hist, 0,
                     where=[a > 0 for a in approach_hist],
                     color="green", alpha=0.2, label="Approach")
    ax.fill_between(range(T), approach_hist, 0,
                     where=[a < 0 for a in approach_hist],
                     color="red", alpha=0.2, label="Flee")
    ax.set_ylabel("Approach Gain")
    ax.set_xlabel("Time step")
    ax.set_title("Approach (+) / Flee (-)")
    ax.legend(fontsize=8)
    for et in eaten_times:
        ax.axvline(et, color="gold", linewidth=1, alpha=0.4)

    # Panel 4: Speed over time
    ax = axes[1, 1]
    ax.plot(speed_hist, color="steelblue", alpha=0.7)
    ax.set_ylabel("Swim Speed")
    ax.set_xlabel("Time step")
    ax.set_title("Speed (classification-modulated)")
    for ft in flee_times:
        ax.axvline(ft, color="red", linewidth=1, alpha=0.3)
    for et in eaten_times:
        ax.axvline(et, color="gold", linewidth=1, alpha=0.3)

    # Panel 5: Dopamine + Free Energy
    ax = axes[2, 0]
    ax.plot(F_hist, color="steelblue", label="Free Energy", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(dopa_hist, color="orange", alpha=0.7, label="Dopamine")
    ax.set_ylabel("Free Energy")
    ax2.set_ylabel("Dopamine", color="orange")
    ax.set_xlabel("Time step")
    ax.set_title("Free Energy & Dopamine")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # Panel 6: Performance bar chart
    ax = axes[2, 1]
    labels = ["Food\nEaten", "Flee\nEvents", "Efficiency\n(food/km)"]
    eff = total_eaten / (dist_traveled / 1000 + 0.01)
    values = [total_eaten, len(flee_times), eff]
    colors_bar = ["green", "red", "steelblue"]
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Performance Summary")
    ax.set_ylabel("Count / Metric")

    fig.suptitle("Step 12: Classification-Guided Behavior",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                              "v60_step12_classification_guided_behavior.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


if __name__ == "__main__":
    run_step12()
