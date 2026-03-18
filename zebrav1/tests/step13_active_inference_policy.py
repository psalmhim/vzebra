"""
Step 13: Active Inference Policy Selection

Transitions from reactive (step 12) to deliberative behavior via Expected
Free Energy (EFE)-based policy selection. The fish maintains three goal
states (FORAGE, FLEE, EXPLORE), selects between them by minimizing EFE,
and sustains chosen goals through a persistence mechanism.

New modules:
  - GoalPolicy:    EFE computation, softmax posterior, persistence timer
  - WorkingMemory: Goal-conditioned memory trace with dopamine modulation

Run: python -m zebrav1.tests.step13_active_inference_policy
Output: plots/v1_step13_active_inference_policy.png
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

from zebrav1.brain.zebrafish_snn import ZebrafishSNN
from zebrav1.brain.dopamine import DopamineSystem
from zebrav1.brain.basal_ganglia import BasalGanglia
from zebrav1.brain.optic_tectum import OpticTectum
from zebrav1.brain.thalamus import ThalamusRelay
from zebrav1.brain.goal_policy import (
    GoalPolicy, goal_to_behavior,
    GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, GOAL_NAMES,
)
from zebrav1.brain.working_memory import WorkingMemory
from zebrav1.brain.device_util import get_device
from zebrav1.world.world_env import WorldEnv
from zebrav1.tests.step1_vision_pursuit import compute_retinal_turn, TurnSmoother

# Class indices (must match step 11)
CLASS_NOTHING = 0
CLASS_FOOD = 1
CLASS_ENEMY = 2
CLASS_COLLEAGUE = 3
CLASS_ENVIRONMENT = 4
CLASS_NAMES = ["nothing", "food", "enemy", "colleague", "environment"]


def run_step13(T=1000, swim_speed=1.5, base_turn_gain=0.15):
    print("=" * 60)
    print("Step 13: Active Inference Policy Selection")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN(device=device)

    # Load classifier weights
    cls_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights", "classifier.pt")
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

    # Existing modules (unchanged from step 12)
    dopa_sys = DopamineSystem()
    bg = BasalGanglia(mode="exploratory")
    ot = OpticTectum()
    thal = ThalamusRelay()
    smoother = TurnSmoother(alpha=0.35)

    # NEW: Active inference modules
    goal_policy = GoalPolicy(n_goals=3, beta=2.0, persist_steps=8)
    wm = WorkingMemory(n_latent=16, n_goals=3, buffer_len=20)

    # Rich world: 10 food, 4 enemies, 5 colleagues
    world = WorldEnv(xmin=-500, xmax=500, ymin=-500, ymax=500, n_food=0)
    np.random.seed(42)

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

    # History tracking
    pos_x, pos_y = [], []
    goal_hist = []           # active goal per step
    posterior_hist = []      # goal posterior per step [T, 3]
    efe_hist = []            # EFE per goal per step [T, 3]
    confidence_hist = []
    dopa_hist = []
    cls_probs_hist = []      # classifier probs per step [T, 5]
    eaten_times = []
    flee_times = []
    total_eaten = 0
    total_fled = 0
    prev_oF = None

    # Initialize goal_vec for first step
    goal_vec = np.array([0.0, 0.0, 1.0])  # start in EXPLORE

    for t in range(T):
        fish_pos = np.array([fish_x, fish_y])
        effective_heading = heading + ot.eye_pos * 0.25

        with torch.no_grad():
            out = model.forward(fish_pos, effective_heading, world)

        # Classification
        cls_logits = out["cls"]  # [1, 5]
        cls_probs = F.softmax(cls_logits, dim=1)[0].cpu().numpy()

        # Retinal turn (hardwired direction signal)
        raw_turn = compute_retinal_turn(out)

        # Free energy and neuromodulation
        F_visual = model.compute_free_energy()
        F_audio = 0.1 * abs(math.sin(0.05 * t))
        cms = thal.step(F_visual, F_audio)
        dopa_sys.beta = thal.modulate_dopamine_gain()

        oL_mean = float(out["oL"].abs().mean())
        oR_mean = float(out["oR"].abs().mean())
        eaten, _ = world.try_eat(fish_x, fish_y)
        dopa, rpe, valL, valR = dopa_sys.step(F_visual, oL_mean, oR_mean,
                                               eaten=eaten)

        pi_OT = out["pi_OT"]
        pi_PC = out["pi_PC"]

        # Working memory update
        mem_state, alpha_eff, cls_summary = wm.step(
            cls_probs, goal_vec, dopa, cms, F_visual, pi_OT, pi_PC)

        # Active inference policy selection
        choice, goal_vec, posterior, confidence, efe_vec = goal_policy.step(
            cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual, wm.get_mean())

        # Goal-conditioned behavior
        approach_gain, speed_mod, explore_mod, turn_strategy = goal_to_behavior(
            choice, cls_probs, posterior, confidence)

        # Modulated turn
        modulated_turn = raw_turn * approach_gain
        turn = smoother.step(modulated_turn)

        valL_eff = valL - 0.1 * turn
        valR_eff = valR + 0.1 * turn

        # BG exploration modulated by goal
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

        # Eat food → respawn
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

        # Enemy proximity → flee event
        enemy_dist = world.check_enemy_proximity(fish_x, fish_y, danger_radius=35)
        if enemy_dist is not None:
            total_fled += 1
            if len(flee_times) == 0 or t - flee_times[-1] > 10:
                flee_times.append(t)

        # Move enemies (patrol)
        if t % 5 == 0:
            new_enemies = []
            for (ex, ey) in world.enemies:
                ex += np.random.uniform(-3, 3)
                ey += np.random.uniform(-3, 3)
                ex = max(world.xmin + 20, min(world.xmax - 20, ex))
                ey = max(world.ymin + 20, min(world.ymax - 20, ey))
                new_enemies.append((ex, ey))
            world.enemies = new_enemies

        # Move colleagues (schooling drift)
        if t % 3 == 0:
            new_colleagues = []
            for (cx, cy) in world.colleagues:
                dx = fish_x - cx
                dy = fish_y - cy
                dist = math.sqrt(dx * dx + dy * dy) + 1e-8
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

        # Record history
        pos_x.append(fish_x)
        pos_y.append(fish_y)
        goal_hist.append(choice)
        posterior_hist.append(posterior.copy())
        efe_hist.append(efe_vec.copy())
        confidence_hist.append(confidence)
        dopa_hist.append(dopa)
        cls_probs_hist.append(cls_probs.copy())

        if t % 100 == 0:
            goal_name = GOAL_NAMES[choice]
            pred_cls = CLASS_NAMES[int(cls_logits.argmax(dim=1).item())]
            print(f"  t={t:4d}  pos=({fish_x:+6.1f},{fish_y:+6.1f})  "
                  f"goal={goal_name:8s}  cls={pred_cls:10s}  "
                  f"conf={confidence:.2f}  dopa={dopa:.2f}  "
                  f"eaten={total_eaten}  fled={total_fled}")

    # === SUMMARY ===
    goal_counts = [goal_hist.count(g) for g in range(3)]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Food eaten:       {total_eaten}")
    print(f"  Flee events:      {len(flee_times)}")
    print(f"  Goal distribution:")
    for g in range(3):
        pct = 100 * goal_counts[g] / T
        print(f"    {GOAL_NAMES[g]:10s}: {goal_counts[g]:4d} ({pct:.1f}%)")

    # Count goal switches
    goal_switches = sum(1 for i in range(1, T) if goal_hist[i] != goal_hist[i-1])
    print(f"  Goal switches:    {goal_switches}")

    # Average goal episode length
    episodes = []
    ep_len = 1
    for i in range(1, T):
        if goal_hist[i] == goal_hist[i-1]:
            ep_len += 1
        else:
            episodes.append(ep_len)
            ep_len = 1
    episodes.append(ep_len)
    avg_episode = np.mean(episodes) if episodes else 0

    print(f"  Avg episode len:  {avg_episode:.1f}")
    print(f"  Avg confidence:   {np.mean(confidence_hist):.3f}")
    print(f"{'='*60}")

    # === PASS/FAIL CRITERIA ===
    print("\n--- Pass/Fail Criteria ---")
    results = {}

    # 1. Goal switching: >=5 switches
    results["goal_switching"] = goal_switches >= 5
    print(f"  [{'PASS' if results['goal_switching'] else 'FAIL'}] "
          f"Goal switching: {goal_switches} >= 5")

    # 2. Goal persistence: avg >=5 steps per episode
    results["goal_persistence"] = avg_episode >= 5
    print(f"  [{'PASS' if results['goal_persistence'] else 'FAIL'}] "
          f"Goal persistence: avg {avg_episode:.1f} >= 5 steps")

    # 3. Goal-context alignment: FORAGE when food or FLEE when enemy (>30%)
    aligned = 0
    context_count = 0
    for i in range(T):
        cp = cls_probs_hist[i]
        if cp[CLASS_FOOD] > 0.3:
            context_count += 1
            if goal_hist[i] == GOAL_FORAGE:
                aligned += 1
        elif cp[CLASS_ENEMY] > 0.3:
            context_count += 1
            if goal_hist[i] == GOAL_FLEE:
                aligned += 1
    alignment_pct = aligned / max(context_count, 1)
    results["goal_alignment"] = alignment_pct > 0.30
    print(f"  [{'PASS' if results['goal_alignment'] else 'FAIL'}] "
          f"Goal-context alignment: {alignment_pct:.1%} > 30% "
          f"({aligned}/{context_count})")

    # 4. Exploration during "nothing" periods (>20%)
    #    Include both nothing AND environment — classifier uses environment
    #    for empty/featureless areas
    nothing_steps = sum(
        1 for i in range(T)
        if (cls_probs_hist[i][CLASS_NOTHING]
            + cls_probs_hist[i][CLASS_ENVIRONMENT]) > 0.3)
    explore_during_nothing = sum(
        1 for i in range(T)
        if (cls_probs_hist[i][CLASS_NOTHING]
            + cls_probs_hist[i][CLASS_ENVIRONMENT]) > 0.3
        and goal_hist[i] == GOAL_EXPLORE)
    nothing_explore_pct = explore_during_nothing / max(nothing_steps, 1)
    results["explore_nothing"] = nothing_explore_pct > 0.20
    print(f"  [{'PASS' if results['explore_nothing'] else 'FAIL'}] "
          f"Exploration during nothing: {nothing_explore_pct:.1%} > 20% "
          f"({explore_during_nothing}/{nothing_steps})")

    # 5. System stability: no NaN
    all_arrays = (posterior_hist + efe_hist + confidence_hist + dopa_hist)
    has_nan = any(np.any(np.isnan(v)) for v in all_arrays
                  if isinstance(v, np.ndarray))
    has_nan = has_nan or any(math.isnan(v) for v in all_arrays
                             if isinstance(v, (int, float)))
    results["stability"] = not has_nan
    print(f"  [{'PASS' if results['stability'] else 'FAIL'}] "
          f"System stability: no NaN")

    # 6. Foraging: >=3 food eaten
    results["foraging"] = total_eaten >= 3
    print(f"  [{'PASS' if results['foraging'] else 'FAIL'}] "
          f"Foraging: {total_eaten} >= 3")

    all_pass = all(results.values())
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} "
          f"({sum(results.values())}/6)")
    print("=" * 60)

    # === PLOT: 3x2 panels ===
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    posterior_arr = np.array(posterior_hist)  # [T, 3]
    efe_arr = np.array(efe_hist)             # [T, 3]
    cls_arr = np.array(cls_probs_hist)       # [T, 5]

    # Panel (0,0): Trajectory colored by goal
    ax = axes[0, 0]
    goal_colors = {GOAL_FORAGE: "green", GOAL_FLEE: "red", GOAL_EXPLORE: "blue"}
    for i in range(1, len(pos_x)):
        c = goal_colors[goal_hist[i]]
        ax.plot([pos_x[i-1], pos_x[i]], [pos_y[i-1], pos_y[i]],
                color=c, alpha=0.5, linewidth=1)
    ax.plot(pos_x[0], pos_y[0], "ko", markersize=10, label="Start", zorder=6)
    ax.plot(pos_x[-1], pos_y[-1], "ks", markersize=10, label="End", zorder=6)
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
    ax.set_title(f"Trajectory by Goal (eaten:{total_eaten}, fled:{len(flee_times)})")
    ax.set_aspect("equal")
    # Legend for goal colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="FORAGE"),
        Line2D([0], [0], color="red", lw=2, label="FLEE"),
        Line2D([0], [0], color="blue", lw=2, label="EXPLORE"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    # Panel (0,1): Goal posterior stacked area
    ax = axes[0, 1]
    ax.stackplot(range(T),
                 posterior_arr[:, GOAL_FORAGE],
                 posterior_arr[:, GOAL_FLEE],
                 posterior_arr[:, GOAL_EXPLORE],
                 labels=["FORAGE", "FLEE", "EXPLORE"],
                 colors=["green", "red", "blue"], alpha=0.7)
    ax.set_ylabel("Goal Posterior")
    ax.set_xlabel("Time step")
    ax.set_title("Goal Posterior (stacked)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1)

    # Panel (1,0): EFE per goal over time
    ax = axes[1, 0]
    ax.plot(efe_arr[:, GOAL_FORAGE], color="green", alpha=0.7, label="G_forage")
    ax.plot(efe_arr[:, GOAL_FLEE], color="red", alpha=0.7, label="G_flee")
    ax.plot(efe_arr[:, GOAL_EXPLORE], color="blue", alpha=0.7, label="G_explore")
    ax.set_ylabel("EFE (lower = preferred)")
    ax.set_xlabel("Time step")
    ax.set_title("Expected Free Energy per Goal")
    ax.legend(fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    # Panel (1,1): Confidence + dopamine dual-axis
    ax = axes[1, 1]
    ax.plot(confidence_hist, color="purple", alpha=0.7, label="Confidence")
    ax.set_ylabel("Confidence", color="purple")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(dopa_hist, color="orange", alpha=0.7, label="Dopamine")
    ax2.set_ylabel("Dopamine", color="orange")
    ax2.set_ylim(0, 1)
    ax.set_xlabel("Time step")
    ax.set_title("Confidence & Dopamine")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # Panel (2,0): Classification stacked area (what fish sees)
    ax = axes[2, 0]
    # Smooth for visibility
    window = 20
    cls_smooth = np.zeros((T, 5))
    for i in range(T):
        start = max(0, i - window)
        cls_smooth[i] = cls_arr[start:i+1].mean(axis=0)
    colors_cls = ["gray", "green", "red", "blue", "brown"]
    ax.stackplot(range(T), cls_smooth.T, labels=CLASS_NAMES,
                 colors=colors_cls, alpha=0.7)
    ax.set_ylabel("Classification (smoothed)")
    ax.set_xlabel("Time step")
    ax.set_title("What the Fish Sees")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1)

    # Panel (2,1): Performance bars
    ax = axes[2, 1]
    labels = ["Food\nEaten", "Flee\nEvents", "Goal\nSwitches",
              "Avg\nEpisode", "Alignment\n%", "Explore\nNothing%"]
    values = [total_eaten, len(flee_times), goal_switches,
              avg_episode, alignment_pct * 100, nothing_explore_pct * 100]
    colors_bar = ["green", "red", "steelblue", "purple", "orange", "blue"]
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontweight="bold", fontsize=8)
    ax.set_title("Performance Summary")
    ax.set_ylabel("Count / Metric")

    fig.suptitle("Step 13: Active Inference Policy Selection",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                              "v1_step13_active_inference_policy.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")

    return all_pass


if __name__ == "__main__":
    run_step13()
