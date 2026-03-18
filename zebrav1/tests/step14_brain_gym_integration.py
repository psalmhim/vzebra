"""
Step 14: Brain-Gym Integration

Bridges the deliberative zebrafish brain (Steps 1-13) with the Gymnasium
environment. Adds 5 new features: hunger mechanic, 1D vision strip rendering,
depth-shaded vision, obstacle collision physics, and habit network shortcut.

Run: python -m zebrav1.tests.step14_brain_gym_integration
Output: plots/v1_step14_brain_gym_integration.png
"""
import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav1.gym_env.brain_agent import BrainAgent
from zebrav1.brain.goal_policy import GOAL_NAMES, GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE


def run_step14(T=600):
    print("=" * 60)
    print("Step 14: Brain-Gym Integration")
    print("=" * 60)

    # Deterministic seeding for all random sources
    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    # Create environment and agent
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=T + 100)

    cls_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights", "classifier.pt")
    agent = BrainAgent(device="auto", cls_weights_path=cls_path, use_habit=True)

    obs, info = env.reset(seed=42)
    agent.reset()

    # History tracking
    pos_x, pos_y = [], []
    goal_hist = []
    posterior_hist = []
    confidence_hist = []
    dopa_hist = []
    energy_hist = []
    reward_hist = []
    cumulative_reward = []
    eaten_times = []
    shortcut_times = []
    retL_hist = []
    retR_hist = []

    total_eaten = 0
    cum_reward = 0.0
    obstacle_penetrations = 0
    vision_max = 0.0
    has_nan = False

    print(f"  Obstacles: {len(env.obstacles)}")
    print(f"  Initial energy: {env.fish_energy:.1f}")

    for t in range(T):
        # Agent produces action
        action = agent.act(obs, env)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Feed back actual eaten count
        agent.update_post_step(info, env=env)

        # Track diagnostics
        diag = agent.last_diagnostics
        pos_x.append(info["fish_pos"][0])
        pos_y.append(info["fish_pos"][1])
        goal_hist.append(diag.get("goal", 2))
        posterior_hist.append(diag.get("posterior", np.array([0.33, 0.33, 0.34])))
        confidence_hist.append(diag.get("confidence", 0.0))
        dopa_hist.append(diag.get("dopa", 0.5))
        energy_hist.append(info.get("fish_energy", 0.0))
        reward_hist.append(reward)
        cum_reward += reward
        cumulative_reward.append(cum_reward)

        # Food eaten
        eaten_this = info.get("food_eaten_this_step", 0)
        if eaten_this > 0:
            total_eaten += eaten_this
            eaten_times.append(t)

        # Habit shortcuts
        if diag.get("shortcut_active", False):
            shortcut_times.append(t)

        # Vision strip max
        retL_max = diag.get("retL_max", 0.0)
        retR_max = diag.get("retR_max", 0.0)
        vision_max = max(vision_max, retL_max, retR_max)

        # Retinal history (subsample for heatmap)
        if "retL_intensity" in diag:
            # Subsample to 20 bins for the heatmap
            retL_full = diag["retL_intensity"]
            retR_full = diag["retR_intensity"]
            n = len(retL_full)
            bin_size = max(1, n // 20)
            retL_binned = np.array([retL_full[i:i+bin_size].mean()
                                    for i in range(0, n, bin_size)])[:20]
            retR_binned = np.array([retR_full[i:i+bin_size].mean()
                                    for i in range(0, n, bin_size)])[:20]
            retL_hist.append(retL_binned)
            retR_hist.append(retR_binned)

        # Check obstacle penetration
        fish_r = env.fish_size * 0.5
        for obs_item in env.obstacles:
            dx = info["fish_pos"][0] - obs_item["x"]
            dy = info["fish_pos"][1] - obs_item["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < obs_item["r"] - fish_r:
                obstacle_penetrations += 1

        # NaN check
        for key in ["cls_probs", "posterior", "efe_vec"]:
            val = diag.get(key)
            if val is not None and np.any(np.isnan(val)):
                has_nan = True
        for key in ["dopa", "rpe", "F_visual", "confidence"]:
            val = diag.get(key)
            if val is not None and math.isnan(val):
                has_nan = True

        # Progress log
        if t % 100 == 0:
            goal_name = GOAL_NAMES[diag.get("goal", 2)]
            sc = "H" if diag.get("shortcut_active", False) else "E"
            print(f"  t={t:4d}  pos=({info['fish_pos'][0]:6.1f},"
                  f"{info['fish_pos'][1]:6.1f})  "
                  f"goal={goal_name:8s}[{sc}]  "
                  f"energy={info['fish_energy']:5.1f}  "
                  f"dopa={diag.get('dopa', 0):.2f}  "
                  f"eaten={total_eaten}  "
                  f"reward={cum_reward:.1f}")

        if terminated or truncated:
            print(f"  Episode ended at step {t}: "
                  f"terminated={terminated}, truncated={truncated}")
            break

    final_step = t + 1
    habit_stats = agent.habit.get_stats()

    # === SUMMARY ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Steps survived:     {final_step}")
    print(f"  Food eaten:         {total_eaten}")
    print(f"  Final energy:       {energy_hist[-1]:.1f}")
    print(f"  Cumulative reward:  {cum_reward:.1f}")
    print(f"  Obstacle penetrations: {obstacle_penetrations}")
    print(f"  Vision max:         {vision_max:.4f}")
    print(f"  Habit shortcuts:    {habit_stats['total_habit_uses']}")
    print(f"  Habit rate:         {habit_stats['habit_rate']:.1%}")
    print(f"  Any NaN:            {has_nan}")
    print(f"{'='*60}")

    # === PASS/FAIL CRITERIA (7 checks) ===
    print("\n--- Pass/Fail Criteria ---")
    results = {}

    # 1. Survival: episode >= 300 steps
    results["survival"] = final_step >= 300
    print(f"  [{'PASS' if results['survival'] else 'FAIL'}] "
          f"Survival: {final_step} >= 300 steps")

    # 2. Foraging: >= 3 food eaten
    results["foraging"] = total_eaten >= 3
    print(f"  [{'PASS' if results['foraging'] else 'FAIL'}] "
          f"Foraging: {total_eaten} >= 3")

    # 3. Energy management: energy > 0 at step 200
    energy_at_200 = energy_hist[min(199, len(energy_hist) - 1)]
    results["energy_mgmt"] = energy_at_200 > 0
    print(f"  [{'PASS' if results['energy_mgmt'] else 'FAIL'}] "
          f"Energy at step 200: {energy_at_200:.1f} > 0")

    # 4. Collision physics: 0 illegal obstacle penetrations
    results["collision"] = obstacle_penetrations == 0
    print(f"  [{'PASS' if results['collision'] else 'FAIL'}] "
          f"Collision physics: {obstacle_penetrations} penetrations")

    # 5. Vision strip: retinal data max > 0.01
    results["vision"] = vision_max > 0.01
    print(f"  [{'PASS' if results['vision'] else 'FAIL'}] "
          f"Vision strip: max={vision_max:.4f} > 0.01")

    # 6. Habit activations: >= 1 habit shortcut used
    results["habit"] = habit_stats["total_habit_uses"] >= 1
    print(f"  [{'PASS' if results['habit'] else 'FAIL'}] "
          f"Habit shortcuts: {habit_stats['total_habit_uses']} >= 1")

    # 7. System stability: no NaN
    results["stability"] = not has_nan
    print(f"  [{'PASS' if results['stability'] else 'FAIL'}] "
          f"System stability: no NaN = {not has_nan}")

    all_pass = all(results.values())
    n_pass = sum(results.values())
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} ({n_pass}/7)")
    print("=" * 60)

    # === PLOT: 3x2 panels ===
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    steps = range(len(pos_x))
    posterior_arr = np.array(posterior_hist)

    # Panel (0,0): Trajectory colored by goal + obstacles
    ax = axes[0, 0]
    goal_colors = {GOAL_FORAGE: "green", GOAL_FLEE: "red", GOAL_EXPLORE: "blue"}
    for i in range(1, len(pos_x)):
        c = goal_colors.get(goal_hist[i], "gray")
        ax.plot([pos_x[i-1], pos_x[i]], [pos_y[i-1], pos_y[i]],
                color=c, alpha=0.5, linewidth=1)
    ax.plot(pos_x[0], pos_y[0], "ko", markersize=10, label="Start", zorder=6)
    if len(pos_x) > 1:
        ax.plot(pos_x[-1], pos_y[-1], "ks", markersize=10, label="End", zorder=6)
    for i, et in enumerate(eaten_times):
        if et < len(pos_x):
            ax.plot(pos_x[et], pos_y[et], "y*", markersize=14,
                    zorder=7, label="Eat" if i == 0 else None)
    # Draw obstacles
    for obs_item in env.obstacles:
        circle = plt.Circle((obs_item["x"], obs_item["y"]), obs_item["r"],
                             color="brown", alpha=0.4, zorder=5)
        ax.add_patch(circle)
    ax.set_xlim(0, env.arena_w)
    ax.set_ylim(env.arena_h, 0)  # y-down for gym coords
    ax.set_title(f"Trajectory (eaten:{total_eaten}, steps:{final_step})")
    ax.set_aspect("equal")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="FORAGE"),
        Line2D([0], [0], color="red", lw=2, label="FLEE"),
        Line2D([0], [0], color="blue", lw=2, label="EXPLORE"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    # Panel (0,1): Energy over time + food-eat events
    ax = axes[0, 1]
    ax.plot(steps, energy_hist, color="green", linewidth=1.5, label="Energy")
    ax.axhline(20, color="orange", linestyle="--", alpha=0.5, label="Low threshold")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="Starvation")
    for et in eaten_times:
        ax.axvline(et, color="gold", alpha=0.3, linewidth=1)
    ax.set_ylabel("Energy")
    ax.set_xlabel("Time step")
    ax.set_title("Energy Over Time (gold lines = food eaten)")
    ax.legend(fontsize=7)
    ax.set_ylim(-5, 105)

    # Panel (1,0): Retinal vision strip heatmap (L/R eyes over time)
    ax = axes[1, 0]
    if len(retL_hist) > 0:
        retL_arr = np.array(retL_hist)
        retR_arr = np.array(retR_hist)
        combined = np.concatenate([retL_arr, retR_arr], axis=1)  # [T, 40]
        ax.imshow(combined.T, aspect="auto", cmap="hot",
                  extent=[0, len(retL_hist), 40, 0])
        ax.axhline(20, color="white", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Retinal pixel (L:0-19, R:20-39)")
        ax.set_xlabel("Time step")
        ax.set_title("Retinal Vision Strip (L/R eyes)")
    else:
        ax.text(0.5, 0.5, "No retinal data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Retinal Vision Strip")

    # Panel (1,1): Goal posterior stacked area + habit shortcuts marked
    ax = axes[1, 1]
    if len(posterior_arr) > 0:
        ax.stackplot(steps,
                     posterior_arr[:, GOAL_FORAGE],
                     posterior_arr[:, GOAL_FLEE],
                     posterior_arr[:, GOAL_EXPLORE],
                     labels=["FORAGE", "FLEE", "EXPLORE"],
                     colors=["green", "red", "blue"], alpha=0.7)
        for st in shortcut_times:
            ax.axvline(st, color="gold", alpha=0.4, linewidth=1)
    ax.set_ylabel("Goal Posterior")
    ax.set_xlabel("Time step")
    ax.set_title("Goal Posterior + Habit Shortcuts (gold)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1)

    # Panel (2,0): Cumulative reward + dopamine
    ax = axes[2, 0]
    ax.plot(steps, cumulative_reward, color="steelblue", linewidth=1.5,
            label="Cum. Reward")
    ax.set_ylabel("Cumulative Reward", color="steelblue")
    ax2 = ax.twinx()
    ax2.plot(steps, dopa_hist, color="orange", alpha=0.7, linewidth=1,
             label="Dopamine")
    ax2.set_ylabel("Dopamine", color="orange")
    ax2.set_ylim(0, 1)
    ax.set_xlabel("Time step")
    ax.set_title("Cumulative Reward & Dopamine")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # Panel (2,1): Performance bars (7 criteria)
    ax = axes[2, 1]
    labels = ["Survival\n>=300", "Foraging\n>=3", "Energy\n@200",
              "Collision\n=0", "Vision\n>0.01", "Habit\n>=1", "No NaN"]
    passes = [results["survival"], results["foraging"], results["energy_mgmt"],
              results["collision"], results["vision"], results["habit"],
              results["stability"]]
    colors_bar = ["green" if p else "red" for p in passes]
    values = [final_step, total_eaten, energy_at_200,
              obstacle_penetrations, vision_max,
              habit_stats["total_habit_uses"], 1 if not has_nan else 0]
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="black", alpha=0.8)
    for bar, val, p in zip(bars, values, passes):
        label_text = f"{val:.2f}" if isinstance(val, float) else str(val)
        status = "PASS" if p else "FAIL"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{label_text}\n{status}",
                ha="center", va="bottom", fontweight="bold", fontsize=7)
    ax.set_title("7 Pass/Fail Criteria")
    ax.set_ylabel("Value")

    fig.suptitle("Step 14: Brain-Gym Integration",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                              "v1_step14_brain_gym_integration.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")

    env.close()
    return all_pass


if __name__ == "__main__":
    run_step14()
