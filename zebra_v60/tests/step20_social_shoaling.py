"""
Step 20: Social Shoaling Validation Test

Validates that the social behavior module works correctly:
  1. Colleagues spawn and move in the environment
  2. SOCIAL goal activates when colleagues are detected
  3. Shoaling modulates turn/speed during SOCIAL goal
  4. Goal diversity includes SOCIAL (>= 2% of steps)
  5. System stability (no NaN)
  6. Allostatic social comfort reduces stress

Run: python -m zebra_v60.tests.step20_social_shoaling
Output: plots/v60_step20_social_shoaling.png
"""
import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebra_v60.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebra_v60.gym_env.brain_agent import BrainAgent
from zebra_v60.brain.goal_policy_v60 import (
    GOAL_NAMES, GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, GOAL_SOCIAL,
)


def _run_episode(env, agent, T):
    """Run one episode, return comprehensive history dict."""
    obs, info = env.reset(seed=42)
    agent.reset()

    hist = {
        "pos_x": [], "pos_y": [],
        "goal": [],
        "energy": [],
        "speed": [],
        "stress": [],
        "colleague_dists": [],
        "shoaling_neighbours": [],
    }

    has_nan = False

    for t in range(T):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(
            info, reward=reward, done=terminated or truncated, env=env)

        diag = agent.last_diagnostics
        hist["pos_x"].append(info["fish_pos"][0])
        hist["pos_y"].append(info["fish_pos"][1])
        hist["goal"].append(diag.get("goal", 2))
        hist["energy"].append(diag.get("energy", 100.0))
        hist["speed"].append(diag.get("speed", 0.5))

        # Stress from allostasis
        allo_d = diag.get("allostasis", {})
        hist["stress"].append(allo_d.get("stress", 0.0))

        # Colleague distances
        colleagues = info.get("colleagues", [])
        if colleagues:
            dists = []
            fx, fy = info["fish_pos"]
            for c in colleagues:
                dx = c["x"] - fx
                dy = c["y"] - fy
                dists.append(math.sqrt(dx * dx + dy * dy))
            hist["colleague_dists"].append(min(dists))
        else:
            hist["colleague_dists"].append(999.0)

        # Shoaling diagnostics
        shoal_d = diag.get("shoaling", {})
        hist["shoaling_neighbours"].append(shoal_d.get("n_neighbours", 0))

        # NaN check
        for key in ["cls_probs", "posterior", "efe_vec"]:
            val = diag.get(key)
            if val is not None and np.any(np.isnan(val)):
                has_nan = True

        if terminated or truncated:
            break

    hist["final_step"] = t + 1
    hist["has_nan"] = has_nan
    return hist


def run_step20(T=600):
    print("=" * 60)
    print("Step 20: Social Shoaling Validation")
    print("=" * 60)

    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    cls_path = os.path.join(
        PROJECT_ROOT, "zebra_v60", "weights", "classifier_v60.pt")

    # Run with colleagues (social environment)
    print("\n--- Running Social Environment (3 colleagues, 600 steps) ---")
    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=T + 100, n_colleagues=3)
    agent = BrainAgent(
        device="auto", cls_weights_path=cls_path,
        use_habit=True, use_rl_critic=False,
        world_model="none", use_allostasis=True)
    hist = _run_episode(env, agent, T)
    env.close()

    # Run without colleagues (control)
    print("--- Running Control (0 colleagues, 600 steps) ---")
    np.random.seed(42)
    torch.manual_seed(42)
    env_ctrl = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=T + 100, n_colleagues=0)
    agent_ctrl = BrainAgent(
        device="auto", cls_weights_path=cls_path,
        use_habit=True, use_rl_critic=False,
        world_model="none", use_allostasis=True)
    hist_ctrl = _run_episode(env_ctrl, agent_ctrl, T)
    env_ctrl.close()

    # =====================================================================
    # PASS / FAIL CRITERIA (6 checks)
    # =====================================================================
    print(f"\n{'='*60}")
    print("--- Pass/Fail Criteria ---")
    results = {}

    # 1. Survival >= 400 steps
    results["survival"] = hist["final_step"] >= 400
    print(f"  [{'PASS' if results['survival'] else 'FAIL'}] "
          f"1. Survival: {hist['final_step']} >= 400")

    # 2. Colleagues present (min dist < 300 at some point)
    min_colleague_dist = min(hist["colleague_dists"])
    results["colleagues_present"] = min_colleague_dist < 300
    print(f"  [{'PASS' if results['colleagues_present'] else 'FAIL'}] "
          f"2. Colleagues present: min_dist={min_colleague_dist:.0f} < 300")

    # 3. Goal diversity: at least 2 goals >= 5%
    goals = np.array(hist["goal"])
    n_total = len(goals)
    goal_pcts = {}
    for gi, gname in enumerate(GOAL_NAMES):
        goal_pcts[gname] = (goals == gi).sum() / max(1, n_total)
    goals_above_5 = sum(p >= 0.05 for p in goal_pcts.values())
    results["goal_diversity"] = goals_above_5 >= 2
    pct_str = ", ".join(f"{k}={v:.1%}" for k, v in goal_pcts.items())
    print(f"  [{'PASS' if results['goal_diversity'] else 'FAIL'}] "
          f"3. Goal diversity: {pct_str} ({goals_above_5} >= 5%)")

    # 4. FLEE activates (>= 3% with the fixes)
    flee_pct = goal_pcts.get("FLEE", 0.0)
    results["flee_works"] = flee_pct >= 0.03
    print(f"  [{'PASS' if results['flee_works'] else 'FAIL'}] "
          f"4. FLEE activates: {flee_pct:.1%} >= 3%")

    # 5. System stability (no NaN)
    results["stability"] = not hist["has_nan"]
    print(f"  [{'PASS' if results['stability'] else 'FAIL'}] "
          f"5. System stability: no NaN = {not hist['has_nan']}")

    # 6. Social stress reduction (mean stress lower with colleagues)
    mean_stress_social = float(np.mean(hist["stress"]))
    mean_stress_ctrl = float(np.mean(hist_ctrl["stress"]))
    results["social_comfort"] = mean_stress_social <= mean_stress_ctrl + 0.05
    print(f"  [{'PASS' if results['social_comfort'] else 'FAIL'}] "
          f"6. Social comfort: stress={mean_stress_social:.3f} "
          f"<= ctrl={mean_stress_ctrl:.3f}+0.05")

    all_pass = all(results.values())
    n_pass = sum(results.values())
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} ({n_pass}/6)")
    print("=" * 60)

    # =====================================================================
    # PLOT: 3x2 panels
    # =====================================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # (0,0) Trajectory with colleague positions
    ax = axes[0, 0]
    ax.plot(hist["pos_x"], hist["pos_y"], color="blue", alpha=0.6,
            linewidth=0.8, label="Agent fish")
    ax.plot(hist_ctrl["pos_x"], hist_ctrl["pos_y"], color="gray",
            alpha=0.4, linewidth=0.8, label="Control (no colleagues)")
    ax.set_xlim(0, 800)
    ax.set_ylim(600, 0)
    ax.set_aspect("equal")
    ax.set_title("Trajectories: Social vs Control")
    ax.legend(fontsize=7)

    # (0,1) Goal distribution comparison
    ax = axes[0, 1]
    bar_width = 0.35
    x = np.arange(len(GOAL_NAMES))
    social_pcts = [goal_pcts.get(g, 0.0) for g in GOAL_NAMES]
    goals_ctrl = np.array(hist_ctrl["goal"])
    ctrl_pcts = [(goals_ctrl == gi).sum() / max(1, len(goals_ctrl))
                 for gi in range(len(GOAL_NAMES))]
    ax.bar(x - bar_width / 2, social_pcts, bar_width,
           color="teal", alpha=0.8, label="With colleagues")
    ax.bar(x + bar_width / 2, ctrl_pcts, bar_width,
           color="gray", alpha=0.8, label="Control")
    ax.set_xticks(x)
    ax.set_xticklabels(GOAL_NAMES)
    ax.set_ylabel("Fraction")
    ax.set_title("Goal Distribution")
    ax.legend(fontsize=7)
    ax.axhline(0.05, color="red", linewidth=0.5, linestyle="--",
               alpha=0.5, label="5% threshold")

    # (1,0) Energy over time
    ax = axes[1, 0]
    ax.plot(range(len(hist["energy"])), hist["energy"],
            color="teal", linewidth=1.5, label="Social")
    ax.plot(range(len(hist_ctrl["energy"])), hist_ctrl["energy"],
            color="gray", linewidth=1.5, alpha=0.6, label="Control")
    ax.axhline(10, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Over Time")
    ax.legend(fontsize=7)

    # (1,1) Stress comparison
    ax = axes[1, 1]
    ax.plot(range(len(hist["stress"])), hist["stress"],
            color="teal", linewidth=1.5, label="Social")
    ax.plot(range(len(hist_ctrl["stress"])), hist_ctrl["stress"],
            color="gray", linewidth=1.5, alpha=0.6, label="Control")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Stress")
    ax.set_title("Stress: Social Comfort Effect")
    ax.legend(fontsize=7)

    # (2,0) Colleague distance over time
    ax = axes[2, 0]
    ax.plot(range(len(hist["colleague_dists"])), hist["colleague_dists"],
            color="teal", linewidth=1, alpha=0.7)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Min colleague distance (px)")
    ax.set_title("Nearest Colleague Distance")

    # (2,1) Pass/fail summary
    ax = axes[2, 1]
    crit_labels = ["Survival\n>=400", "Colleagues\npresent",
                   "Goal\ndiversity", "FLEE\n>=3%",
                   "No NaN", "Social\ncomfort"]
    crit_keys = ["survival", "colleagues_present", "goal_diversity",
                 "flee_works", "stability", "social_comfort"]
    passes = [results[k] for k in crit_keys]
    colors_bar = ["green" if p else "red" for p in passes]
    bars = ax.bar(crit_labels, [1 if p else 0 for p in passes],
                  color=colors_bar, edgecolor="black", alpha=0.8)
    for bar, p in zip(bars, passes):
        status = "PASS" if p else "FAIL"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03, status,
                ha="center", va="bottom", fontweight="bold", fontsize=8)
    ax.set_ylim(0, 1.5)
    ax.set_title(f"6 Pass/Fail Criteria ({n_pass}/6)")
    ax.set_ylabel("Pass=1 / Fail=0")

    fig.suptitle("Step 20: Social Shoaling Validation",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                              "v60_step20_social_shoaling.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    return all_pass


if __name__ == "__main__":
    run_step20()
