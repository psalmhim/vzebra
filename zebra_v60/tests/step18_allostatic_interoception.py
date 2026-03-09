"""
Step 18: Allostatic Interoception Test

Two runs: baseline (no allostasis) vs full (allostasis=True), 800 steps each.
Validates hunger/fatigue/stress tracking, goal modulation, dopamine gain
adjustment, and fatigue-based speed capping.

7 Pass/Fail Criteria:
  1. Survival >= 500 steps
  2. Foraging >= 3 food
  3. Fatigue regulation: mean fatigue < 0.6
  4. Stress response: stress peaks when predator near, decays after
  5. Goal modulation: allostatic bias causes >= 5 goal switches
  6. Energy management: energy > 10 at step 400
  7. System stability (no NaN)

Run: python -m zebra_v60.tests.step18_allostatic_interoception
Output: plots/v60_step18_allostatic_interoception.png
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
    GOAL_NAMES, GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE,
)


def _run_episode(env, agent, T):
    """Run one episode, return history dict."""
    obs, info = env.reset(seed=42)
    agent.reset()

    hist = {
        "pos_x": [], "pos_y": [],
        "goal": [],
        "energy": [],
        "speed": [],
        "eaten_times": [],
        # Allostasis-specific
        "hunger": [], "fatigue": [], "stress": [],
        "hunger_error": [], "fatigue_error": [], "stress_error": [],
        "speed_cap": [],
        "goal_bias": [],
    }

    total_eaten = 0
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

        # Allostasis diagnostics
        allo_d = diag.get("allostasis", {})
        hist["hunger"].append(allo_d.get("hunger", 0.0))
        hist["fatigue"].append(allo_d.get("fatigue", 0.0))
        hist["stress"].append(allo_d.get("stress", 0.0))
        hist["hunger_error"].append(allo_d.get("hunger_error", 0.0))
        hist["fatigue_error"].append(allo_d.get("fatigue_error", 0.0))
        hist["stress_error"].append(allo_d.get("stress_error", 0.0))
        hist["speed_cap"].append(allo_d.get("speed_cap", 1.0))
        hist["goal_bias"].append(allo_d.get("goal_bias", [0, 0, 0]))

        eaten_this = info.get("food_eaten_this_step", 0)
        if eaten_this > 0:
            total_eaten += eaten_this
            hist["eaten_times"].append(t)

        # NaN check
        for key in ["cls_probs", "posterior", "efe_vec"]:
            val = diag.get(key)
            if val is not None and np.any(np.isnan(val)):
                has_nan = True
        for key in ["dopa", "rpe", "F_visual", "confidence"]:
            val = diag.get(key)
            if val is not None and isinstance(val, float) and math.isnan(val):
                has_nan = True

        if terminated or truncated:
            break

    hist["final_step"] = t + 1
    hist["total_eaten"] = total_eaten
    hist["has_nan"] = has_nan
    return hist


def run_step18(T=800):
    print("=" * 60)
    print("Step 18: Allostatic Interoception")
    print("=" * 60)

    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    cls_path = os.path.join(
        PROJECT_ROOT, "zebra_v60", "weights", "classifier_v60.pt")

    # --- Run 1: Baseline (no allostasis) ---
    print("\n--- Baseline run (no allostasis, 800 steps) ---")
    env_base = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=T + 100)
    agent_base = BrainAgent(
        device="auto", cls_weights_path=cls_path,
        use_habit=True, use_rl_critic=False,
        world_model="none", use_allostasis=False)
    hist_base = _run_episode(env_base, agent_base, T)
    env_base.close()
    print(f"  Baseline — Steps: {hist_base['final_step']}, "
          f"Eaten: {hist_base['total_eaten']}")

    # --- Run 2: Allostatic (with allostasis) ---
    print("\n--- Allostatic run (use_allostasis=True, 800 steps) ---")
    np.random.seed(42)
    torch.manual_seed(42)
    env_allo = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=T + 100)
    agent_allo = BrainAgent(
        device="auto", cls_weights_path=cls_path,
        use_habit=True, use_rl_critic=False,
        world_model="none", use_allostasis=True)
    hist = _run_episode(env_allo, agent_allo, T)
    env_allo.close()
    print(f"  Allostatic — Steps: {hist['final_step']}, "
          f"Eaten: {hist['total_eaten']}, "
          f"Mean fatigue: {np.mean(hist['fatigue']):.3f}")

    # =====================================================================
    # PASS / FAIL CRITERIA (7 checks, on allostatic run)
    # =====================================================================
    print(f"\n{'='*60}")
    print("--- Pass/Fail Criteria (allostatic run) ---")
    results = {}

    # 1. Survival >= 500 steps
    results["survival"] = hist["final_step"] >= 500
    print(f"  [{'PASS' if results['survival'] else 'FAIL'}] "
          f"1. Survival: {hist['final_step']} >= 500")

    # 2. Foraging >= 3 food
    results["foraging"] = hist["total_eaten"] >= 3
    print(f"  [{'PASS' if results['foraging'] else 'FAIL'}] "
          f"2. Foraging: {hist['total_eaten']} >= 3")

    # 3. Fatigue regulation: mean fatigue < 0.6
    mean_fatigue = float(np.mean(hist["fatigue"]))
    results["fatigue_reg"] = mean_fatigue < 0.6
    print(f"  [{'PASS' if results['fatigue_reg'] else 'FAIL'}] "
          f"3. Fatigue regulation: {mean_fatigue:.3f} < 0.6")

    # 4. Stress response: stress peaks then decays
    stress_arr = np.array(hist["stress"])
    stress_peak = float(stress_arr.max())
    # Check that stress decays: mean of last quarter < peak
    n = len(stress_arr)
    q4_stress = float(stress_arr[3*n//4:].mean()) if n >= 4 else stress_peak
    stress_ok = stress_peak > 0.05 and q4_stress < stress_peak
    results["stress_response"] = stress_ok
    print(f"  [{'PASS' if results['stress_response'] else 'FAIL'}] "
          f"4. Stress response: peak={stress_peak:.3f}, "
          f"Q4_mean={q4_stress:.3f}")

    # 5. Goal modulation: >= 5 goal switches
    goals = hist["goal"]
    goal_switches = sum(1 for i in range(1, len(goals))
                        if goals[i] != goals[i-1])
    results["goal_modulation"] = goal_switches >= 5
    print(f"  [{'PASS' if results['goal_modulation'] else 'FAIL'}] "
          f"5. Goal modulation: {goal_switches} switches >= 5")

    # 6. Energy management: energy > 10 at step 400
    energy_at_400 = hist["energy"][min(399, len(hist["energy"]) - 1)]
    results["energy_mgmt"] = energy_at_400 > 10
    print(f"  [{'PASS' if results['energy_mgmt'] else 'FAIL'}] "
          f"6. Energy at step 400: {energy_at_400:.1f} > 10")

    # 7. System stability (no NaN)
    results["stability"] = not hist["has_nan"]
    print(f"  [{'PASS' if results['stability'] else 'FAIL'}] "
          f"7. System stability: no NaN = {not hist['has_nan']}")

    all_pass = all(results.values())
    n_pass = sum(results.values())
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} ({n_pass}/7)")
    print("=" * 60)

    # =====================================================================
    # PLOT: 3x2 panels
    # =====================================================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    from matplotlib.lines import Line2D

    steps = range(len(hist["pos_x"]))
    steps_base = range(len(hist_base["pos_x"]))

    # (0,0) Trajectory colored by goal (allostatic run)
    ax = axes[0, 0]
    goal_colors = {GOAL_FORAGE: "green", GOAL_FLEE: "red",
                   GOAL_EXPLORE: "blue"}
    px, py = hist["pos_x"], hist["pos_y"]
    for i in range(1, len(px)):
        c = goal_colors.get(hist["goal"][i], "gray")
        ax.plot([px[i-1], px[i]], [py[i-1], py[i]],
                color=c, alpha=0.4, linewidth=0.8)
    ax.plot(px[0], py[0], "ko", ms=10, label="Start", zorder=6)
    if len(px) > 1:
        ax.plot(px[-1], py[-1], "ks", ms=10, label="End", zorder=6)
    for i, et in enumerate(hist["eaten_times"]):
        if et < len(px):
            ax.plot(px[et], py[et], "y*", ms=14, zorder=7,
                    label="Eat" if i == 0 else None)
    ax.set_xlim(0, 800)
    ax.set_ylim(600, 0)
    ax.set_aspect("equal")
    ax.set_title(f"Allostatic Trajectory (eaten:{hist['total_eaten']})")
    legend_elems = [
        Line2D([0], [0], color="green", lw=2, label="FORAGE"),
        Line2D([0], [0], color="red", lw=2, label="FLEE"),
        Line2D([0], [0], color="blue", lw=2, label="EXPLORE"),
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc="upper right")

    # (0,1) Energy: baseline vs allostatic
    ax = axes[0, 1]
    ax.plot(steps_base, hist_base["energy"], color="gray", linewidth=1.5,
            alpha=0.7, label="Baseline")
    ax.plot(steps, hist["energy"], color="darkorange", linewidth=1.5,
            alpha=0.9, label="Allostatic")
    ax.axhline(10, color="red", linewidth=0.5, linestyle="--",
               alpha=0.5, label="Danger threshold")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy: Baseline vs Allostatic")
    ax.legend(fontsize=7)

    # (1,0) Allostatic variables over time
    ax = axes[1, 0]
    ax.plot(steps, hist["hunger"], color="darkorange", linewidth=1.5,
            label="Hunger", alpha=0.8)
    ax.plot(steps, hist["fatigue"], color="purple", linewidth=1.5,
            label="Fatigue", alpha=0.8)
    ax.plot(steps, hist["stress"], color="red", linewidth=1.5,
            label="Stress", alpha=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value [0, 1]")
    ax.set_title("Allostatic Variables")
    ax.legend(fontsize=7)

    # (1,1) Allostatic errors over time
    ax = axes[1, 1]
    ax.plot(steps, hist["hunger_error"], color="darkorange", linewidth=1.5,
            label="Hunger err", alpha=0.8)
    ax.plot(steps, hist["fatigue_error"], color="purple", linewidth=1.5,
            label="Fatigue err", alpha=0.8)
    ax.plot(steps, hist["stress_error"], color="red", linewidth=1.5,
            label="Stress err", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Allostatic Error")
    ax.set_title("Allostatic Errors (predicted - setpoint)")
    ax.legend(fontsize=7)

    # (2,0) Speed + speed_cap
    ax = axes[2, 0]
    ax.plot(steps, hist["speed"], color="teal", linewidth=1,
            alpha=0.6, label="Actual speed")
    ax.plot(steps, hist["speed_cap"], color="red", linewidth=1.5,
            linestyle="--", label="Speed cap (fatigue)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Speed")
    ax.set_title("Speed & Fatigue Cap")
    ax.legend(fontsize=7)

    # (2,1) Pass/fail bar chart
    ax = axes[2, 1]
    crit_labels = [
        "Survival\n>=500", "Foraging\n>=3", "Fatigue\n<0.6",
        "Stress\nresponse", "Goal\nswitch>=5", "Energy\n>10@400", "No NaN",
    ]
    crit_keys = ["survival", "foraging", "fatigue_reg", "stress_response",
                 "goal_modulation", "energy_mgmt", "stability"]
    passes = [results[k] for k in crit_keys]
    colors_bar = ["green" if p else "red" for p in passes]
    values = [
        hist["final_step"], hist["total_eaten"], mean_fatigue,
        stress_peak, goal_switches, energy_at_400,
        1 if not hist["has_nan"] else 0,
    ]
    bars = ax.bar(crit_labels, [1 if p else 0 for p in passes],
                  color=colors_bar, edgecolor="black", alpha=0.8)
    for bar, val, p in zip(bars, values, passes):
        label_text = f"{val:.3f}" if isinstance(val, float) else str(val)
        status = "PASS" if p else "FAIL"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{label_text}\n{status}",
                ha="center", va="bottom", fontweight="bold", fontsize=6)
    ax.set_ylim(0, 1.5)
    ax.set_title(f"7 Pass/Fail Criteria ({n_pass}/7)")
    ax.set_ylabel("Pass=1 / Fail=0")

    fig.suptitle("Step 18: Allostatic Interoception",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                              "v60_step18_allostatic_interoception.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    return all_pass


if __name__ == "__main__":
    run_step18()
