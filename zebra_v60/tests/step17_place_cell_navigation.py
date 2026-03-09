"""
Step 17: Hippocampal Place Cell Navigation Test

Validates the place cell world model pipeline: cell allocation, path
integration, replay consolidation, and planning bonus generation.
Runs 600 steps with world_model='place_cell'.

7 Pass/Fail Criteria:
  1. Survival >= 400 steps
  2. Foraging >= 2 food
  3. Place cell allocation >= 15 cells
  4. PI accuracy: mean error < 100px
  5. G_plan non-zero: max|G| > 0.01 after step 100
  6. Food map learning: cells near food have higher food_rate
  7. System stability (no NaN)

Run: python -m zebra_v60.tests.step17_place_cell_navigation
Output: plots/v60_step17_place_cell_navigation.png
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
    """Run one episode with place cell model, return history dict."""
    obs, info = env.reset(seed=42)
    agent.reset()

    hist = {
        "pos_x": [], "pos_y": [],
        "pi_x": [], "pi_y": [],
        "goal": [],
        "n_allocated": [],
        "pi_error": [],
        "G_plan": [],
        "blend_weight": [],
        "eaten_times": [],
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

        # Place cell diagnostics
        pc_d = diag.get("place_cells", {})
        hist["n_allocated"].append(pc_d.get("n_allocated", 0))
        hist["pi_error"].append(pc_d.get("pi_error", 0.0))
        hist["G_plan"].append(pc_d.get("G_plan", np.zeros(3)))
        hist["blend_weight"].append(pc_d.get("blend_weight", 0.0))

        pi_pos = pc_d.get("pi_pos", np.array([400.0, 300.0]))
        hist["pi_x"].append(float(pi_pos[0]))
        hist["pi_y"].append(float(pi_pos[1]))

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
        g_plan = pc_d.get("G_plan", np.zeros(3))
        if np.any(np.isnan(g_plan)):
            has_nan = True

        if terminated or truncated:
            break

    hist["final_step"] = t + 1
    hist["total_eaten"] = total_eaten
    hist["has_nan"] = has_nan
    return hist


def run_step17(T=600):
    print("=" * 60)
    print("Step 17: Hippocampal Place Cell Navigation")
    print("=" * 60)

    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    cls_path = os.path.join(
        PROJECT_ROOT, "zebra_v60", "weights", "classifier_v60.pt")

    print("\n--- Running with place cell world model (600 steps) ---")
    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=T + 100)
    agent = BrainAgent(
        device="auto", cls_weights_path=cls_path,
        use_habit=True, use_rl_critic=False,
        world_model="place_cell")
    hist = _run_episode(env, agent, T)
    env.close()

    # Get final place cell state for analysis
    pc = agent.place_cells

    print(f"  Steps: {hist['final_step']}, "
          f"Eaten: {hist['total_eaten']}, "
          f"Cells: {hist['n_allocated'][-1]}, "
          f"Mean PI err: {np.mean(hist['pi_error']):.1f}px")

    # =====================================================================
    # PASS / FAIL CRITERIA (7 checks)
    # =====================================================================
    print(f"\n{'='*60}")
    print("--- Pass/Fail Criteria ---")
    results = {}

    # 1. Survival >= 400 steps
    results["survival"] = hist["final_step"] >= 400
    print(f"  [{'PASS' if results['survival'] else 'FAIL'}] "
          f"1. Survival: {hist['final_step']} >= 400")

    # 2. Foraging >= 2 food
    results["foraging"] = hist["total_eaten"] >= 2
    print(f"  [{'PASS' if results['foraging'] else 'FAIL'}] "
          f"2. Foraging: {hist['total_eaten']} >= 2")

    # 3. Place cell allocation >= 15 cells
    final_cells = hist["n_allocated"][-1]
    results["allocation"] = final_cells >= 15
    print(f"  [{'PASS' if results['allocation'] else 'FAIL'}] "
          f"3. Cell allocation: {final_cells} >= 15")

    # 4. PI accuracy: mean error < 100px
    mean_pi_err = float(np.mean(hist["pi_error"]))
    results["pi_accuracy"] = mean_pi_err < 100.0
    print(f"  [{'PASS' if results['pi_accuracy'] else 'FAIL'}] "
          f"4. PI accuracy: {mean_pi_err:.1f}px < 100px")

    # 5. G_plan non-zero after warmup (step 100+)
    G_plan_arr = np.array(hist["G_plan"])
    n_steps = len(G_plan_arr)
    warmup_idx = min(100, n_steps)
    if n_steps > warmup_idx:
        G_post = G_plan_arr[warmup_idx:]
        max_G = float(np.abs(G_post).max())
    else:
        max_G = 0.0
    results["plan_active"] = max_G > 0.01
    print(f"  [{'PASS' if results['plan_active'] else 'FAIL'}] "
          f"5. G_plan non-zero: max|G|={max_G:.4f} > 0.01")

    # 6. Food map learning: cells near eaten locations have higher food_rate
    food_map_ok = False
    if pc.n_allocated > 0 and len(hist["eaten_times"]) > 0:
        # Average food_rate of cells near eaten locations vs overall
        near_food_rates = []
        for et in hist["eaten_times"]:
            if et < len(hist["pos_x"]):
                eat_pos = np.array([hist["pos_x"][et], hist["pos_y"][et]])
                food_r, _ = pc.query(eat_pos)
                near_food_rates.append(food_r)
        overall_food = float(pc.food_rate[:pc.n_allocated].mean())
        near_mean = float(np.mean(near_food_rates)) if near_food_rates else 0.0
        food_map_ok = near_mean >= overall_food
    results["food_map"] = food_map_ok
    print(f"  [{'PASS' if results['food_map'] else 'FAIL'}] "
          f"6. Food map learning: near_food={near_mean:.4f} >= "
          f"overall={overall_food:.4f}")

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

    # (0,0) Trajectory + place cell centroids
    ax = axes[0, 0]
    goal_colors = {GOAL_FORAGE: "green", GOAL_FLEE: "red",
                   GOAL_EXPLORE: "blue"}
    px, py = hist["pos_x"], hist["pos_y"]
    for i in range(1, len(px)):
        c = goal_colors.get(hist["goal"][i], "gray")
        ax.plot([px[i-1], px[i]], [py[i-1], py[i]],
                color=c, alpha=0.4, linewidth=0.8)
    # Plot place cell centroids
    if pc.n_allocated > 0:
        centroids = pc.centroids[:pc.n_allocated]
        visits = pc.visit_count[:pc.n_allocated]
        sizes = 10 + 200 * (visits / (visits.max() + 1e-8))
        ax.scatter(centroids[:, 0], centroids[:, 1], s=sizes,
                   c="orange", alpha=0.5, edgecolors="black",
                   linewidths=0.5, zorder=5, label="Place cells")
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
    ax.set_title(f"Trajectory + Place Cells "
                 f"(eaten:{hist['total_eaten']}, cells:{final_cells})")
    ax.legend(fontsize=7, loc="upper right")

    # (0,1) Cell allocation + PI error (dual y-axis)
    ax = axes[0, 1]
    ax.plot(steps, hist["n_allocated"], color="brown", linewidth=2,
            label="Cells allocated")
    ax.set_ylabel("# Cells", color="brown")
    ax.set_xlabel("Time step")
    ax2 = ax.twinx()
    ax2.plot(steps, hist["pi_error"], color="teal", linewidth=1,
             alpha=0.6, label="PI error")
    if len(hist["pi_error"]) > 20:
        k = 20
        pi_smooth = np.convolve(
            hist["pi_error"], np.ones(k) / k, mode="valid")
        ax2.plot(range(k - 1, k - 1 + len(pi_smooth)),
                 pi_smooth, color="teal", linewidth=2, label=f"PI MA-{k}")
    ax2.set_ylabel("PI error (px)", color="teal")
    ax.set_title("Cell Allocation & Path Integration Error")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # (1,0) G_plan per goal over time
    ax = axes[1, 0]
    G_arr = np.array(hist["G_plan"])
    if len(G_arr) > 0:
        ax.plot(steps, G_arr[:, 0], color="green", linewidth=1.5,
                label="FORAGE", alpha=0.8)
        ax.plot(steps, G_arr[:, 1], color="red", linewidth=1.5,
                label="FLEE", alpha=0.8)
        ax.plot(steps, G_arr[:, 2], color="blue", linewidth=1.5,
                label="EXPLORE", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(100, color="gray", linewidth=0.5, linestyle=":",
               label="Warmup end")
    ax.set_xlabel("Time step")
    ax.set_ylabel("G_plan")
    ax.set_title("Planning Bonus Per Goal")
    ax.legend(fontsize=7)

    # (1,1) Place cell map: food_rate heatmap + risk overlay
    ax = axes[1, 1]
    if pc.n_allocated > 0:
        centroids = pc.centroids[:pc.n_allocated]
        food_rates = pc.food_rate[:pc.n_allocated]
        risks = pc.risk[:pc.n_allocated]
        # Food rate as color, risk as marker size
        sc = ax.scatter(centroids[:, 0], centroids[:, 1],
                        c=food_rates, cmap="YlGn", s=80,
                        edgecolors="black", linewidths=0.5,
                        vmin=0, vmax=max(0.01, food_rates.max()),
                        zorder=5)
        plt.colorbar(sc, ax=ax, label="food_rate")
        # Overlay risk as red circles
        risk_sizes = 20 + 300 * risks
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   s=risk_sizes, facecolors="none", edgecolors="red",
                   linewidths=1, alpha=0.6, zorder=4, label="Risk (size)")
    ax.set_xlim(0, 800)
    ax.set_ylim(600, 0)
    ax.set_aspect("equal")
    ax.set_title("Place Cell Map: Food (color) + Risk (red ring)")
    ax.legend(fontsize=7)

    # (2,0) Actual trajectory vs PI-estimated trajectory
    ax = axes[2, 0]
    ax.plot(hist["pos_x"], hist["pos_y"], color="blue", alpha=0.5,
            linewidth=1, label="Actual")
    ax.plot(hist["pi_x"], hist["pi_y"], color="red", alpha=0.5,
            linewidth=1, linestyle="--", label="Path Integration")
    ax.set_xlim(0, 800)
    ax.set_ylim(600, 0)
    ax.set_aspect("equal")
    ax.set_title("Actual vs Path-Integrated Trajectory")
    ax.legend(fontsize=7)

    # (2,1) Pass/fail bar chart
    ax = axes[2, 1]
    crit_labels = [
        "Survival\n>=400", "Foraging\n>=2", "Cells\n>=15",
        "PI err\n<100", "G_plan\n>0.01", "Food\nmap", "No NaN",
    ]
    crit_keys = ["survival", "foraging", "allocation", "pi_accuracy",
                 "plan_active", "food_map", "stability"]
    passes = [results[k] for k in crit_keys]
    colors_bar = ["green" if p else "red" for p in passes]
    values = [
        hist["final_step"], hist["total_eaten"], final_cells,
        mean_pi_err, max_G,
        near_mean if 'near_mean' in dir() else 0.0,
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

    fig.suptitle("Step 17: Hippocampal Place Cell Navigation",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                              "v60_step17_place_cell_navigation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    return all_pass


if __name__ == "__main__":
    run_step17()
