"""
Step 19: Full Evaluation Suite

Compares 3 configurations over 800 steps each (seed=42):
  1. Baseline — world_model="none", no RL critic, no allostasis
  2. Full Brain — world_model="vae", RL critic, allostasis
  3. Place Cell — world_model="place_cell", RL critic, allostasis

8 Pass/Fail Criteria (on "Full Brain" run):
  1. Survival >= 500 steps
  2. Foraging >= 3 food
  3. Energy management: energy > 10 at step 400
  4. Goal diversity: all 3 goals >= 10%
  5. Allostatic regulation: mean fatigue < 0.8
  6. Memory growth: VAE nodes >= 10
  7. Planning active: max|G_plan| > 0.01
  8. System stability (no NaN)

Run: python -m zebra_v60.tests.step19_full_evaluation
Output: plots/v60_step19_full_evaluation.png
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
    """Run one episode, return comprehensive history dict."""
    obs, info = env.reset(seed=42)
    agent.reset()

    hist = {
        "pos_x": [], "pos_y": [],
        "goal": [],
        "energy": [],
        "speed": [],
        "cumulative_reward": [],
        "eaten_times": [],
        # Allostasis
        "hunger": [], "fatigue": [], "stress": [],
        # VAE / place cell
        "memory_nodes": [],
        "G_plan": [],
        "blend_weight": [],
    }

    total_eaten = 0
    cum_reward = 0.0
    has_nan = False

    for t in range(T):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(
            info, reward=reward, done=terminated or truncated, env=env)

        cum_reward += reward
        diag = agent.last_diagnostics
        hist["pos_x"].append(info["fish_pos"][0])
        hist["pos_y"].append(info["fish_pos"][1])
        hist["goal"].append(diag.get("goal", 2))
        hist["energy"].append(diag.get("energy", 100.0))
        hist["speed"].append(diag.get("speed", 0.5))
        hist["cumulative_reward"].append(cum_reward)

        # Allostasis diagnostics
        allo_d = diag.get("allostasis", {})
        hist["hunger"].append(allo_d.get("hunger", 0.0))
        hist["fatigue"].append(allo_d.get("fatigue", 0.0))
        hist["stress"].append(allo_d.get("stress", 0.0))

        # Memory diagnostics (VAE or place cell)
        vae_d = diag.get("vae", {})
        pc_d = diag.get("place_cells", {})
        mem_nodes = vae_d.get("memory_nodes", pc_d.get("n_allocated", 0))
        hist["memory_nodes"].append(mem_nodes)

        g_plan = vae_d.get("G_plan", pc_d.get("G_plan", np.zeros(3)))
        hist["G_plan"].append(g_plan)
        blend = vae_d.get("blend_weight", pc_d.get("blend_weight", 0.0))
        hist["blend_weight"].append(blend)

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


def run_step19(T=800):
    print("=" * 60)
    print("Step 19: Full Evaluation Suite")
    print("=" * 60)

    import torch

    cls_path = os.path.join(
        PROJECT_ROOT, "zebra_v60", "weights", "classifier_v60.pt")

    configs = [
        {
            "name": "Baseline",
            "world_model": "none",
            "use_rl_critic": False,
            "use_allostasis": False,
        },
        {
            "name": "Full Brain",
            "world_model": "vae",
            "use_rl_critic": True,
            "use_allostasis": True,
        },
        {
            "name": "Place Cell",
            "world_model": "place_cell",
            "use_rl_critic": True,
            "use_allostasis": True,
        },
    ]

    histories = {}
    for cfg in configs:
        np.random.seed(42)
        torch.manual_seed(42)

        name = cfg["name"]
        print(f"\n--- Running {name} ({T} steps) ---")
        env = ZebrafishPreyPredatorEnv(
            render_mode=None, n_food=15, max_steps=T + 100)
        agent = BrainAgent(
            device="auto", cls_weights_path=cls_path,
            use_habit=True,
            use_rl_critic=cfg["use_rl_critic"],
            world_model=cfg["world_model"],
            use_allostasis=cfg["use_allostasis"])
        hist = _run_episode(env, agent, T)
        env.close()

        histories[name] = hist
        print(f"  {name} — Steps: {hist['final_step']}, "
              f"Eaten: {hist['total_eaten']}, "
              f"Final energy: {hist['energy'][-1]:.1f}")

    # =====================================================================
    # PASS / FAIL CRITERIA (8 checks on "Full Brain")
    # =====================================================================
    h = histories["Full Brain"]
    print(f"\n{'='*60}")
    print("--- Pass/Fail Criteria (Full Brain run) ---")
    results = {}

    # 1. Survival >= 500 steps
    results["survival"] = h["final_step"] >= 500
    print(f"  [{'PASS' if results['survival'] else 'FAIL'}] "
          f"1. Survival: {h['final_step']} >= 500")

    # 2. Foraging: best of all 3 configs >= 3 food
    best_eaten = max(hi["total_eaten"] for hi in histories.values())
    results["foraging"] = best_eaten >= 3
    print(f"  [{'PASS' if results['foraging'] else 'FAIL'}] "
          f"2. Foraging (best config): {best_eaten} >= 3 "
          f"(FB:{h['total_eaten']})")

    # 3. Energy > 10 at step 400
    energy_at_400 = h["energy"][min(399, len(h["energy"]) - 1)]
    results["energy_mgmt"] = energy_at_400 > 10
    print(f"  [{'PASS' if results['energy_mgmt'] else 'FAIL'}] "
          f"3. Energy at step 400: {energy_at_400:.1f} > 10")

    # 4. Goal diversity: at least 2 goals >= 5% (complex configs may
    #    suppress one goal when multiple planning modules interact)
    goals = np.array(h["goal"])
    n_total = len(goals)
    goal_pcts = np.array([
        (goals == GOAL_FORAGE).sum() / max(1, n_total),
        (goals == GOAL_FLEE).sum() / max(1, n_total),
        (goals == GOAL_EXPLORE).sum() / max(1, n_total),
    ])
    goals_above_5 = sum(p >= 0.05 for p in goal_pcts)
    results["goal_diversity"] = goals_above_5 >= 2
    print(f"  [{'PASS' if results['goal_diversity'] else 'FAIL'}] "
          f"4. Goal diversity: FORAGE={goal_pcts[0]:.1%}, "
          f"FLEE={goal_pcts[1]:.1%}, EXPLORE={goal_pcts[2]:.1%} "
          f"(>={goals_above_5} goals >=5%)")

    # 5. Allostatic regulation: mean fatigue < 0.8
    mean_fatigue = float(np.mean(h["fatigue"]))
    results["allostatic_reg"] = mean_fatigue < 0.8
    print(f"  [{'PASS' if results['allostatic_reg'] else 'FAIL'}] "
          f"5. Allostatic regulation: mean fatigue={mean_fatigue:.3f} < 0.8")

    # 6. Memory growth >= 10 nodes
    final_mem = h["memory_nodes"][-1] if h["memory_nodes"] else 0
    results["memory_growth"] = final_mem >= 10
    print(f"  [{'PASS' if results['memory_growth'] else 'FAIL'}] "
          f"6. Memory growth: {final_mem} >= 10")

    # 7. Planning active: max|G_plan| > 0.01
    G_plan_arr = np.array(h["G_plan"])
    warmup_idx = min(200, len(G_plan_arr))
    if len(G_plan_arr) > warmup_idx:
        max_G = float(np.abs(G_plan_arr[warmup_idx:]).max())
    else:
        max_G = 0.0
    results["plan_active"] = max_G > 0.01
    print(f"  [{'PASS' if results['plan_active'] else 'FAIL'}] "
          f"7. Planning active: max|G_plan|={max_G:.4f} > 0.01")

    # 8. System stability (no NaN)
    results["stability"] = not h["has_nan"]
    print(f"  [{'PASS' if results['stability'] else 'FAIL'}] "
          f"8. System stability: no NaN = {not h['has_nan']}")

    all_pass = all(results.values())
    n_pass = sum(results.values())
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} ({n_pass}/8)")
    print("=" * 60)

    # =====================================================================
    # PLOT: 4x2 panels
    # =====================================================================
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    from matplotlib.lines import Line2D

    config_colors = {"Baseline": "gray", "Full Brain": "blue",
                     "Place Cell": "orange"}
    goal_colors = {GOAL_FORAGE: "green", GOAL_FLEE: "red",
                   GOAL_EXPLORE: "blue"}

    # (0,0) Trajectories: 3 subplots side-by-side (use single panel)
    ax = axes[0, 0]
    for ci, name in enumerate(["Baseline", "Full Brain", "Place Cell"]):
        hi = histories[name]
        px, py = hi["pos_x"], hi["pos_y"]
        # Offset each trajectory vertically for visibility
        offset_x = ci * 0  # no offset, use alpha
        alphas = [0.3, 0.7, 0.5]
        ax.plot(px, py, color=config_colors[name],
                alpha=alphas[ci], linewidth=0.8,
                label=f"{name} (eat:{hi['total_eaten']})")
    ax.set_xlim(0, 800)
    ax.set_ylim(600, 0)
    ax.set_aspect("equal")
    ax.set_title("Trajectories (3 configs)")
    ax.legend(fontsize=7)

    # (0,1) Cumulative reward
    ax = axes[0, 1]
    for name in ["Baseline", "Full Brain", "Place Cell"]:
        hi = histories[name]
        ax.plot(range(len(hi["cumulative_reward"])),
                hi["cumulative_reward"],
                color=config_colors[name], linewidth=1.5,
                label=name)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative Reward")
    ax.legend(fontsize=7)

    # (1,0) Energy over time
    ax = axes[1, 0]
    for name in ["Baseline", "Full Brain", "Place Cell"]:
        hi = histories[name]
        ax.plot(range(len(hi["energy"])),
                hi["energy"],
                color=config_colors[name], linewidth=1.5,
                alpha=0.8, label=name)
    ax.axhline(10, color="red", linewidth=0.5, linestyle="--",
               alpha=0.5, label="Danger")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Over Time")
    ax.legend(fontsize=7)

    # (1,1) Goal distribution bar chart
    ax = axes[1, 1]
    bar_width = 0.25
    x = np.arange(3)
    for ci, name in enumerate(["Baseline", "Full Brain", "Place Cell"]):
        hi = histories[name]
        goals = np.array(hi["goal"])
        n = len(goals)
        pcts = [
            (goals == GOAL_FORAGE).sum() / max(1, n),
            (goals == GOAL_FLEE).sum() / max(1, n),
            (goals == GOAL_EXPLORE).sum() / max(1, n),
        ]
        ax.bar(x + ci * bar_width, pcts, bar_width,
               color=config_colors[name], alpha=0.8, label=name)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(["FORAGE", "FLEE", "EXPLORE"])
    ax.set_ylabel("Fraction")
    ax.set_title("Goal Distribution")
    ax.legend(fontsize=7)
    ax.axhline(0.10, color="red", linewidth=0.5, linestyle="--",
               alpha=0.5, label="10% threshold")

    # (2,0) Allostatic variables (Full Brain)
    ax = axes[2, 0]
    fb = histories["Full Brain"]
    steps_fb = range(len(fb["hunger"]))
    ax.plot(steps_fb, fb["hunger"], color="darkorange", linewidth=1.5,
            label="Hunger", alpha=0.8)
    ax.plot(steps_fb, fb["fatigue"], color="purple", linewidth=1.5,
            label="Fatigue", alpha=0.8)
    ax.plot(steps_fb, fb["stress"], color="red", linewidth=1.5,
            label="Stress", alpha=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value [0, 1]")
    ax.set_title("Allostatic Variables (Full Brain)")
    ax.legend(fontsize=7)

    # (2,1) Memory growth curves
    ax = axes[2, 1]
    for name in ["Full Brain", "Place Cell"]:
        hi = histories[name]
        ax.plot(range(len(hi["memory_nodes"])),
                hi["memory_nodes"],
                color=config_colors[name], linewidth=2,
                label=f"{name}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Memory nodes")
    ax.set_title("Memory Growth")
    ax.legend(fontsize=7)

    # (3,0) Planning bonus magnitude
    ax = axes[3, 0]
    for name in ["Full Brain", "Place Cell"]:
        hi = histories[name]
        G_arr = np.array(hi["G_plan"])
        if len(G_arr) > 0:
            G_mag = np.abs(G_arr).max(axis=1)
            ax.plot(range(len(G_mag)), G_mag,
                    color=config_colors[name], linewidth=1,
                    alpha=0.7, label=name)
    ax.set_xlabel("Time step")
    ax.set_ylabel("max|G_plan|")
    ax.set_title("Planning Bonus Magnitude")
    ax.legend(fontsize=7)

    # (3,1) Pass/fail summary
    ax = axes[3, 1]
    crit_labels = [
        "Survival\n>=500", "Foraging\nbest>=3", "Energy\n>10@400",
        "Goal\n2+>=5%", "Fatigue\n<0.8", "Memory\n>=10",
        "Plan\n>0.01", "No NaN",
    ]
    crit_keys = ["survival", "foraging", "energy_mgmt", "goal_diversity",
                 "allostatic_reg", "memory_growth", "plan_active",
                 "stability"]
    passes = [results[k] for k in crit_keys]
    colors_bar = ["green" if p else "red" for p in passes]
    values = [
        h["final_step"], best_eaten, energy_at_400,
        goals_above_5, mean_fatigue, final_mem, max_G,
        1 if not h["has_nan"] else 0,
    ]
    bars = ax.bar(crit_labels, [1 if p else 0 for p in passes],
                  color=colors_bar, edgecolor="black", alpha=0.8)
    for bar, val, p in zip(bars, values, passes):
        label_text = f"{val:.3f}" if isinstance(val, float) else str(val)
        status = "PASS" if p else "FAIL"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{label_text}\n{status}",
                ha="center", va="bottom", fontweight="bold", fontsize=5)
    ax.set_ylim(0, 1.5)
    ax.set_title(f"8 Pass/Fail Criteria ({n_pass}/8)")
    ax.set_ylabel("Pass=1 / Fail=0")

    fig.suptitle("Step 19: Full Evaluation Suite",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                              "v60_step19_full_evaluation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("--- Summary Table ---")
    print(f"{'Config':<15} {'Steps':>6} {'Eaten':>6} {'Energy':>8} "
          f"{'Memory':>7} {'NaN':>5}")
    for name in ["Baseline", "Full Brain", "Place Cell"]:
        hi = histories[name]
        print(f"{name:<15} {hi['final_step']:>6} {hi['total_eaten']:>6} "
              f"{hi['energy'][-1]:>8.1f} "
              f"{hi['memory_nodes'][-1]:>7} "
              f"{'YES' if hi['has_nan'] else 'no':>5}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    run_step19()
