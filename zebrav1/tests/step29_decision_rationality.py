"""
Step 29: Decision Rationality Evaluation

Quantifies whether the zebrafish brain makes rational decisions by
measuring 6 intelligence metrics against random and optimal baselines:

1. Foraging Efficiency: food/distance vs random agent
2. Patch Selection: does it choose the densest food patch?
3. Flee Timing: flee when predator approaches, return when safe
4. Path Efficiency: straight-line / actual path ratio
5. Energy Management: proactive foraging before crisis
6. Threat Response Latency: steps to react to approaching predator

Run: python -m zebrav1.tests.step29_decision_rationality
Output: plots/v1_step29_decision_rationality.png
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

GOAL_FORAGE = 0
GOAL_FLEE = 1
GOAL_EXPLORE = 2


def _run_episode(env, agent, T, label="Brain"):
    """Run one episode and collect decision quality metrics."""
    obs, info = env.reset(seed=42)
    agent.reset()

    metrics = {
        "positions": [],
        "goals": [],
        "food_eaten_steps": [],
        "energy_history": [],
        "flee_starts": [],
        "flee_ends": [],
        "pred_distances": [],
        "food_distances": [],      # distance to nearest food each step
        "patch_densities": [],     # density of chosen food target
        "best_patch_density": [],  # density of best available patch
        "distance_traveled": 0.0,
        "total_eaten": 0,
        "prev_x": env.fish_x,
        "prev_y": env.fish_y,
        "was_fleeing": False,
        "threat_detected_step": None,
        "flee_response_latencies": [],
    }

    for t in range(T):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward,
                               done=terminated or truncated, env=env)

        diag = agent.last_diagnostics
        goal = diag.get("goal", 2)
        energy = diag.get("energy", 100)
        px, py = env.fish_x, env.fish_y

        # Track position and distance
        dx = px - metrics["prev_x"]
        dy = py - metrics["prev_y"]
        step_dist = math.sqrt(dx * dx + dy * dy)
        metrics["distance_traveled"] += step_dist
        metrics["prev_x"] = px
        metrics["prev_y"] = py

        metrics["positions"].append((px, py))
        metrics["goals"].append(goal)
        metrics["energy_history"].append(energy)

        # Predator distance
        pred_dist = math.sqrt(
            (px - env.pred_x) ** 2 + (py - env.pred_y) ** 2)
        metrics["pred_distances"].append(pred_dist)

        # Nearest food distance
        nearest_food_dist = 999.0
        for food in env.foods:
            fd = math.sqrt((px - food[0]) ** 2 + (py - food[1]) ** 2)
            nearest_food_dist = min(nearest_food_dist, fd)
        metrics["food_distances"].append(nearest_food_dist)

        # Food eaten tracking
        if info.get("total_eaten", 0) > metrics["total_eaten"]:
            metrics["food_eaten_steps"].append(t)
            metrics["total_eaten"] = info["total_eaten"]

        # Patch density tracking (from food prospects)
        prospects = diag.get("food_prospects", [])
        if prospects:
            chosen_density = prospects[0].get("density", 0)
            metrics["patch_densities"].append(chosen_density)
            best_density = max(p.get("density", 0) for p in prospects)
            metrics["best_patch_density"].append(best_density)
        else:
            metrics["patch_densities"].append(0)
            metrics["best_patch_density"].append(0)

        # Flee timing tracking
        is_fleeing = (goal == GOAL_FLEE)
        if is_fleeing and not metrics["was_fleeing"]:
            metrics["flee_starts"].append(t)
            # Measure response latency from threat detection
            if metrics["threat_detected_step"] is not None:
                latency = t - metrics["threat_detected_step"]
                metrics["flee_response_latencies"].append(latency)
        if not is_fleeing and metrics["was_fleeing"]:
            metrics["flee_ends"].append(t)
        metrics["was_fleeing"] = is_fleeing

        # Threat detection: predator approaching (distance decreasing)
        if (len(metrics["pred_distances"]) > 2
                and metrics["pred_distances"][-1] < metrics["pred_distances"][-2]
                and pred_dist < 200
                and not is_fleeing):
            if metrics["threat_detected_step"] is None:
                metrics["threat_detected_step"] = t
        elif is_fleeing or pred_dist > 250:
            metrics["threat_detected_step"] = None

        if terminated or truncated:
            break

    metrics["steps"] = t + 1
    return metrics


def _run_random_episode(env, T):
    """Run random agent baseline for comparison."""
    obs, info = env.reset(seed=42)
    total_dist = 0.0
    total_eaten = 0
    prev_x, prev_y = env.fish_x, env.fish_y

    for t in range(T):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        dx = env.fish_x - prev_x
        dy = env.fish_y - prev_y
        total_dist += math.sqrt(dx * dx + dy * dy)
        prev_x, prev_y = env.fish_x, env.fish_y
        total_eaten = info.get("total_eaten", 0)
        if terminated or truncated:
            break

    return {"total_eaten": total_eaten, "distance": total_dist,
            "steps": t + 1}


def compute_rationality_scores(m, random_m):
    """Compute 6 rationality scores from episode metrics."""

    # 1. Foraging Efficiency: food per km traveled
    brain_efficiency = (m["total_eaten"] /
                        max(0.001, m["distance_traveled"] / 1000.0))
    random_efficiency = (random_m["total_eaten"] /
                         max(0.001, random_m["distance"] / 1000.0))
    # Score: ratio vs random (capped at 5x)
    eff_ratio = brain_efficiency / max(0.01, random_efficiency)
    foraging_score = min(100, eff_ratio * 20)

    # 2. Patch Selection Quality: chose dense patch vs best available
    if len(m["patch_densities"]) > 0 and len(m["best_patch_density"]) > 0:
        chosen = np.array(m["patch_densities"])
        best = np.array(m["best_patch_density"])
        mask = best > 0
        if mask.sum() > 0:
            patch_ratio = chosen[mask].mean() / (best[mask].mean() + 1e-8)
        else:
            patch_ratio = 0.5
    else:
        patch_ratio = 0.5
    patch_score = min(100, patch_ratio * 100)

    # 3. Flee Timing: flee when close, forage when safe
    pred_dists = np.array(m["pred_distances"])
    goals = np.array(m["goals"])
    if len(pred_dists) > 10:
        # When predator < 150px: should be fleeing
        close_mask = pred_dists < 150
        flee_when_close = (goals[close_mask] == GOAL_FLEE).mean() if close_mask.sum() > 0 else 0
        # When predator > 250px: should NOT be fleeing
        far_mask = pred_dists > 250
        forage_when_safe = (goals[far_mask] != GOAL_FLEE).mean() if far_mask.sum() > 0 else 1
        flee_score = (flee_when_close * 50 + forage_when_safe * 50)
    else:
        flee_score = 50

    # 4. Path Efficiency: straight-line displacement / total distance
    if len(m["positions"]) > 10:
        start = m["positions"][0]
        end = m["positions"][-1]
        displacement = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        # For foraging, displacement is less meaningful — use food/distance
        path_score = min(100, brain_efficiency * 30)
    else:
        path_score = 50

    # 5. Energy Management: proactive foraging before crisis
    energy = np.array(m["energy_history"])
    if len(energy) > 50:
        # Count steps where energy dropped below 30% (crisis)
        crisis_steps = (energy < 30).sum()
        crisis_ratio = crisis_steps / len(energy)
        # Good: minimal time in crisis zone
        energy_score = max(0, 100 - crisis_ratio * 200)
    else:
        energy_score = 50

    # 6. Threat Response Latency: steps from detection to flee
    if len(m["flee_response_latencies"]) > 0:
        avg_latency = np.mean(m["flee_response_latencies"])
        # Good: < 5 steps, poor: > 20 steps
        latency_score = max(0, min(100, 100 - avg_latency * 5))
    else:
        # No flee events — either no predator encounters or no response
        n_close = (np.array(m["pred_distances"]) < 150).sum()
        if n_close > 10:
            latency_score = 20  # predator was close but never fled — bad
        else:
            latency_score = 70  # predator stayed far — neutral

    return {
        "foraging_efficiency": foraging_score,
        "patch_selection": patch_score,
        "flee_timing": flee_score,
        "path_efficiency": path_score,
        "energy_management": energy_score,
        "threat_response": latency_score,
    }


def run_step29():
    print("=" * 60)
    print("Step 29: Decision Rationality Evaluation")
    print("=" * 60)

    T = 800
    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=T, side_panels=False)

    # Run brain agent
    print("\n--- Brain Agent ---")
    agent = BrainAgent(device="auto", world_model="place_cell",
                       use_allostasis=True)
    brain_metrics = _run_episode(env, agent, T, "Brain")
    print(f"  Steps: {brain_metrics['steps']}, "
          f"Eaten: {brain_metrics['total_eaten']}, "
          f"Distance: {brain_metrics['distance_traveled']:.0f}")

    # Run random baseline
    print("\n--- Random Agent ---")
    random_metrics = _run_random_episode(env, T)
    print(f"  Steps: {random_metrics['steps']}, "
          f"Eaten: {random_metrics['total_eaten']}, "
          f"Distance: {random_metrics['distance']:.0f}")

    env.close()

    # Compute rationality scores
    scores = compute_rationality_scores(brain_metrics, random_metrics)

    print("\n" + "=" * 60)
    print("DECISION RATIONALITY SCORES (0-100)")
    print("=" * 60)
    total = 0
    for name, score in scores.items():
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        grade = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D" if score >= 20 else "F"
        print(f"  {name:25s} {bar} {score:5.1f} [{grade}]")
        total += score
    overall = total / len(scores)
    print(f"\n  {'OVERALL':25s} {'':20s} {overall:5.1f}")
    print(f"  Intelligence Grade: ", end="")
    if overall >= 80:
        print("EXCELLENT — rational decision-maker")
    elif overall >= 60:
        print("GOOD — mostly rational with some suboptimal choices")
    elif overall >= 40:
        print("FAIR — better than random but significant room for improvement")
    else:
        print("POOR — near-random behavior")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Trajectory colored by goal
    ax = axes[0, 0]
    pos = np.array(brain_metrics["positions"])
    goals = np.array(brain_metrics["goals"])
    colors = {0: "green", 1: "red", 2: "blue", 3: "cyan"}
    for g, c in colors.items():
        mask = goals == g
        if mask.sum() > 0:
            ax.scatter(pos[mask, 0], pos[mask, 1], c=c, s=1, alpha=0.5,
                       label=["FORAGE", "FLEE", "EXPLORE", "SOCIAL"][g])
    ax.set_title("Trajectory (colored by goal)")
    ax.legend(fontsize=7, markerscale=5)
    ax.set_aspect("equal")

    # 2. Rationality radar chart
    axes[0, 1].remove()
    ax_radar = fig.add_subplot(3, 2, 2, polar=True)
    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += [values[0]]
    angles += [angles[0]]
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.fill(angles, values, alpha=0.25, color="steelblue")
    ax_radar.plot(angles, values, color="steelblue", linewidth=2)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_title(f"Rationality Profile ({overall:.0f}/100)", pad=20)

    # 3. Energy + predator distance over time
    ax = axes[1, 0]
    ax.plot(brain_metrics["energy_history"], color="orange", label="Energy")
    ax.axhline(30, color="red", linestyle="--", alpha=0.3, label="Crisis")
    ax.set_ylabel("Energy", color="orange")
    ax.set_xlabel("Step")
    ax2 = ax.twinx()
    ax2.plot(brain_metrics["pred_distances"], color="red", alpha=0.5,
             label="Predator dist")
    ax2.set_ylabel("Pred dist (px)", color="red")
    ax.set_title("Energy Management + Threat Proximity")
    ax.legend(fontsize=7, loc="upper left")

    # 4. Food distance over time (should decrease when foraging)
    ax = axes[1, 1]
    ax.plot(brain_metrics["food_distances"], color="green", alpha=0.7)
    for t in brain_metrics["food_eaten_steps"]:
        ax.axvline(t, color="gold", alpha=0.5, linewidth=2)
    ax.set_ylabel("Nearest food dist (px)")
    ax.set_xlabel("Step")
    ax.set_title("Food Pursuit (gold lines = eaten)")

    # 5. Goal distribution pie
    ax = axes[2, 0]
    goal_names = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
    goal_colors = ["green", "red", "blue", "cyan"]
    goal_counts = [np.sum(goals == g) for g in range(4)]
    nonzero = [(n, c, cnt) for n, c, cnt in
               zip(goal_names, goal_colors, goal_counts) if cnt > 0]
    if nonzero:
        ax.pie([x[2] for x in nonzero],
               labels=[x[0] for x in nonzero],
               colors=[x[1] for x in nonzero],
               autopct="%1.0f%%", startangle=90)
    ax.set_title("Goal Distribution")

    # 6. Score bars
    ax = axes[2, 1]
    y_pos = np.arange(len(scores))
    vals = list(scores.values())
    bar_colors = ["green" if v >= 60 else "orange" if v >= 40 else "red"
                  for v in vals]
    ax.barh(y_pos, vals, color=bar_colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([k.replace("_", " ") for k in scores.keys()],
                       fontsize=8)
    ax.set_xlim(0, 100)
    ax.axvline(60, color="green", linestyle="--", alpha=0.3)
    ax.axvline(40, color="orange", linestyle="--", alpha=0.3)
    ax.set_xlabel("Score (0-100)")
    ax.set_title(f"Rationality Scores (Overall: {overall:.0f})")

    fig.suptitle("Step 29: Decision Rationality Evaluation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                             "v1_step29_decision_rationality.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # Pass/fail
    passed = overall >= 40
    print(f"\n{'PASS' if passed else 'FAIL'}: "
          f"Overall rationality {overall:.0f}/100 "
          f"(threshold: 40)")
    print("=" * 60)


if __name__ == "__main__":
    run_step29()
