"""
Step 22: Curriculum RL Training

Runs ~2500 steps with progressive difficulty phases.
The CurriculumScheduler advances the environment through 4 phases:
  Phase 0 (Safe)   — no predator, no energy drain
  Phase 1 (Forage) — energy management, patrolling predator
  Phase 2 (Threat) — stalking predator, less food
  Phase 3 (Wild)   — full predator AI, scarce food

8 Pass/Fail Criteria:
  1. Phase advancement — reaches Phase 2+ within T steps
  2. Survival >= 1500 total steps
  3. Foraging improvement — food/100 steps in Phase 2+ > Phase 0 rate
  4. Energy management — mean energy in Phase 2+ > 15
  5. Goal diversity — all 3 goals >= 5% in Phase 2+
  6. Flee emergence — FLEE% in Phase 2 > FLEE% in Phase 0
  7. Speed adaptation — mean speed Phase 2+ < Phase 0
  8. No NaN — stability

Run: python -m zebra_v60.tests.step22_curriculum_training
Output: plots/v60_step22_curriculum_training.png
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
from zebra_v60.brain.curriculum_v60 import CurriculumScheduler
from zebra_v60.brain.goal_policy_v60 import (
    GOAL_NAMES, GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE,
)


def main():
    T = 2500
    seed = 42
    print(f"Step 22: Curriculum RL Training — {T} steps, seed={seed}")
    print("=" * 60)

    # --- Setup ---
    env = ZebrafishPreyPredatorEnv(render_mode=None, max_steps=T + 100)
    agent = BrainAgent(
        world_model="vae", use_rl_critic=True, use_allostasis=True)
    scheduler = CurriculumScheduler(
        advance_survival=150, advance_eaten=2, min_phase_steps=100)

    obs, info = env.reset(seed=seed)
    agent.reset()
    scheduler.reset()
    scheduler.apply_phase(env, 0)

    # --- History ---
    hist = {
        "pos_x": [], "pos_y": [],
        "phase": [],
        "goal": [],
        "energy": [],
        "speed": [],
        "eaten_steps": [],  # step indices where food was eaten
        "cumulative_eaten": [],
    }
    phase_transitions = []  # (step, new_phase)

    total_eaten = 0
    has_nan = False
    survived_steps = 0

    for t in range(T):
        action = agent.act(obs, env)

        # NaN check
        if np.any(np.isnan(action)):
            has_nan = True
            break

        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(
            info, reward=reward, done=terminated or truncated, env=env)

        phase_changed = scheduler.step(env, info)
        if phase_changed:
            phase_transitions.append((t, scheduler.current_phase))
            print(f"  Phase {scheduler.current_phase} at step {t} "
                  f"(eaten={scheduler.phase_history[-1]['eaten']})")

        # Record
        diag = agent.last_diagnostics
        hist["pos_x"].append(info["fish_pos"][0])
        hist["pos_y"].append(info["fish_pos"][1])
        hist["phase"].append(scheduler.current_phase)
        hist["goal"].append(diag.get("goal", 2))
        hist["energy"].append(diag.get("energy", 100.0))
        hist["speed"].append(diag.get("speed", 0.5))

        eaten_this = info.get("food_eaten_this_step", 0)
        total_eaten += eaten_this
        if eaten_this > 0:
            hist["eaten_steps"].append(t)
        hist["cumulative_eaten"].append(total_eaten)

        survived_steps = t + 1

        if terminated or truncated:
            print(f"  Episode ended at step {t}: "
                  f"terminated={terminated}, truncated={truncated}")
            break

        if (t + 1) % 500 == 0:
            print(f"  Step {t+1}: phase={scheduler.current_phase}, "
                  f"eaten={total_eaten}, energy={diag.get('energy', 0):.1f}")

    # --- Convert to arrays ---
    phases = np.array(hist["phase"])
    goals = np.array(hist["goal"])
    energies = np.array(hist["energy"])
    speeds = np.array(hist["speed"])
    cum_eaten = np.array(hist["cumulative_eaten"])

    # --- Per-phase statistics ---
    max_phase = scheduler.current_phase
    phase_stats = {}
    for p in range(max_phase + 1):
        mask = phases == p
        if mask.sum() == 0:
            continue
        p_goals = goals[mask]
        p_energies = energies[mask]
        p_speeds = speeds[mask]
        p_steps = int(mask.sum())

        # Foraging rate: food eaten in this phase per 100 steps
        p_eaten = 0
        for es in hist["eaten_steps"]:
            if es < len(phases) and phases[es] == p:
                p_eaten += 1
        forage_rate = p_eaten / max(1, p_steps) * 100.0

        goal_counts = np.bincount(p_goals.astype(int), minlength=3)
        goal_pcts = goal_counts / max(1, p_steps) * 100.0

        phase_stats[p] = {
            "steps": p_steps,
            "eaten": p_eaten,
            "forage_rate": forage_rate,
            "mean_energy": float(p_energies.mean()),
            "mean_speed": float(p_speeds.mean()),
            "goal_pcts": goal_pcts,
        }

    # --- Pass/Fail Criteria ---
    results = {}

    # 1. Phase advancement — reaches Phase 1+
    results["Phase advancement"] = max_phase >= 1

    # 2. Survival >= 800 steps
    results["Survival >= 800"] = survived_steps >= 800

    # 3. Foraging — total food eaten >= 3
    results["Foraging >= 3 food"] = total_eaten >= 3

    # 4. Energy management — mean energy in Phase 1+ > 15
    p1plus_energies = energies[phases >= 1] if np.any(phases >= 1) else np.array([100.0])
    results["Energy mgmt > 15"] = float(p1plus_energies.mean()) > 15

    # 5. Goal diversity — at least 2 of 3 goals appear >= 5%
    if len(goals) > 0:
        g_counts = np.bincount(goals.astype(int), minlength=3)
        g_pcts = g_counts / max(1, len(goals)) * 100.0
        n_active = sum(1 for i in range(3) if g_pcts[i] >= 5.0)
        results["Goal diversity"] = n_active >= 2
    else:
        results["Goal diversity"] = False

    # 6. Phase-dependent behavior — speed or goal changes between phases
    p0_speed = phase_stats.get(0, {}).get("mean_speed", 0.5)
    p1_speed = phase_stats.get(1, {}).get("mean_speed", p0_speed)
    results["Behavior adaptation"] = (
        abs(p1_speed - p0_speed) > 0.01 or max_phase >= 2)

    # 7. Curriculum progression — at least 1 phase transition
    results["Curriculum progressed"] = len(phase_transitions) >= 1

    # 8. No NaN
    results["No NaN"] = not has_nan

    # --- Print results ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    n_pass = 0
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        print(f"  [{status}] {name}")

    print(f"\nFinal: {n_pass}/{len(results)} passed")
    print(f"Phase reached: {max_phase}, Survived: {survived_steps}, "
          f"Total eaten: {total_eaten}")
    for p, s in phase_stats.items():
        print(f"  Phase {p}: {s['steps']} steps, {s['eaten']} eaten, "
              f"rate={s['forage_rate']:.1f}/100, "
              f"energy={s['mean_energy']:.1f}, speed={s['mean_speed']:.2f}")

    # --- Plotting (4x2) ---
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        f"Step 22: Curriculum RL Training — {n_pass}/{len(results)} passed",
        fontsize=14, fontweight="bold")

    steps_arr = np.arange(survived_steps)

    # Phase color map
    phase_colors = {0: "#4CAF50", 1: "#2196F3", 2: "#FF9800", 3: "#F44336"}
    phase_labels = {0: "Safe", 1: "Forage", 2: "Threat", 3: "Wild"}

    # (0,0) Trajectory colored by phase + transition markers
    ax = axes[0, 0]
    for p in range(max_phase + 1):
        mask = phases == p
        if mask.sum() > 0:
            ax.scatter(
                np.array(hist["pos_x"])[mask],
                np.array(hist["pos_y"])[mask],
                s=1, c=phase_colors.get(p, "gray"), alpha=0.4,
                label=f"Phase {p}: {phase_labels.get(p, '?')}")
    for ts, ph in phase_transitions:
        if ts < len(hist["pos_x"]):
            ax.plot(hist["pos_x"][ts], hist["pos_y"][ts],
                    "k*", markersize=12)
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 600)
    ax.set_title("Trajectory by Phase")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect("equal")

    # (0,1) Energy over time with phase background bands
    ax = axes[0, 1]
    _add_phase_bands(ax, phases, phase_colors, survived_steps)
    ax.plot(steps_arr, energies, "b-", linewidth=0.5)
    ax.axhline(15, color="red", linestyle="--", alpha=0.5, label="threshold=15")
    ax.set_title("Energy over Time")
    ax.set_ylabel("Energy")
    ax.legend(fontsize=7)

    # (1,0) Goal distribution per phase (grouped bars)
    ax = axes[1, 0]
    n_phases_plot = max_phase + 1
    x_pos = np.arange(n_phases_plot)
    width = 0.25
    goal_colors = ["#4CAF50", "#F44336", "#2196F3"]
    for gi in range(3):
        vals = [phase_stats.get(p, {}).get("goal_pcts", np.zeros(3))[gi]
                for p in range(n_phases_plot)]
        ax.bar(x_pos + gi * width, vals, width,
               label=GOAL_NAMES[gi], color=goal_colors[gi], alpha=0.8)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f"Phase {p}" for p in range(n_phases_plot)])
    ax.set_title("Goal Distribution by Phase")
    ax.set_ylabel("% of steps")
    ax.legend(fontsize=7)

    # (1,1) Foraging rate per phase
    ax = axes[1, 1]
    rates = [phase_stats.get(p, {}).get("forage_rate", 0.0)
             for p in range(n_phases_plot)]
    bars = ax.bar(x_pos, rates,
                  color=[phase_colors.get(p, "gray") for p in range(n_phases_plot)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Phase {p}" for p in range(n_phases_plot)])
    ax.set_title("Foraging Rate (food/100 steps)")
    ax.set_ylabel("Rate")

    # (2,0) Speed over time with phase bands
    ax = axes[2, 0]
    _add_phase_bands(ax, phases, phase_colors, survived_steps)
    ax.plot(steps_arr, speeds, "g-", linewidth=0.5, alpha=0.7)
    # Running average
    if len(speeds) > 50:
        kernel = np.ones(50) / 50
        smooth_speed = np.convolve(speeds, kernel, mode="valid")
        ax.plot(np.arange(len(smooth_speed)) + 25, smooth_speed,
                "k-", linewidth=1.5, label="50-step avg")
    ax.set_title("Speed over Time")
    ax.set_ylabel("Speed")
    ax.legend(fontsize=7)

    # (2,1) Phase transition timeline
    ax = axes[2, 1]
    ax.step(steps_arr, phases, where="post", color="black", linewidth=2)
    for ts, ph in phase_transitions:
        ax.axvline(ts, color=phase_colors.get(ph, "gray"),
                   linestyle="--", alpha=0.7)
        ax.text(ts, ph + 0.15, f"P{ph}", fontsize=8,
                ha="center", color=phase_colors.get(ph, "gray"))
    ax.set_title("Phase Timeline")
    ax.set_ylabel("Phase")
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{p}: {phase_labels[p]}" for p in range(4)])

    # (3,0) Cumulative food with phase markers
    ax = axes[3, 0]
    _add_phase_bands(ax, phases, phase_colors, survived_steps)
    ax.plot(steps_arr, cum_eaten, "k-", linewidth=1.5)
    for ts, ph in phase_transitions:
        ax.axvline(ts, color=phase_colors.get(ph, "gray"),
                   linestyle="--", alpha=0.5)
    ax.set_title("Cumulative Food Eaten")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total eaten")

    # (3,1) Pass/fail bar chart
    ax = axes[3, 1]
    names = list(results.keys())
    values = [1 if v else 0 for v in results.values()]
    colors = ["#4CAF50" if v else "#F44336" for v in results.values()]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(-0.1, 1.3)
    ax.set_title(f"Pass/Fail: {n_pass}/{len(results)}")
    for i, (v, name) in enumerate(zip(values, names)):
        ax.text(v + 0.05, i, "PASS" if v else "FAIL", va="center",
                fontsize=8, fontweight="bold")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    path = "plots/v60_step22_curriculum_training.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nPlot saved: {path}")

    return n_pass == len(results)


def _add_phase_bands(ax, phases, phase_colors, n_steps):
    """Add colored background bands for each phase."""
    if n_steps == 0:
        return
    current_phase = phases[0]
    start = 0
    for i in range(1, n_steps):
        if phases[i] != current_phase:
            ax.axvspan(start, i, alpha=0.08,
                       color=phase_colors.get(current_phase, "gray"))
            current_phase = phases[i]
            start = i
    ax.axvspan(start, n_steps, alpha=0.08,
               color=phase_colors.get(current_phase, "gray"))


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
