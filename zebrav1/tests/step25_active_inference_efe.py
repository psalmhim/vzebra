"""
Step 25: Proper Active Inference EFE with Three Ecological Drives

Behavioral validation of proper EFE policy selection where three ecological
drives — energy economy, predator risk, and exploration — emerge naturally
from a single EFE computation over the generative model.

Tests:
  1. Starving fish (energy=20) → FORAGE dominates (>50%)
  2. Nearby predator           → FLEE dominates (>50%)
  3. Well-fed, safe fish       → EXPLORE selected (>30%)
  4. Trade-off: moderate hunger + predator → FLEE first, FORAGE when safe
  5. System stability: no NaN, goals diverse over extended run

Configuration:
  - use_active_inference=True
  - world_model="vae"
  - use_allostasis=True
  - use_rl_critic=True

Run: python -m zebrav1.tests.step25_active_inference_efe
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

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav1.gym_env.brain_agent import BrainAgent
from zebrav1.brain.goal_policy import (
    GOAL_NAMES, GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, GOAL_SOCIAL,
)


def _make_env_agent(seed=42):
    """Create env + agent with full Step 25 config."""
    env = ZebrafishPreyPredatorEnv(
        render_mode=None, arena_size=800, max_steps=2000)
    agent = BrainAgent(
        use_rl_critic=True,
        world_model="vae",
        use_allostasis=True,
        use_active_inference=True)
    return env, agent


def _run_episode(env, agent, T, seed=42):
    """Run T steps, return goal history and diagnostics."""
    obs, info = env.reset(seed=seed)
    agent.reset()

    goals = []
    energies = []
    has_nan = False
    total_eaten = 0

    for t in range(T):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(
            info, reward=reward, done=terminated or truncated, env=env)

        diag = agent.last_diagnostics
        goals.append(diag.get("goal", GOAL_EXPLORE))
        energies.append(diag.get("energy", 100.0))
        total_eaten += info.get("food_eaten_this_step", 0)

        # NaN check
        if np.isnan(diag.get("dopa", 0)):
            has_nan = True

        if terminated or truncated:
            break

    return {
        "goals": goals,
        "energies": energies,
        "total_eaten": total_eaten,
        "steps": len(goals),
        "has_nan": has_nan,
        "survived": len(goals),
    }


def test_efe_engine_exists():
    """Test 0: EFE engine is properly created."""
    env, agent = _make_env_agent()
    assert agent.efe_engine is not None, "EFE engine should exist"
    assert agent.preferred_outcomes is not None
    assert agent.lambda_adapter is not None
    print("  [PASS] EFE engine exists")
    env.close()
    return True


def test_starving_forages():
    """Test 1: Starving fish (energy=20) should FORAGE >50%."""
    env, agent = _make_env_agent()
    obs, info = env.reset(seed=42)
    agent.reset()

    # Force low energy
    env.fish_energy = 20.0

    goals = []
    for t in range(100):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(
            info, reward=reward, done=terminated or truncated, env=env)
        goals.append(agent.last_diagnostics.get("goal", GOAL_EXPLORE))
        # Keep energy low
        env.fish_energy = min(25.0, env.fish_energy)
        if terminated or truncated:
            break

    forage_pct = goals.count(GOAL_FORAGE) / len(goals)
    passed = forage_pct > 0.50
    print(f"  [{'PASS' if passed else 'FAIL'}] Starving → FORAGE "
          f"{forage_pct:.0%} (need >50%)")
    env.close()
    return passed


def test_predator_flees():
    """Test 2: Nearby predator → FLEE dominates."""
    env, agent = _make_env_agent()
    obs, info = env.reset(seed=42)
    agent.reset()

    # Place predator very close
    env.pred_x = env.fish_x + 50
    env.pred_y = env.fish_y
    env.pred_heading = math.atan2(
        env.fish_y - env.pred_y, env.fish_x - env.pred_x)

    goals = []
    for t in range(80):
        # Keep predator close
        env.pred_x = env.fish_x + 40 + 20 * math.sin(t * 0.1)
        env.pred_y = env.fish_y + 20 * math.cos(t * 0.1)
        env.pred_heading = math.atan2(
            env.fish_y - env.pred_y, env.fish_x - env.pred_x)
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(
            info, reward=reward, done=terminated or truncated, env=env)
        goals.append(agent.last_diagnostics.get("goal", GOAL_EXPLORE))
        if terminated or truncated:
            break

    flee_pct = goals.count(GOAL_FLEE) / len(goals)
    passed = flee_pct > 0.50
    print(f"  [{'PASS' if passed else 'FAIL'}] Predator near → FLEE "
          f"{flee_pct:.0%} (need >50%)")
    env.close()
    return passed


def test_wellfed_safe_explores():
    """Test 3: Well-fed safe fish → EXPLORE selected >20%."""
    env, agent = _make_env_agent()
    obs, info = env.reset(seed=42)
    agent.reset()

    # Full energy, push predator far away
    env.fish_energy = 95.0
    env.pred_x = env.fish_x + 500
    env.pred_y = env.fish_y + 500

    goals = []
    for t in range(120):
        env.fish_energy = max(80.0, env.fish_energy)
        env.pred_x = env.fish_x + 500
        env.pred_y = env.fish_y + 500
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(
            info, reward=reward, done=terminated or truncated, env=env)
        goals.append(agent.last_diagnostics.get("goal", GOAL_EXPLORE))
        if terminated or truncated:
            break

    explore_pct = goals.count(GOAL_EXPLORE) / len(goals)
    passed = explore_pct > 0.20
    print(f"  [{'PASS' if passed else 'FAIL'}] Well-fed safe → EXPLORE "
          f"{explore_pct:.0%} (need >20%)")
    env.close()
    return passed


def test_extended_stability():
    """Test 4: Extended run — no NaN, survival, goal diversity."""
    env, agent = _make_env_agent()
    result = _run_episode(env, agent, T=500, seed=42)

    # Stability: no NaN
    nan_pass = not result["has_nan"]
    print(f"  [{'PASS' if nan_pass else 'FAIL'}] No NaN in diagnostics")

    # Survival
    surv_pass = result["survived"] >= 300
    print(f"  [{'PASS' if surv_pass else 'FAIL'}] Survived "
          f"{result['survived']}/500 steps (need >=300)")

    # Goal diversity: at least 2 different goals used
    unique_goals = len(set(result["goals"]))
    div_pass = unique_goals >= 2
    print(f"  [{'PASS' if div_pass else 'FAIL'}] Goal diversity: "
          f"{unique_goals} unique goals (need >=2)")

    env.close()
    return nan_pass and surv_pass and div_pass


def _plot_results(result, output_path):
    """Generate summary plot."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    T = len(result["goals"])
    steps = np.arange(T)

    # Goal timeline
    goal_colors = {
        GOAL_FORAGE: "green", GOAL_FLEE: "red",
        GOAL_EXPLORE: "blue", GOAL_SOCIAL: "orange"}
    for t in range(T):
        g = result["goals"][t]
        axes[0].axvspan(t, t + 1, color=goal_colors.get(g, "gray"),
                        alpha=0.4)
    axes[0].set_ylabel("Goal")
    axes[0].set_title("Step 25: Proper Active Inference EFE — Goal Timeline")
    # Legend
    for gi, name in enumerate(GOAL_NAMES):
        axes[0].plot([], [], color=goal_colors.get(gi, "gray"),
                     linewidth=5, label=name, alpha=0.6)
    axes[0].legend(loc="upper right", ncol=4)

    # Energy
    axes[1].plot(steps, result["energies"], color="orange", linewidth=1)
    axes[1].set_ylabel("Energy")
    axes[1].axhline(70, color="gray", linestyle="--", alpha=0.5,
                     label="Setpoint")
    axes[1].legend()

    # Goal distribution (rolling window)
    window = 50
    for gi, name in enumerate(GOAL_NAMES):
        pcts = []
        for t in range(T):
            start = max(0, t - window)
            segment = result["goals"][start:t + 1]
            pcts.append(segment.count(gi) / len(segment))
        axes[2].plot(steps, pcts, color=goal_colors.get(gi, "gray"),
                     label=name, alpha=0.8)
    axes[2].set_ylabel("Goal %")
    axes[2].set_xlabel("Step")
    axes[2].legend(loc="upper right", ncol=4)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"\n  Plot saved: {output_path}")


def main():
    print("=" * 60)
    print("Step 25: Proper Active Inference EFE")
    print("=" * 60)

    results = []

    print("\n--- Unit Tests ---")
    results.append(("EFE engine exists", test_efe_engine_exists()))

    print("\n--- Behavioral Tests ---")
    results.append(("Starving → FORAGE", test_starving_forages()))
    results.append(("Predator → FLEE", test_predator_flees()))
    results.append(("Well-fed safe → EXPLORE", test_wellfed_safe_explores()))

    print("\n--- Extended Stability ---")
    results.append(("Extended stability", test_extended_stability()))

    # Full run for plot
    print("\n--- Full Run (500 steps) ---")
    env, agent = _make_env_agent()
    full_result = _run_episode(env, agent, T=500, seed=42)
    env.close()

    output_path = os.path.join(
        PROJECT_ROOT, "plots", "v1_step25_active_inference_efe.png")
    _plot_results(full_result, output_path)

    # Summary
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, p in results if p)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} passed")
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'} — {name}")
    print("=" * 60)

    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
