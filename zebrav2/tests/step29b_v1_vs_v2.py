"""
v1 vs v2 Decision Scenario Comparison

Runs the same 5 decision scenarios from step29b using ZebrafishBrainV2
and compares to v1's 89/100 RATIONAL score.

Run: python -m zebrav2.tests.step29b_v1_vs_v2
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
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory as _inject_sensory

GOAL_FORAGE = 0
GOAL_FLEE = 1
GOAL_EXPLORE = 2


class BrainV2Agent:
    """Wraps ZebrafishBrainV2 in a v1 BrainAgent-compatible interface."""

    def __init__(self, device=None):
        from zebrav2.spec import DEVICE
        dev = device or DEVICE
        self.brain = ZebrafishBrainV2(device=dev)
        self.last_diagnostics = {}

    def reset(self):
        self.brain.reset()
        self.last_diagnostics = {}

    def act(self, obs, env):
        # Signal previous goal's flee state to env BEFORE step()
        # (v1 BrainAgent calls set_flee_active before env.step)
        is_flee = (self.brain.current_goal == GOAL_FLEE)
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(is_flee, panic_intensity=0.8 if is_flee else 0.0)

        # Inject geometric sensory inputs that v2 brain expects from env
        _inject_sensory(env)
        out = self.brain.step(obs, env)
        self.last_diagnostics = {
            'goal': out['goal'],
            'turn_rate': out['turn'],
            'speed': out['speed'],
        }
        return np.array([out['turn'], out['speed']], dtype=np.float32)

    def update_post_step(self, info, reward=0, done=False, env=None):
        # v2 brain handles updates internally in step()
        pass


def _setup_scenario(env, scenario):
    """Configure environment for a specific decision scenario (same as v1 step29b)."""
    _seed_map = {"A": 101, "B": 102, "C": 103, "D": 104, "E": 105}
    np.random.seed(_seed_map.get(scenario, 100))
    env.foods = []

    if scenario == "A":
        env.fish_x = env.arena_w * 0.5
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x = env.arena_w * 0.8
        env.pred_y = env.arena_h * 0.2
        env.pred_heading = math.pi
        env.pred_state = "STALK"
        for _ in range(8):
            fx = env.pred_x + np.random.uniform(-40, 40)
            fy = env.pred_y + np.random.uniform(-40, 40)
            env.foods.append([fx, fy, "small"])
        safe_x = env.arena_w * 0.2
        safe_y = env.arena_h * 0.8
        for _ in range(3):
            fx = safe_x + np.random.uniform(-30, 30)
            fy = safe_y + np.random.uniform(-30, 30)
            env.foods.append([fx, fy, "small"])

    elif scenario == "B":
        env.fish_x = env.arena_w * 0.5
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x = 30
        env.pred_y = 30
        env.pred_state = "PATROL"
        for rock in getattr(env, 'rock_formations', []):
            rc = (rock["cx"], rock["cy"])
            for _ in range(5):
                fx = rc[0] + np.random.uniform(-20, 20)
                fy = rc[1] + np.random.uniform(-20, 20)
                env.foods.append([fx, fy, "small"])
            break
        open_x = env.arena_w * 0.3
        open_y = env.arena_h * 0.3
        for _ in range(4):
            fx = open_x + np.random.uniform(-25, 25)
            fy = open_y + np.random.uniform(-25, 25)
            env.foods.append([fx, fy, "small"])

    elif scenario == "C":
        env.fish_x = env.arena_w * 0.5
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.fish_energy = 80.0
        env.pred_x = env.fish_x + 120
        env.pred_y = env.fish_y
        env.pred_heading = math.pi
        env.pred_state = "HUNT"
        env.pred_stamina = 1.0
        for _ in range(6):
            fx = env.fish_x + np.random.uniform(-80, 80)
            fy = env.fish_y + np.random.uniform(-80, 80)
            env.foods.append([fx, fy, "small"])

    elif scenario == "D":
        env.fish_x = env.arena_w * 0.5
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.fish_energy = 15.0
        env.pred_x = env.fish_x + 100
        env.pred_y = env.fish_y + 50
        env.pred_heading = math.atan2(-50, -100)
        env.pred_state = "STALK"
        for _ in range(5):
            fx = env.fish_x - np.random.uniform(30, 80)
            fy = env.fish_y - np.random.uniform(30, 80)
            fx = max(20, fx)
            fy = max(20, fy)
            env.foods.append([fx, fy, "small"])

    elif scenario == "E":
        env.fish_x = env.arena_w * 0.3
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x = 30
        env.pred_y = 30
        env.pred_state = "PATROL"
        for rock in getattr(env, 'rock_formations', []):
            rc = (rock["cx"], rock["cy"])
            dx = rc[0] - env.fish_x
            dy = rc[1] - env.fish_y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-8
            far_x = rc[0] + (dx / dist) * 50
            far_y = rc[1] + (dy / dist) * 50
            far_x = max(20, min(env.arena_w - 20, far_x))
            far_y = max(20, min(env.arena_h - 20, far_y))
            for _ in range(4):
                fx = far_x + np.random.uniform(-20, 20)
                fy = far_y + np.random.uniform(-20, 20)
                env.foods.append([fx, fy, "small"])
            break


def _run_scenario(env, agent, scenario, T=200):
    """Run a single scenario and collect metrics."""
    import random
    _seed_map = {"A": 101, "B": 102, "C": 103, "D": 104, "E": 105}
    _s = _seed_map.get(scenario, 100)
    random.seed(_s)
    np.random.seed(_s)

    obs, info = env.reset(seed=42)
    agent.reset()

    # Warm up 20 steps
    for _ in range(20):
        action = agent.act(obs, env)
        obs, _, _, _, info = env.step(action)
        agent.update_post_step(info, reward=0, done=False, env=env)

    _setup_scenario(env, scenario)

    positions = []
    goals = []
    pred_dists = []
    food_eaten = 0
    survived = True

    for t in range(T):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward,
                               done=terminated or truncated, env=env)

        diag = agent.last_diagnostics
        positions.append((env.fish_x, env.fish_y))
        goals.append(diag.get("goal", 2))
        pred_dists.append(math.sqrt(
            (env.fish_x - env.pred_x) ** 2 +
            (env.fish_y - env.pred_y) ** 2))
        food_eaten = info.get("total_eaten", 0)

        if terminated:
            survived = False
            break

    return {
        "positions": np.array(positions),
        "goals": np.array(goals),
        "pred_dists": np.array(pred_dists),
        "food_eaten": food_eaten,
        "survived": survived,
        "steps": t + 1,
    }


def _score_scenario(scenario, result):
    """Same scoring as v1 step29b."""
    if scenario == "A":
        score = 0
        if result["survived"]:
            score += 40
        mean_pred_dist = result["pred_dists"].mean()
        score += min(30, mean_pred_dist / 150.0 * 30)
        score += min(30, result["food_eaten"] * 10)
        rationale = (f"survived={'Y' if result['survived'] else 'N'}, "
                     f"mean_pred_dist={mean_pred_dist:.0f}, "
                     f"food={result['food_eaten']}")

    elif scenario == "B":
        score = 0
        if result["survived"]:
            score += 30
        score += min(40, result["food_eaten"] * 10)
        forage_pct = (result["goals"] == GOAL_FORAGE).mean()
        score += min(30, forage_pct * 50)
        rationale = (f"food={result['food_eaten']}, "
                     f"forage%={forage_pct:.0%}")

    elif scenario == "C":
        score = 0
        if result["survived"]:
            score += 50
        first_flee = None
        for i, g in enumerate(result["goals"]):
            if g == GOAL_FLEE:
                first_flee = i
                break
        if first_flee is not None:
            score += max(0, 30 - first_flee * 3)
        if len(result["pred_dists"]) > 20:
            dist_increase = result["pred_dists"][-10:].mean() - result["pred_dists"][:10].mean()
            score += min(20, max(0, dist_increase / 100.0 * 20))
        rationale = (f"survived={'Y' if result['survived'] else 'N'}, "
                     f"first_flee={first_flee}, "
                     f"flee%={(result['goals'] == GOAL_FLEE).mean():.0%}")

    elif scenario == "D":
        score = 0
        if result["survived"]:
            score += 30
        forage_pct = (result["goals"] == GOAL_FORAGE).mean()
        score += min(40, forage_pct * 60)
        score += min(30, result["food_eaten"] * 15)
        rationale = (f"forage%={forage_pct:.0%}, "
                     f"food={result['food_eaten']}, "
                     f"survived={'Y' if result['survived'] else 'N'}")

    elif scenario == "E":
        score = 0
        if result["survived"]:
            score += 20
        score += min(50, result["food_eaten"] * 15)
        explore_pct = (result["goals"] == GOAL_EXPLORE).mean()
        forage_pct = (result["goals"] == GOAL_FORAGE).mean()
        score += min(30, (explore_pct + forage_pct) * 40)
        rationale = (f"food={result['food_eaten']}, "
                     f"forage%={forage_pct:.0%}, "
                     f"explore%={explore_pct:.0%}")

    return score, rationale


# v1 reference scores
V1_SCORES = {"A": 100, "B": 85, "C": 80, "D": 100, "E": 79}
V1_OVERALL = 89


def run_comparison():
    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("v1 vs v2: Structured Decision Scenarios")
    print("=" * 60)
    print(f"v1 baseline: {V1_OVERALL}/100 RATIONAL")
    print("=" * 60)

    scenarios = {
        "A": "Safe vs Risky food patch",
        "B": "Occluded vs Open food",
        "C": "Predator charge — flee timing",
        "D": "Starvation dilemma — forage vs flee",
        "E": "Detour around obstacle",
    }

    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=300, side_panels=False)
    agent = BrainV2Agent()

    np.random.seed(42)
    torch.manual_seed(42)

    results = {}
    scores = {}

    for scenario_id, description in scenarios.items():
        print(f"\n--- Scenario {scenario_id}: {description} ---")
        result = _run_scenario(env, agent, scenario_id, T=200)
        score, rationale = _score_scenario(scenario_id, result)
        results[scenario_id] = result
        scores[scenario_id] = score
        v1_score = V1_SCORES[scenario_id]
        delta = score - v1_score
        grade = ("A" if score >= 80 else "B" if score >= 60
                 else "C" if score >= 40 else "D" if score >= 20 else "F")
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        print(f"  v2: {score:.0f}/100 [{grade}]  ({rationale})")
        print(f"  v1: {v1_score}/100  delta: {delta_str}")

    env.close()

    overall_v2 = np.mean(list(scores.values()))
    delta_overall = overall_v2 - V1_OVERALL
    delta_str = f"+{delta_overall:.1f}" if delta_overall >= 0 else f"{delta_overall:.1f}"

    print(f"\n{'=' * 60}")
    print(f"v2 OVERALL DECISION QUALITY: {overall_v2:.0f}/100  (v1: {V1_OVERALL})  delta: {delta_str}")
    if overall_v2 >= 92:
        print("VERDICT: EXCEEDED target (≥92/100)")
    elif overall_v2 >= 89:
        print("VERDICT: MATCHED v1 RATIONAL score")
    elif overall_v2 >= 70:
        print("VERDICT: RATIONAL — below target, needs tuning")
    else:
        print("VERDICT: ADEQUATE — significant gap vs v1")
    print(f"{'=' * 60}")

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, (sid, desc) in enumerate(scenarios.items()):
        ax = axes_flat[idx]
        r = results[sid]
        pos = r["positions"]
        goals = r["goals"]

        colors_map = {0: "green", 1: "red", 2: "blue", 3: "cyan"}
        for g, c in colors_map.items():
            mask = goals == g
            if mask.sum() > 0:
                ax.scatter(pos[mask, 0], pos[mask, 1], c=c, s=3, alpha=0.6)
        ax.plot(pos[0, 0], pos[0, 1], "ko", markersize=8, label="Start")
        ax.plot(pos[-1, 0], pos[-1, 1], "ks", markersize=8, label="End")
        v2s = scores[sid]
        v1s = V1_SCORES[sid]
        grade = ("A" if v2s >= 80 else "B" if v2s >= 60 else "C")
        ax.set_title(f"{sid}: {desc}\nv2={v2s:.0f} [{grade}]  v1={v1s}", fontsize=9)
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 600)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    # Summary bar comparison
    ax = axes_flat[5]
    sids = list(scores.keys())
    v2_vals = [scores[s] for s in sids]
    v1_vals = [V1_SCORES[s] for s in sids]
    x = np.arange(len(sids))
    width = 0.35
    ax.barh(x + width/2, v2_vals, width, color="steelblue", alpha=0.8, label="v2")
    ax.barh(x - width/2, v1_vals, width, color="orange", alpha=0.8, label="v1")
    ax.set_yticks(x)
    ax.set_yticklabels(sids, fontsize=10)
    ax.set_xlim(0, 100)
    ax.axvline(60, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Score")
    ax.set_title(f"v2: {overall_v2:.0f}/100  v1: {V1_OVERALL}/100")
    ax.legend(loc="lower right")

    fig.suptitle("v1 vs v2 Decision Scenario Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v2_vs_v1_decision_scenarios.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    return scores, overall_v2


if __name__ == "__main__":
    run_comparison()
