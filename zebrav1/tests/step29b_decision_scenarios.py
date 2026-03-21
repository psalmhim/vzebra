"""
Step 29b: Structured Decision Scenarios

Tests zebrafish intelligence with 5 designed dilemmas that require
rational decision-making. Each scenario has a clear optimal choice.

Scenarios:
  A. Safe vs Risky food: dense food near predator vs sparse food far away
  B. Occluded vs Open food: food behind rock vs food in open water
  C. Predator charge: predator suddenly approaches — flee or freeze?
  D. Starvation dilemma: low energy + predator nearby — forage or flee?
  E. Detour planning: food visible but obstacle between — navigate around?

Run: python -m zebrav1.tests.step29b_decision_scenarios
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


def _setup_scenario(env, scenario):
    """Configure environment for a specific decision scenario."""

    # Seed numpy for reproducible food placement across runs/code versions
    _seed_map = {"A": 101, "B": 102, "C": 103, "D": 104, "E": 105}
    np.random.seed(_seed_map.get(scenario, 100))

    # Clear existing food
    env.foods = []

    if scenario == "A":
        # SAFE vs RISKY: dense food near predator (risky), sparse food far (safe)
        # Fish starts at center. Predator near top-right.
        # Dense patch (8 food) at top-right near predator.
        # Sparse patch (3 food) at bottom-left, far from predator.
        env.fish_x = env.arena_w * 0.5
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x = env.arena_w * 0.8
        env.pred_y = env.arena_h * 0.2
        env.pred_heading = math.pi  # facing fish
        env.pred_state = "STALK"
        # Risky dense patch (near predator)
        for _ in range(8):
            fx = env.pred_x + np.random.uniform(-40, 40)
            fy = env.pred_y + np.random.uniform(-40, 40)
            env.foods.append([fx, fy, "small"])
        # Safe sparse patch (far from predator)
        safe_x = env.arena_w * 0.2
        safe_y = env.arena_h * 0.8
        for _ in range(3):
            fx = safe_x + np.random.uniform(-30, 30)
            fy = safe_y + np.random.uniform(-30, 30)
            env.foods.append([fx, fy, "small"])

    elif scenario == "B":
        # OCCLUDED vs OPEN: food behind rock vs food in open water
        env.fish_x = env.arena_w * 0.5
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x = 30
        env.pred_y = 30
        env.pred_state = "PATROL"
        # Food behind rock (occluded)
        rock_x = env.arena_w * 0.7
        rock_y = env.arena_h * 0.5
        # Place food directly behind the first rock formation
        for rock in getattr(env, 'rock_formations', []):
            rc = (rock["cx"], rock["cy"])
            for _ in range(5):
                fx = rc[0] + np.random.uniform(-20, 20)
                fy = rc[1] + np.random.uniform(-20, 20)
                env.foods.append([fx, fy, "small"])
            break
        # Open food (no obstacles between)
        open_x = env.arena_w * 0.3
        open_y = env.arena_h * 0.3
        for _ in range(4):
            fx = open_x + np.random.uniform(-25, 25)
            fy = open_y + np.random.uniform(-25, 25)
            env.foods.append([fx, fy, "small"])

    elif scenario == "C":
        # PREDATOR CHARGE: predator suddenly rushes fish. Must flee.
        env.fish_x = env.arena_w * 0.5
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.fish_energy = 80.0
        # Predator at 120px away, facing fish, in HUNT mode
        env.pred_x = env.fish_x + 120
        env.pred_y = env.fish_y
        env.pred_heading = math.pi  # facing fish
        env.pred_state = "HUNT"
        env.pred_stamina = 1.0
        # Some food around (test if fish flees instead of foraging)
        for _ in range(6):
            fx = env.fish_x + np.random.uniform(-80, 80)
            fy = env.fish_y + np.random.uniform(-80, 80)
            env.foods.append([fx, fy, "small"])

    elif scenario == "D":
        # STARVATION DILEMMA: very low energy + predator nearby
        # Rational: must forage despite predator (will die of starvation otherwise)
        env.fish_x = env.arena_w * 0.5
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.fish_energy = 15.0  # critically low
        env.pred_x = env.fish_x + 100
        env.pred_y = env.fish_y + 50
        env.pred_heading = math.atan2(-50, -100)
        env.pred_state = "STALK"
        # Food in opposite direction from predator
        for _ in range(5):
            fx = env.fish_x - np.random.uniform(30, 80)
            fy = env.fish_y - np.random.uniform(30, 80)
            fx = max(20, fx)
            fy = max(20, fy)
            env.foods.append([fx, fy, "small"])

    elif scenario == "E":
        # DETOUR: food visible but obstacle between fish and food
        env.fish_x = env.arena_w * 0.3
        env.fish_y = env.arena_h * 0.5
        env.fish_heading = 0.0
        env.pred_x = 30
        env.pred_y = 30
        env.pred_state = "PATROL"
        # Food on other side of nearest rock
        for rock in getattr(env, 'rock_formations', []):
            rc = (rock["cx"], rock["cy"])
            # Place food on far side of rock from fish
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
    """Run a single scenario and evaluate the decision."""
    obs, info = env.reset(seed=42)
    agent.reset()

    # Let the brain warm up for 20 steps before setting scenario
    for _ in range(20):
        action = agent.act(obs, env)
        obs, _, _, _, info = env.step(action)
        agent.update_post_step(info, reward=0, done=False, env=env)

    # Setup the scenario
    _setup_scenario(env, scenario)

    # Run and collect metrics
    positions = []
    goals = []
    pred_dists = []
    food_eaten = 0
    initial_food = len(env.foods)
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

    goals = np.array(goals)
    pred_dists = np.array(pred_dists)
    positions = np.array(positions)

    return {
        "positions": positions,
        "goals": goals,
        "pred_dists": pred_dists,
        "food_eaten": food_eaten,
        "survived": survived,
        "steps": t + 1,
    }


def _score_scenario(scenario, result):
    """Score the decision quality for each scenario (0-100)."""

    if scenario == "A":
        # SAFE vs RISKY: should avoid dense-risky patch, eat from safe patch
        # Good: survived + ate food while keeping distance from predator
        score = 0
        if result["survived"]:
            score += 40
        # Did it keep distance from predator? (mean > 150px = safe)
        mean_pred_dist = result["pred_dists"].mean()
        score += min(30, mean_pred_dist / 150.0 * 30)
        # Did it eat anything?
        score += min(30, result["food_eaten"] * 10)
        rationale = (f"survived={'Y' if result['survived'] else 'N'}, "
                     f"mean_pred_dist={mean_pred_dist:.0f}, "
                     f"food={result['food_eaten']}")

    elif scenario == "B":
        # OCCLUDED vs OPEN: should prefer open food over occluded
        score = 0
        if result["survived"]:
            score += 30
        # Did it eat food? (open food should be eaten first)
        score += min(40, result["food_eaten"] * 10)
        # Did it spend most time exploring, not stuck at rock?
        forage_pct = (result["goals"] == GOAL_FORAGE).mean()
        score += min(30, forage_pct * 50)
        rationale = (f"food={result['food_eaten']}, "
                     f"forage%={forage_pct:.0%}")

    elif scenario == "C":
        # PREDATOR CHARGE: must flee immediately
        score = 0
        if result["survived"]:
            score += 50
        # How quickly did it switch to FLEE? (within first 10 steps = best)
        first_flee = None
        for i, g in enumerate(result["goals"]):
            if g == GOAL_FLEE:
                first_flee = i
                break
        if first_flee is not None:
            latency_score = max(0, 30 - first_flee * 3)
            score += latency_score
        # Did it increase distance from predator?
        if len(result["pred_dists"]) > 20:
            dist_increase = result["pred_dists"][-10:].mean() - result["pred_dists"][:10].mean()
            score += min(20, max(0, dist_increase / 100.0 * 20))
        rationale = (f"survived={'Y' if result['survived'] else 'N'}, "
                     f"first_flee={first_flee}, "
                     f"flee%={(result['goals'] == GOAL_FLEE).mean():.0%}")

    elif scenario == "D":
        # STARVATION DILEMMA: must forage despite predator
        score = 0
        if result["survived"]:
            score += 30
        # Did it forage? (critical test — FORAGE should dominate)
        forage_pct = (result["goals"] == GOAL_FORAGE).mean()
        score += min(40, forage_pct * 60)
        # Did it eat food? (must eat to survive)
        score += min(30, result["food_eaten"] * 15)
        rationale = (f"forage%={forage_pct:.0%}, "
                     f"food={result['food_eaten']}, "
                     f"survived={'Y' if result['survived'] else 'N'}")

    elif scenario == "E":
        # DETOUR: navigate around obstacle to reach food
        score = 0
        if result["survived"]:
            score += 20
        # Did it eat food? (requires successful detour)
        score += min(50, result["food_eaten"] * 15)
        # Did it explore (needed for detour)?
        explore_pct = (result["goals"] == GOAL_EXPLORE).mean()
        forage_pct = (result["goals"] == GOAL_FORAGE).mean()
        score += min(30, (explore_pct + forage_pct) * 40)
        rationale = (f"food={result['food_eaten']}, "
                     f"forage%={forage_pct:.0%}, "
                     f"explore%={explore_pct:.0%}")

    return score, rationale


def run_step29b():
    import torch
    # Global seeds for fully reproducible evaluation across code versions
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Step 29b: Structured Decision Scenarios")
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
    agent = BrainAgent(device="auto", world_model="place_cell",
                       use_allostasis=True)

    # Load trained classifier weights for consistent evaluation
    # (brain_checkpoint may have old CLS head dimensions — skip it here)
    _cls_ckpt = os.path.join(PROJECT_ROOT, "zebrav1", "weights", "classifier_wfb.pt")
    if os.path.exists(_cls_ckpt):
        _cls_state = torch.load(_cls_ckpt, map_location="cpu", weights_only=False)
        _snn_state = _cls_state.get("snn", _cls_state)
        _current = agent.model.state_dict()
        _filtered = {k: v for k, v in _snn_state.items()
                     if k in _current and v.shape == _current[k].shape}
        agent.model.load_state_dict(_filtered, strict=False)

    # Re-seed after initialization to ensure deterministic warm-up
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
        grade = ("A" if score >= 80 else "B" if score >= 60
                 else "C" if score >= 40 else "D" if score >= 20 else "F")
        print(f"  Score: {score:.0f}/100 [{grade}]  ({rationale})")

    env.close()

    # Overall
    overall = np.mean(list(scores.values()))
    print(f"\n{'=' * 60}")
    print(f"OVERALL DECISION QUALITY: {overall:.0f}/100")
    if overall >= 70:
        print("VERDICT: RATIONAL — makes intelligent trade-offs")
    elif overall >= 50:
        print("VERDICT: ADEQUATE — better than random, room for improvement")
    else:
        print("VERDICT: POOR — needs improvement in decision-making")
    print(f"{'=' * 60}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, (sid, desc) in enumerate(scenarios.items()):
        if idx >= 5:
            break
        ax = axes_flat[idx]
        r = results[sid]
        s = scores[sid]
        pos = r["positions"]
        goals = r["goals"]

        colors_map = {0: "green", 1: "red", 2: "blue", 3: "cyan"}
        for g, c in colors_map.items():
            mask = goals == g
            if mask.sum() > 0:
                ax.scatter(pos[mask, 0], pos[mask, 1], c=c, s=3, alpha=0.6)
        ax.plot(pos[0, 0], pos[0, 1], "ko", markersize=8, label="Start")
        ax.plot(pos[-1, 0], pos[-1, 1], "ks", markersize=8, label="End")
        grade = ("A" if s >= 80 else "B" if s >= 60
                 else "C" if s >= 40 else "D" if s >= 20 else "F")
        ax.set_title(f"{sid}: {desc}\nScore: {s:.0f} [{grade}]", fontsize=9)
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 600)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    # Summary bar chart in last panel
    ax = axes_flat[5]
    sids = list(scores.keys())
    vals = [scores[s] for s in sids]
    bar_colors = ["green" if v >= 60 else "orange" if v >= 40 else "red"
                  for v in vals]
    ax.barh(range(len(sids)), vals, color=bar_colors, alpha=0.7)
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels([f"{s}: {scenarios[s][:20]}" for s in sids], fontsize=8)
    ax.set_xlim(0, 100)
    ax.axvline(60, color="green", linestyle="--", alpha=0.3)
    ax.set_xlabel("Score")
    ax.set_title(f"Overall: {overall:.0f}/100")

    fig.suptitle("Step 29b: Decision Scenario Evaluation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                             "v1_step29b_decision_scenarios.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")


if __name__ == "__main__":
    run_step29b()
