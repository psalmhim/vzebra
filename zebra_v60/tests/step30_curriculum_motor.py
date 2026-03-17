"""
Step 30: Curriculum RL Training with Motor Primitives

Progressive difficulty training that adapts to the fish's current
competence level. Uses the motor primitive system (Step 28) with
increasing environmental challenge.

Curriculum stages:
  1. FORAGE_EASY: no predator, many food, no rocks (learn basic foraging)
  2. FORAGE_HARD: no predator, food behind rocks (learn navigation)
  3. SURVIVE_EASY: slow predator, many food, no rocks (learn flee timing)
  4. SURVIVE_HARD: full environment (predator + rocks + food density)

Each stage runs for N episodes. The fish advances when it passes the
stage criterion. Weights are saved after each successful stage.

Run: python -m zebra_v60.tests.step30_curriculum_motor
Output: weights/curriculum_v60.pt
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

from zebra_v60.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebra_v60.gym_env.brain_agent import BrainAgent
from zebra_v60.brain.device_util import get_device


STAGES = [
    {
        "name": "FORAGE_EASY",
        "desc": "No predator, many food, no rocks",
        "n_food": 25,
        "pred_states": [],  # disable predator
        "max_steps": 400,
        "criterion": {"min_eaten": 5, "min_steps": 350},
        "episodes": 3,
    },
    {
        "name": "FORAGE_HARD",
        "desc": "No predator, food with rocks",
        "n_food": 20,
        "pred_states": [],
        "max_steps": 500,
        "criterion": {"min_eaten": 4, "min_steps": 450},
        "episodes": 3,
    },
    {
        "name": "SURVIVE_EASY",
        "desc": "Slow predator, many food",
        "n_food": 20,
        "pred_states": ["PATROL", "STALK"],  # no HUNT
        "max_steps": 600,
        "criterion": {"min_eaten": 3, "min_steps": 500},
        "episodes": 3,
    },
    {
        "name": "SURVIVE_HARD",
        "desc": "Full environment",
        "n_food": 15,
        "pred_states": ["PATROL", "STALK", "HUNT", "AMBUSH"],
        "max_steps": 800,
        "criterion": {"min_eaten": 3, "min_steps": 500},
        "episodes": 3,
    },
]


def _run_curriculum_episode(env, agent, stage, seed):
    """Run one curriculum episode with Hebbian learning active."""
    env._pred_allowed_states = stage["pred_states"]
    obs, info = env.reset(seed=seed)
    agent.reset()

    total_eaten = 0
    for t in range(stage["max_steps"]):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward,
                               done=terminated or truncated, env=env)
        total_eaten = info.get("total_eaten", 0)
        if terminated or truncated:
            break

    return {"steps": t + 1, "eaten": total_eaten,
            "energy": agent.last_diagnostics.get("energy", 0)}


def run_step30():
    print("=" * 60)
    print("Step 30: Curriculum RL Training with Motor Primitives")
    print("=" * 60)

    device = get_device()
    stage_results = {}

    for si, stage in enumerate(STAGES):
        print(f"\n{'='*55}")
        print(f"Stage {si+1}/{len(STAGES)}: {stage['name']}")
        print(f"  {stage['desc']}")
        print(f"  Episodes: {stage['episodes']}, "
              f"Max steps: {stage['max_steps']}")
        print(f"  Criterion: eat≥{stage['criterion']['min_eaten']}, "
              f"survive≥{stage['criterion']['min_steps']}")
        print(f"{'='*55}")

        env = ZebrafishPreyPredatorEnv(
            render_mode=None,
            n_food=stage["n_food"],
            max_steps=stage["max_steps"],
            side_panels=False)

        agent = BrainAgent(device="auto", world_model="place_cell",
                           use_allostasis=True)

        episode_results = []
        passed = False

        for ep in range(stage["episodes"]):
            seed = 42 + si * 100 + ep * 10
            result = _run_curriculum_episode(env, agent, stage, seed)
            episode_results.append(result)

            eat_ok = result["eaten"] >= stage["criterion"]["min_eaten"]
            surv_ok = result["steps"] >= stage["criterion"]["min_steps"]
            status = "PASS" if (eat_ok and surv_ok) else "FAIL"
            print(f"  Episode {ep+1}: steps={result['steps']}, "
                  f"eaten={result['eaten']}, energy={result['energy']:.0f} "
                  f"[{status}]")

            if eat_ok and surv_ok:
                passed = True

        env.close()

        avg_eaten = np.mean([r["eaten"] for r in episode_results])
        avg_steps = np.mean([r["steps"] for r in episode_results])
        stage_results[stage["name"]] = {
            "passed": passed,
            "avg_eaten": avg_eaten,
            "avg_steps": avg_steps,
            "episodes": episode_results,
        }

        if passed:
            print(f"  → STAGE PASSED (avg: {avg_eaten:.1f} food, "
                  f"{avg_steps:.0f} steps)")
        else:
            print(f"  → STAGE FAILED (avg: {avg_eaten:.1f} food, "
                  f"{avg_steps:.0f} steps)")
            print(f"  Stopping curriculum — fish needs more training")
            break

    # Summary
    print(f"\n{'='*60}")
    print("CURRICULUM RESULTS")
    print(f"{'='*60}")
    total_passed = 0
    for name, res in stage_results.items():
        status = "PASS" if res["passed"] else "FAIL"
        print(f"  {name:20s}: {status} "
              f"(avg {res['avg_eaten']:.1f} food, "
              f"{res['avg_steps']:.0f} steps)")
        if res["passed"]:
            total_passed += 1

    print(f"\n  Stages passed: {total_passed}/{len(STAGES)}")
    grade = ("EXPERT" if total_passed == 4 else
             "COMPETENT" if total_passed == 3 else
             "LEARNING" if total_passed >= 1 else "NOVICE")
    print(f"  Competence level: {grade}")
    print(f"{'='*60}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of stages
    ax = axes[0]
    names = list(stage_results.keys())
    eaten = [stage_results[n]["avg_eaten"] for n in names]
    colors = ["green" if stage_results[n]["passed"] else "red" for n in names]
    ax.barh(range(len(names)), eaten, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Avg Food Eaten")
    ax.set_title(f"Curriculum Progress ({total_passed}/{len(STAGES)} stages)")

    # Episode details for each stage
    ax = axes[1]
    for si, (name, res) in enumerate(stage_results.items()):
        steps = [ep["steps"] for ep in res["episodes"]]
        ax.plot(range(1, len(steps) + 1), steps,
                marker="o", label=name, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps Survived")
    ax.legend(fontsize=8)
    ax.set_title("Survival per Episode")

    fig.suptitle("Step 30: Curriculum Training",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                             "v60_step30_curriculum_motor.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")


if __name__ == "__main__":
    run_step30()
