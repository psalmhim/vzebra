"""
Step 26b: Compare original vs W_FB-trained weights

Quick 800-step evaluation with Place Cell config to compare:
  - Foraging performance (food eaten, energy)
  - Prediction error (PE) levels
  - Goal distribution

Run: python -m zebrav1.tests.step26b_compare_wfb
"""
import os
import sys
import math

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav1.gym_env.brain_agent import BrainAgent


def run_eval(weight_label, weight_override=None, T=800, seed=42):
    """Run one episode and return stats."""
    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=T, side_panels=False)
    agent = BrainAgent(device="auto", world_model="place_cell",
                       use_allostasis=True)

    # Override weights if specified
    if weight_override and os.path.exists(weight_override):
        state = torch.load(weight_override, map_location=agent.model.device,
                           weights_only=True)
        agent.model.load_saveable_state(state)

    obs, info = env.reset(seed=seed)
    agent.reset()

    goals = []
    energies = []
    pe_vals = []

    for t in range(T):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward,
                               done=terminated or truncated, env=env)

        diag = agent.last_diagnostics
        goals.append(diag.get("goal", 2))
        energies.append(diag.get("energy", 100.0))

        # Collect PE from SNN layers
        model = agent.model
        pe_total = 0.0
        for layer in [model.OT_F, model.PT_L, model.PC_per, model.PC_int]:
            if hasattr(layer, 'pred_error') and layer.pred_error is not None:
                pe_total += layer.pred_error.pow(2).mean().item()
        pe_vals.append(pe_total)

        if terminated or truncated:
            if terminated:
                print(f"  [{weight_label}] CAUGHT at step {t}!")
            break

    env.close()

    goals_arr = np.array(goals)
    goal_names = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
    goal_pcts = {goal_names[i]: 100 * np.sum(goals_arr == i) / len(goals_arr)
                 for i in range(4)}

    result = {
        "label": weight_label,
        "steps": t + 1,
        "eaten": info["total_eaten"],
        "final_energy": energies[-1] if energies else 0,
        "mean_energy": np.mean(energies),
        "mean_pe": np.mean(pe_vals),
        "final_pe": np.mean(pe_vals[-50:]) if len(pe_vals) >= 50 else np.mean(pe_vals),
        "goal_pcts": goal_pcts,
    }
    return result


def main():
    print("=" * 65)
    print("Step 26b: Compare Original vs W_FB-Trained Weights")
    print("=" * 65)

    cls_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights",
                            "classifier.pt")
    wfb_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights",
                            "classifier_wfb.pt")

    results = []

    # Run with original weights (classifier.pt is default)
    print("\n--- Original (classifier.pt) ---")
    r1 = run_eval("Original", weight_override=None)
    results.append(r1)

    # Run with W_FB-trained weights
    if os.path.exists(wfb_path):
        print(f"\n--- W_FB-trained (classifier_wfb.pt) ---")
        r2 = run_eval("W_FB-trained", weight_override=wfb_path)
        results.append(r2)
    else:
        print(f"\nWARNING: {wfb_path} not found, skipping W_FB comparison")

    # Print comparison table
    print(f"\n{'=' * 65}")
    print(f"{'Metric':<20} ", end="")
    for r in results:
        print(f"{r['label']:>20} ", end="")
    print()
    print("-" * 65)

    for metric in ["steps", "eaten", "final_energy", "mean_energy",
                    "mean_pe", "final_pe"]:
        print(f"{metric:<20} ", end="")
        for r in results:
            val = r[metric]
            if isinstance(val, float):
                print(f"{val:>20.4f} ", end="")
            else:
                print(f"{val:>20} ", end="")
        print()

    print("-" * 65)
    for goal in ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]:
        print(f"  {goal:<18} ", end="")
        for r in results:
            print(f"{r['goal_pcts'][goal]:>19.1f}% ", end="")
        print()

    print("=" * 65)

    # Pass/fail assessment
    if len(results) >= 2:
        orig, wfb = results[0], results[1]
        pe_improved = wfb["mean_pe"] < orig["mean_pe"]
        food_ok = wfb["eaten"] >= orig["eaten"] - 2  # allow 2 food tolerance
        energy_ok = wfb["final_energy"] > 20
        print(f"\nAssessment:")
        print(f"  PE improved:     {'YES' if pe_improved else 'NO'} "
              f"({orig['mean_pe']:.4f} → {wfb['mean_pe']:.4f})")
        print(f"  Food preserved:  {'YES' if food_ok else 'NO'} "
              f"({orig['eaten']} → {wfb['eaten']})")
        print(f"  Energy safe:     {'YES' if energy_ok else 'NO'} "
              f"(final={wfb['final_energy']:.1f})")
        passed = pe_improved and food_ok and energy_ok
        print(f"  Overall: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
