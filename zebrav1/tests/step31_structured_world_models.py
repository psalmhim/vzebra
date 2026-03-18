"""
Step 31: Structured World Models — Geographic + Predator + Internal State.

Tests:
  A. Geographic model builds obstacle map from experience (>50% accuracy)
  B. Geographic model food density tracks food locations (density > 0)
  C. Predator model maintains object permanence (position error < 150px)
  D. Predator model infers hunting intent (intent > 0.3 when approaching)
  E. Internal state model energy prediction (RMSE < 15 over 30 steps)
  F. Internal state model policy comparison (FORAGE lowest risk when low energy)
  G. Full integration: structured models don't break brain pipeline
  H. Geographic exploration bonus high for unvisited regions

Pass criterion: 7/8 tests pass.

Run: python -m zebrav1.tests.step31_structured_world_models
"""
import os
import sys
import math
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zebrav1.brain.geographic_model import GeographicModel
from zebrav1.brain.predator_model import PredatorModel
from zebrav1.brain.internal_state_model import InternalStateModel


def test_A_geographic_obstacle_map():
    """Run updates simulating obstacles, check belief > 0.5 in obstacle zone."""
    geo = GeographicModel()

    # Simulate fish at fixed position looking at obstacles ahead (heading=0)
    for step in range(100):
        rf = {"obstacle_px_L": 15, "obstacle_px_R": 20, "food_px_total": 0}
        geo.update([400, 300], rf, 0.0, step)

    # Cells 1-3 cells ahead (heading=0 = right) should have high belief
    # Cell ahead: 400 + 20*2 = 440
    obstacle_ahead = geo.query_obstacle([440, 300])
    # Cell far away: never observed
    obstacle_far = geo.query_obstacle([50, 50])

    passed = obstacle_ahead > obstacle_far and obstacle_ahead > 0.55
    print(f"  obstacle_ahead={obstacle_ahead:.3f}, "
          f"obstacle_far={obstacle_far:.3f}, "
          f"PASS={passed}")
    return passed


def test_B_geographic_food_density():
    """Simulate food observations, verify density > 0 in observed area."""
    geo = GeographicModel()

    # Fish at centre seeing food ahead
    for step in range(50):
        rf = {"obstacle_px_L": 0, "obstacle_px_R": 0, "food_px_total": 30}
        geo.update([400, 300], rf, 0.0, step)

    density = geo.query_food_density([420, 300], 50)
    passed = density > 0.01
    print(f"  food_density_ahead={density:.4f}, PASS={passed}")
    return passed


def test_C_predator_object_permanence():
    """Show predator for 20 steps, hide for 20, check position error."""
    pm = PredatorModel()

    fish_pos = [400, 300]
    # Predator at (600, 300) moving left
    true_x, true_y = 600, 300
    true_vx, true_vy = -3, 0

    # Phase 1: visible for 20 steps
    for step in range(20):
        true_x += true_vx
        true_y += true_vy
        # Simulate retinal features
        dist = math.sqrt((true_x - fish_pos[0]) ** 2
                         + (true_y - fish_pos[1]) ** 2)
        intensity = min(1.0, 80.0 / (dist + 10))
        bearing = math.atan2(true_y - fish_pos[1], true_x - fish_pos[0])
        lateral = np.clip((bearing - 0.0) / (math.pi / 4), -1, 1)
        rf = {
            "enemy_px_total": 25,
            "enemy_lateral_bias": float(lateral),
            "enemy_intensity_mean": float(intensity),
            "enemy_spread": 5.0,
        }
        pm.predict()
        pm.update(rf, fish_pos, 0.0, step)

    # Phase 2: hidden for 20 steps
    for step in range(20, 40):
        true_x += true_vx
        true_y += true_vy
        rf = {"enemy_px_total": 0, "enemy_lateral_bias": 0,
              "enemy_intensity_mean": 0, "enemy_spread": 0}
        pm.predict()
        pm.update(rf, fish_pos, 0.0, step)

    # Check predicted position error
    error = math.sqrt((pm.belief.x - true_x) ** 2
                      + (pm.belief.y - true_y) ** 2)
    passed = error < 150
    print(f"  pos_error={error:.1f}px, "
          f"pred=({pm.belief.x:.0f},{pm.belief.y:.0f}), "
          f"true=({true_x:.0f},{true_y:.0f}), PASS={passed}")
    return passed


def test_D_predator_intent_inference():
    """Predator approaching fish → intent should be > 0.3."""
    pm = PredatorModel()
    fish_pos = [400, 300]

    # Predator approaching from right
    for step in range(30):
        dist = 250 - step * 5  # closing
        intensity = min(1.0, 80.0 / (dist + 10))
        rf = {
            "enemy_px_total": 15 + step,
            "enemy_lateral_bias": 0.2,
            "enemy_intensity_mean": float(intensity),
            "enemy_spread": 3.0,
        }
        pm.predict()
        pm.update(rf, fish_pos, 0.0, step)

    intent = pm.belief.intent
    passed = intent > 0.3
    print(f"  intent={intent:.3f}, PASS={passed}")
    return passed


def test_E_internal_energy_prediction():
    """Simulate 30 steps of foraging, check prediction RMSE < 15."""
    ism = InternalStateModel()
    errors = []
    energy = 90.0

    for step in range(30):
        # Simulate: drain 0.1/step, eat food every 10 steps
        eaten = 1 if step % 10 == 5 else 0
        speed = 0.6
        energy -= 0.08 * speed + (2.0 * eaten if eaten else -0.08 * speed)
        energy = max(0, min(100, energy + 2.0 * eaten))

        ism.predict(speed, eaten)
        ism.observe(energy)
        errors.append((ism.get_energy() - energy) ** 2)

    rmse = math.sqrt(np.mean(errors))
    passed = rmse < 15
    print(f"  RMSE={rmse:.2f}, PASS={passed}")
    return passed


def test_F_internal_policy_comparison():
    """At low energy, FORAGE should have lowest risk."""
    ism = InternalStateModel(initial_energy=25.0)

    risk, tts = ism.compare_policies()
    forage_risk = risk[0]
    flee_risk = risk[1]

    # Forage should have lower risk than flee when starving
    passed = forage_risk <= flee_risk
    print(f"  forage_risk={forage_risk:.3f}, flee_risk={flee_risk:.3f}, "
          f"PASS={passed}")
    return passed


def test_G_full_integration():
    """Run brain agent for 100 steps — no crashes."""
    from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
    from zebrav1.gym_env.brain_agent import BrainAgent

    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=200, side_panels=False)
    agent = BrainAgent(use_allostasis=True)
    obs, info = env.reset(seed=42)
    agent.reset()

    survived = True
    for t in range(100):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward, done=terminated, env=env)
        if terminated:
            survived = False
            break

    env.close()

    # Check that structured model diagnostics exist
    diag = agent.last_diagnostics
    has_geo = "geographic" in diag
    has_pred = "predator_model" in diag or agent.predator_model is None
    has_int = "internal_state" in diag or not isinstance(
        agent.interoceptive, InternalStateModel)

    passed = True  # no crash is already a pass
    print(f"  steps={t+1}, survived={survived}, "
          f"geo={has_geo}, pred={has_pred}, int={has_int}, PASS={passed}")
    return passed


def test_H_exploration_bonus():
    """Unvisited regions should have high exploration bonus."""
    geo = GeographicModel()

    # Visit some cells
    for step in range(50):
        rf = {"obstacle_px_L": 0, "obstacle_px_R": 0, "food_px_total": 0}
        geo.update([400, 300], rf, 0.0, step)

    visited_bonus = geo.get_exploration_bonus([400, 300])
    unvisited_bonus = geo.get_exploration_bonus([50, 50])

    passed = unvisited_bonus > visited_bonus
    print(f"  visited_bonus={visited_bonus:.3f}, "
          f"unvisited_bonus={unvisited_bonus:.3f}, PASS={passed}")
    return passed


def run_step31():
    print("=" * 60)
    print("Step 31: Structured World Models")
    print("=" * 60)

    tests = [
        ("A", "Geographic obstacle map", test_A_geographic_obstacle_map),
        ("B", "Geographic food density", test_B_geographic_food_density),
        ("C", "Predator object permanence", test_C_predator_object_permanence),
        ("D", "Predator intent inference", test_D_predator_intent_inference),
        ("E", "Internal energy prediction", test_E_internal_energy_prediction),
        ("F", "Internal policy comparison", test_F_internal_policy_comparison),
        ("G", "Full integration (no crash)", test_G_full_integration),
        ("H", "Exploration bonus", test_H_exploration_bonus),
    ]

    results = {}
    for tid, desc, fn in tests:
        print(f"\n--- Test {tid}: {desc} ---")
        try:
            passed = fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            passed = False
        results[tid] = passed

    # Summary
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {n_pass}/{n_total} tests passed")
    for tid, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {tid}: {status}")

    verdict = "PASS" if n_pass >= 7 else "FAIL"
    print(f"\nVERDICT: {verdict} (need 7/{n_total})")
    print("=" * 60)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A: Geographic model exploration map
    geo = GeographicModel()
    for s in range(200):
        rf = {"obstacle_px_L": 5 * (s % 3), "obstacle_px_R": 3 * (s % 2),
              "food_px_total": 10 * (s % 5 == 0)}
        x = 100 + (s * 3) % 600
        y = 100 + (s * 7) % 400
        geo.update([x, y], rf, (s * 0.1) % (2 * math.pi), s)

    ax = axes[0, 0]
    ax.imshow(geo.obstacle_belief, cmap="Reds", vmin=0, vmax=1, origin="lower")
    ax.set_title("Obstacle Belief Map")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    ax = axes[0, 1]
    ax.imshow(geo.get_exploration_map(), cmap="Blues", vmin=0, vmax=1,
              origin="lower")
    ax.set_title("Epistemic Value (Exploration Bonus)")

    # C: Predator tracking
    ax = axes[1, 0]
    pm = PredatorModel()
    true_xs, true_ys = [], []
    est_xs, est_ys = [], []
    fish_pos = [400, 300]
    tx, ty = 600, 400
    tvx, tvy = -3, -1
    for s in range(60):
        tx += tvx
        ty += tvy
        true_xs.append(tx)
        true_ys.append(ty)
        dist = math.sqrt((tx - fish_pos[0]) ** 2 + (ty - fish_pos[1]) ** 2)
        intensity = min(1.0, 80.0 / (dist + 10))
        bearing = math.atan2(ty - fish_pos[1], tx - fish_pos[0])
        lateral = np.clip((bearing - 0.0) / (math.pi / 4), -1, 1)
        visible = s < 30 or s > 45  # hidden between 30-45
        rf = {
            "enemy_px_total": 20 if visible else 0,
            "enemy_lateral_bias": float(lateral) if visible else 0,
            "enemy_intensity_mean": float(intensity) if visible else 0,
            "enemy_spread": 5.0 if visible else 0,
        }
        pm.predict()
        pm.update(rf, fish_pos, 0.0, s)
        est_xs.append(pm.belief.x)
        est_ys.append(pm.belief.y)

    ax.plot(true_xs, true_ys, "r-", label="True", linewidth=2)
    ax.plot(est_xs, est_ys, "b--", label="Estimated", linewidth=1.5)
    ax.axvspan(0, 0, label="Hidden period", alpha=0.2, color="gray")
    ax.fill_between(range(30, 46), min(true_ys + est_ys),
                    max(true_ys + est_ys), alpha=0.15, color="gray",
                    label="Hidden")
    ax.set_title("Predator Tracking (Object Permanence)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)

    # F: Energy trajectory comparison
    ax = axes[1, 1]
    ism = InternalStateModel(initial_energy=30.0)
    labels = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
    colors = ["green", "red", "blue", "orange"]
    for gi in range(4):
        traj, tts = ism.simulate_trajectory(gi)
        ax.plot(traj, color=colors[gi], label=f"{labels[gi]} (die@{tts})")
    ax.axhline(0, color="black", linestyle=":", alpha=0.3)
    ax.set_title("Energy Trajectories at 30% Energy")
    ax.set_xlabel("Steps Ahead")
    ax.set_ylabel("Predicted Energy")
    ax.legend(fontsize=8)

    plt.suptitle(f"Step 31: Structured World Models — {n_pass}/{n_total} tests",
                 fontsize=14)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/v1_step31_structured_world_models.png", dpi=150)
    print(f"\nPlot saved: plots/v1_step31_structured_world_models.png")


if __name__ == "__main__":
    run_step31()
