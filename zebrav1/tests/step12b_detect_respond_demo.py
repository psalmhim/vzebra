"""
Step 12b: Detection → Identification → Response Demo
Clear demonstration that the fish can:
  1. Detect an object via retinal input
  2. Identify it as food or enemy via classifier
  3. Approach food / Flee from enemy

Three scenarios run sequentially:
  A) Food only → fish approaches and eats
  B) Enemy only → fish detects and flees
  C) Both food + enemy → fish approaches food, flees enemy

Run: python -m zebrav1.tests.step12b_detect_respond_demo
Output: plots/v1_step12b_detect_respond_demo.png
"""
import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as Func
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.brain.zebrafish_snn import ZebrafishSNN
from zebrav1.brain.device_util import get_device
from zebrav1.world.world_env import WorldEnv
from zebrav1.tests.step1_vision_pursuit import compute_retinal_turn, TurnSmoother

CLASS_NAMES = ["nothing", "food", "enemy", "colleague", "environment"]
CLASS_FOOD = 1
CLASS_ENEMY = 2

# Type encoding values (must match retina_sampling.py)
TYPE_FOOD = 1.0
TYPE_ENEMY = 0.5


def compute_valence_turn(out):
    """Compute a turn signal from type channels that steers TOWARD food and
    AWAY from enemy.

    Each retinal pixel in the type channel ([400:]) carries the entity type
    value: food=1.0, enemy=0.5, nothing=0.0, etc.

    For each eye, count food-weighted and enemy-weighted pixel activations.
    Then: valence_turn = food_direction - enemy_direction
      food_direction  = (food_R - food_L)  → positive = food is right
      enemy_direction = (enemy_R - enemy_L) → positive = enemy is right
      We APPROACH food (same sign) and FLEE enemy (inverted sign).
    """
    typeL = out["retL_full"][0, 400:].cpu().numpy()  # [400]
    typeR = out["retR_full"][0, 400:].cpu().numpy()  # [400]

    # Food pixels: type ≈ 1.0 (threshold > 0.75 to avoid obstacle 0.75)
    food_L = float(np.sum(typeL > 0.75))
    food_R = float(np.sum(typeR > 0.75))

    # Enemy pixels: type ≈ 0.5 (0.35 < type < 0.65)
    enemy_L = float(np.sum((typeL > 0.35) & (typeL < 0.65)))
    enemy_R = float(np.sum((typeR > 0.35) & (typeR < 0.65)))

    # Normalize food and enemy directions independently
    food_total = food_L + food_R + 1e-8
    enemy_total = enemy_L + enemy_R + 1e-8

    # Approach food (positive = food is right), flee enemy (invert: turn away)
    food_dir = (food_R - food_L) / food_total    # [-1, 1]
    enemy_dir = (enemy_R - enemy_L) / enemy_total  # [-1, 1]

    # Combined: steer toward food, steer away from enemy
    return food_dir - enemy_dir


def run_scenario(model, world, fish_start, heading_start, T=200,
                 swim_speed=2.0, turn_gain=0.18):
    """Run a single scenario and return trajectory + classification history."""
    model.reset()
    smoother = TurnSmoother(alpha=0.3)

    fish_x, fish_y = fish_start
    heading = heading_start

    pos_x, pos_y = [], []
    cls_hist = []
    cls_prob_hist = []
    approach_hist = []
    dist_to_food_hist = []
    dist_to_enemy_hist = []
    food_eaten = False
    food_eaten_step = -1
    # Remember original food positions for distance tracking after eaten
    original_foods = list(world.foods) if world.foods else []

    # Temporal smoothing of classification (prevents single-frame whiplash)
    smooth_probs = np.zeros(5)
    cls_alpha = 0.3  # smoothing factor (lower = more smoothing)

    for t in range(T):
        fish_pos = np.array([fish_x, fish_y])

        with torch.no_grad():
            out = model.forward(fish_pos, heading, world)

        # Classification with temporal smoothing
        cls_probs = Func.softmax(out["cls"], dim=1)[0].cpu().numpy()
        smooth_probs = cls_alpha * cls_probs + (1 - cls_alpha) * smooth_probs
        predicted = int(smooth_probs.argmax())
        p_food = smooth_probs[CLASS_FOOD]
        p_enemy = smooth_probs[CLASS_ENEMY]

        # Detect mixed scene (both food and enemy have significant smoothed probability)
        both_visible = p_food > 0.1 and p_enemy > 0.1

        # Behavioral response
        if both_visible:
            # Mixed scene: use valence turn (type-channel compass)
            # Steers toward food pixels, away from enemy pixels
            approach_gain = 1.3
            speed = swim_speed
            retinal = compute_retinal_turn(out)
            valence = compute_valence_turn(out)
            raw_turn = retinal + 0.5 * valence
        elif p_enemy > 0.4:
            # Enemy dominant → flee (invert retinal turn)
            approach_gain = -1.0
            speed = swim_speed * 1.5
            raw_turn = compute_retinal_turn(out)
        elif p_food > 0.2:
            # Food dominant → amplified approach
            approach_gain = 1.0 + 0.3 * p_food
            speed = swim_speed * (1.0 + 0.2 * p_food)
            raw_turn = compute_retinal_turn(out)
        else:
            # Baseline approach
            approach_gain = 1.0
            speed = swim_speed * 0.8
            raw_turn = compute_retinal_turn(out)

        modulated_turn = raw_turn * approach_gain
        turn = smoother.step(modulated_turn)

        # Locomotion
        heading += turn_gain * turn
        heading = math.atan2(math.sin(heading), math.cos(heading))
        fish_x += speed * math.cos(heading)
        fish_y += speed * math.sin(heading)

        # Clamp to world
        fish_x = max(world.xmin + 5, min(world.xmax - 5, fish_x))
        fish_y = max(world.ymin + 5, min(world.ymax - 5, fish_y))

        # Distance to nearest food/enemy (BEFORE eating, so we see approach)
        food_ref = world.foods if world.foods else original_foods
        if food_ref:
            d_food = min(math.sqrt((fish_x - fx)**2 + (fish_y - fy)**2)
                         for fx, fy in food_ref)
        else:
            d_food = float('nan')
        if world.enemies:
            d_enemy = min(math.sqrt((fish_x - ex)**2 + (fish_y - ey)**2)
                          for ex, ey in world.enemies)
        else:
            d_enemy = float('nan')
        dist_to_food_hist.append(d_food)
        dist_to_enemy_hist.append(d_enemy)

        # Eat food
        n_before = len(world.foods)
        world.try_eat(fish_x, fish_y)
        if len(world.foods) < n_before and not food_eaten:
            food_eaten = True
            food_eaten_step = t

        # Record
        pos_x.append(fish_x)
        pos_y.append(fish_y)
        cls_hist.append(predicted)
        cls_prob_hist.append(cls_probs.copy())
        approach_hist.append(approach_gain)

    return {
        "pos_x": pos_x, "pos_y": pos_y,
        "cls_hist": cls_hist,
        "cls_prob_hist": np.array(cls_prob_hist),
        "approach_hist": approach_hist,
        "dist_food": dist_to_food_hist,
        "dist_enemy": dist_to_enemy_hist,
        "food_eaten": food_eaten,
        "food_eaten_step": food_eaten_step,
    }


def run_step12b():
    print("=" * 60)
    print("Step 12b: Detection → Identification → Response Demo")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN(device=device)

    cls_path = os.path.join(PROJECT_ROOT, "zebrav1", "weights", "classifier.pt")
    if os.path.exists(cls_path):
        model.load_state_dict(
            torch.load(cls_path, weights_only=True, map_location=device))
        print(f"  Loaded classifier weights")
    else:
        print(f"  ERROR: No classifier weights at {cls_path}")
        return

    model.eval()

    # ===== Scenario A: Food only =====
    print("\n--- Scenario A: Food at (80, 0), fish starts at origin heading right ---")
    world_a = WorldEnv(xmin=-300, xmax=300, ymin=-300, ymax=300, n_food=0)
    world_a.foods = [(80, 0)]
    world_a._orig_foods = list(world_a.foods)
    res_a = run_scenario(model, world_a, (0, 0), 0.0, T=150)

    min_dist_food_a = min(res_a["dist_food"])
    food_detected = sum(1 for c in res_a["cls_hist"] if c == CLASS_FOOD)
    eaten_a = res_a["food_eaten"]
    print(f"  Food classified {food_detected}/{len(res_a['cls_hist'])} steps")
    print(f"  Min distance to food: {min_dist_food_a:.1f}")
    print(f"  Food eaten: {'YES (step ' + str(res_a['food_eaten_step']) + ')' if eaten_a else 'NO'}")

    # ===== Scenario B: Enemy only =====
    print("\n--- Scenario B: Enemy at (80, 0), fish starts at origin heading right ---")
    world_b = WorldEnv(xmin=-300, xmax=300, ymin=-300, ymax=300, n_food=0)
    world_b.enemies = [(80, 0)]
    res_b = run_scenario(model, world_b, (0, 0), 0.0, T=150)

    final_dist_enemy_b = res_b["dist_enemy"][-1]
    min_dist_enemy_b = min(res_b["dist_enemy"])
    enemy_detected = sum(1 for c in res_b["cls_hist"] if c == CLASS_ENEMY)
    print(f"  Enemy classified {enemy_detected}/{len(res_b['cls_hist'])} steps")
    print(f"  Min distance to enemy: {min_dist_enemy_b:.1f}")
    print(f"  Final distance to enemy: {final_dist_enemy_b:.1f}")
    print(f"  Fled: {'YES' if final_dist_enemy_b > 80 else 'NO'}")

    # ===== Scenario C: Food and Enemy BOTH directly ahead =====
    # Hard case: both at similar distance ahead but different sides.
    # Fish must use type-channel compass to steer toward food, away from enemy.
    print("\n--- Scenario C: Food at (60, -40), Enemy at (60, 40), heading right ---")
    world_c = WorldEnv(xmin=-300, xmax=300, ymin=-300, ymax=300, n_food=0)
    world_c.foods = [(60, -40)]
    world_c.enemies = [(60, 40)]
    world_c._orig_foods = list(world_c.foods)
    res_c = run_scenario(model, world_c, (0, 0), 0.0, T=200)

    min_dist_food_c = min(res_c["dist_food"])
    final_dist_food_c = res_c["dist_food"][-1]
    final_dist_enemy_c = res_c["dist_enemy"][-1]
    eaten_c = res_c["food_eaten"]
    food_cls_c = sum(1 for c in res_c["cls_hist"] if c == CLASS_FOOD)
    enemy_cls_c = sum(1 for c in res_c["cls_hist"] if c == CLASS_ENEMY)
    print(f"  Food classified: {food_cls_c}, Enemy classified: {enemy_cls_c}")
    print(f"  Min dist to food: {min_dist_food_c:.1f}, Min dist to enemy: {min(res_c['dist_enemy']):.1f}")
    print(f"  Final dist to food: {final_dist_food_c:.1f}, to enemy: {final_dist_enemy_c:.1f}")
    print(f"  Food eaten: {'YES (step ' + str(res_c['food_eaten_step']) + ')' if eaten_c else 'NO'}")

    # ===== SUMMARY =====
    pass_a = eaten_a or min_dist_food_a < 25
    pass_b = final_dist_enemy_b > 80
    pass_c = (eaten_c or min_dist_food_c < 25) and final_dist_enemy_c > 50

    print(f"\n{'='*60}")
    print("DETECTION → RESPONSE SUMMARY")
    print(f"  Scenario A (food):  {'PASS' if pass_a else 'FAIL'} "
          f"- {'EATEN at step ' + str(res_a['food_eaten_step']) if eaten_a else 'min_dist=' + f'{min_dist_food_a:.0f}'}, "
          f"detected={food_detected}/{len(res_a['cls_hist'])}")
    print(f"  Scenario B (enemy): {'PASS' if pass_b else 'FAIL'} "
          f"- final_dist={final_dist_enemy_b:.0f}, detected={enemy_detected}/{len(res_b['cls_hist'])}")
    print(f"  Scenario C (both):  {'PASS' if pass_c else 'FAIL'} "
          f"- {'food EATEN' if eaten_c else 'food_dist=' + f'{min_dist_food_c:.0f}'}, "
          f"enemy_dist={final_dist_enemy_c:.0f}")
    all_pass = pass_a and pass_b and pass_c
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'} ({sum([pass_a, pass_b, pass_c])}/3)")
    print(f"{'='*60}")

    # ===== PLOT: 3 rows × 3 cols =====
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    scenarios = [
        (res_a, world_a, "A: Food Only"),
        (res_b, world_b, "B: Enemy Only"),
        (res_c, world_c, "C: Food + Enemy"),
    ]

    for row, (res, world, title) in enumerate(scenarios):
        T_s = len(res["pos_x"])

        # Col 0: Trajectory
        ax = axes[row, 0]
        # Color path by classification
        for i in range(1, T_s):
            c = {0: "gray", 1: "green", 2: "red", 3: "blue", 4: "brown"}[res["cls_hist"][i]]
            ax.plot([res["pos_x"][i-1], res["pos_x"][i]],
                    [res["pos_y"][i-1], res["pos_y"][i]],
                    color=c, alpha=0.7, linewidth=2)
        ax.plot(res["pos_x"][0], res["pos_y"][0], "ko", ms=12, label="Start", zorder=6)
        ax.plot(res["pos_x"][-1], res["pos_y"][-1], "k^", ms=12, label="End", zorder=6)
        # Mark food eaten position
        if res["food_eaten"] and res["food_eaten_step"] >= 0:
            et = res["food_eaten_step"]
            ax.plot(res["pos_x"][et], res["pos_y"][et], "gD", ms=14,
                    label="Ate food!", zorder=8, markeredgecolor="black")
        # Show food/enemy positions (use original positions even if eaten)
        food_positions = getattr(world, '_orig_foods', world.foods)
        for fx, fy in food_positions:
            ax.plot(fx, fy, "g*", ms=18, label="Food", zorder=7)
        for ex, ey in world.enemies:
            ax.plot(ex, ey, "rv", ms=14, label="Enemy", zorder=7)
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.set_aspect("equal")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

        # Col 1: Classification probabilities
        ax = axes[row, 1]
        ax.plot(res["cls_prob_hist"][:, 1], color="green", label="food", lw=2)
        ax.plot(res["cls_prob_hist"][:, 2], color="red", label="enemy", lw=2)
        ax.plot(res["cls_prob_hist"][:, 4], color="brown", label="environ", lw=1, alpha=0.5)
        ax.plot(res["cls_prob_hist"][:, 0], color="gray", label="nothing", lw=1, alpha=0.5)
        ax.set_ylabel("Class Probability")
        ax.set_xlabel("Time step")
        ax.set_title("Classification")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

        # Col 2: Distance to food/enemy
        ax = axes[row, 2]
        if not all(math.isnan(d) for d in res["dist_food"]):
            ax.plot(res["dist_food"], color="green", lw=2, label="Dist to food")
        if not all(math.isnan(d) for d in res["dist_enemy"]):
            ax.plot(res["dist_enemy"], color="red", lw=2, label="Dist to enemy")
        ax.axhline(18, color="green", ls="--", alpha=0.4, label="Eat radius")
        ax.set_ylabel("Distance")
        ax.set_xlabel("Time step")
        ax.set_title("Distance to Entities")
        ax.legend(fontsize=7)

    fig.suptitle("Step 12b: Detect → Identify → Approach/Flee",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots",
                              "v1_step12b_detect_respond_demo.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


if __name__ == "__main__":
    run_step12b()
