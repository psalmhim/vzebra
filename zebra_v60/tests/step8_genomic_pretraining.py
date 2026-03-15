"""
Step 8: Genomic Pre-training (Supervised)
Train the SNN vision-to-motor pathway using simulated food positions
as supervision. This mimics genomic initialization — evolution selected
for circuits that can orient toward prey from birth.

The motor pathway (Retina→OT→PT→PC→Motor) learns to produce correct
turn signals from visual input. This is the signal used in closed-loop
foraging (steps 9-10).

Run: python -m zebra_v60.tests.step8_genomic_pretraining
Output: plots/v60_step8_genomic_pretraining.png
        zebra_v60/weights/genomic_v60.pt
"""
import os
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebra_v60.brain.zebrafish_snn_v60 import ZebrafishSNN_v60
from zebra_v60.brain.device_util import get_device
from zebra_v60.world.world_env import WorldEnv


def compute_motor_turn(out):
    """Decode motor output into turn signal (same as used in foraging)."""
    motor = out["motor"]
    pred_left = motor[0, :100].sigmoid().mean()
    pred_right = motor[0, 100:].sigmoid().mean()
    return pred_right - pred_left


def run_step8(n_epochs=120, samples_per_epoch=50, n_integration=5, lr=0.005):
    print("=" * 60)
    print("Step 8: Genomic Pre-training (Supervised)")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN_v60(device=device)

    world = WorldEnv(xmin=-500, xmax=500, ymin=-500, ymax=500, n_food=0)

    # Trainable: everything except topographic OT_L/OT_R and classifier
    trainable_params = []
    for name, param in model.named_parameters():
        if "OT_L" in name or "OT_R" in name or "cls_" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params.append(param)

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loss_hist = []
    turn_accuracy_hist = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        correct_direction = 0
        total_count = 0

        for s in range(samples_per_epoch):
            model.reset()
            optimizer.zero_grad()

            heading = np.random.uniform(-math.pi, math.pi)
            fish_pos = np.array([0.0, 0.0])

            # Diverse goal contexts: 70% FORAGE, 20% FLEE, 10% EXPLORE
            r = np.random.random()
            if r < 0.70:
                # FORAGE: turn toward food
                goal_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
                entity_angle = heading + np.random.uniform(-2.5, 2.5)
                entity_dist = np.random.uniform(30, 120)
                entity_pos = np.array([
                    fish_pos[0] + entity_dist * math.cos(entity_angle),
                    fish_pos[1] + entity_dist * math.sin(entity_angle),
                ])
                world.foods = [tuple(entity_pos)]
                world.enemies = []
                target_sign = 1.0  # toward entity
            elif r < 0.90:
                # FLEE: turn away from enemy
                goal_probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
                entity_angle = heading + np.random.uniform(-2.5, 2.5)
                entity_dist = np.random.uniform(30, 120)
                entity_pos = np.array([
                    fish_pos[0] + entity_dist * math.cos(entity_angle),
                    fish_pos[1] + entity_dist * math.sin(entity_angle),
                ])
                world.foods = []
                world.enemies = [tuple(entity_pos)]
                target_sign = -1.0  # away from entity
            else:
                # EXPLORE: no target, turn toward center (mild)
                goal_probs = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
                world.foods = []
                world.enemies = []
                entity_angle = heading  # straight ahead
                entity_pos = np.array([0.0, 0.0])
                target_sign = 1.0

            # Multi-step integration with goal context
            for _ in range(n_integration):
                out = model.forward(fish_pos, heading, world,
                                    goal_probs=goal_probs)

            # Motor output → predicted turn
            pred_turn = compute_motor_turn(out)

            # Target: angle to entity → desired turn (toward or away)
            dx = entity_pos[0] - fish_pos[0]
            dy = entity_pos[1] - fish_pos[1]
            angle_to_entity = math.atan2(dy, dx) - heading
            angle_to_entity = math.atan2(math.sin(angle_to_entity),
                                         math.cos(angle_to_entity))
            target_turn = math.tanh(2.0 * angle_to_entity) * target_sign
            target = torch.tensor(target_turn, device=device)

            # Motor loss
            loss = F.mse_loss(pred_turn, target)

            # Eye angle loss (auxiliary)
            eye_out = out["eye"]
            pred_angle = eye_out.mean()
            angle_target = torch.tensor(
                float(angle_to_entity * target_sign / math.pi), device=device)
            loss += 0.3 * F.mse_loss(pred_angle, angle_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy tracking
            total_count += 1
            if abs(target_turn) < 0.05:
                correct_direction += 1
            elif (target_turn > 0 and pred_turn.item() > 0) or \
                 (target_turn < 0 and pred_turn.item() < 0):
                correct_direction += 1

        scheduler.step()
        avg_loss = epoch_loss / samples_per_epoch
        accuracy = correct_direction / total_count
        loss_hist.append(avg_loss)
        turn_accuracy_hist.append(accuracy)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}  "
                  f"direction_acc={accuracy:.1%}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

    # === VALIDATION ===
    print("\n--- Validation (from reset, 5 steps) ---")
    model.eval()
    fish_pos = np.array([0.0, 0.0])

    test_angles = np.linspace(-math.pi * 0.9, math.pi * 0.9, 13)
    val_true_angles = []
    val_pred_turns = []

    forage_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    for angle in test_angles:
        heading = 0.0
        food = (80 * math.cos(angle), 80 * math.sin(angle))
        model.reset()
        world.foods = [food]
        world.enemies = []
        with torch.no_grad():
            for _ in range(n_integration):
                out = model.forward(fish_pos, heading, world,
                                    goal_probs=forage_goal)
            pred_turn = float(compute_motor_turn(out))

        true_angle = math.atan2(math.sin(angle), math.cos(angle))
        target = math.tanh(2.0 * true_angle)
        sign_ok = (target * pred_turn > 0) or abs(target) < 0.05
        label = "OK" if sign_ok else "MISS"

        val_true_angles.append(true_angle)
        val_pred_turns.append(pred_turn)

        deg = math.degrees(true_angle)
        print(f"  angle={deg:+7.1f}°  target={target:+.3f}  "
              f"pred={pred_turn:+.4f}  {label}")

    passed = sum(1 for ta, pt in zip(val_true_angles, val_pred_turns)
                 if (math.tanh(2 * ta) * pt > 0) or abs(math.tanh(2 * ta)) < 0.05)
    print(f"\nValidation (reset): {passed}/{len(test_angles)} sign-correct")

    # Save weights
    weights_dir = os.path.join(PROJECT_ROOT, "zebra_v60", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "genomic_v60.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path}")

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(loss_hist, color="steelblue")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(turn_accuracy_hist, color="green")
    axes[1].set_ylabel("Direction Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].axhline(0.8, color="red", linestyle="--", alpha=0.5, label="80% target")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Direction Accuracy")
    axes[1].legend()

    # Validation scatter
    target_curve = [math.tanh(2.0 * a) for a in val_true_angles]
    axes[2].scatter(np.degrees(val_true_angles), val_pred_turns,
                     color="coral", s=60, zorder=5, label="Predicted")
    axes[2].plot(np.degrees(val_true_angles), target_curve,
                  color="gray", linestyle="--", alpha=0.7, label="Target")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].axvline(0, color="black", linewidth=0.5)
    axes[2].set_xlabel("True Angle (deg)")
    axes[2].set_ylabel("Turn Signal")
    axes[2].set_title(f"Validation ({passed}/13 sign-correct)")
    axes[2].legend(fontsize=8)

    fig.suptitle("Step 8: Genomic Pre-training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v60_step8_genomic_pretraining.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


if __name__ == "__main__":
    run_step8()
