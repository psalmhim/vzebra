"""
Step 11: Object Classification Training
Train the classifier head to identify: nothing, food, enemy, colleague, environment
from retinal input with >80% accuracy per class.

The classifier uses the same SNN pathway (retina→OT→PT→PC→classifier head).
Each training sample places a SINGLE entity type in the visual field.

Run: python -m zebrav1.tests.step11_object_classification
Output: plots/v1_step11_object_classification.png
        zebrav1/weights/classifier.pt
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

from zebrav1.brain.zebrafish_snn import ZebrafishSNN
from zebrav1.brain.device_util import get_device
from zebrav1.world.world_env import WorldEnv

# Class labels matching SNN output head order
CLASS_NAMES = ["nothing", "food", "enemy", "colleague", "environment"]
CLASS_NOTHING = 0
CLASS_FOOD = 1
CLASS_ENEMY = 2
CLASS_COLLEAGUE = 3
CLASS_ENVIRONMENT = 4


def make_single_entity_scene(world, entity_class, fish_pos, heading):
    """Place a single entity in the world at a visible angle/distance.

    Returns the entity class label.
    """
    # Clear all entities
    world.foods = []
    world.enemies = []
    world.colleagues = []
    world.obstacles = []

    if entity_class == CLASS_NOTHING:
        # Empty scene — no entities, fish is centered (away from walls)
        return CLASS_NOTHING

    # Random position in front of fish: angle ±80°, distance 30-100
    angle = heading + np.random.uniform(-1.4, 1.4)  # ±80°
    dist = np.random.uniform(30, 100)
    ex = float(fish_pos[0] + dist * math.cos(angle))
    ey = float(fish_pos[1] + dist * math.sin(angle))

    if entity_class == CLASS_FOOD:
        world.foods.append((ex, ey))
    elif entity_class == CLASS_ENEMY:
        world.enemies.append((ex, ey))
    elif entity_class == CLASS_COLLEAGUE:
        world.colleagues.append((ex, ey))
    elif entity_class == CLASS_ENVIRONMENT:
        # 50% wall proximity, 50% obstacle (rock)
        if np.random.random() < 0.5:
            # Place fish near a wall
            wall = np.random.choice(["left", "right", "top", "bottom"])
            if wall == "right":
                fish_pos[0] = world.xmax - 40
            elif wall == "left":
                fish_pos[0] = world.xmin + 40
            elif wall == "top":
                fish_pos[1] = world.ymax - 40
            elif wall == "bottom":
                fish_pos[1] = world.ymin + 40
        else:
            # Place obstacle (rock) in visual field
            hw = np.random.uniform(10, 30)
            hh = np.random.uniform(8, 20)
            world.add_obstacle(ex, ey, hw, hh)

    return entity_class


def run_step11(n_epochs=250, samples_per_epoch=150, lr=0.01, n_integration=3):
    print("=" * 60)
    print("Step 11: Object Classification Training")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN(device=device)

    # Large world for training
    world = WorldEnv(xmin=-300, xmax=300, ymin=-300, ymax=300, n_food=0)

    # Only train classifier weights — freeze everything else
    trainable_params = []
    for name, param in model.named_parameters():
        if "cls_" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Class weights: 3x for environment (hardest class)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0], device=device)

    loss_hist = []
    acc_hist = []
    class_acc_hist = {name: [] for name in CLASS_NAMES}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        class_correct = [0] * 5
        class_total = [0] * 5

        for s in range(samples_per_epoch):
            model.reset()
            optimizer.zero_grad()

            # Random class (balanced sampling)
            target_class = s % 5
            heading = np.random.uniform(-math.pi, math.pi)
            fish_pos = np.array([0.0, 0.0])

            make_single_entity_scene(world, target_class, fish_pos, heading)

            # Goal context: match entity type to plausible goal
            if target_class == CLASS_FOOD:
                goal_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
            elif target_class == CLASS_ENEMY:
                goal_probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
            else:
                goal_probs = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)

            # Multi-step integration (extra steps for environment class)
            n_steps = 8 if target_class == CLASS_ENVIRONMENT else n_integration
            for _ in range(n_steps):
                out = model.forward(fish_pos, heading, world,
                                    goal_probs=goal_probs)

            # Classification loss (class-weighted)
            cls_logits = out["cls"]  # [1, 5]
            target = torch.tensor([target_class], device=device)
            loss = F.cross_entropy(cls_logits, target, weight=class_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy tracking
            pred = cls_logits.argmax(dim=1).item()
            if pred == target_class:
                correct += 1
            class_correct[target_class] += int(pred == target_class)
            class_total[target_class] += 1

        scheduler.step()
        avg_loss = epoch_loss / samples_per_epoch
        acc = correct / samples_per_epoch
        loss_hist.append(avg_loss)
        acc_hist.append(acc)

        for i, name in enumerate(CLASS_NAMES):
            ca = class_correct[i] / max(1, class_total[i])
            class_acc_hist[name].append(ca)

        if epoch % 15 == 0:
            per_class = "  ".join(f"{CLASS_NAMES[i]}={class_correct[i]/max(1,class_total[i]):.0%}"
                                   for i in range(5))
            print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}  acc={acc:.1%}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")
            print(f"    {per_class}")

    # === VALIDATION ===
    print("\n--- Validation (200 samples) ---")
    model.eval()
    val_correct = [0] * 5
    val_total = [0] * 5
    val_confusion = np.zeros((5, 5), dtype=int)

    np.random.seed(999)
    for v in range(200):
        target_class = v % 5
        heading = np.random.uniform(-math.pi, math.pi)
        fish_pos = np.array([0.0, 0.0])

        model.reset()
        make_single_entity_scene(world, target_class, fish_pos, heading)

        if target_class == CLASS_FOOD:
            goal_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        elif target_class == CLASS_ENEMY:
            goal_probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
        else:
            goal_probs = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)

        with torch.no_grad():
            n_steps = 8 if target_class == CLASS_ENVIRONMENT else n_integration
            for _ in range(n_steps):
                out = model.forward(fish_pos, heading, world,
                                    goal_probs=goal_probs)
            pred = out["cls"].argmax(dim=1).item()

        val_confusion[target_class, pred] += 1
        val_total[target_class] += 1
        val_correct[target_class] += int(pred == target_class)

    print("\n  Confusion Matrix:")
    print(f"  {'':12s}", end="")
    for name in CLASS_NAMES:
        print(f"  {name[:6]:>6s}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s}", end="")
        for j in range(5):
            print(f"  {val_confusion[i, j]:6d}", end="")
        acc = val_correct[i] / max(1, val_total[i])
        print(f"  | {acc:.1%}")

    total_val_acc = sum(val_correct) / sum(val_total)
    print(f"\n  Overall accuracy: {total_val_acc:.1%}")
    all_above_80 = all(val_correct[i] / max(1, val_total[i]) >= 0.8 for i in range(5))
    print(f"  All classes ≥80%: {'YES' if all_above_80 else 'NO'}")

    for i, name in enumerate(CLASS_NAMES):
        acc = val_correct[i] / max(1, val_total[i])
        status = "OK" if acc >= 0.8 else "BELOW"
        print(f"    {name:12s}: {acc:.1%} {status}")

    # Save weights
    weights_dir = os.path.join(PROJECT_ROOT, "zebrav1", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "classifier.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"\nWeights saved: {weights_path}")

    # === PLOT: 4 panels ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Loss
    axes[0, 0].plot(loss_hist, color="steelblue")
    axes[0, 0].set_ylabel("Cross-Entropy Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_title("Training Loss")

    # Panel 2: Overall accuracy
    axes[0, 1].plot(acc_hist, color="green")
    axes[0, 1].axhline(0.8, color="red", linestyle="--", alpha=0.5, label="80% target")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_title("Overall Accuracy")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()

    # Panel 3: Per-class accuracy over time
    colors = ["gray", "gold", "red", "blue", "brown"]
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        axes[1, 0].plot(class_acc_hist[name], color=color, label=name, alpha=0.8)
    axes[1, 0].axhline(0.8, color="red", linestyle="--", alpha=0.3)
    axes[1, 0].set_ylabel("Per-Class Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_title("Per-Class Learning Curves")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(fontsize=8)

    # Panel 4: Confusion matrix
    ax = axes[1, 1]
    im = ax.imshow(val_confusion, cmap="Blues", aspect="auto")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([n[:6] for n in CLASS_NAMES], rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (acc={total_val_acc:.1%})")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(val_confusion[i, j]),
                    ha="center", va="center",
                    color="white" if val_confusion[i, j] > val_confusion.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Step 11: Object Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v1_step11_object_classification.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


if __name__ == "__main__":
    run_step11()
