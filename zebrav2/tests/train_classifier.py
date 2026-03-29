"""
Train ClassifierV2 on gameplay data collected from v2 demo runs.

1. Collect labeled retinal samples by running episodes
2. Label each frame using ground-truth entity positions
3. Train 804→128→5 spiking classifier with class-weighted loss
4. Save weights

Run: python -m zebrav2.tests.train_classifier
"""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.classifier import ClassifierV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav2.spec import DEVICE

# Classes: 0=nothing, 1=food, 2=enemy, 3=colleague, 4=environment
CLASS_NAMES = ['nothing', 'food', 'enemy', 'colleague', 'environment']
N_COLLECT_EPISODES = 20
STEPS_PER_EPISODE = 200
N_EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3


def _label_frame(env):
    """Determine dominant class from env ground truth."""
    fish_x = getattr(env, 'fish_x', 400)
    fish_y = getattr(env, 'fish_y', 300)

    # Check enemy proximity
    pred_x = getattr(env, 'pred_x', -999)
    pred_y = getattr(env, 'pred_y', -999)
    pred_dist = math.sqrt((pred_x - fish_x) ** 2 + (pred_y - fish_y) ** 2)

    # Check nearest food
    food_dist = 999.0
    for food in getattr(env, 'foods', []):
        d = math.sqrt((food[0] - fish_x) ** 2 + (food[1] - fish_y) ** 2)
        food_dist = min(food_dist, d)

    # Check rocks
    rock_dist = 999.0
    for rock in getattr(env, 'rock_formations', []):
        d = math.sqrt((rock['cx'] - fish_x) ** 2 + (rock['cy'] - fish_y) ** 2)
        rock_dist = min(rock_dist, d - rock.get('radius', 30))

    # Label based on what's closest and visible (within ~200px FoV)
    if pred_dist < 150:
        return 2  # enemy
    elif food_dist < 120:
        return 1  # food
    elif rock_dist < 80:
        return 4  # environment (obstacle)
    else:
        return 0  # nothing


def _build_input(env):
    """Build 804-dim classifier input from env retinal arrays."""
    brain_L = getattr(env, 'brain_L', np.zeros(800))
    brain_R = getattr(env, 'brain_R', np.zeros(800))

    L = torch.tensor(brain_L, dtype=torch.float32, device=DEVICE)
    R = torch.tensor(brain_R, dtype=torch.float32, device=DEVICE)

    type_L = L[400:]
    type_R = R[400:]
    type_all = torch.cat([type_L, type_R])

    obs_px = (torch.abs(type_all - 0.75) < 0.1).float().sum().unsqueeze(0)
    ene_px = (torch.abs(type_all - 0.5) < 0.1).float().sum().unsqueeze(0)
    food_px = (torch.abs(type_all - 1.0) < 0.15).float().sum().unsqueeze(0)
    bound_px = (torch.abs(type_all - 0.12) < 0.05).float().sum().unsqueeze(0)

    x = torch.cat([type_all, obs_px, ene_px, food_px, bound_px])
    return x


def collect_data():
    """Run episodes and collect labeled (input, label) pairs."""
    print(f"Collecting data: {N_COLLECT_EPISODES} episodes × {STEPS_PER_EPISODE} steps...")
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15,
                                    max_steps=STEPS_PER_EPISODE, side_panels=False)
    brain = ZebrafishBrainV2(device=DEVICE)

    inputs = []
    labels = []

    for ep in range(N_COLLECT_EPISODES):
        obs, info = env.reset(seed=ep * 7)
        brain.reset()

        for t in range(STEPS_PER_EPISODE):
            is_flee = (brain.current_goal == 1)
            if hasattr(env, 'set_flee_active'):
                env.set_flee_active(is_flee, panic_intensity=0.8 if is_flee else 0.0)

            inject_sensory(env)
            out = brain.step(obs, env)
            action = np.array([out['turn'], out['speed']], dtype=np.float32)

            # Collect sample
            x = _build_input(env)
            label = _label_frame(env)
            inputs.append(x.detach())
            labels.append(label)

            obs, reward, terminated, truncated, info = env.step(action)
            env._eaten_now = info.get('food_eaten_this_step', 0)

            if terminated or truncated:
                break

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep + 1}/{N_COLLECT_EPISODES} done, "
                  f"{len(inputs)} samples collected")

    env.close()

    X = torch.stack(inputs)
    Y = torch.tensor(labels, dtype=torch.long, device=DEVICE)

    # Class distribution
    for c in range(5):
        count = (Y == c).sum().item()
        print(f"  Class {c} ({CLASS_NAMES[c]}): {count} samples "
              f"({100 * count / len(Y):.1f}%)")

    return X, Y


def train_classifier(X, Y):
    """Train ClassifierV2 with class-weighted cross-entropy."""
    print(f"\nTraining classifier: {len(X)} samples, {N_EPOCHS} epochs...")

    classifier = ClassifierV2(device=DEVICE)

    # Class weights (inverse frequency, 3x boost for rare classes)
    class_counts = torch.bincount(Y, minlength=5).float()
    class_counts = class_counts.clamp(min=1)
    weights = (1.0 / class_counts)
    weights = weights / weights.sum() * 5.0
    # Extra boost for environment class (often underrepresented)
    weights[4] = weights[4] * 3.0
    print(f"  Class weights: {weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(classifier.parameters(), lr=LR)

    # Split train/val
    n = len(X)
    perm = torch.randperm(n)
    split = int(0.85 * n)
    train_idx = perm[:split]
    val_idx = perm[split:]

    best_val_acc = 0.0
    best_state = None

    for epoch in range(N_EPOCHS):
        classifier.train()
        # Shuffle
        shuf = train_idx[torch.randperm(len(train_idx))]
        total_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(shuf), BATCH_SIZE):
            batch_idx = shuf[i:i + BATCH_SIZE]
            xb = X[batch_idx]
            yb = Y[batch_idx]

            optimizer.zero_grad()
            # Forward each sample (classifier uses LIF integration)
            logits_list = []
            for j in range(len(xb)):
                classifier.reset()
                logit = classifier(xb[j])
                logits_list.append(logit)
            logits = torch.cat(logits_list, dim=0)

            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(xb)

        train_acc = 100.0 * correct / max(total, 1)

        # Validation
        classifier.eval()
        val_correct = 0
        with torch.no_grad():
            for j in range(len(val_idx)):
                classifier.reset()
                logit = classifier(X[val_idx[j]])
                pred = logit.argmax(dim=1)
                if pred.item() == Y[val_idx[j]].item():
                    val_correct += 1
        val_acc = 100.0 * val_correct / max(len(val_idx), 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}: loss={total_loss / total:.4f} "
                  f"train_acc={train_acc:.1f}% val_acc={val_acc:.1f}% "
                  f"(best={best_val_acc:.1f}%)")

    # Load best
    if best_state is not None:
        classifier.load_state_dict(best_state)

    # Per-class accuracy
    classifier.eval()
    class_correct = [0] * 5
    class_total = [0] * 5
    with torch.no_grad():
        for j in range(len(X)):
            classifier.reset()
            logit = classifier(X[j])
            pred = logit.argmax(dim=1).item()
            label = Y[j].item()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    print(f"\n  Per-class accuracy (best val={best_val_acc:.1f}%):")
    for c in range(5):
        if class_total[c] > 0:
            acc = 100.0 * class_correct[c] / class_total[c]
            print(f"    {CLASS_NAMES[c]:12s}: {acc:.1f}% ({class_correct[c]}/{class_total[c]})")

    return classifier


def save_weights(classifier):
    out_dir = os.path.join(PROJECT_ROOT, "zebrav2", "weights")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "classifier_v2.pt")
    torch.save(classifier.state_dict(), path)
    print(f"\n  Weights saved: {path}")
    return path


def main():
    print("=" * 60)
    print("ClassifierV2 Training on Gameplay Data")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    X, Y = collect_data()
    classifier = train_classifier(X, Y)
    save_weights(classifier)

    print("\nDone. Classifier ready for integration.")


if __name__ == "__main__":
    main()
