"""
Step 11b: Gameplay-Based Classifier Data Collection & Retraining

Phase 1: Run 10 episodes of 300 steps with BrainAgent, capturing retinal
          snapshots every 3 steps with ground-truth labels from the environment.
Phase 2: Retrain the classifier on gameplay data for 50 epochs.

Run: python -m zebrav1.tests.step11b_gameplay_classifier
Output: zebrav1/weights/gameplay_classifier_data.npz
        zebrav1/weights/classifier.pt
        zebrav1/weights/classifier_wfb.pt
"""
import os
import sys
import math
import shutil

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav1.gym_env.brain_agent import BrainAgent
from zebrav1.brain.zebrafish_snn import ZebrafishSNN
from zebrav1.brain.device_util import get_device

CLASS_NAMES = ["nothing", "food", "enemy", "colleague", "environment"]
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "zebrav1", "weights")


def collect_gameplay_data(n_episodes=10, max_steps=300, sample_interval=3):
    """Phase 1: Collect retinal snapshots with ground-truth labels."""
    print("=" * 60)
    print("Phase 1: Collecting Gameplay Classifier Data")
    print(f"  Episodes: {n_episodes}, Steps/ep: {max_steps}, "
          f"Sample every {sample_interval} steps")
    print("=" * 60)

    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=max_steps,
        n_colleagues=3, side_panels=False)
    agent = BrainAgent(use_allostasis=True)

    X_all, y_all = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 7)
        agent.reset()

        ep_labels = {c: 0 for c in range(5)}

        for t in range(max_steps):
            action = agent.act(obs, env)
            obs, rew, term, trunc, info = env.step(action)
            agent.update_post_step(info, reward=rew, done=term, env=env)

            if t % sample_interval == 0:
                # --- Extract retinal input (same 804-dim as classifier) ---
                out = agent._last_snn_out
                type_L = out["retL_full"][0, 400:].cpu().numpy()
                type_R = out["retR_full"][0, 400:].cpu().numpy()
                type_features = np.concatenate([type_L, type_R])  # 800

                # Aggregate pixel counts (normalized by 50)
                obs_count = (
                    np.sum(np.abs(type_L - 0.75) < 0.1)
                    + np.sum(np.abs(type_R - 0.75) < 0.1)
                ) / 50.0
                ene_count = (
                    np.sum(np.abs(type_L - 0.5) < 0.1)
                    + np.sum(np.abs(type_R - 0.5) < 0.1)
                ) / 50.0
                food_count = (
                    np.sum(np.abs(type_L - 1.0) < 0.1)
                    + np.sum(np.abs(type_R - 1.0) < 0.1)
                ) / 50.0
                bnd_count = (
                    np.sum(np.abs(type_L - 0.12) < 0.05)
                    + np.sum(np.abs(type_R - 0.12) < 0.05)
                ) / 50.0

                x = np.concatenate([
                    type_features,
                    [obs_count, ene_count, food_count, bnd_count]
                ])

                # --- Ground truth label from environment ---
                pred_dist = math.sqrt(
                    (env.fish_x - env.pred_x) ** 2
                    + (env.fish_y - env.pred_y) ** 2
                )
                if env.foods:
                    nearest_food = min(
                        math.sqrt(
                            (env.fish_x - f[0]) ** 2
                            + (env.fish_y - f[1]) ** 2
                        )
                        for f in env.foods
                    )
                else:
                    nearest_food = 999.0

                obs_px = (
                    np.sum(np.abs(type_L - 0.75) < 0.1)
                    + np.sum(np.abs(type_R - 0.75) < 0.1)
                )

                # Priority: enemy > food > environment > colleague > nothing
                if pred_dist < 150 and ene_count * 50 > 3:
                    label = 2  # enemy
                elif nearest_food < 50 and food_count * 50 > 3:
                    label = 1  # food
                elif obs_px > 30:
                    label = 4  # environment
                elif any(
                    math.sqrt(
                        (env.fish_x - c["x"]) ** 2
                        + (env.fish_y - c["y"]) ** 2
                    ) < 80
                    for c in getattr(env, 'colleagues', [])
                ):
                    label = 3  # colleague
                else:
                    label = 0  # nothing

                X_all.append(x)
                y_all.append(label)
                ep_labels[label] += 1

            if term or trunc:
                break

        print(f"  Episode {ep:2d}: {t + 1:3d} steps | "
              f"nothing={ep_labels[0]} food={ep_labels[1]} "
              f"enemy={ep_labels[2]} colleague={ep_labels[3]} "
              f"env={ep_labels[4]}")

    env.close()

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)
    dist = [int((y == c).sum()) for c in range(5)]
    print(f"\nDataset: {len(X)} samples")
    print(f"  Class distribution: {dict(zip(CLASS_NAMES, dist))}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, "gameplay_classifier_data.npz")
    np.savez(save_path, X=X, y=y)
    print(f"  Saved: {save_path}")

    return X, y


def retrain_classifier(X_np, y_np, n_epochs=50, batch_size=32, lr=0.001):
    """Phase 2: Retrain classifier on gameplay data."""
    print("\n" + "=" * 60)
    print("Phase 2: Retraining Classifier on Gameplay Data")
    print(f"  Samples: {len(X_np)}, Epochs: {n_epochs}, "
          f"Batch: {batch_size}, LR: {lr}")
    print("=" * 60)

    device = get_device()
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.int64)

    # Load model with genomic+hebbian weights
    model = ZebrafishSNN(device=str(device))
    hebbian_path = os.path.join(WEIGHTS_DIR, "genomic_hebbian.pt")
    if os.path.exists(hebbian_path):
        state = torch.load(hebbian_path, map_location=device, weights_only=False)
        model.load_saveable_state(state)
        print(f"  Loaded base weights: {hebbian_path}")
    else:
        print(f"  WARNING: {hebbian_path} not found, using random init")

    # Freeze all except classifier head
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if "cls" in name:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    print(f"  Frozen: {frozen_count} params, Trainable: {trainable_count} params")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    # Class weights: inverse frequency balancing
    counts = torch.bincount(y, minlength=5).float()
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * 5.0
    print(f"  Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        correct = 0
        n_batches = 0

        for i in range(0, len(X), batch_size):
            batch_x = X[perm[i:i + batch_size]].to(device)
            batch_y = y[perm[i:i + batch_size]].to(device)

            logits = model.cls_out(torch.relu(model.cls_hidden(batch_x)))
            loss = F.cross_entropy(logits, batch_y, weight=weights.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == batch_y).sum().item()
            n_batches += 1

        acc = correct / len(X)
        avg_loss = total_loss / max(1, n_batches)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  acc={acc:.1%}")

    # === Per-class accuracy ===
    model.eval()
    print("\n--- Final Per-Class Accuracy ---")
    class_correct = [0] * 5
    class_total = [0] * 5

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i + batch_size].to(device)
            batch_y = y[i:i + batch_size].to(device)
            logits = model.cls_out(torch.relu(model.cls_hidden(batch_x)))
            preds = logits.argmax(1)
            for c in range(5):
                mask = (batch_y == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == c) & mask).sum().item()

    overall_correct = sum(class_correct)
    overall_total = sum(class_total)
    overall_acc = overall_correct / max(1, overall_total)

    for c in range(5):
        ca = class_correct[c] / max(1, class_total[c])
        status = "OK" if ca >= 0.8 else ("WARN" if ca >= 0.5 else "LOW")
        print(f"  {CLASS_NAMES[c]:12s}: {ca:6.1%}  "
              f"({class_correct[c]}/{class_total[c]}) {status}")
    print(f"  {'Overall':12s}: {overall_acc:6.1%}  "
          f"({overall_correct}/{overall_total})")

    # Save weights
    state = model.get_saveable_state()
    cls_path = os.path.join(WEIGHTS_DIR, "classifier.pt")
    torch.save(state, cls_path)
    print(f"\n  Saved: {cls_path}")

    wfb_path = os.path.join(WEIGHTS_DIR, "classifier_wfb.pt")
    shutil.copy(cls_path, wfb_path)
    print(f"  Copied: {wfb_path}")

    print(f"\nClassifier retrained. Final overall accuracy: {overall_acc:.1%}")
    return overall_acc


def main():
    # Phase 1: Collect data
    X, y = collect_gameplay_data(
        n_episodes=10, max_steps=300, sample_interval=3)

    # Phase 2: Retrain classifier
    retrain_classifier(X, y, n_epochs=50, batch_size=32, lr=0.001)


if __name__ == "__main__":
    main()
