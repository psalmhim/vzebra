# ============================================================
# train_agent.py
# Full training of zebrafish agent:
#   1) Supervised classifier training
#   2) Dopaminergic RL policy adaptation
# ============================================================

import torch
from dataset_builder import build_dataset
from zebrafish_agent import ZebrafishAgent
from prey_predator_env import PreyPredatorEnv
import matplotlib.pyplot as plt
import os


def train_full(device="cpu"):
    """
    Full training pipeline:
    1. Build supervised dataset from environment
    2. Train prey/predator classifier
    3. RL fine-tuning of policy
    """
    
    print("=" * 70)
    print("ZEBRAFISH AGENT TRAINING PIPELINE")
    print("=" * 70)
    print()

    # 1) Build supervised dataset
    print("STAGE 1: Building supervised dataset")
    print("-" * 70)
    X, y = build_dataset(N=2000, device=device)

    # 2) Build agent and train classifier
    print("STAGE 2: Training prey/predator classifier")
    print("-" * 70)
    agent = ZebrafishAgent(device=device)
    agent.train_classifier(X, y, epochs=100, lr=5e-3, device=device)

    # 3) RL fine-tuning with classifier-guided policy
    print("STAGE 3: RL fine-tuning (classifier-guided policy)")
    print("-" * 70)
    
    # Create new agent with classifier-guided mode enabled
    agent_rl = ZebrafishAgent(device=device, use_classifier_policy=True)
    agent_rl.prey_pred.load_state_dict(agent.prey_pred.state_dict())  # Copy trained classifier weights
    
    env = PreyPredatorEnv(T=300)
    rewards = []
    correct_count = 0
    total_count = 0

    for t in range(1500):
        obs = env.reset()
        
        # Skip neutral objects for policy training
        if obs["type"] == 0:
            continue
            
        label = 0 if obs["type"] == 1 else 1

        dx = obs["dx"]
        dy = obs["dy"]
        size = obs["size"]

        retL = agent_rl.renderer.render(dx, dy, size)
        retR = agent_rl.renderer.render(dx, dy, size)
        out = agent_rl.step(retL, retR)

        prey_prob = out["prey_prob"]
        pred_prob = out["pred_prob"]
        choice = out["policy"]

        r = agent_rl.rl_update_policy(prey_prob, pred_prob, choice, label)
        rewards.append(r)
        
        if r > 0:
            correct_count += 1
        total_count += 1

        if t % 300 == 0:
            recent_acc = correct_count / max(total_count, 1)
            print(f"[RL] Step {t:4d}  Reward={r:+.1f}  Running acc={recent_acc*100:.1f}%")

    # Final RL stats
    final_acc = correct_count / max(total_count, 1)
    print()
    print("=" * 70)
    print("✓ RL fine-tuning complete")
    print(f"  Final accuracy: {final_acc*100:.1f}%")
    print(f"  Correct: {correct_count}/{total_count}")
    print("=" * 70)
    print()

    # Save RL rewards curve
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(rewards, alpha=0.6, label="Reward")
    
    # Add running average
    window = 50
    if len(rewards) >= window:
        running_avg = [sum(rewards[max(0,i-window):i+1])/min(i+1,window) for i in range(len(rewards))]
        plt.plot(running_avg, linewidth=2, color='red', label=f'Running avg ({window})')
    
    plt.title("RL Rewards During Training")
    plt.xlabel("Training step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/rl_rewards.png", dpi=150)
    print("✓ Saved RL reward curve: plots/rl_rewards.png")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.prey_pred.state_dict(), "models/prey_pred.pt")
    print("✓ Saved trained classifier: models/prey_pred.pt")
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    train_full()

