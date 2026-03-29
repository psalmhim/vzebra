"""
Training pipelines for ZebrafishBrainV2.

Pipeline 1: Hebbian STDP fine-tuning (tectum→thalamus→pallium weights)
Pipeline 2: Online RL training (reward-modulated STDP during gameplay)
Pipeline 3: Classifier retraining on v2 gameplay data
Pipeline 4: Curriculum training (staged difficulty)

Run: .venv/bin/python -u -m zebrav2.tests.train_all_pipelines
"""
import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'v2_training')
WEIGHT_DIR = os.path.join(PROJECT_ROOT, 'zebrav2', 'weights')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)

GOAL_NAMES = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']


def pipeline1_hebbian(n_episodes=3, steps_per_ep=200):
    """Pipeline 1: Hebbian STDP fine-tuning on tectum→thalamus→pallium."""
    print("\n=== Pipeline 1: Hebbian STDP Fine-Tuning ===")
    from zebrav2.brain.plasticity import EligibilitySTDP

    brain = ZebrafishBrainV2(device=DEVICE)
    brain.reset()

    # Create STDP learners for key projections
    stdp_tc = EligibilitySTDP(brain.thalamus.W_tect_tc.weight, device=DEVICE,
                               A_plus=0.003, A_minus=0.003)
    stdp_pal = EligibilitySTDP(brain.pallium.W_tc_pals.weight, device=DEVICE,
                                A_plus=0.002, A_minus=0.002)

    weight_norms_tc = []
    weight_norms_pal = []
    rewards_all = []

    for ep in range(n_episodes):
        env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=steps_per_ep)
        obs, info = env.reset(seed=42 + ep)
        brain.reset()
        ep_reward = 0.0

        for t in range(steps_per_ep):
            if hasattr(env, 'set_flee_active'):
                env.set_flee_active(brain.current_goal == 1, 0.8 if brain.current_goal == 1 else 0.0)
            inject_sensory(env)
            out = brain.step(obs, env)
            action = np.array([out['turn'], out['speed']], dtype=np.float32)
            obs, reward, term, trunc, info = env.step(action)
            env._eaten_now = info.get('food_eaten_this_step', 0)
            ep_reward += reward

            # STDP update: use tectum→thalamus spike pairs
            tect_spikes = (brain.tectum.sfgs_b.spike_E > 0).float()
            thal_spikes = (brain.thalamus.TC.rate > 0.01).float()
            pal_spikes = (brain.pallium.pal_s.spike_E > 0).float()
            tc_rate = brain.thalamus.TC.rate

            stdp_tc.update_traces(tect_spikes[:stdp_tc.W.shape[1]], thal_spikes)
            stdp_pal.update_traces(tc_rate, pal_spikes[:stdp_pal.W.shape[0]])

            # Consolidate with DA every 10 steps
            if t % 10 == 0:
                DA = brain.neuromod.DA.item()
                ACh = brain.neuromod.ACh.item()
                stdp_tc.consolidate(DA, ACh, eta=0.0005)
                stdp_pal.consolidate(DA, ACh, eta=0.0005)

            if term or trunc:
                break

        env.close()
        rewards_all.append(ep_reward)
        weight_norms_tc.append(float(brain.thalamus.W_tect_tc.weight.data.abs().mean()))
        weight_norms_pal.append(float(brain.pallium.W_tc_pals.weight.data.abs().mean()))
        print(f"  Episode {ep+1}/{n_episodes}: reward={ep_reward:.1f}, "
              f"W_tc={weight_norms_tc[-1]:.4f}, W_pal={weight_norms_pal[-1]:.4f}")

    # Save weights
    torch.save(brain.thalamus.state_dict(),
               os.path.join(WEIGHT_DIR, 'thalamus_hebbian.pt'))
    torch.save(brain.pallium.state_dict(),
               os.path.join(WEIGHT_DIR, 'pallium_hebbian.pt'))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Pipeline 1: Hebbian STDP Fine-Tuning', fontweight='bold')
    axes[0].plot(rewards_all, 'b-o')
    axes[0].set_title('Episode Reward')
    axes[1].plot(weight_norms_tc, 'r-o', label='TC')
    axes[1].plot(weight_norms_pal, 'g-o', label='Pal')
    axes[1].set_title('Weight Norms')
    axes[1].legend()
    axes[2].text(0.1, 0.5, f'Episodes: {n_episodes}\nSteps/ep: {steps_per_ep}\n'
                 f'Final reward: {rewards_all[-1]:.1f}', fontsize=11,
                 transform=axes[2].transAxes)
    axes[2].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'pipeline1_hebbian.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved weights and figure.")


def pipeline2_online_rl(n_episodes=3, steps_per_ep=300):
    """Pipeline 2: Online RL with reward-modulated STDP."""
    print("\n=== Pipeline 2: Online RL Training ===")
    brain = ZebrafishBrainV2(device=DEVICE)

    rewards_per_ep = []
    food_per_ep = []
    survival_per_ep = []
    critic_values_per_ep = []

    for ep in range(n_episodes):
        env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=steps_per_ep)
        obs, info = env.reset(seed=100 + ep)
        brain.reset()
        ep_reward = 0.0
        total_eaten = 0
        critic_vals = []

        for t in range(steps_per_ep):
            if hasattr(env, 'set_flee_active'):
                env.set_flee_active(brain.current_goal == 1, 0.8 if brain.current_goal == 1 else 0.0)
            inject_sensory(env)
            out = brain.step(obs, env)
            action = np.array([out['turn'], out['speed']], dtype=np.float32)
            obs, reward, term, trunc, info = env.step(action)
            env._eaten_now = info.get('food_eaten_this_step', 0)
            total_eaten += env._eaten_now
            ep_reward += reward
            critic_vals.append(out.get('critic_value', 0))

            if term or trunc:
                break

        env.close()
        rewards_per_ep.append(ep_reward)
        food_per_ep.append(total_eaten)
        survival_per_ep.append(t + 1)
        critic_values_per_ep.append(np.mean(critic_vals))
        print(f"  Episode {ep+1}/{n_episodes}: reward={ep_reward:.1f}, "
              f"food={total_eaten}, survived={t+1}, critic={np.mean(critic_vals):.3f}")

    # Save critic weights
    torch.save(brain.critic.state_dict(),
               os.path.join(WEIGHT_DIR, 'critic_online_rl.pt'))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Pipeline 2: Online RL Training', fontweight='bold')
    axes[0].plot(rewards_per_ep, 'b-o')
    axes[0].set_title('Episode Reward')
    axes[1].plot(food_per_ep, 'g-o')
    axes[1].set_title('Food Eaten')
    axes[2].plot(survival_per_ep, 'r-o')
    axes[2].set_title('Survival Steps')
    axes[3].plot(critic_values_per_ep, 'm-o')
    axes[3].set_title('Mean Critic Value')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'pipeline2_online_rl.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved weights and figure.")


def pipeline3_classifier(n_samples=200, n_epochs=5):
    """Pipeline 3: Retrain classifier on v2 gameplay data."""
    print("\n=== Pipeline 3: Classifier Retraining ===")
    from zebrav2.brain.classifier import ClassifierV2

    # Collect gameplay data
    print("  Collecting gameplay data...")
    brain = ZebrafishBrainV2(device=DEVICE)
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=n_samples)
    obs, info = env.reset(seed=42)
    brain.reset()

    X_data = []
    Y_data = []

    for t in range(n_samples):
        inject_sensory(env)
        L = torch.tensor(env.brain_L, dtype=torch.float32, device=DEVICE)
        R = torch.tensor(env.brain_R, dtype=torch.float32, device=DEVICE)

        # Build classifier input
        type_all = torch.cat([L[400:], R[400:]])
        obs_px = ((torch.abs(type_all - 0.75) < 0.1).float().sum()).unsqueeze(0)
        ene_px = ((torch.abs(type_all - 0.5) < 0.1).float().sum()).unsqueeze(0)
        food_px_count = ((torch.abs(type_all - 1.0) < 0.15).float().sum()).unsqueeze(0)
        bound_px = ((torch.abs(type_all - 0.12) < 0.05).float().sum()).unsqueeze(0)
        x = torch.cat([type_all, obs_px, ene_px, food_px_count, bound_px])

        # Ground truth label from pixel analysis
        food_px = float(food_px_count)
        enemy_px = float(ene_px)
        rock_px = float(obs_px)
        if food_px > 5:
            label = 1  # food
        elif enemy_px > 3:
            label = 2  # enemy
        elif rock_px > 5:
            label = 4  # environment
        else:
            label = 0  # nothing

        X_data.append(x.detach().cpu())
        Y_data.append(label)

        out = brain.step(obs, env)
        action = np.array([out['turn'], out['speed']], dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
        env._eaten_now = info.get('food_eaten_this_step', 0)
        if term or trunc:
            break
    env.close()

    X = torch.stack(X_data).to(DEVICE)
    Y = torch.tensor(Y_data, dtype=torch.long, device=DEVICE)
    print(f"  Collected {len(X)} samples. Class distribution: {np.bincount(Y_data, minlength=5)}")

    # Train
    classifier = ClassifierV2(device=DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    losses = []
    accuracies = []

    for epoch in range(n_epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        correct = 0
        for i in range(0, len(X), 16):
            batch_x = X[perm[i:i+16]]
            batch_y = Y[perm[i:i+16]]
            batch_loss = 0.0
            batch_correct = 0
            for j in range(len(batch_x)):
                classifier.reset()
                logits = classifier(batch_x[j])
                loss = torch.nn.functional.cross_entropy(logits, batch_y[j:j+1])
                batch_loss += loss
                pred = logits.argmax(dim=-1)
                batch_correct += (pred == batch_y[j]).sum().item()
            batch_loss = batch_loss / len(batch_x)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * len(batch_x)
            correct += batch_correct

        avg_loss = total_loss / len(X)
        accuracy = correct / len(X)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.1%}")

    # Save
    torch.save(classifier.state_dict(), os.path.join(WEIGHT_DIR, 'classifier_v2.pt'))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Pipeline 3: Classifier Retraining', fontweight='bold')
    axes[0].plot(losses, 'r-o')
    axes[0].set_title('Training Loss')
    axes[1].plot([a * 100 for a in accuracies], 'g-o')
    axes[1].set_title('Accuracy (%)')
    axes[1].set_ylim(0, 100)
    axes[2].bar(range(5), np.bincount(Y_data, minlength=5),
                tick_label=['nothing', 'food', 'enemy', 'colleague', 'env'])
    axes[2].set_title('Training Data Distribution')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'pipeline3_classifier.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved classifier weights. Final accuracy: {accuracies[-1]:.1%}")


def pipeline4_curriculum(stages=3, steps_per_stage=200):
    """Pipeline 4: Curriculum training (staged difficulty)."""
    print("\n=== Pipeline 4: Curriculum Training ===")

    stage_configs = [
        {'n_food': 20, 'pred_state': 'PATROL', 'label': 'Easy (lots of food, passive predator)'},
        {'n_food': 10, 'pred_state': 'STALK', 'label': 'Medium (normal food, stalking predator)'},
        {'n_food': 5, 'pred_state': 'HUNT', 'label': 'Hard (scarce food, hunting predator)'},
    ]

    brain = ZebrafishBrainV2(device=DEVICE)
    stage_results = []

    for stage_idx in range(min(stages, len(stage_configs))):
        config = stage_configs[stage_idx]
        print(f"  Stage {stage_idx+1}: {config['label']}")

        env = ZebrafishPreyPredatorEnv(render_mode=None,
                                        n_food=config['n_food'],
                                        max_steps=steps_per_stage)
        obs, info = env.reset(seed=42 + stage_idx * 10)
        brain.reset()
        total_eaten = 0
        total_reward = 0.0
        goals_log = []

        for t in range(steps_per_stage):
            if hasattr(env, 'set_flee_active'):
                env.set_flee_active(brain.current_goal == 1, 0.8 if brain.current_goal == 1 else 0.0)
            inject_sensory(env)
            out = brain.step(obs, env)
            action = np.array([out['turn'], out['speed']], dtype=np.float32)
            obs, reward, term, trunc, info = env.step(action)
            env._eaten_now = info.get('food_eaten_this_step', 0)
            total_eaten += env._eaten_now
            total_reward += reward
            goals_log.append(brain.current_goal)
            if term or trunc:
                break

        env.close()
        survived = t + 1
        from collections import Counter
        gc = Counter(goals_log)
        result = {
            'stage': stage_idx + 1,
            'label': config['label'],
            'survived': survived,
            'food': total_eaten,
            'reward': total_reward,
            'goal_dist': {GOAL_NAMES[g]: gc.get(g, 0) for g in range(4)},
            'pass': survived >= steps_per_stage * 0.8 and total_eaten >= 1,
        }
        stage_results.append(result)
        status = 'PASS' if result['pass'] else 'LEARNING'
        print(f"    survived={survived}, food={total_eaten}, reward={total_reward:.1f} → {status}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Pipeline 4: Curriculum Training', fontweight='bold')

    labels = [f"S{r['stage']}" for r in stage_results]
    axes[0].bar(labels, [r['survived'] for r in stage_results],
                color=['green' if r['pass'] else 'orange' for r in stage_results])
    axes[0].set_title('Survival (steps)')
    axes[0].axhline(steps_per_stage * 0.8, color='k', linestyle='--')

    axes[1].bar(labels, [r['food'] for r in stage_results],
                color=['green' if r['food'] > 0 else 'red' for r in stage_results])
    axes[1].set_title('Food Eaten')

    axes[2].bar(labels, [r['reward'] for r in stage_results],
                color=['green' if r['reward'] > 0 else 'red' for r in stage_results])
    axes[2].set_title('Total Reward')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'pipeline4_curriculum.png'), dpi=150)
    plt.close(fig)

    n_pass = sum(1 for r in stage_results if r['pass'])
    print(f"  Curriculum: {n_pass}/{len(stage_results)} stages PASS")


if __name__ == '__main__':
    t0 = time.time()
    print(f"Device: {DEVICE}")

    pipeline1_hebbian(n_episodes=3, steps_per_ep=200)
    pipeline2_online_rl(n_episodes=3, steps_per_ep=300)
    pipeline3_classifier(n_samples=200, n_epochs=5)
    pipeline4_curriculum(stages=3, steps_per_stage=200)

    print(f"\nTotal training time: {time.time()-t0:.1f}s")
    print(f"Figures saved to: {PLOT_DIR}/")
    print(f"Weights saved to: {WEIGHT_DIR}/")
