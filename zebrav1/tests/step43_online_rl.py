"""
Step 43: Online Reinforcement Learning

Runs multiple episodes with the brain agent, using the Hebbian learning
system and RL critic to improve behavior over time. After each episode
batch, saves a checkpoint with updated weights.

Reward: +10 per food eaten, -50 for getting caught, +0.1 per survival step.

Run: python -m zebrav1.tests.step43_online_rl
"""
import os, sys, math, numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav1.gym_env.brain_agent import BrainAgent


def run_online_rl(n_episodes=20, max_steps=500, save_interval=5):
    print("=" * 60)
    print("Step 43: Online Reinforcement Learning")
    print("=" * 60)

    env = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=15, max_steps=max_steps, side_panels=False)
    agent = BrainAgent(use_allostasis=True, use_rl_critic=True)

    # Load checkpoint if exists
    ckpt_path = "zebrav1/weights/brain_checkpoint.pt"
    if os.path.exists(ckpt_path):
        try:
            agent.load_checkpoint(ckpt_path)
            print(f"Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    episode_stats = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 13 + 7)
        agent.reset()

        total_reward = 0.0
        for t in range(max_steps):
            action = agent.act(obs, env)
            obs, rew, term, trunc, info = env.step(action)

            # Shaped reward
            food_eaten = info.get("food_eaten_this_step", 0)
            step_reward = food_eaten * 10.0 + 0.1  # survival + food
            if term:
                step_reward -= 50.0  # death penalty

            agent.update_post_step(info, reward=step_reward, done=term, env=env)
            total_reward += step_reward

            if term or trunc:
                break

        eaten = env.total_eaten
        stats = {
            "episode": ep,
            "steps": t + 1,
            "eaten": eaten,
            "reward": total_reward,
            "caught": term,
        }
        episode_stats.append(stats)
        caught = "CAUGHT" if term else "survived"
        print(f"  Ep {ep:2d}: {t+1:4d} steps, {eaten:2d} food, "
              f"reward={total_reward:+6.1f} [{caught}]")

        # Save checkpoint periodically
        if (ep + 1) % save_interval == 0:
            agent.save_checkpoint(ckpt_path)

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    # Compare first half vs second half
    half = n_episodes // 2
    first = episode_stats[:half]
    second = episode_stats[half:]

    f_eaten = np.mean([s["eaten"] for s in first])
    s_eaten = np.mean([s["eaten"] for s in second])
    f_steps = np.mean([s["steps"] for s in first])
    s_steps = np.mean([s["steps"] for s in second])
    f_caught = np.mean([s["caught"] for s in first])
    s_caught = np.mean([s["caught"] for s in second])

    print(f"  First half:  {f_eaten:.1f} food, {f_steps:.0f} steps, "
          f"{f_caught:.0%} caught")
    print(f"  Second half: {s_eaten:.1f} food, {s_steps:.0f} steps, "
          f"{s_caught:.0%} caught")

    improved = s_eaten > f_eaten or s_steps > f_steps
    print(f"\n  Learning: {'YES' if improved else 'NO'}")

    # Final save
    agent.save_checkpoint(ckpt_path)
    print(f"  Final checkpoint saved to {ckpt_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        eps = range(n_episodes)

        axes[0].plot(eps, [s["eaten"] for s in episode_stats], 'g.-')
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Food Eaten")
        axes[0].set_title("Foraging")

        axes[1].plot(eps, [s["steps"] for s in episode_stats], 'b.-')
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Survival Steps")
        axes[1].set_title("Survival")

        axes[2].plot(eps, [s["reward"] for s in episode_stats], 'r.-')
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Total Reward")
        axes[2].set_title("Reward")

        plt.suptitle("Online RL Training", fontsize=14)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/v1_step43_online_rl.png", dpi=150)
        print(f"  Plot saved: plots/v1_step43_online_rl.png")
    except Exception as e:
        print(f"  Plot failed: {e}")


if __name__ == "__main__":
    run_online_rl()
