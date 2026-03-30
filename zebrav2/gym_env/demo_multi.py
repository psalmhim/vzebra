"""
Multi-agent demo: v2 brain as focal fish in v1's multi-agent environment.

Run: .venv/bin/python -u -m zebrav2.gym_env.demo_multi
"""
import sys, os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.multi_agent_env import MultiAgentZebrafishEnv
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav2.spec import DEVICE


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--n-fish', type=int, default=5)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--render', action='store_true', default=True)
    args = parser.parse_args()

    env = MultiAgentZebrafishEnv(
        n_fish=args.n_fish,
        render_mode='human' if args.render else None,
        n_food=15,
        max_steps=args.steps)
    brain = ZebrafishBrainV2(device=DEVICE)
    obs, info = env.reset(seed=42)
    brain.reset()

    frames = [] if args.record else None
    total_eaten = 0

    for t in range(args.steps):
        # Inject sensory for focal fish
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(brain.current_goal == 1,
                                0.8 if brain.current_goal == 1 else 0.0)
        inject_sensory(env)

        # v2 brain decides for focal fish
        out = brain.step(obs, env)
        action = np.array([out['turn'], out['speed']], dtype=np.float32)

        # Step all fish (focal + conspecifics)
        obs, reward, terminated, truncated, info = env.step(action)
        env._eaten_now = info.get('food_eaten_this_step', 0)
        total_eaten += env._eaten_now

        if args.record and hasattr(env, 'render'):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if t % 100 == 0:
            goal = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL'][brain.current_goal]
            n_alive = len([f for f in env.all_fish if f.get('alive', True)])
            print(f"Step {t:4d}: goal={goal}, food={total_eaten}, "
                  f"energy={brain.energy:.0f}, fish_alive={n_alive}/{args.n_fish}")

        if terminated or truncated:
            print(f"Terminated at step {t}" +
                  (" (caught)" if terminated and not truncated else ""))
            break

    env.close()
    survived = t + 1
    print(f"\nMulti-agent: {survived} steps, {total_eaten} food, "
          f"{args.n_fish} fish")

    if frames and len(frames) > 10:
        import imageio
        outpath = os.path.join(PROJECT_ROOT, 'plots', 'zebrafish_v2_multi.mp4')
        writer = imageio.get_writer(outpath, fps=20,
                                     codec='libx264', quality=7)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"Video saved: {outpath} ({len(frames)} frames)")


if __name__ == '__main__':
    main()
