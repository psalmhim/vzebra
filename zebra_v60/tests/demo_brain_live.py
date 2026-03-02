"""
Live visualization of the full brain agent in the gym environment.

Shows the zebrafish with its vision strip, goal state, and critic diagnostics.
Press Q or close the window to quit.

Run: python -m zebra_v60.tests.demo_brain_live
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebra_v60.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebra_v60.gym_env.brain_agent import BrainAgent
from zebra_v60.brain.goal_policy_v60 import GOAL_NAMES


def run_live(T=2000, use_rl_critic=True):
    env = ZebrafishPreyPredatorEnv(
        render_mode="human", n_food=15, max_steps=T)

    cls_path = os.path.join(
        PROJECT_ROOT, "zebra_v60", "weights", "classifier_v60.pt")
    agent = BrainAgent(
        device="auto", cls_weights_path=cls_path,
        use_habit=True, use_rl_critic=use_rl_critic)

    obs, info = env.reset(seed=42)
    agent.reset()

    total_eaten = 0

    for t in range(T):
        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward,
                               done=terminated or truncated, env=env)

        diag = agent.last_diagnostics
        eaten = info.get("food_eaten_this_step", 0)
        total_eaten += eaten

        if t % 50 == 0:
            goal = GOAL_NAMES[diag.get("goal", 2)]
            sc = "H" if diag.get("shortcut_active", False) else "E"
            conf = diag.get("confidence", 0.0)
            critic_v = diag.get("critic_value", 0.0)
            td = diag.get("td_error", 0.0)
            print(f"  t={t:4d}  goal={goal:8s}[{sc}]  "
                  f"conf={conf:.2f}  V={critic_v:.3f}  "
                  f"td={td:.4f}  eaten={total_eaten}")

        env.render()

        if terminated or truncated:
            print(f"Episode ended at step {t}: "
                  f"terminated={terminated}, truncated={truncated}")
            break

    env.close()
    print(f"\nDone. Steps: {t+1}, Food eaten: {total_eaten}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-critic", action="store_true",
                        help="Run pure AIF without RL critic")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    run_live(T=args.steps, use_rl_critic=not args.no_critic)
