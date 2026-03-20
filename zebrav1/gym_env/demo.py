"""
Demo: Zebrafish Predator-Prey Gymnasium Environment

Modes:
  1. Random agent (default) — shows the environment with random actions
  2. Heuristic agent — rule-based flee/forage using observations
  3. Brain agent — full SNN brain pipeline with optional neural monitor
  4. Human mode — render to screen

Run:
  python -m zebrav1.gym_env.demo             # random, headless
  python -m zebrav1.gym_env.demo --heuristic  # heuristic, headless
  python -m zebrav1.gym_env.demo --render      # heuristic, pygame window
  python -m zebrav1.gym_env.demo --brain --monitor  # brain agent + neural viz
  python -m zebrav1.gym_env.demo --brain --render --record  # save MP4 video
  python -m zebrav1.gym_env.demo --brain --monitor --record --steps 800

Output: plots/v1_gym_env_demo.png, plots/zebrafish_brain_demo.mp4
"""
import os
import sys
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv


def heuristic_action(obs):
    """Simple rule-based agent: flee predator, seek food, avoid walls."""
    # obs indices:
    #   0-3: fish (x, y, heading, speed)
    #   4-8: pred (x, y, heading, dist, angle)
    #   9-10: nearest food (dist, angle)
    #   11-14: walls (N, S, E, W)
    #   15: eaten count

    pred_dist_n = obs[7]   # normalized predator distance [-1, 1]
    pred_angle_n = obs[8]  # normalized predator angle [-1, 1]
    food_dist_n = obs[9]
    food_angle_n = obs[10]
    wall_n = obs[11]       # distance to top wall
    wall_s = obs[12]
    wall_e = obs[13]
    wall_w = obs[14]

    # Predator distance: -1 = close, 1 = far
    pred_close = pred_dist_n < -0.3  # predator is close

    if pred_close:
        # FLEE: turn away from predator, max speed
        turn = -pred_angle_n  # turn opposite to predator
        speed = 1.0
    else:
        # FORAGE: turn toward nearest food
        turn = food_angle_n * 0.8
        speed = 0.5 + 0.3 * max(0, -food_dist_n)  # faster when food close

    # Wall avoidance override
    wall_min = min(wall_n, wall_s, wall_e, wall_w)
    if wall_min < -0.6:  # very close to a wall
        if wall_n == wall_min:
            turn += 0.5   # turn away from top
        elif wall_s == wall_min:
            turn -= 0.5
        elif wall_w == wall_min:
            turn -= 0.3
        elif wall_e == wall_min:
            turn += 0.3
        speed = max(0.3, speed)

    turn = np.clip(turn, -1, 1)
    speed = np.clip(speed, 0, 1)

    return np.array([turn, speed], dtype=np.float32)


def run_demo(use_heuristic=False, render=False, T=1000):
    render_mode = "human" if render else None
    env = ZebrafishPreyPredatorEnv(render_mode=render_mode, n_food=15,
                                    max_steps=T)

    obs, info = env.reset(seed=42)

    # History
    fish_xs, fish_ys = [], []
    pred_xs, pred_ys = [], []
    rewards = []
    food_eaten = []

    agent_name = "heuristic" if use_heuristic else "random"
    print(f"Running {agent_name} agent for {T} steps...")

    cumulative_reward = 0.0
    for t in range(T):
        if use_heuristic:
            action = heuristic_action(obs)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        fish_xs.append(info["fish_pos"][0])
        fish_ys.append(info["fish_pos"][1])
        pred_xs.append(info["pred_pos"][0])
        pred_ys.append(info["pred_pos"][1])
        rewards.append(reward)
        food_eaten.append(info["total_eaten"])

        if t % 200 == 0:
            print(f"  t={t:4d}  fish=({info['fish_pos'][0]:.0f},"
                  f"{info['fish_pos'][1]:.0f})  "
                  f"pred=({info['pred_pos'][0]:.0f},"
                  f"{info['pred_pos'][1]:.0f})  "
                  f"eaten={info['total_eaten']}  reward={cumulative_reward:.1f}")

        if terminated or truncated:
            if terminated:
                print(f"  CAUGHT by predator at step {t}!")
            break

    env.close()

    print(f"\nSummary: survived {t+1} steps, "
          f"ate {info['total_eaten']} food, "
          f"total reward = {cumulative_reward:.1f}")

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Trajectory
    ax = axes[0, 0]
    ax.plot(fish_xs, fish_ys, "b-", alpha=0.5, linewidth=1, label="Zebrafish")
    ax.plot(pred_xs, pred_ys, "r-", alpha=0.3, linewidth=1, label="Predator")
    ax.plot(fish_xs[0], fish_ys[0], "bo", markersize=10, label="Fish start")
    ax.plot(pred_xs[0], pred_ys[0], "ro", markersize=10, label="Pred start")
    if len(fish_xs) > 1:
        ax.plot(fish_xs[-1], fish_ys[-1], "bs", markersize=10, label="Fish end")
    for fx, fy in env.foods:
        ax.plot(fx, fy, "g.", markersize=8, alpha=0.5)
    ax.set_xlim(0, env.arena_w)
    ax.set_ylim(0, env.arena_h)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"Trajectory ({agent_name} agent)")
    ax.legend(fontsize=7, loc="upper right")

    # Panel 2: Cumulative food
    ax = axes[0, 1]
    ax.plot(food_eaten, color="green", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative food eaten")
    ax.set_title("Foraging Performance")

    # Panel 3: Reward over time
    ax = axes[1, 0]
    cum_rewards = np.cumsum(rewards)
    ax.plot(cum_rewards, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative Reward")

    # Panel 4: Fish-predator distance
    ax = axes[1, 1]
    dists = [math.sqrt((fx - px) ** 2 + (fy - py) ** 2)
             for fx, fy, px, py in zip(fish_xs, fish_ys, pred_xs, pred_ys)]
    ax.plot(dists, color="red", alpha=0.7, linewidth=1)
    ax.axhline(env.pred_catch_radius, color="darkred", linestyle="--",
               label=f"Catch radius ({env.pred_catch_radius})")
    ax.axhline(env.pred_chase_radius, color="orange", linestyle="--",
               label=f"Chase radius ({env.pred_chase_radius})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.set_title("Fish-Predator Distance")
    ax.legend(fontsize=8)

    fig.suptitle(f"Zebrafish Gym Environment — {agent_name} agent",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v1_gym_env_demo.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


def run_brain_demo(render=False, monitor=False, record=False, T=1000,
                   autosave=False, load_checkpoint=None,
                   predator_brain=False, sound=False,
                   multi_agent=False):
    """Run the full brain agent with optional neural activity monitor.

    When --monitor is used, a single combined window is created:
      Left side:  environment arena (800x600)
      Right side: neural activity monitor (500x600)

    When --record is used, frames are captured and saved to an MP4 video.
    Recording implies rendering (rgb_array mode is used automatically).
    """
    import pygame
    from zebrav1.gym_env.brain_agent import BrainAgent

    if multi_agent:
        from zebrav1.gym_env.multi_agent_env import MultiAgentZebrafishEnv
        EnvClass = MultiAgentZebrafishEnv
        env_kwargs = {"n_fish": 5}
    else:
        EnvClass = ZebrafishPreyPredatorEnv
        env_kwargs = {}

    # Recording requires rgb_array rendering
    if record:
        render = True

    mon = None
    combined_screen = None
    clock = None
    frames = []  # for video recording

    if monitor and render:
        # Combined window: env renders off-screen, we composite
        from zebrav1.viz.neural_monitor import NeuralMonitor

        env = EnvClass(
            render_mode="rgb_array", n_food=15, max_steps=T,
            side_panels=True, use_predator_brain=predator_brain,
            **env_kwargs)
        mon = NeuralMonitor()
        MON_W = mon.WIDTH  # 500
        RENDER_W = env.render_width
        RENDER_H = env.render_height

        COMBINED_H = max(RENDER_H, mon.HEIGHT)

        pygame.init()
        combined_screen = pygame.display.set_mode(
            (RENDER_W + MON_W, COMBINED_H))
        pygame.display.set_caption(
            "Zebrafish Brain Agent + Neural Monitor")
        clock = pygame.time.Clock()

    elif render:
        # Env-only window (standard human mode)
        env = EnvClass(
            render_mode="human" if not record else "rgb_array",
            n_food=15, max_steps=T, side_panels=True,
            use_predator_brain=predator_brain, **env_kwargs)
        RENDER_W = env.render_width
        RENDER_H = env.render_height
        if record:
            pygame.init()
            combined_screen = pygame.display.set_mode(
                (RENDER_W, RENDER_H))
            pygame.display.set_caption("Zebrafish Brain Agent")
            clock = pygame.time.Clock()

    elif monitor:
        # Monitor-only (no arena rendering)
        from zebrav1.viz.neural_monitor import NeuralMonitor

        env = EnvClass(
            render_mode=None, n_food=15, max_steps=T,
            use_predator_brain=predator_brain, **env_kwargs)
        mon = NeuralMonitor()
        MON_W = mon.WIDTH
        RENDER_W = MON_W
        RENDER_H = mon.HEIGHT

        pygame.init()
        combined_screen = pygame.display.set_mode(
            (RENDER_W, RENDER_H))
        pygame.display.set_caption("Zebrafish Neural Monitor")
        clock = pygame.time.Clock()

    else:
        # Headless
        env = EnvClass(
            render_mode=None, n_food=15, max_steps=T,
            use_predator_brain=predator_brain, **env_kwargs)
        RENDER_W = env.render_width
        RENDER_H = env.render_height

    agent = BrainAgent(device="auto", world_model="vae",
                       use_allostasis=True)
    if load_checkpoint is not None:
        agent.load_checkpoint(load_checkpoint)
    obs, info = env.reset(seed=42)
    agent.reset()

    # Sound engines
    spike_audio = None
    sound_engine = None
    if sound:
        try:
            from zebrav1.viz.spike_audio import SpikeAudioEngine
            spike_audio = SpikeAudioEngine(master_volume=0.4)
        except ImportError:
            pass
        from zebrav1.viz.sound_engine import SoundEngine
        sound_engine = SoundEngine(enabled=True)

    rec_msg = " [RECORDING]" if record else ""
    pred_msg = " + predator brain" if predator_brain else ""
    snd_msg = " + spike audio" if sound else ""
    print(f"Running brain agent for {T} steps"
          f"{' + neural monitor' if monitor else ''}"
          f"{pred_msg}{snd_msg}{rec_msg}...")

    cumulative_reward = 0.0
    running = True

    for t in range(T):
        if not running:
            break

        action = agent.act(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update_post_step(info, reward=reward,
                               done=terminated or truncated, env=env)
        cumulative_reward += reward

        # --- Rendering ---
        if combined_screen is not None:
            # Handle pygame events on the combined window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            if running:
                combined_screen.fill((0, 0, 0))

                # Left: env arena (if rendering)
                if render:
                    env_frame = env.render()  # rgb_array: [H, W, 3]
                    if env_frame is not None:
                        env_surf = pygame.surfarray.make_surface(
                            np.transpose(env_frame, (1, 0, 2)))
                        combined_screen.blit(env_surf, (0, 0))

                # Right (or full): neural monitor
                if mon is not None and hasattr(agent, '_last_snn_out'):
                    mon.update(
                        agent._last_snn_out, agent.last_diagnostics)
                    mon.render()
                    mon_x = RENDER_W if render else 0
                    combined_screen.blit(mon.get_surface(), (mon_x, 0))

                # Spike sonification
                if spike_audio is not None and hasattr(agent, '_last_snn_out'):
                    spike_audio.update(agent._last_snn_out)

                pygame.display.flip()

                # Capture frame for recording
                if record:
                    w, h = combined_screen.get_size()
                    frame = np.transpose(
                        np.array(
                            pygame.surfarray.pixels3d(combined_screen)),
                        (1, 0, 2)).copy()
                    frames.append(frame)

                clock.tick(30)

        elif render:
            # Standard human-mode rendering (env handles its own window)
            env.render()

        # Spike audio (works with or without monitor/combined_screen)
        if (spike_audio is not None
                and combined_screen is None
                and hasattr(agent, '_last_snn_out')):
            spike_audio.update(agent._last_snn_out)

        # Biological sound effects (heartbeat, splash, rumble)
        if sound_engine is not None:
            d = agent.last_diagnostics
            sound_engine.update(
                step=t,
                heart_rate=d.get("heart_rate", 0.3),
                is_fleeing=(d.get("goal", 2) == 1),
                mauthner_fired=d.get("mauthner_active", False),
                food_eaten=(info.get("food_eaten_this_step", 0) > 0),
                enemy_proximity=d.get("cls_probs", [0]*5)[2])

        if t % 100 == 0:
            vae = agent.last_diagnostics.get("vae", {})
            ep = vae.get("epistemic_per_goal", [0, 0, 0])
            goal_names = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
            goal = agent.last_diagnostics.get("goal", 2)
            ep_str = ",".join(f"{v:.3f}" for v in ep)
            print(f"  t={t:4d}  goal={goal_names[goal]:7s}  "
                  f"eaten={info['total_eaten']}  "
                  f"ep=[{ep_str}]  "
                  f"reward={cumulative_reward:.1f}")

        if terminated or truncated:
            if terminated:
                print(f"  CAUGHT by predator at step {t}!")
            break

    # Prevent env.close() from calling pygame.quit() when we own the display
    if combined_screen is not None:
        env._screen = None
    env.close()
    if combined_screen is not None:
        pygame.quit()

    print(f"\nSummary: survived {t+1} steps, "
          f"ate {info['total_eaten']} food, "
          f"total reward = {cumulative_reward:.1f}")

    # Save checkpoint
    if autosave:
        ckpt_dir = os.path.join(PROJECT_ROOT, "zebrav1", "weights")
        ckpt_path = os.path.join(ckpt_dir, "brain_checkpoint.pt")
        agent.save_checkpoint(ckpt_path)

    # Save video
    if record and frames:
        import imageio
        video_dir = os.path.join(PROJECT_ROOT, "plots")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, "zebrafish_brain_demo.mp4")
        print(f"\nSaving {len(frames)} frames to {video_path} ...")
        imageio.mimwrite(video_path, frames, fps=30, quality=8)
        print(f"Video saved: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic", action="store_true",
                        help="Use heuristic agent instead of random")
    parser.add_argument("--brain", action="store_true",
                        help="Use full brain agent (SNN + world model)")
    parser.add_argument("--monitor", action="store_true",
                        help="Show neural activity monitor window")
    parser.add_argument("--render", action="store_true",
                        help="Render with pygame (requires display)")
    parser.add_argument("--record", action="store_true",
                        help="Record to MP4 video (plots/zebrafish_brain_demo.mp4)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--autosave", action="store_true",
                        help="Save checkpoint after run "
                             "(zebrav1/weights/brain_checkpoint.pt)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Load checkpoint before run")
    parser.add_argument("--predator-brain", action="store_true",
                        help="Use SNN brain agent for predator "
                             "(instead of state machine)")
    parser.add_argument("--sound", action="store_true",
                        help="Enable spike sonification "
                             "(different sounds per neuron group)")
    parser.add_argument("--multi-agent", action="store_true",
                        help="5-fish multi-agent mode "
                             "(1 focal + 4 conspecific brains)")
    args = parser.parse_args()

    if args.brain:
        run_brain_demo(render=args.render, monitor=args.monitor,
                       record=args.record, T=args.steps,
                       autosave=args.autosave,
                       load_checkpoint=args.load_checkpoint,
                       predator_brain=args.predator_brain,
                       sound=args.sound,
                       multi_agent=args.multi_agent)
    else:
        run_demo(use_heuristic=args.heuristic, render=args.render,
                 T=args.steps)
