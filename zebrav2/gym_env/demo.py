"""
ZebrafishSNN v2 Demo

Runs ZebrafishBrainV2 in the v1 environment with rendering and optional
neural activity monitor (matching v1 UI layout).

Run:
  python -m zebrav2.gym_env.demo --render --steps 500
  python -m zebrav2.gym_env.demo --render --monitor --steps 500
  python -m zebrav2.gym_env.demo --render --monitor --record --steps 500
  python -m zebrav2.gym_env.demo --record --steps 1500
"""
import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.spec import DEVICE
from zebrav2.brain.sensory_bridge import inject_sensory as _inject_sensory

GOAL_NAMES = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]


def run_v2_demo(render=False, monitor=False, record=False, T=1500, ckpt=None, device=None):
    import pygame

    if record:
        render = True

    # CPU is 12x faster than MPS for small per-step tensors; use it when recording
    _device = device or ('cpu' if record else DEVICE)

    # --- Environment ---
    use_panels = render  # side_panels shows retina/goal on env side (like v1)
    env = ZebrafishPreyPredatorEnv(
        render_mode="rgb_array" if render else None,
        n_food=15, max_steps=T, side_panels=use_panels)

    brain = ZebrafishBrainV2(device=_device)

    if ckpt and os.path.exists(ckpt):
        import torch
        state = torch.load(ckpt, map_location=_device, weights_only=False)
        brain.load_state_dict(state.get('brain', state), strict=False)
        print(f"Loaded checkpoint: {ckpt}")

    # --- Window setup (matching v1 layout) ---
    mon = None
    combined_screen = None
    clock = None

    RENDER_W = env.render_width
    RENDER_H = env.render_height

    STATS_H = 100  # bottom stats bar height

    if monitor and render:
        from zebrav2.viz.neural_monitor import NeuralMonitorV2
        mon = NeuralMonitorV2()
        MON_W = mon.WIDTH
        COMBINED_H = max(RENDER_H + STATS_H, mon.HEIGHT)

        pygame.init()
        pygame.font.init()
        combined_screen = pygame.display.set_mode((RENDER_W + MON_W, COMBINED_H))
        pygame.display.set_caption("Zebrafish v2 Brain Agent + Neural Monitor")
        clock = pygame.time.Clock()

    elif render:
        pygame.init()
        pygame.font.init()
        combined_screen = pygame.display.set_mode((RENDER_W, RENDER_H))
        pygame.display.set_caption("Zebrafish v2 Brain Agent")
        clock = pygame.time.Clock()

    elif monitor:
        from zebrav2.viz.neural_monitor import NeuralMonitorV2
        mon = NeuralMonitorV2()
        pygame.init()
        pygame.font.init()
        combined_screen = pygame.display.set_mode((mon.WIDTH, mon.HEIGHT))
        pygame.display.set_caption("Zebrafish v2 Neural Monitor")
        clock = pygame.time.Clock()

    obs, info = env.reset(seed=42)
    brain.reset()

    cumulative_reward = 0.0
    running = True
    mode_str = ""
    if render:
        mode_str += " [RENDER]"
    if monitor:
        mode_str += " [MONITOR]"
    if record:
        mode_str += f" [RECORDING/{_device.upper()}]"
    print(f"Running ZebrafishBrainV2 for {T} steps{mode_str}...")

    # Streaming video writer (avoids buffering all frames in RAM)
    _writer = None
    if record:
        import imageio
        os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
        video_path = os.path.join(PROJECT_ROOT, "plots", "zebrafish_v2_demo.mp4")
        _writer = imageio.get_writer(video_path, fps=30, quality=8, macro_block_size=16)
        print(f"Streaming to {video_path} ...")

    for t in range(T):
        if not running:
            break

        # Signal flee state BEFORE env step
        is_flee = (brain.current_goal == 1)
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(is_flee, panic_intensity=0.8 if is_flee else 0.0)

        # Inject sensory
        _inject_sensory(env)

        # Brain step
        out = brain.step(obs, env)
        action = np.array([out['turn'], out['speed']], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        env._eaten_now = info.get('food_eaten_this_step', 0)
        cumulative_reward += reward

        # --- Rendering ---
        if combined_screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    # Interactive commands (matching v1)
                    elif event.key == pygame.K_p:
                        env.pred_state = 'HUNT'
                    elif event.key == pygame.K_r:
                        env.pred_state = 'PATROL'
                    elif event.key == pygame.K_f:
                        for _ in range(5):
                            import random
                            fx = random.uniform(100, 700)
                            fy = random.uniform(100, 500)
                            env.foods.append([fx, fy, 'small'])

            if running:
                combined_screen.fill((0, 0, 0))

                # Left: env arena (with side panels if render)
                if render:
                    env_frame = env.render()
                    if env_frame is not None:
                        env_surf = pygame.surfarray.make_surface(
                            np.transpose(env_frame, (1, 0, 2)))
                        combined_screen.blit(env_surf, (0, 0))

                # Right: neural monitor
                if mon is not None:
                    mon.update(brain, out, info)
                    mon.render()
                    mon_x = RENDER_W if render else 0
                    combined_screen.blit(mon.get_surface(), (mon_x, 0))

                # Bottom stats bar (below arena, fills gap)
                if render:
                    sy = RENDER_H
                    sw = RENDER_W
                    pygame.draw.rect(combined_screen, (15, 15, 25), (0, sy, sw, STATS_H))
                    pygame.draw.line(combined_screen, (40, 40, 55), (0, sy), (sw, sy))
                    sf = pygame.font.SysFont("monospace", 12, bold=True)
                    sf2 = pygame.font.SysFont("monospace", 11)

                    # Row 1: Goal + Energy + Step
                    goal = brain.current_goal
                    gn = GOAL_NAMES[goal]
                    gc = [(50,200,50),(220,50,50),(100,150,255),(0,180,170)][goal]
                    combined_screen.blit(sf.render(f"GOAL: {gn}", True, gc), (8, sy+4))
                    ef = max(0, min(1, brain.energy/100))
                    ec = (50,200,50) if ef > 0.3 else (220,100,50)
                    pygame.draw.rect(combined_screen, (30,30,40), (120, sy+6, 100, 12))
                    pygame.draw.rect(combined_screen, ec, (120, sy+6, int(100*ef), 12))
                    combined_screen.blit(sf2.render(f"E:{brain.energy:.0f}%", True, (180,180,180)), (225, sy+4))
                    eaten = info.get('total_eaten', 0)
                    combined_screen.blit(sf2.render(f"t={t:4d}  food={eaten}  reward={cumulative_reward:.1f}", True, (180,180,180)), (290, sy+4))

                    # Row 2: Neuromod bars
                    bx = 8
                    by2 = sy + 22
                    bw, bh = 80, 10
                    for i, (lbl, val, col) in enumerate([
                        ("DA", brain.neuromod.DA.item(), (255,200,50)),
                        ("NA", brain.neuromod.NA.item(), (100,200,255)),
                        ("5HT", brain.neuromod.HT5.item(), (200,100,255)),
                        ("ACh", brain.neuromod.ACh.item(), (100,255,150))]):
                        x = bx + i * 140
                        combined_screen.blit(sf2.render(lbl, True, (180,180,180)), (x, by2))
                        pygame.draw.rect(combined_screen, (30,30,40), (x+30, by2+1, bw, bh))
                        pygame.draw.rect(combined_screen, col, (x+30, by2+1, int(bw*val), bh))
                        combined_screen.blit(sf2.render(f"{val:.2f}", True, (150,150,150)), (x+30+bw+3, by2))

                    # Row 3: Allostasis + Predator + Amygdala
                    by3 = sy + 38
                    for i, (lbl, val, col) in enumerate([
                        ("Hunger", brain.allostasis.hunger, (255,140,50)),
                        ("Fatigue", brain.allostasis.fatigue, (100,180,255)),
                        ("Stress", brain.allostasis.stress, (255,80,80)),
                        ("Amy", brain.amygdala_alpha, (220,120,80))]):
                        x = bx + i * 140
                        combined_screen.blit(sf2.render(lbl, True, (180,180,180)), (x, by3))
                        pygame.draw.rect(combined_screen, (30,30,40), (x+50, by3+1, bw-10, bh))
                        pygame.draw.rect(combined_screen, col, (x+50, by3+1, int((bw-10)*min(1,val)), bh))

                    # Row 4: Motor + Predator model
                    by4 = sy + 54
                    combined_screen.blit(sf2.render(
                        f"Turn:{out['turn']:+.2f}  Spd:{out['speed']:.1f}  "
                        f"Pred: d={brain._pred_dist_gt:.0f} int={brain.pred_model.intent:.2f} "
                        f"vis={'Y' if brain.pred_model.visible else 'N'}  "
                        f"BG gate={brain.bg.gate.item():.2f}",
                        True, (160,160,180)), (bx, by4))

                    # Row 5: Keys help
                    by5 = sy + 72
                    combined_screen.blit(sf2.render(
                        "Keys: P=predator attack  R=retreat  F=spawn food  Q=quit",
                        True, (100,100,120)), (bx, by5))

                pygame.display.flip()

                # Capture frame for recording (stream directly to disk)
                if record and _writer is not None:
                    frame = np.transpose(
                        np.array(pygame.surfarray.pixels3d(combined_screen)),
                        (1, 0, 2)).copy()
                    _writer.append_data(frame)
                    if not running:
                        break

                if not record:
                    clock.tick(30)

        if t % 100 == 0:
            g = brain.current_goal
            da = brain.neuromod.DA.item()
            ht5 = brain.neuromod.HT5.item()
            print(f"  t={t:4d}  goal={GOAL_NAMES[g]:7s}  "
                  f"eaten={info['total_eaten']}  "
                  f"DA={da:.2f}  5HT={ht5:.2f}  "
                  f"reward={cumulative_reward:.1f}")

        if terminated or truncated:
            if terminated:
                print(f"  CAUGHT by predator at step {t}!")
            break

    if _writer is not None:
        _writer.close()
        print(f"Video saved: {video_path}")

    if combined_screen is not None:
        env._screen = None
    env.close()
    if combined_screen is not None:
        pygame.quit()

    print(f"\nSummary: survived {t+1} steps, "
          f"ate {info['total_eaten']} food, "
          f"total reward = {cumulative_reward:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true",
                        help="Show pygame window with arena")
    parser.add_argument("--monitor", action="store_true",
                        help="Show neural activity monitor panel")
    parser.add_argument("--record", action="store_true",
                        help="Save MP4 video")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to .pt checkpoint (default: random weights)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu/mps/cuda (default: cpu when recording, mps otherwise)")
    args = parser.parse_args()
    run_v2_demo(render=args.render, monitor=args.monitor,
                record=args.record, T=args.steps, ckpt=args.ckpt, device=args.device)
