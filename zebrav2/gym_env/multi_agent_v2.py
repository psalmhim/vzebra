"""
Multi-agent environment: all fish use full ZebrafishBrainV2.

Each fish has its own v2 brain instance (34 modules, 7000+ spiking neurons).
Fish compete for food, collaborate in predator avoidance, and exhibit
emergent shoaling behavior.

Run: .venv/bin/python -u -m zebrav2.gym_env.multi_agent_v2
"""
import os, sys, math, time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.personality import assign_personalities
from zebrav2.brain.predator_brain import PredatorBrain
from zebrav2.spec import DEVICE


def inject_sensory_for_fish(env, fish_idx, all_fish):
    """Compute retinal arrays for a specific fish (not just fish[0])."""
    fish = all_fish[fish_idx]
    fx, fy, fh = fish['x'], fish['y'], fish['heading']
    arena_w = getattr(env, 'arena_w', 800)
    arena_h = getattr(env, 'arena_h', 600)
    fov_rad = math.radians(200)

    brain_L = np.zeros(800, dtype=np.float32)
    brain_R = np.zeros(800, dtype=np.float32)

    def _project(ox, oy, radius, type_val):
        dx, dy = ox - fx, oy - fy
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6
        if dist > 500:
            return
        angle_to = math.atan2(dy, dx)
        rel = math.atan2(math.sin(angle_to - fh), math.cos(angle_to - fh))
        if abs(rel) > fov_rad / 2:
            return
        ang_hw = math.atan(radius / dist)
        col_c = int((rel + fov_rad / 2) / fov_rad * 400)
        col_hw = max(1, int(ang_hw / fov_rad * 400))
        intensity = min(1.0, (radius / dist) * 15)
        for c in range(max(0, col_c - col_hw), min(400, col_c + col_hw)):
            if rel <= 0:
                brain_L[c] = max(brain_L[c], intensity)
                brain_L[400 + c] = type_val
            else:
                brain_R[c] = max(brain_R[c], intensity)
                brain_R[400 + c] = type_val

    # Project predator
    _project(getattr(env, 'pred_x', -9999), getattr(env, 'pred_y', -9999), 20.0, 0.5)

    # Project food
    for food in getattr(env, 'foods', []):
        _project(food[0], food[1], 12.0, 0.8)

    # Project rocks
    for rock in getattr(env, 'rock_formations', []):
        _project(rock['cx'], rock['cy'], rock.get('radius', 30), 0.3)

    # Project other fish as conspecifics (type=0.25)
    for j, other in enumerate(all_fish):
        if j == fish_idx or not other.get('alive', True):
            continue
        _project(other['x'], other['y'], 8.0, 0.25)

    enemy_px = (int(np.sum(np.abs(brain_L[400:] - 0.5) < 0.1))
                + int(np.sum(np.abs(brain_R[400:] - 0.5) < 0.1)))

    # Set on env temporarily for brain.step()
    env.brain_L = brain_L
    env.brain_R = brain_R
    env._enemy_pixels_total = enemy_px
    env.fish_x = fx
    env.fish_y = fy
    env.fish_heading = fh


def run_multi_v2(n_fish=5, max_steps=500, seed=42, record=False, personality_mode='mixed'):
    env = ZebrafishPreyPredatorEnv(render_mode='rgb_array' if record else None,
                                    n_food=20, max_steps=max_steps, side_panels=record)
    obs, info = env.reset(seed=seed)

    # Assign personalities
    personalities = assign_personalities(n_fish, mode=personality_mode)
    types = ['bold', 'shy', 'explorer', 'social', 'default']

    # Initialize all fish
    all_fish = []
    brains = []
    np.random.seed(seed)
    for i in range(n_fish):
        x = np.random.uniform(150, 650)
        y = np.random.uniform(150, 450)
        h = np.random.uniform(-math.pi, math.pi)
        ptype = types[i % len(types)] if personality_mode == 'mixed' else 'default'
        all_fish.append({'x': x, 'y': y, 'heading': h, 'speed': 0.5,
                         'alive': True, 'food_eaten': 0, 'energy': 100.0,
                         'personality': ptype})
        brain = ZebrafishBrainV2(device=DEVICE, personality=personalities[i])
        brain.reset()
        brains.append(brain)

    print(f"  Personalities: {[f['personality'] for f in all_fish]}")

    # Intelligent predator brain
    pred_brain = PredatorBrain(arena_w=env.arena_w, arena_h=env.arena_h)

    # Fish 0 = env's focal fish
    env.fish_x, env.fish_y, env.fish_heading = all_fish[0]['x'], all_fish[0]['y'], all_fish[0]['heading']

    frames = [] if record else None
    total_food = [0] * n_fish
    import torch
    torch_device = DEVICE
    t0 = time.time()

    for t in range(max_steps):
        for i in range(n_fish):
            if not all_fish[i]['alive']:
                continue

            # Set env state to this fish's position
            inject_sensory_for_fish(env, i, all_fish)
            env.fish_energy = all_fish[i]['energy']

            # Brain step
            if hasattr(env, 'set_flee_active'):
                env.set_flee_active(brains[i].current_goal == 1,
                                    0.8 if brains[i].current_goal == 1 else 0.0)
            out = brains[i].step(obs, env)

            # Update fish position
            turn = out['turn']
            speed = out['speed']
            fish = all_fish[i]
            turn_max = 0.15
            if brains[i].current_goal == 1:
                turn_max = 0.45
            fish['heading'] += turn * turn_max
            fish['heading'] = math.atan2(math.sin(fish['heading']), math.cos(fish['heading']))
            fish_speed_base = 3.0
            fish['x'] += fish_speed_base * speed * math.cos(fish['heading'])
            fish['y'] += fish_speed_base * speed * math.sin(fish['heading'])
            # Wall clamp
            fish['x'] = max(10, min(env.arena_w - 10, fish['x']))
            fish['y'] = max(10, min(env.arena_h - 10, fish['y']))
            fish['speed'] = speed

            # Food eating
            eaten_this = 0
            remaining_foods = []
            for food in env.foods:
                dx, dy = food[0] - fish['x'], food[1] - fish['y']
                if math.sqrt(dx*dx + dy*dy) < 35:
                    eaten_this += 1
                    total_food[i] += 1
                    fish['energy'] = min(100, fish['energy'] + 8)
                else:
                    remaining_foods.append(food)
            env.foods = remaining_foods
            env._eaten_now = eaten_this
            fish['food_eaten'] = total_food[i]

            # Energy decay
            actual_speed = min(2.0, speed)
            speed_cost = 0.03 * actual_speed * (0.5 + 0.5 * actual_speed)
            fish['energy'] -= (0.05 + speed_cost)
            if fish['energy'] <= 0:
                fish['alive'] = False

            # Predator catch check
            pred_dx = getattr(env, 'pred_x', -999) - fish['x']
            pred_dy = getattr(env, 'pred_y', -999) - fish['y']
            if math.sqrt(pred_dx**2 + pred_dy**2) < 20:
                fish['alive'] = False
                pred_brain.on_catch()

        # Update env focal fish to fish[0] for predator AI
        if all_fish[0]['alive']:
            env.fish_x, env.fish_y = all_fish[0]['x'], all_fish[0]['y']
            env.fish_heading = all_fish[0]['heading']

        # Intelligent predator AI
        fish_info = [{'x': f['x'], 'y': f['y'], 'energy': f['energy'],
                       'alive': f['alive'], 'speed': f['speed'],
                       'heading': f.get('heading', 0)}
                      for f in all_fish]
        pdx, pdy, pspeed, pstate = pred_brain.step(env.pred_x, env.pred_y, fish_info)
        env.pred_x += pdx
        env.pred_y += pdy
        env.pred_x = max(10, min(env.arena_w - 10, env.pred_x))
        env.pred_y = max(10, min(env.arena_h - 10, env.pred_y))
        env.pred_state = pstate

        # Record frame
        if record and hasattr(env, 'render'):
            # Sync focal fish position for rendering
            if all_fish[0]['alive']:
                env.fish_x, env.fish_y = all_fish[0]['x'], all_fish[0]['y']
                env.fish_heading = all_fish[0]['heading']
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Respawn food
        if len(env.foods) < 5:
            for _ in range(3):
                env.foods.append([np.random.uniform(50, 750),
                                  np.random.uniform(50, 550), 'small'])

        n_alive = sum(1 for f in all_fish if f['alive'])
        if t % 50 == 0:
            elapsed = time.time() - t0
            fps = (t + 1) / elapsed if elapsed > 0 else 0
            foods_str = '/'.join(str(total_food[i]) for i in range(n_fish))
            goals_str = '/'.join(['F','L','E','S'][brains[i].current_goal] for i in range(n_fish) if all_fish[i]['alive'])
            print(f"  t={t:4d}  alive={n_alive}/{n_fish}  food=[{foods_str}]  "
                  f"goals=[{goals_str}]  pred={pstate:6s} stam={pred_brain.stamina:.1f}  "
                  f"{fps:.2f} steps/s")

        if n_alive == 0:
            break

    elapsed = time.time() - t0
    print(f"\nSummary: {t+1} steps in {elapsed:.0f}s ({(t+1)/elapsed:.2f} steps/s)")
    print(f"  Fish alive: {sum(1 for f in all_fish if f['alive'])}/{n_fish}")
    for i in range(n_fish):
        status = 'ALIVE' if all_fish[i]['alive'] else 'DEAD'
        ptype = all_fish[i].get('personality', 'default')
        print(f"  Fish {i} ({ptype:8s}): {status}, food={total_food[i]}, energy={all_fish[i]['energy']:.0f}")

    env.close()

    if record and frames and len(frames) > 5:
        import imageio
        outpath = os.path.join(PROJECT_ROOT, 'plots', 'zebrafish_v2_multi.mp4')
        print(f"\n  Saving {len(frames)} frames to {outpath}...")
        imageio.mimwrite(outpath, frames, fps=20, quality=7)
        print(f"  Video saved: {outpath}")

    for b in brains:
        del b
    return all_fish, total_food


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-fish', type=int, default=5)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--personality', type=str, default='mixed',
                        choices=['mixed', 'random', 'uniform'])
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    print(f"Multi-agent v2: {args.n_fish} fish × full v2 brain, {args.steps} steps")
    print(f"Device: {DEVICE}, Personality: {args.personality}")
    run_multi_v2(n_fish=args.n_fish, max_steps=args.steps, seed=args.seed,
                 personality_mode=args.personality, record=args.record)
