"""
Training engine: runs episodes, saves checkpoints, publishes metrics.

Decoupled from UI — communicates via callback functions.
Web dashboard or CLI can observe training progress.
"""
import os
import sys
import time
import math
import json
import numpy as np
import torch
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav2.brain.personality import get_personality, assign_personalities
from zebrav2.brain.predator_brain import PredatorBrain
from zebrav2.gym_env.multi_agent_v2 import inject_sensory_for_fish
from zebrav2.engine.config import TrainingConfig
from zebrav2.engine.checkpoint import CheckpointManager
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

GOAL_NAMES = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']


class TrainingEngine:
    """Headless training engine. UI-independent."""

    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.checkpoint_mgr = CheckpointManager(
            os.path.join(PROJECT_ROOT, 'zebrav2', 'checkpoints'))
        self.brain = None
        self.env = None
        self.pred_brain = None

        # Metrics history (per round)
        self.round_history = []
        # Current step metrics (for real-time streaming)
        self.current_step_data = {}
        self.current_round = 0
        self.total_rounds_done = 0
        self.running = False
        self._thread = None

        # Callbacks for UI
        self.on_step = None       # fn(step_data) called each step
        self.on_round_end = None  # fn(round_metrics) called at round end
        self.on_training_done = None  # fn(history)

        # Replay recording
        self._recording = []
        self.saved_replays = []   # list of {round, metrics, steps}
        self.max_replays = 5

    def _create_brain(self):
        personality_name = self.config.get('fish.personality_mode', 'default')
        personality = get_personality(personality_name)
        brain = ZebrafishBrainV2(device=DEVICE, personality=personality)

        # Load checkpoint if specified
        ckpt_path = self.config.get('training.load_checkpoint')
        if ckpt_path and os.path.exists(ckpt_path):
            rnd, metrics = self.checkpoint_mgr.load(brain, ckpt_path)
            self.total_rounds_done = rnd
            print(f"  Loaded checkpoint: round {rnd}, metrics={metrics}")
        else:
            brain.reset()
        return brain

    def _create_env(self):
        cfg = self.config
        env = ZebrafishPreyPredatorEnv(
            render_mode=None,
            n_food=0,  # we spawn food manually in patches
            max_steps=cfg.get('env.max_steps', 500))
        return env

    def _spawn_food_patches(self, env, n_total=20, seed=None):
        """Spawn 3 food-dense areas surrounded by rocks + sparse scatter."""
        rng = np.random.RandomState(seed)
        aw = getattr(env, 'arena_w', 800)
        ah = getattr(env, 'arena_h', 600)
        env.foods = []

        # Clear existing rocks and create new ones
        env.rock_formations = []

        # 3 fixed food-dense areas in different regions
        patch_centers = [
            (aw * 0.2, ah * 0.25),   # top-left
            (aw * 0.75, ah * 0.3),   # top-right
            (aw * 0.4, ah * 0.75),   # bottom-center
        ]

        food_per_patch = max(3, n_total // 4)  # ~75% in patches

        for pi, (pcx, pcy) in enumerate(patch_centers):
            # Jitter patch center
            pcx += rng.uniform(-30, 30)
            pcy += rng.uniform(-30, 30)
            pcx = float(np.clip(pcx, 80, aw - 80))
            pcy = float(np.clip(pcy, 80, ah - 80))

            # Spawn food cluster inside the patch
            for _ in range(food_per_patch):
                fx = pcx + rng.normal(0, 25)
                fy = pcy + rng.normal(0, 25)
                fx = float(np.clip(fx, 30, aw - 30))
                fy = float(np.clip(fy, 30, ah - 30))
                env.foods.append([fx, fy, 'small'])

            # Surround each patch with rocks forming a wall with 1-2 gaps
            n_rocks = 8  # wall segments
            gap_idx = rng.randint(0, n_rocks)  # one gap for fish to enter
            gap_idx2 = (gap_idx + n_rocks // 2) % n_rocks  # second gap opposite
            for ri in range(n_rocks):
                if ri == gap_idx or ri == gap_idx2:
                    continue  # gap — no rock here
                angle = 2 * math.pi * ri / n_rocks + rng.uniform(-0.1, 0.1)
                dist = 60 + rng.uniform(0, 10)
                rx = pcx + dist * math.cos(angle)
                ry = pcy + dist * math.sin(angle)
                rx = float(np.clip(rx, 40, aw - 40))
                ry = float(np.clip(ry, 40, ah - 40))
                base_r = float(rng.uniform(15, 35))
                # Generate polygon vertices (irregular shape)
                n_verts = rng.randint(5, 9)
                vertices = []
                for vi in range(n_verts):
                    va = 2 * math.pi * vi / n_verts + rng.uniform(-0.3, 0.3)
                    vr = base_r * (0.6 + rng.uniform(0, 0.8))
                    vertices.append([float(rx + vr * math.cos(va)),
                                     float(ry + vr * math.sin(va))])
                env.rock_formations.append({
                    'cx': rx, 'cy': ry, 'radius': base_r,
                    'base_r': base_r,
                    'vertices': vertices, 'type': 'polygon',
                    'aabbs': [{'x': rx, 'y': ry,
                               'hw': base_r, 'hh': base_r}],
                    'lobes': [{'cx': rx, 'cy': ry, 'r': base_r}],
                })

        # Add scattered rocks between patches (obstacles)
        for _ in range(rng.randint(5, 10)):
            rx = float(rng.uniform(60, aw - 60))
            ry = float(rng.uniform(60, ah - 60))
            # Avoid being too close to patches
            too_close = False
            for pcx, pcy in patch_centers:
                if math.sqrt((rx-pcx)**2 + (ry-pcy)**2) < 80:
                    too_close = True
                    break
            if too_close:
                continue
            base_r = float(rng.uniform(20, 45))
            n_verts = rng.randint(4, 8)
            vertices = []
            for vi in range(n_verts):
                va = 2 * math.pi * vi / n_verts + rng.uniform(-0.4, 0.4)
                vr = base_r * (0.5 + rng.uniform(0, 0.9))
                vertices.append([float(rx + vr * math.cos(va)),
                                 float(ry + vr * math.sin(va))])
            env.rock_formations.append({
                'cx': rx, 'cy': ry, 'radius': base_r,
                'base_r': base_r,
                'vertices': vertices, 'type': 'polygon',
                'aabbs': [{'x': rx, 'y': ry,
                           'hw': base_r, 'hh': base_r}],
                'lobes': [{'cx': rx, 'cy': ry, 'r': base_r}],
            })

        # Sparse food outside patches (25%)
        n_scatter = max(0, n_total - food_per_patch * 3)
        for _ in range(n_scatter):
            fx = float(rng.uniform(50, aw - 50))
            fy = float(rng.uniform(50, ah - 50))
            env.foods.append([fx, fy, 'small'])

    def _compute_fitness(self, survived, food_eaten, mean_efe, energy_final,
                          geo_coverage):
        """Compute fitness score from objectives config."""
        obj = self.config.data.get('objectives', {})
        fitness = (
            obj.get('survival_weight', 1.5) * survived +           # raised from 1.0
            obj.get('food_weight', 50.0) * food_eaten +
            obj.get('efe_weight', -5.0) * mean_efe +               # softer: was -10.0
            obj.get('energy_weight', 0.5) * energy_final +
            obj.get('exploration_weight', 5.0) * min(5.0, geo_coverage * 100)  # capped
        )
        return float(fitness)

    def run_round(self, round_num):
        """Run one episode/round. Returns metrics dict."""
        self.current_round = round_num
        cfg = self.config

        seed = cfg.get('env.seed')
        if seed is None:
            seed = round_num * 7 + 1

        env = self._create_env()
        obs, info = env.reset(seed=seed)
        # Spawn food in patches (clustered)
        n_food = cfg.get('env.n_food', 20)
        self._spawn_food_patches(env, n_total=n_food, seed=seed)
        self.brain.reset()
        # Don't reset learned weights — keep from previous rounds
        self.brain._apply_personality()

        # Predator — always spawned; aggression depends on predator_ai setting
        predator_ai = cfg.get('env.predator_ai', 'none')
        pred_brain = PredatorBrain()
        if predator_ai == 'none':
            # Passive patrol: no hunting, just visible in arena
            pred_brain.hunt_speed = 0.0
            pred_brain.stalk_dist = 9999
            pred_brain.detect_dist = 0
            pred_brain.state = 'PATROL'
        elif predator_ai == 'simple':
            pred_brain.detect_dist = 200
            pred_brain.hunt_speed = 2.5

        total_eaten = 0
        total_efe = 0.0
        goals_log = []
        step_data_list = []

        for t in range(cfg.get('env.max_steps', 500)):
            if not self.running:
                break

            if hasattr(env, 'set_flee_active'):
                env.set_flee_active(self.brain.current_goal == 1,
                                    0.8 if self.brain.current_goal == 1 else 0.0)
            inject_sensory(env)
            out = self.brain.step(obs, env)
            action = np.array([out['turn'], out['speed']], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            env._eaten_now = info.get('food_eaten_this_step', 0)
            total_eaten += env._eaten_now

            # Predator AI: always step (predator always visible in arena)
            fish_info = [{'x': float(getattr(env, 'fish_x', 400)),
                          'y': float(getattr(env, 'fish_y', 300)),
                          'energy': float(self.brain.energy),
                          'alive': True,
                          'speed': float(out['speed']),
                          'heading': float(getattr(env, 'fish_heading', 0))}]
            pdx, pdy, pspeed, pstate = pred_brain.step(
                getattr(env, 'pred_x', 50),
                getattr(env, 'pred_y', 50),
                fish_info)
            if hasattr(env, 'pred_x'):
                env.pred_x = max(10, min(getattr(env, 'arena_w', 800) - 10,
                                         env.pred_x + pdx))
                env.pred_y = max(10, min(getattr(env, 'arena_h', 600) - 10,
                                         env.pred_y + pdy))
                # Catch only when predator is active (not passive patrol)
                if predator_ai == 'intelligent':
                    catch_dx = env.pred_x - getattr(env, 'fish_x', 0)
                    catch_dy = env.pred_y - getattr(env, 'fish_y', 0)
                    if math.sqrt(catch_dx**2 + catch_dy**2) < 20:
                        pred_brain.on_catch()

            # Rock collision: push fish out of rocks
            fx, fy = getattr(env, 'fish_x', 400), getattr(env, 'fish_y', 300)
            for rock in getattr(env, 'rock_formations', []):
                rcx, rcy, rr = rock['cx'], rock['cy'], rock['radius']
                dx, dy = fx - rcx, fy - rcy
                dist = math.sqrt(dx*dx + dy*dy) + 1e-8
                if dist < rr + 8:  # fish radius ~8
                    # Push fish out
                    push = (rr + 10 - dist)
                    env.fish_x = fx + push * dx / dist
                    env.fish_y = fy + push * dy / dist

            efe = out.get('free_energy', 0)
            total_efe += efe
            goals_log.append(self.brain.current_goal)

            # Step data for real-time streaming
            # Collect food positions
            food_positions = []
            for food in getattr(env, 'foods', []):
                food_positions.append([float(food[0]), float(food[1])])
            # Collect rock positions (with polygon vertices if available)
            rock_positions = []
            for rock in getattr(env, 'rock_formations', []):
                r = [float(rock.get('cx', 0)), float(rock.get('cy', 0)),
                     float(rock.get('radius', 30))]
                if 'vertices' in rock:
                    r.append(rock['vertices'])  # polygon vertices
                rock_positions.append(r)

            step_data = {
                'round': round_num,
                'step': t,
                'goal': GOAL_NAMES[self.brain.current_goal],
                'turn': float(out['turn']),
                'speed': float(out['speed']),
                'energy': float(self.brain.energy),
                'food_total': total_eaten,
                'free_energy': float(efe),
                'DA': float(out['DA']),
                'NA': float(out['NA']),
                '5HT': float(out['5HT']),
                'ACh': float(out['ACh']),
                'amygdala': float(out['amygdala']),
                'critic_value': float(out.get('critic_value', 0)),
                'surprise': float(out.get('predictive_surprise', 0)),
                'novelty': float(out.get('novelty', 0)),
                'heart_rate': float(out.get('insula_heart_rate', 2)),
                'valence': float(out.get('insula_valence', 0)),
                'vae_nodes': int(out.get('vae_memory_nodes', 0)),
                'fish_x': float(getattr(env, 'fish_x', 400)),
                'fish_y': float(getattr(env, 'fish_y', 300)),
                'fish_heading': float(getattr(env, 'fish_heading', 0)),
                'pred_x': float(getattr(env, 'pred_x', -100)),
                'pred_y': float(getattr(env, 'pred_y', -100)),
                'pred_heading': float(getattr(env, 'pred_heading', 0)),
                'pred_energy': float(pred_brain.energy) if pred_brain else 0,
                'pred_stamina': float(pred_brain.stamina) if pred_brain else 0,
                'pred_state': pred_brain.state if pred_brain else '',
                'pred_heatmap': [[float(pred_brain.place.cx[j]),
                                  float(pred_brain.place.cy[j]),
                                  float(pred_brain.place.prey_density[j])]
                                 for j in range(pred_brain.place.n_cells)] if pred_brain else [],
                'foods': food_positions,
                'rocks': rock_positions,
                'arena_w': int(getattr(env, 'arena_w', 800)),
                'arena_h': int(getattr(env, 'arena_h', 600)),
                # Retinal view (subsampled for dashboard — 40 pixels per eye)
                'retina_L': [float(x) for x in env.brain_L[:400:10]] if hasattr(env, 'brain_L') else [],
                'retina_R': [float(x) for x in env.brain_R[:400:10]] if hasattr(env, 'brain_R') else [],
                'retina_L_type': [float(x) for x in env.brain_L[400::10]] if hasattr(env, 'brain_L') else [],
                'retina_R_type': [float(x) for x in env.brain_R[400::10]] if hasattr(env, 'brain_R') else [],
                # Regional spike counts (for raster plots)
                'spikes': {
                    'sfgs_b': float(self.brain.tectum.sfgs_b.spike_E.sum()),
                    'sfgs_d': float(self.brain.tectum.sfgs_d.spike_E.sum()),
                    'sgc': float(self.brain.tectum.sgc.spike_E.sum()),
                    'so': float(self.brain.tectum.so.spike_E.sum()),
                    'tc': float(self.brain.thalamus.TC.rate.sum()),
                    'trn': float(self.brain.thalamus.TRN.rate.sum()),
                    'pal_s': float(self.brain.pallium.pal_s.spike_E.sum()),
                    'pal_d': float(self.brain.pallium.pal_d.spike_E.sum()),
                    'amygdala': float(self.brain.amygdala.CeA.rate.sum()),
                    'cerebellum': float(self.brain.cerebellum.gc_rate.sum()),
                    'habenula': float(self.brain.habenula.lhb_rate.sum()),
                    'd1': float(self.brain.bg.d1_rate.sum()),
                    'd2': float(self.brain.bg.d2_rate.sum()),
                    'critic': float(self.brain.critic.hidden_rate.sum()),
                    'insula': float(self.brain.insula.hunger_rate.sum() + self.brain.insula.stress_rate.sum()),
                },
            }
            self.current_step_data = step_data
            if self.on_step:
                self.on_step(step_data)
            # Record for replay (skip large arrays to save memory)
            self._recording.append({k: v for k, v in step_data.items()
                                     if k not in ('spikes', 'retina_L', 'retina_R',
                                                  'retina_L_type', 'retina_R_type')})

            # Food respawn: when food runs low, add a new small patch
            if cfg.get('env.food_respawn', True) and len(getattr(env, 'foods', [])) < cfg.get('env.food_respawn_min', 5):
                aw = getattr(env, 'arena_w', 800)
                ah = getattr(env, 'arena_h', 600)
                cx = float(np.random.uniform(100, aw - 100))
                cy = float(np.random.uniform(100, ah - 100))
                for _ in range(np.random.randint(3, 6)):
                    fx = float(np.clip(cx + np.random.normal(0, 30), 30, aw - 30))
                    fy = float(np.clip(cy + np.random.normal(0, 30), 30, ah - 30))
                    env.foods.append([fx, fy, 'small'])

            if terminated or truncated:
                break

        env.close()
        survived = t + 1
        mean_efe = total_efe / max(1, survived)
        geo_coverage = float(np.mean(self.brain.geo_model.visit_count > 0))

        from collections import Counter
        gc = Counter(goals_log)

        metrics = {
            'round': round_num,
            'survived': survived,
            'food_eaten': total_eaten,
            'mean_efe': float(mean_efe),
            'energy_final': float(self.brain.energy),
            'caught': terminated and not truncated,
            'geo_coverage': geo_coverage,
            'goal_distribution': {GOAL_NAMES[g]: gc.get(g, 0) for g in range(4)},
            'fitness': self._compute_fitness(survived, total_eaten, mean_efe,
                                              self.brain.energy, geo_coverage),
            'vae_memory_nodes': self.brain.vae.memory.n_allocated,
            'critic_mean_value': float(self.brain.critic.values.mean()),
            'personality': self.brain.personality.get('description', 'default'),
        }
        # Save replay
        self.saved_replays.append({
            'round': round_num, 'metrics': metrics,
            'steps': self._recording,
        })
        if len(self.saved_replays) > self.max_replays:
            self.saved_replays.pop(0)
        self._recording = []
        return metrics

    def run_round_multi(self, round_num, brains, n_fish=5):
        """Run one multi-agent episode. Returns metrics dict.

        All n_fish brains are stepped each timestep, with sensory injection
        per-fish via inject_sensory_for_fish. step_data includes an
        'other_fish' array so the dashboard can render all fish.
        """
        self.current_round = round_num
        cfg = self.config
        seed = cfg.get('env.seed')
        if seed is None:
            seed = round_num * 7 + 1

        env = self._create_env()
        obs, info = env.reset(seed=seed)
        n_food = cfg.get('env.n_food', 20)
        self._spawn_food_patches(env, n_total=n_food, seed=seed)

        # Personality type labels for logging
        ptypes = ['bold', 'shy', 'explorer', 'social', 'default']

        # Initialize fish states
        rng = np.random.RandomState(seed)
        all_fish = []
        for i in range(n_fish):
            brains[i].reset()
            brains[i]._apply_personality()
            all_fish.append({
                'x': float(rng.uniform(100, 700)),
                'y': float(rng.uniform(100, 500)),
                'heading': float(rng.uniform(0, 2 * math.pi)),
                'energy': 100.0,
                'alive': True,
                'speed': 0.0,
                'goal': 'EXPLORE',
                'food_eaten': 0,
                'personality': ptypes[i % len(ptypes)],
            })

        # Predator — always spawned; aggression depends on predator_ai setting
        predator_ai = cfg.get('env.predator_ai', 'none')
        pred_brain = PredatorBrain()
        if predator_ai == 'none':
            pred_brain.hunt_speed = 0.0
            pred_brain.stalk_dist = 9999
            pred_brain.detect_dist = 0
            pred_brain.state = 'PATROL'
        elif predator_ai == 'simple':
            pred_brain.detect_dist = 200
            pred_brain.hunt_speed = 2.5

        total_eaten = [0] * n_fish
        total_efe = 0.0
        goals_log = []
        focal_out = {}  # captured from fish[0]'s brain.step()

        for t in range(cfg.get('env.max_steps', 500)):
            if not self.running:
                break

            # Step each fish brain
            for i in range(n_fish):
                fish = all_fish[i]
                if not fish['alive']:
                    continue

                # Inject per-fish sensory
                inject_sensory_for_fish(env, i, all_fish)
                env.fish_energy = fish['energy']

                if hasattr(env, 'set_flee_active'):
                    env.set_flee_active(
                        brains[i].current_goal == 1,
                        0.8 if brains[i].current_goal == 1 else 0.0)

                out = brains[i].step(obs, env)

                # Capture focal fish output for step_data
                if i == 0:
                    focal_out = out

                # Update position
                turn = out['turn']
                speed = out['speed']
                turn_max = 0.45 if brains[i].current_goal == 1 else 0.15
                fish['heading'] += turn * turn_max
                fish['heading'] = math.atan2(
                    math.sin(fish['heading']), math.cos(fish['heading']))
                fish_speed_base = 3.0
                fish['x'] += fish_speed_base * speed * math.cos(fish['heading'])
                fish['y'] += fish_speed_base * speed * math.sin(fish['heading'])
                fish['x'] = max(10, min(getattr(env, 'arena_w', 800) - 10, fish['x']))
                fish['y'] = max(10, min(getattr(env, 'arena_h', 600) - 10, fish['y']))
                fish['speed'] = speed
                fish['goal'] = GOAL_NAMES[brains[i].current_goal]

                # Food eating
                eaten_this = 0
                remaining_foods = []
                for food in getattr(env, 'foods', []):
                    dx, dy = food[0] - fish['x'], food[1] - fish['y']
                    if math.sqrt(dx * dx + dy * dy) < 35:
                        eaten_this += 1
                        total_eaten[i] += 1
                        fish['energy'] = min(100, fish['energy'] + 8)
                    else:
                        remaining_foods.append(food)
                env.foods = remaining_foods
                env._eaten_now = eaten_this
                fish['food_eaten'] = total_eaten[i]

                # Energy decay
                actual_speed = min(2.0, speed)
                speed_cost = 0.03 * actual_speed * (0.5 + 0.5 * actual_speed)
                fish['energy'] -= (0.05 + speed_cost)
                if fish['energy'] <= 0:
                    fish['alive'] = False

                # Rock collision
                for rock in getattr(env, 'rock_formations', []):
                    rcx, rcy, rr = rock['cx'], rock['cy'], rock['radius']
                    dx, dy = fish['x'] - rcx, fish['y'] - rcy
                    dist = math.sqrt(dx * dx + dy * dy) + 1e-8
                    if dist < rr + 8:
                        push = (rr + 10 - dist)
                        fish['x'] += push * dx / dist
                        fish['y'] += push * dy / dist

            # Update env focal fish to fish[0] for predator
            if all_fish[0]['alive']:
                env.fish_x = all_fish[0]['x']
                env.fish_y = all_fish[0]['y']
                env.fish_heading = all_fish[0]['heading']

            # Predator AI — always step (always visible in arena)
            fish_info = [{'x': f['x'], 'y': f['y'],
                          'energy': f['energy'], 'alive': f['alive'],
                          'speed': f['speed'],
                          'heading': f.get('heading', 0)}
                         for f in all_fish]
            pdx, pdy, pspeed, pstate = pred_brain.step(
                getattr(env, 'pred_x', 50),
                getattr(env, 'pred_y', 50), fish_info)
            if hasattr(env, 'pred_x'):
                env.pred_x = max(10, min(
                    getattr(env, 'arena_w', 800) - 10, env.pred_x + pdx))
                env.pred_y = max(10, min(
                    getattr(env, 'arena_h', 600) - 10, env.pred_y + pdy))

            # Predator catch check (only when actively hunting)
            if predator_ai == 'intelligent':
                for i in range(n_fish):
                    if not all_fish[i]['alive']:
                        continue
                    cdx = env.pred_x - all_fish[i]['x']
                    cdy = env.pred_y - all_fish[i]['y']
                    if math.sqrt(cdx ** 2 + cdy ** 2) < 20:
                        all_fish[i]['alive'] = False
                        pred_brain.on_catch()

            # Focal fish data for step_data
            focal = all_fish[0]
            focal_brain = brains[0]

            efe = focal_out.get('free_energy', 0)
            total_efe += efe
            goals_log.append(focal_brain.current_goal)

            # Collect food/rock positions
            food_positions = [[float(f[0]), float(f[1])]
                              for f in getattr(env, 'foods', [])]
            rock_positions = []
            for rock in getattr(env, 'rock_formations', []):
                r = [float(rock.get('cx', 0)), float(rock.get('cy', 0)),
                     float(rock.get('radius', 30))]
                if 'vertices' in rock:
                    r.append(rock['vertices'])
                rock_positions.append(r)

            # Build other_fish array (fish 1..n-1)
            other_fish_data = []
            for i in range(1, n_fish):
                f = all_fish[i]
                other_fish_data.append({
                    'x': float(f['x']),
                    'y': float(f['y']),
                    'heading': float(f['heading']),
                    'goal': f['goal'],
                    'energy': float(f['energy']),
                    'alive': f['alive'],
                    'personality': f.get('personality', 'default'),
                    'food_eaten': f['food_eaten'],
                })

            step_data = {
                'round': round_num,
                'step': t,
                'mode': 'multi-agent',
                'goal': GOAL_NAMES[focal_brain.current_goal],
                'turn': float(focal_out.get('turn', 0)),
                'speed': float(focal_out.get('speed', 0)),
                'energy': float(focal_brain.energy),
                'food_total': total_eaten[0],
                'free_energy': float(efe),
                'DA': float(focal_out.get('DA', 0)),
                'NA': float(focal_out.get('NA', 0)),
                '5HT': float(focal_out.get('5HT', 0)),
                'ACh': float(focal_out.get('ACh', 0)),
                'amygdala': float(focal_out.get('amygdala', 0)),
                'critic_value': float(focal_out.get('critic_value', 0)),
                'surprise': float(focal_out.get('predictive_surprise', 0)),
                'novelty': float(focal_out.get('novelty', 0)),
                'heart_rate': float(focal_out.get('insula_heart_rate', 2)),
                'valence': float(focal_out.get('insula_valence', 0)),
                'vae_nodes': int(focal_out.get('vae_memory_nodes', 0)),
                'fish_x': float(focal['x']),
                'fish_y': float(focal['y']),
                'fish_heading': float(focal['heading']),
                'pred_x': float(getattr(env, 'pred_x', -100)),
                'pred_y': float(getattr(env, 'pred_y', -100)),
                'pred_heading': float(getattr(env, 'pred_heading', 0)),
                'pred_energy': float(pred_brain.energy) if pred_brain else 0,
                'pred_stamina': float(pred_brain.stamina) if pred_brain else 0,
                'pred_state': pred_brain.state if pred_brain else '',
                'pred_heatmap': (
                    [[float(pred_brain.place.centroids[j, 0]),
                      float(pred_brain.place.centroids[j, 1]),
                      float(pred_brain.place.prey_density[j])]
                     for j in range(pred_brain.place.n_cells)]
                    if pred_brain else []),
                'foods': food_positions,
                'rocks': rock_positions,
                'arena_w': int(getattr(env, 'arena_w', 800)),
                'arena_h': int(getattr(env, 'arena_h', 600)),
                'retina_L': ([float(x) for x in env.brain_L[:400:10]]
                             if hasattr(env, 'brain_L') else []),
                'retina_R': ([float(x) for x in env.brain_R[:400:10]]
                             if hasattr(env, 'brain_R') else []),
                'retina_L_type': ([float(x) for x in env.brain_L[400::10]]
                                  if hasattr(env, 'brain_L') else []),
                'retina_R_type': ([float(x) for x in env.brain_R[400::10]]
                                  if hasattr(env, 'brain_R') else []),
                'spikes': {
                    'sfgs_b': float(focal_brain.tectum.sfgs_b.spike_E.sum()),
                    'sfgs_d': float(focal_brain.tectum.sfgs_d.spike_E.sum()),
                    'sgc': float(focal_brain.tectum.sgc.spike_E.sum()),
                    'so': float(focal_brain.tectum.so.spike_E.sum()),
                    'tc': float(focal_brain.thalamus.TC.rate.sum()),
                    'trn': float(focal_brain.thalamus.TRN.rate.sum()),
                    'pal_s': float(focal_brain.pallium.pal_s.spike_E.sum()),
                    'pal_d': float(focal_brain.pallium.pal_d.spike_E.sum()),
                    'amygdala': float(focal_brain.amygdala.CeA.rate.sum()),
                    'cerebellum': float(focal_brain.cerebellum.gc_rate.sum()),
                    'habenula': float(focal_brain.habenula.lhb_rate.sum()),
                    'd1': float(focal_brain.bg.d1_rate.sum()),
                    'd2': float(focal_brain.bg.d2_rate.sum()),
                    'critic': float(focal_brain.critic.hidden_rate.sum()),
                    'insula': float(
                        focal_brain.insula.hunger_rate.sum()
                        + focal_brain.insula.stress_rate.sum()),
                },
                'other_fish': other_fish_data,
            }
            self.current_step_data = step_data
            if self.on_step:
                self.on_step(step_data)

            # Food respawn
            if (cfg.get('env.food_respawn', True)
                    and len(getattr(env, 'foods', []))
                    < cfg.get('env.food_respawn_min', 5)):
                aw = getattr(env, 'arena_w', 800)
                ah = getattr(env, 'arena_h', 600)
                cx = float(np.random.uniform(100, aw - 100))
                cy = float(np.random.uniform(100, ah - 100))
                for _ in range(np.random.randint(3, 6)):
                    fx = float(np.clip(
                        cx + np.random.normal(0, 30), 30, aw - 30))
                    fy = float(np.clip(
                        cy + np.random.normal(0, 30), 30, ah - 30))
                    env.foods.append([fx, fy, 'small'])

            n_alive = sum(1 for f in all_fish if f['alive'])
            if n_alive == 0:
                break

        env.close()
        survived = t + 1
        mean_efe = total_efe / max(1, survived)
        geo_coverage = float(np.mean(brains[0].geo_model.visit_count > 0))

        from collections import Counter
        gc = Counter(goals_log)

        metrics = {
            'round': round_num,
            'mode': 'multi-agent',
            'n_fish': n_fish,
            'survived': survived,
            'food_eaten': sum(total_eaten),
            'food_per_fish': total_eaten,
            'fish_alive': sum(1 for f in all_fish if f['alive']),
            'mean_efe': float(mean_efe),
            'energy_final': float(brains[0].energy),
            'caught': False,
            'geo_coverage': geo_coverage,
            'goal_distribution': {
                GOAL_NAMES[g]: gc.get(g, 0) for g in range(4)},
            'fitness': self._compute_fitness(
                survived, sum(total_eaten), mean_efe,
                brains[0].energy, geo_coverage),
            'vae_memory_nodes': brains[0].vae.memory.n_allocated,
            'critic_mean_value': float(brains[0].critic.values.mean()),
            'personality': 'mixed',
        }
        return metrics

    def train_multi(self, n_rounds=None, n_fish=5):
        """Run multiple rounds of multi-agent training."""
        if n_rounds is None:
            n_rounds = self.config.get('training.n_rounds', 10)
        save_every = self.config.get('training.save_every', 5)

        self.running = True

        # Create n_fish brains with different personalities (once)
        personalities = assign_personalities(n_fish, mode='mixed')
        brains = []
        for i in range(n_fish):
            brain = ZebrafishBrainV2(device=DEVICE, personality=personalities[i])
            # Load checkpoint for focal fish (brain 0) only
            if i == 0:
                ckpt_path = self.config.get('training.load_checkpoint')
                if ckpt_path and os.path.exists(ckpt_path):
                    rnd, metrics = self.checkpoint_mgr.load(brain, ckpt_path)
                    self.total_rounds_done = rnd
                    print(f"  Loaded checkpoint: round {rnd}")
                else:
                    brain.reset()
            else:
                brain.reset()
            brains.append(brain)

        # Keep focal brain reference for checkpointing
        self.brain = brains[0]

        ptypes = ['bold', 'shy', 'explorer', 'social', 'default']
        print(f"\n{'=' * 60}")
        print(f"  Multi-Agent Training: {n_rounds} rounds, {n_fish} fish")
        print(f"  Personalities: {[ptypes[i % len(ptypes)] for i in range(n_fish)]}")
        print(f"  Config: predator={self.config.get('env.predator_ai')}, "
              f"food={self.config.get('env.n_food')}")
        print(f"{'=' * 60}")

        for r in range(1, n_rounds + 1):
            if not self.running:
                break
            round_num = self.total_rounds_done + r
            t0 = time.time()
            metrics = self.run_round_multi(round_num, brains, n_fish=n_fish)
            elapsed = time.time() - t0
            metrics['elapsed_sec'] = elapsed
            self.round_history.append(metrics)

            print(f"  Round {round_num}: survived={metrics['survived']}, "
                  f"food={metrics['food_eaten']} "
                  f"({'/'.join(str(x) for x in metrics['food_per_fish'])}), "
                  f"alive={metrics['fish_alive']}/{n_fish}, "
                  f"fitness={metrics['fitness']:.0f} ({elapsed:.0f}s)")

            if self.on_round_end:
                self.on_round_end(metrics)

            if r % save_every == 0 or r == n_rounds:
                path = self.checkpoint_mgr.save(
                    brains[0], round_num, metrics, self.config)
                print(f"    Checkpoint saved: {path}")

        self.total_rounds_done += n_rounds
        self.running = False

        if self.on_training_done:
            self.on_training_done(self.round_history)

        # Clean up non-focal brains
        for b in brains[1:]:
            del b

        return self.round_history

    def train_async_multi(self, n_rounds=None, n_fish=5):
        """Run multi-agent training in background thread."""
        self._thread = threading.Thread(
            target=self.train_multi,
            args=(n_rounds, n_fish),
            daemon=True)
        self._thread.start()
        return self._thread

    def train(self, n_rounds=None):
        """Run multiple rounds of training."""
        if n_rounds is None:
            n_rounds = self.config.get('training.n_rounds', 10)
        save_every = self.config.get('training.save_every', 5)

        self.running = True
        self.brain = self._create_brain()

        print(f"\n{'='*60}")
        print(f"  Training: {n_rounds} rounds")
        print(f"  Config: {self.config.get('fish.personality_mode')} personality, "
              f"{self.config.get('env.n_food')} food, "
              f"predator={self.config.get('env.predator_ai')}")
        print(f"{'='*60}")

        for r in range(1, n_rounds + 1):
            if not self.running:
                break

            round_num = self.total_rounds_done + r
            t0 = time.time()
            metrics = self.run_round(round_num)
            elapsed = time.time() - t0

            metrics['elapsed_sec'] = elapsed
            self.round_history.append(metrics)

            print(f"  Round {round_num}: survived={metrics['survived']}, "
                  f"food={metrics['food_eaten']}, fitness={metrics['fitness']:.0f}, "
                  f"EFE={metrics['mean_efe']:.4f} ({elapsed:.0f}s)")

            if self.on_round_end:
                self.on_round_end(metrics)

            # Save checkpoint
            if r % save_every == 0 or r == n_rounds:
                path = self.checkpoint_mgr.save(
                    self.brain, round_num, metrics, self.config)
                print(f"    Checkpoint saved: {path}")

        self.total_rounds_done += n_rounds
        self.running = False

        if self.on_training_done:
            self.on_training_done(self.round_history)

        return self.round_history

    def train_async(self, n_rounds=None):
        """Run training in background thread."""
        self._thread = threading.Thread(
            target=self.train, args=(n_rounds,), daemon=True)
        self._thread.start()
        return self._thread

    def stop(self):
        """Stop training gracefully."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)

    def get_status(self):
        """Current training status for dashboard."""
        return {
            'running': self.running,
            'current_round': self.current_round,
            'total_rounds_done': self.total_rounds_done,
            'round_history': self.round_history[-20:],  # last 20
            'current_step': self.current_step_data,
            'config': self.config.data,
            'checkpoints': self.checkpoint_mgr.list_checkpoints()[-10:],
        }
