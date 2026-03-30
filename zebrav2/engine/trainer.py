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
from zebrav2.brain.personality import get_personality
from zebrav2.brain.predator_brain import PredatorBrain
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
            n_food=cfg.get('env.n_food', 15),
            max_steps=cfg.get('env.max_steps', 500))
        return env

    def _compute_fitness(self, survived, food_eaten, mean_efe, energy_final,
                          geo_coverage):
        """Compute fitness score from objectives config."""
        obj = self.config.data.get('objectives', {})
        fitness = (
            obj.get('survival_weight', 1.0) * survived +
            obj.get('food_weight', 50.0) * food_eaten +
            obj.get('efe_weight', -10.0) * mean_efe +
            obj.get('energy_weight', 0.5) * energy_final +
            obj.get('exploration_weight', 5.0) * geo_coverage * 100
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
        self.brain.reset()
        # Don't reset learned weights — keep from previous rounds
        self.brain._apply_personality()

        # Predator
        pred_brain = None
        if cfg.get('env.predator_ai') == 'intelligent':
            pred_brain = PredatorBrain()

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

            efe = out.get('free_energy', 0)
            total_efe += efe
            goals_log.append(self.brain.current_goal)

            # Step data for real-time streaming
            # Collect food positions
            food_positions = []
            for food in getattr(env, 'foods', []):
                food_positions.append([float(food[0]), float(food[1])])
            # Collect rock positions
            rock_positions = []
            for rock in getattr(env, 'rock_formations', []):
                rock_positions.append([float(rock.get('cx', 0)), float(rock.get('cy', 0)),
                                       float(rock.get('radius', 30))])

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
                'foods': food_positions,
                'rocks': rock_positions,
                'arena_w': int(getattr(env, 'arena_w', 800)),
                'arena_h': int(getattr(env, 'arena_h', 600)),
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
        return metrics

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
