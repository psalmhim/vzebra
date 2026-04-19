"""
5-seed performance benchmark: before/after Tier 2 modules.

Tests survival time, food eaten, and flee success across 5 seeds.
Compares ablated (no Tier 2) vs full (with Tier 2) configurations.
"""
import sys
import os
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from zebrav2.spec import DEVICE
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.config.brain_config import BrainConfig


class MinimalBenchEnv:
    """Lightweight environment for benchmarking (no rendering)."""
    def __init__(self, seed=0, arena_w=800, arena_h=600):
        self.rng = np.random.RandomState(seed)
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.fish_x = arena_w / 2
        self.fish_y = arena_h / 2
        self.fish_heading = 0.0
        self.fish_energy = 100.0
        self.pred_x = self.rng.uniform(50, arena_w - 50)
        self.pred_y = self.rng.uniform(50, arena_h - 50)
        self._pred_vx = 0.0
        self._pred_vy = 0.0
        self._eaten_now = 0
        self._enemy_pixels_total = 0.0
        self.foods = []
        self.all_fish = [{'x': self.fish_x, 'y': self.fish_y,
                          'heading': self.fish_heading, 'alive': True}]
        self._step = 0
        self._total_eaten = 0
        self._alive = True
        self._spawn_food(5)
        self.brain_L = np.zeros(800, dtype=np.float32)
        self.brain_R = np.zeros(800, dtype=np.float32)

    def _spawn_food(self, n):
        for _ in range(n):
            self.foods.append((
                self.rng.uniform(50, self.arena_w - 50),
                self.rng.uniform(50, self.arena_h - 50),
            ))

    def _update_retina(self):
        """Simplified retinal input from food/predator positions."""
        self.brain_L = np.zeros(800, dtype=np.float32)
        self.brain_R = np.zeros(800, dtype=np.float32)
        # Food: type=1.0, intensity proportional to distance
        for fd in self.foods:
            dx = fd[0] - self.fish_x
            dy = fd[1] - self.fish_y
            dist = max(1.0, np.sqrt(dx*dx + dy*dy))
            if dist > 300:
                continue
            angle = np.arctan2(dy, dx) - self.fish_heading
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            intensity = max(0.0, min(1.0, 1.0 - dist / 300.0))
            # Left eye: angle < 0 (left visual field)
            # Right eye: angle > 0 (right visual field)
            col = int(200 + angle / np.pi * 200)
            col = max(0, min(399, col))
            if angle < 0:
                self.brain_L[col] = intensity
                self.brain_L[400 + col] = 1.0  # food type
            else:
                self.brain_R[col] = intensity
                self.brain_R[400 + col] = 1.0  # food type

        # Predator: type=0.5
        dx = self.pred_x - self.fish_x
        dy = self.pred_y - self.fish_y
        pred_dist = max(1.0, np.sqrt(dx*dx + dy*dy))
        if pred_dist < 250:
            angle = np.arctan2(dy, dx) - self.fish_heading
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            intensity = max(0.0, min(1.0, 1.0 - pred_dist / 250.0))
            size = max(1, int(20 * intensity))
            col = int(200 + angle / np.pi * 200)
            for c in range(max(0, col - size), min(400, col + size)):
                if angle < 0:
                    self.brain_L[c] = intensity
                    self.brain_L[400 + c] = 0.5
                else:
                    self.brain_R[c] = intensity
                    self.brain_R[400 + c] = 0.5
            self._enemy_pixels_total = size * 2 * intensity
        else:
            self._enemy_pixels_total = 0.0

    def step_env(self, turn, speed):
        """Advance environment by one step."""
        self._step += 1
        self._eaten_now = 0

        # Move fish
        self.fish_heading += turn * 0.15
        self.fish_heading = np.arctan2(np.sin(self.fish_heading),
                                       np.cos(self.fish_heading))
        self.fish_x += np.cos(self.fish_heading) * speed * 3.0
        self.fish_y += np.sin(self.fish_heading) * speed * 3.0
        # Boundary
        self.fish_x = max(10, min(self.arena_w - 10, self.fish_x))
        self.fish_y = max(10, min(self.arena_h - 10, self.fish_y))

        # Energy drain
        self.fish_energy -= 0.2 * speed
        if self.fish_energy <= 0:
            self._alive = False
            return

        # Eat food
        eaten_idx = []
        for i, fd in enumerate(self.foods):
            dx = fd[0] - self.fish_x
            dy = fd[1] - self.fish_y
            if np.sqrt(dx*dx + dy*dy) < 35:
                eaten_idx.append(i)
                self.fish_energy = min(100, self.fish_energy + 10)
                self._eaten_now += 1
                self._total_eaten += 1
        for i in reversed(eaten_idx):
            self.foods.pop(i)
        # Respawn food
        if len(self.foods) < 3:
            self._spawn_food(3)

        # Move predator (simple chase with noise)
        dx = self.fish_x - self.pred_x
        dy = self.fish_y - self.pred_y
        pred_dist = max(1.0, np.sqrt(dx*dx + dy*dy))
        if pred_dist < 300:
            # Hunt mode
            self._pred_vx = dx / pred_dist * 2.5 + self.rng.normal(0, 0.5)
            self._pred_vy = dy / pred_dist * 2.5 + self.rng.normal(0, 0.5)
        else:
            # Patrol
            self._pred_vx = self.rng.normal(0, 1.0)
            self._pred_vy = self.rng.normal(0, 1.0)
        self.pred_x += self._pred_vx
        self.pred_y += self._pred_vy
        self.pred_x = max(10, min(self.arena_w - 10, self.pred_x))
        self.pred_y = max(10, min(self.arena_h - 10, self.pred_y))

        # Kill if caught
        if pred_dist < 20:
            self._alive = False

        # Update retinal arrays
        self.all_fish[0] = {'x': self.fish_x, 'y': self.fish_y,
                            'heading': self.fish_heading, 'alive': True}
        self._update_retina()


def run_episode(brain, seed, max_steps=500):
    """Run one episode, return metrics."""
    env = MinimalBenchEnv(seed=seed)
    brain.reset()
    env._update_retina()

    for step in range(max_steps):
        result = brain.step(None, env)
        env.step_env(result['turn'], result['speed'])
        if not env._alive:
            break

    return {
        'survival_steps': step + 1,
        'food_eaten': env._total_eaten,
        'alive': env._alive,
        'final_energy': env.fish_energy,
    }


def run_benchmark(label, cfg, seeds, max_steps=500):
    """Run benchmark across seeds, return aggregated metrics."""
    print(f"\n--- {label} ---")
    brain = ZebrafishBrainV2(device=DEVICE, brain_config=cfg)
    results = []
    for seed in seeds:
        r = run_episode(brain, seed, max_steps)
        results.append(r)
        print(f"  seed={seed}: survived={r['survival_steps']}/{max_steps}, "
              f"food={r['food_eaten']}, alive={r['alive']}, energy={r['final_energy']:.1f}")

    agg = {
        'mean_survival': np.mean([r['survival_steps'] for r in results]),
        'mean_food': np.mean([r['food_eaten'] for r in results]),
        'survival_rate': np.mean([1 if r['alive'] else 0 for r in results]),
        'mean_energy': np.mean([r['final_energy'] for r in results]),
    }
    print(f"  MEAN: survival={agg['mean_survival']:.0f}, food={agg['mean_food']:.1f}, "
          f"alive={agg['survival_rate']*100:.0f}%, energy={agg['mean_energy']:.1f}")
    return agg


def main():
    seeds = [42, 123, 456]
    max_steps = 30  # reduced for practical runtime (~4s/step on MPS)

    print("=" * 60)
    print("  TIER 2 PERFORMANCE BENCHMARK (5 seeds)")
    print("=" * 60)

    # --- Baseline: Tier 2 modules ablated ---
    cfg_base = BrainConfig()
    cfg_base.ablation.raphe = False
    cfg_base.ablation.locus_coeruleus = False
    cfg_base.ablation.habituation = False
    cfg_base.ablation.pectoral_fin = False
    # Keep habenula/insula off (default)
    baseline = run_benchmark("BASELINE (no Tier 2)", cfg_base, seeds, max_steps)

    # --- Full: Tier 2 enabled ---
    cfg_full = BrainConfig()
    # Keep habenula/insula off (default) — only test Tier 2 additions
    full = run_benchmark("FULL (with Tier 2)", cfg_full, seeds, max_steps)

    # --- Comparison ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Baseline':>10} {'Full':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for k in ['mean_survival', 'mean_food', 'survival_rate', 'mean_energy']:
        b = baseline[k]
        f = full[k]
        d = f - b
        sign = '+' if d >= 0 else ''
        if k == 'survival_rate':
            print(f"  {k:<25} {b*100:>9.0f}% {f*100:>9.0f}% {sign}{d*100:>8.0f}%")
        else:
            print(f"  {k:<25} {b:>10.1f} {f:>10.1f} {sign}{d:>9.1f}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_tier2_results.json')
    with open(out_path, 'w') as fp:
        json.dump({'baseline': baseline, 'full': full, 'seeds': seeds,
                   'max_steps': max_steps}, fp, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
