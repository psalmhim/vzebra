"""
Single-agent training launcher.

Usage:
    .venv/bin/python -m zebrav2.train [options]

Options:
    --rounds N       Number of training rounds (default: 30)
    --checkpoint P   Path to starting checkpoint (default: latest)
    --predator MODE  none / simple / intelligent (default: intelligent)
    --food N         Food items per episode (default: 20)
    --steps N        Max steps per episode (default: 500)
    --save-every N   Save checkpoint every N rounds (default: 5)
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.engine.config import TrainingConfig
from zebrav2.engine.trainer import TrainingEngine


def main():
    parser = argparse.ArgumentParser(description='Single-agent training')
    parser.add_argument('--rounds',     type=int, default=30)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--predator',   type=str, default='intelligent',
                        choices=['none', 'simple', 'intelligent'])
    parser.add_argument('--food',       type=int, default=20)
    parser.add_argument('--steps',      type=int, default=500)
    parser.add_argument('--save-every', type=int, default=5)
    args = parser.parse_args()

    ckpt_dir = os.path.join(PROJECT_ROOT, 'zebrav2', 'checkpoints')
    checkpoint = args.checkpoint
    if checkpoint is None:
        pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
        if pts:
            checkpoint = os.path.join(ckpt_dir, pts[-1])
            print(f"  Auto-selected checkpoint: {pts[-1]}")

    config = TrainingConfig()
    config.data['training']['n_rounds']   = args.rounds
    config.data['training']['save_every'] = args.save_every
    config.data['env']['predator_ai']     = args.predator
    config.data['env']['n_food']          = args.food
    config.data['env']['max_steps']       = args.steps
    if checkpoint:
        config.data['training']['load_checkpoint'] = checkpoint

    engine = TrainingEngine(config=config)

    # Print per-round summary
    def on_round(m):
        goal_dist = m['goal_distribution']
        flee_pct = round(100 * goal_dist.get('FLEE', 0) / max(1, m['survived']), 0)
        print(f"  Round {m['round']:3d}: survived={m['survived']:3d}  "
              f"food={m['food_eaten']}  fitness={m['fitness']:.0f}  "
              f"FLEE={int(flee_pct)}%  ({m.get('elapsed_sec', 0):.0f}s)", flush=True)

    engine.on_round_end = on_round
    engine.train(n_rounds=args.rounds)


if __name__ == '__main__':
    main()
