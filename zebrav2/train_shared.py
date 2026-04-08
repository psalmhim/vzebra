"""
Shared-weight multi-agent training launcher.

Usage:
    .venv/bin/python -m zebrav2.train_shared [options]

Options:
    --rounds N       Number of training rounds (default: 20)
    --workers N      Number of parallel workers per round (default: 5)
    --checkpoint P   Path to starting checkpoint (default: latest)
    --predator MODE  Predator AI: none / simple / intelligent (default: intelligent)
    --food N         Food items per episode (default: 20)
    --steps N        Max steps per episode (default: 500)
    --save-every N   Save checkpoint every N rounds (default: 5)

Example (resume from round 50, run 20 more rounds with 5 workers):
    .venv/bin/python -m zebrav2.train_shared \\
        --checkpoint zebrav2/checkpoints/ckpt_round_0050.pt \\
        --rounds 20 --workers 5 --predator intelligent
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.engine.config import TrainingConfig
from zebrav2.engine.shared_trainer import SharedWeightTrainer


def main():
    parser = argparse.ArgumentParser(description='Shared-weight multi-agent training')
    parser.add_argument('--rounds',      type=int, default=20)
    parser.add_argument('--workers',     type=int, default=5)
    parser.add_argument('--checkpoint',  type=str, default=None)
    parser.add_argument('--predator',    type=str, default='intelligent',
                        choices=['none', 'simple', 'intelligent'])
    parser.add_argument('--food',        type=int, default=20)
    parser.add_argument('--steps',       type=int, default=500)
    parser.add_argument('--save-every',  type=int, default=5)
    args = parser.parse_args()

    # If no checkpoint given, look for the latest one
    ckpt_dir = os.path.join(PROJECT_ROOT, 'zebrav2', 'checkpoints')
    checkpoint = args.checkpoint
    if checkpoint is None:
        pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
        if pts:
            checkpoint = os.path.join(ckpt_dir, pts[-1])
            print(f"  Auto-selected checkpoint: {pts[-1]}")
        else:
            print("  No checkpoint found — starting from scratch")

    # Build config
    config = TrainingConfig()
    config.data['training']['n_rounds']        = args.rounds
    config.data['training']['save_every']      = args.save_every
    config.data['env']['predator_ai']          = args.predator
    config.data['env']['n_food']               = args.food
    config.data['env']['max_steps']            = args.steps
    if checkpoint:
        config.data['training']['load_checkpoint'] = checkpoint

    trainer = SharedWeightTrainer(config=config, n_workers=args.workers)
    trainer.train_shared(n_rounds=args.rounds)


if __name__ == '__main__':
    main()
