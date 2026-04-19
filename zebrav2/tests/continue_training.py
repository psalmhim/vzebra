"""
Continue training from the latest checkpoint, starting at round 21.
Runs 30 more rounds (21-50) with the EFE fixes applied.

Run: .venv/bin/python -u -m zebrav2.tests.continue_training
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.engine.trainer import TrainingEngine
from zebrav2.engine.config import TrainingConfig

CKPT_DIR = os.path.join(PROJECT_ROOT, 'zebrav2', 'checkpoints')

def find_latest_checkpoint():
    pts = sorted([f for f in os.listdir(CKPT_DIR) if f.endswith('.pt')])
    return os.path.join(CKPT_DIR, pts[-1]) if pts else None

if __name__ == '__main__':
    ckpt = find_latest_checkpoint()
    print(f'Resuming from: {ckpt}')

    config = TrainingConfig({
        'training': {
            'n_rounds': 30,
            'save_every': 5,
            'load_checkpoint': ckpt,
        },
        'env': {
            'n_food': 20,
            'max_steps': 500,
            'predator_ai': 'intelligent',
        },
        'fish': {
            'personality_mode': 'default',
        },
    })

    engine = TrainingEngine(config=config)

    def on_round(m):
        goal_dist = m.get('goal_distribution', {})
        flee_pct = goal_dist.get('FLEE', 0) / max(1, m['survived']) * 100
        print(f"    FLEE={flee_pct:.0f}%, food={m['food_eaten']}, "
              f"critic={m['critic_mean_value']:.4f}, geo={m['geo_coverage']*100:.1f}%")

    engine.on_round_end = on_round
    engine.train(n_rounds=30)
    print('\nTraining complete.')
