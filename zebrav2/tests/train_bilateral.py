"""
Train from latest checkpoint with bilateral vision changes.
Validates no regression in survival/food intake after hemispheric tectum/thalamus.

Run: .venv/bin/python -u -m zebrav2.tests.train_bilateral
"""
import os
import sys
import json

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

    # Read previous round number
    meta_files = sorted([f for f in os.listdir(CKPT_DIR) if f.endswith('.json')])
    start_round = 0
    if meta_files:
        with open(os.path.join(CKPT_DIR, meta_files[-1])) as f:
            start_round = json.load(f).get('round', 0)
    print(f'Starting from round {start_round + 1}')

    N_ROUNDS = 15  # rounds 86-100

    config = TrainingConfig({
        'training': {
            'n_rounds': N_ROUNDS,
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

    # Track metrics for regression check
    round_metrics = []

    def on_round(m):
        round_metrics.append(m)
        goal_dist = m.get('goal_distribution', {})
        flee_pct = goal_dist.get('FLEE', 0) / max(1, m['survived']) * 100
        print(f"  Round {m['round']:3d}: survived={m['survived']:3d}  "
              f"food={m['food_eaten']:2d}  FLEE={flee_pct:.0f}%  "
              f"fitness={m['fitness']:.0f}  geo={m['geo_coverage']*100:.1f}%")

    engine.on_round_end = on_round

    print(f'\n{"="*60}')
    print(f'Bilateral Vision Training: {N_ROUNDS} rounds')
    print(f'{"="*60}\n')

    engine.train(n_rounds=N_ROUNDS)

    # Regression check
    print(f'\n{"="*60}')
    print('REGRESSION CHECK')
    print(f'{"="*60}')

    if round_metrics:
        survivals = [m['survived'] for m in round_metrics]
        foods = [m['food_eaten'] for m in round_metrics]
        fitnesses = [m['fitness'] for m in round_metrics]
        avg_surv = sum(survivals) / len(survivals)
        avg_food = sum(foods) / len(foods)
        avg_fit = sum(fitnesses) / len(fitnesses)

        print(f'  Avg survival:  {avg_surv:.0f} steps (baseline: 474)')
        print(f'  Avg food:      {avg_food:.1f} items (baseline: 6.0)')
        print(f'  Avg fitness:   {avg_fit:.0f}')
        print(f'  Caught:        {sum(1 for m in round_metrics if m["caught"])}/{len(round_metrics)}')

        # Check thresholds
        ok_surv = avg_surv >= 400
        ok_food = avg_food >= 4
        print(f'\n  Survival >= 400: {"PASS" if ok_surv else "FAIL"}')
        print(f'  Food >= 4:       {"PASS" if ok_food else "FAIL"}')

    print('\nTraining complete.')
