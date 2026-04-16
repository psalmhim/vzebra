"""
Action-Perception Cycle benchmark — APC on vs off.

Runs matched seeds with:
  - APC OFF: n_inference_passes=1, spontaneity=0, fixed α=0.3
  - APC ON:  n_inference_passes=3 + adaptive α + spontaneity (default)

Reports mean food and survival; verifies APC does not regress baseline.
"""
import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

N_SEEDS = 5
MAX_STEPS = 300


def disable_apc(brain):
    """Force single-pass inference and suppress spontaneity."""
    brain.active_motor.n_inference_passes = 1
    # Monkey-patch _compute_spontaneity to always return 0
    brain._compute_spontaneity = lambda: 0.0


def run_one(brain, seed):
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=MAX_STEPS)
    obs, _ = env.reset(seed=seed)
    brain.reset()
    food = 0
    fe_sum = 0.0
    for t in range(MAX_STEPS):
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(brain.current_goal == 1,
                                0.8 if brain.current_goal == 1 else 0.0)
        inject_sensory(env)
        out = brain.step(obs, env)
        action = np.array([out['turn'], out['speed']], dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
        food += info.get('food_eaten_this_step', 0)
        fe_sum += out.get('free_energy', 0.0)
        if term or trunc:
            return {'survived': t + 1, 'food': food, 'fe_mean': fe_sum / max(1, t + 1)}
    return {'survived': MAX_STEPS, 'food': food, 'fe_mean': fe_sum / MAX_STEPS}


def run_condition(name, setup_fn):
    results = []
    t0 = time.time()
    for i in range(N_SEEDS):
        seed = i * 13 + 7
        brain = ZebrafishBrainV2(device=DEVICE)
        if setup_fn is not None:
            setup_fn(brain)
        r = run_one(brain, seed)
        results.append(r)
        print(f"  [{name}] seed {i+1}/{N_SEEDS}  "
              f"survived={r['survived']}  food={r['food']}  F={r['fe_mean']:.3f}")
    dt = time.time() - t0
    survivals = [r['survived'] for r in results]
    foods = [r['food'] for r in results]
    fes = [r['fe_mean'] for r in results]
    print(f"  {name:>10}: survived={np.mean(survivals):.0f}±{np.std(survivals):.0f}  "
          f"food={np.mean(foods):.2f}±{np.std(foods):.2f}  "
          f"F={np.mean(fes):.3f}  ({dt:.0f}s)")
    return {
        'survival_mean': float(np.mean(survivals)),
        'survival_std': float(np.std(survivals)),
        'food_mean': float(np.mean(foods)),
        'food_std': float(np.std(foods)),
        'fe_mean': float(np.mean(fes)),
    }


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  APC benchmark  (n={N_SEEDS} seeds, {MAX_STEPS} steps)")
    print(f"{'='*60}\n")

    print("Condition: APC OFF (single-pass, α=0.3, no spontaneity)")
    off = run_condition("APC_off", disable_apc)
    print()
    print("Condition: APC ON (3-pass, adaptive α, spontaneity)")
    on = run_condition("APC_on", None)
    print()
    print(f"{'='*60}")
    print(f"  RESULTS  |  APC OFF  →  APC ON   Δ")
    print(f"{'='*60}")
    d_surv = on['survival_mean'] - off['survival_mean']
    d_food = on['food_mean'] - off['food_mean']
    d_fe = on['fe_mean'] - off['fe_mean']
    print(f"  Survival: {off['survival_mean']:.1f} → {on['survival_mean']:.1f}   "
          f"Δ={d_surv:+.1f}")
    print(f"  Food:     {off['food_mean']:.2f} → {on['food_mean']:.2f}   "
          f"Δ={d_food:+.2f}")
    print(f"  F mean:   {off['fe_mean']:.3f} → {on['fe_mean']:.3f}   "
          f"Δ={d_fe:+.3f}")
    regressed = (d_food < -0.3 * max(0.1, off['food_mean'])) or \
                (d_surv < -0.1 * max(1, off['survival_mean']))
    if regressed:
        print("\n  [FAIL] APC regressed baseline")
        sys.exit(1)
    else:
        print("\n  [OK]   APC does not regress baseline")
