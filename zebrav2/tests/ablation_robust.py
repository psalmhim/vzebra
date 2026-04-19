"""
Robust ablation study: n=20 seeds per condition, 300-step episodes.
Run AFTER multi-round training has converged (50+ rounds).

Usage:
  .venv/bin/python -u -m zebrav2.tests.ablation_robust

Saves results to: plots/v2_paper/ablation_robust.json
                  plots/v2_paper/fig4_ablation_robust.png
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from scipy import stats as sp_stats
except ImportError:
    # Lightweight fallback: Welch's t-test via numpy
    class _sp_stats:
        @staticmethod
        def ttest_ind(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            na, nb = len(a), len(b)
            ma, mb = a.mean(), b.mean()
            va = a.var(ddof=1) / na if na > 1 else 1e-9
            vb = b.var(ddof=1) / nb if nb > 1 else 1e-9
            se = np.sqrt(va + vb)
            t = (ma - mb) / max(se, 1e-9)
            df = (va + vb)**2 / (va**2/(na-1) + vb**2/(nb-1)) if (na > 1 and nb > 1) else 1
            # Two-tailed p via normal approximation (adequate for n=20)
            p = 2 * (1 - 0.5 * (1 + np.sign(abs(t)) *
                     np.sqrt(1 - np.exp(-2*t*t/max(df, 1)))))
            return float(t), float(np.clip(p, 0, 1))
    sp_stats = _sp_stats()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'v2_paper')
os.makedirs(PLOT_DIR, exist_ok=True)

N_SEEDS = 20      # seeds per condition
MAX_STEPS = 300   # steps per episode (balance statistical power vs. runtime)

# Ablation conditions: module attribute → whether to ablate
ABLATIONS = {
    'Full model':      {},
    'No olfaction':    {'olfaction': True},
    'No cerebellum':   {'cerebellum': True},
    'No habenula':     {'habenula': True},
    'No amygdala':     {'amygdala': True},
    'No VAE':          {'vae': True},
    'No RL critic':    {'critic': True},
    'No lateral line': {'lateral_line': True},
    'No interoception':{'insula': True},
}


def run_episode(brain, seed: int, max_steps: int = MAX_STEPS) -> dict:
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    brain.reset()
    food_eaten = 0
    for t in range(max_steps):
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(brain.current_goal == 1, 0.8 if brain.current_goal == 1 else 0.0)
        inject_sensory(env)
        out = brain.step(obs, env)
        action = np.array([out['turn'], out['speed']], dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
        food_eaten += info.get('food_eaten_this_step', 0)
        if term or trunc:
            return {'survived': t + 1, 'food': food_eaten}
    return {'survived': max_steps, 'food': food_eaten}


def ablate_module(brain, module_name: str):
    """Disable a module by making its forward() return neutral defaults."""
    mod = getattr(brain, module_name, None)
    if mod is None:
        return False

    if hasattr(mod, 'forward'):
        original_forward = mod.forward
        def noop_forward(*args, _orig=original_forward, **kwargs):
            result = _orig(*args, **kwargs)
            if isinstance(result, dict):
                return {k: (0.0 if isinstance(v, (int, float)) else v)
                        for k, v in result.items()}
            return result
        mod.forward = noop_forward
        return True

    # For _ablated set (habenula / insula style)
    if hasattr(brain, '_ablated'):
        brain._ablated.add(module_name)
        return True
    return False


def run_ablation_condition(condition_name: str, disable: dict) -> dict:
    survivals, foods = [], []
    for seed_idx in range(N_SEEDS):
        seed = seed_idx * 13 + 7   # spread seeds evenly
        brain = ZebrafishBrainV2(device=DEVICE)
        for module_name in disable:
            ablate_module(brain, module_name)
        r = run_episode(brain, seed=seed)
        survivals.append(r['survived'])
        foods.append(r['food'])
        print(f"    [{condition_name}] seed {seed_idx+1}/{N_SEEDS}: "
              f"survived={r['survived']}, food={r['food']}")

    n = len(survivals)
    return {
        'survival_mean': float(np.mean(survivals)),
        'survival_std': float(np.std(survivals, ddof=1)),
        'survival_sem': float(np.std(survivals, ddof=1) / np.sqrt(n)),
        'food_mean': float(np.mean(foods)),
        'food_std': float(np.std(foods, ddof=1)),
        'food_sem': float(np.std(foods, ddof=1) / np.sqrt(n)),
        'survivals': survivals,
        'foods': foods,
    }


def run_ablation_study():
    print(f"\n{'='*60}")
    print(f"  Robust Ablation Study  (n={N_SEEDS} seeds, {MAX_STEPS} steps)")
    print(f"  Conditions: {list(ABLATIONS.keys())}")
    print('='*60)

    results = {}
    t0 = time.time()
    for name, disable in ABLATIONS.items():
        print(f"\n  -- {name} --")
        results[name] = run_ablation_condition(name, disable)
        r = results[name]
        print(f"  SUMMARY  survived={r['survival_mean']:.0f}±{r['survival_std']:.0f}  "
              f"food={r['food_mean']:.2f}±{r['food_std']:.2f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # t-tests vs. full model
    baseline_surv = results['Full model']['survivals']
    baseline_food = results['Full model']['foods']
    print(f"\n{'Condition':<22} {'Survival vs. full':>20}  {'Food vs. full':>16}")
    for name, r in results.items():
        if name == 'Full model':
            continue
        t_s, p_s = sp_stats.ttest_ind(baseline_surv, r['survivals'])
        t_f, p_f = sp_stats.ttest_ind(baseline_food, r['foods'])
        print(f"  {name:<20}  t={t_s:.2f} p={p_s:.3f}{'*' if p_s<0.05 else ' '}   "
              f"t={t_f:.2f} p={p_f:.3f}{'*' if p_f<0.05 else ' '}")

    return results


def plot_ablation(results: dict):
    names = list(results.keys())
    food_means = [results[n]['food_mean'] for n in names]
    food_sems = [results[n]['food_sem'] for n in names]
    surv_means = [results[n]['survival_mean'] for n in names]
    surv_sems = [results[n]['survival_sem'] for n in names]

    baseline_food = results['Full model']['food_mean']
    baseline_surv = results['Full model']['survival_mean']

    colors = ['green' if n == 'Full model' else
              ('red' if results[n]['food_mean'] < baseline_food else 'steelblue')
              for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Ablation Study (n={N_SEEDS} seeds, {MAX_STEPS} steps)', fontsize=13)

    for ax, means, sems, baseline, ylabel in [
        (axes[0], surv_means, surv_sems, baseline_surv, 'Survival steps'),
        (axes[1], food_means, food_sems, baseline_food, 'Food eaten'),
    ]:
        bars = ax.bar(range(len(names)), means, yerr=sems, color=colors,
                      alpha=0.8, capsize=4, error_kw=dict(lw=1.5))
        ax.axhline(baseline, color='green', linestyle='--', linewidth=1.5,
                   label='Full model baseline')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Condition')
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, 'fig4_ablation_robust.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    results = run_ablation_study()

    # Save JSON
    json_path = os.path.join(PLOT_DIR, 'ablation_robust.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    plot_ablation(results)
    print('\nDone.')
