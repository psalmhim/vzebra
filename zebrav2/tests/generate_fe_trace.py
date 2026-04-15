"""
WT vs Schizophrenia free energy trace — Figure for manuscript.

Runs wildtype and schizophrenia brains in a prey-predator environment,
recording total and per-module free energy at each step. Produces a
two-panel figure showing (A) temporal FE trace and (B) module-level
FE comparison.

Usage:
    .venv/bin/python -m zebrav2.tests.generate_fe_trace
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.disorder import apply_disorder
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv


def run_condition(disorder_name, n_steps=200, n_seeds=3):
    """Run brain in environment, return per-step FE traces and module FEs."""
    all_traces = []
    all_module_fes = []  # list of dicts per seed

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        brain = ZebrafishBrainV2()
        if disorder_name != 'wildtype':
            changes = apply_disorder(brain, disorder_name)
            print(f"  {disorder_name} seed {seed}: applied {list(changes.keys())}")

        env = ZebrafishPreyPredatorEnv(render_mode=None)
        obs, _ = env.reset(seed=seed)

        trace = []
        module_fe_accum = {}

        for t in range(n_steps):
            out = brain.step(obs, env=env)
            action = np.array([out['turn'], out['speed']], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            fe = out.get('total_free_energy', 0.0)
            trace.append(float(fe))

            # Accumulate per-module FE
            mfe = brain.get_module_free_energies()
            for k, v in mfe.items():
                if k not in module_fe_accum:
                    module_fe_accum[k] = []
                module_fe_accum[k].append(float(v))

            if terminated or truncated:
                # Pad remaining steps with last value
                for _ in range(n_steps - t - 1):
                    trace.append(trace[-1])
                    for k in module_fe_accum:
                        module_fe_accum[k].append(module_fe_accum[k][-1])
                break

        env.close()
        all_traces.append(trace)
        all_module_fes.append({k: np.mean(v) for k, v in module_fe_accum.items()})
        print(f"  {disorder_name} seed {seed}: mean FE={np.mean(trace):.4f}")

    return np.array(all_traces), all_module_fes


def main():
    print("=" * 60)
    print("WT vs Schizophrenia Free Energy Trace")
    print("=" * 60)

    N_STEPS = 200
    N_SEEDS = 3

    print("\nRunning wildtype...")
    wt_traces, wt_mfe = run_condition('wildtype', N_STEPS, N_SEEDS)

    print("\nRunning schizophrenia...")
    scz_traces, scz_mfe = run_condition('schizophrenia', N_STEPS, N_SEEDS)

    # Compute stats
    wt_mean = wt_traces.mean(axis=0)
    wt_sem = wt_traces.std(axis=0) / np.sqrt(N_SEEDS)
    scz_mean = scz_traces.mean(axis=0)
    scz_sem = scz_traces.std(axis=0) / np.sqrt(N_SEEDS)

    steps = np.arange(N_STEPS)

    # Module-level averages
    modules = list(wt_mfe[0].keys())
    wt_mod_means = {m: np.mean([s[m] for s in wt_mfe]) for m in modules}
    wt_mod_sems = {m: np.std([s[m] for s in wt_mfe]) / np.sqrt(N_SEEDS) for m in modules}
    scz_mod_means = {m: np.mean([s[m] for s in scz_mfe]) for m in modules}
    scz_mod_sems = {m: np.std([s[m] for s in scz_mfe]) / np.sqrt(N_SEEDS) for m in modules}

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Free Energy: Wildtype vs Schizophrenia', fontsize=13, fontweight='bold')

    # Panel A: temporal trace
    ax1.fill_between(steps, wt_mean - wt_sem, wt_mean + wt_sem, alpha=0.2, color='#3498db')
    ax1.fill_between(steps, scz_mean - scz_sem, scz_mean + scz_sem, alpha=0.2, color='#e74c3c')
    ax1.plot(steps, wt_mean, color='#3498db', linewidth=1.5, label='Wildtype')
    ax1.plot(steps, scz_mean, color='#e74c3c', linewidth=1.5, label=f'Schizophrenia (\u03b3_ceil=-1.5)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Total Free Energy (F)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title('(A) Temporal free energy trace', fontsize=10, loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: per-module bar chart
    x = np.arange(len(modules))
    width = 0.35
    short_names = [m.replace('_', '\n') for m in modules]

    bars1 = ax2.bar(x - width/2, [wt_mod_means[m] for m in modules], width,
                     yerr=[wt_mod_sems[m] for m in modules],
                     label='WT', color='#3498db', alpha=0.8, capsize=2)
    bars2 = ax2.bar(x + width/2, [scz_mod_means[m] for m in modules], width,
                     yerr=[scz_mod_sems[m] for m in modules],
                     label='SCZ', color='#e74c3c', alpha=0.8, capsize=2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('Mean FE')
    ax2.legend(fontsize=8)
    ax2.set_title('(B) Per-module free energy', fontsize=10, loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    out_dir = os.path.join(_ROOT, 'plots', 'v2_paper')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fig_fe_trace_wt_vs_scz.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved \u2192 {out_path}")

    # Also save data as JSON
    data_path = os.path.join(out_dir, 'fe_trace_data.json')
    data = {
        'wt_mean_fe': float(wt_mean.mean()),
        'scz_mean_fe': float(scz_mean.mean()),
        'ratio': float(scz_mean.mean() / max(wt_mean.mean(), 1e-8)),
        'wt_module': {m: float(wt_mod_means[m]) for m in modules},
        'scz_module': {m: float(scz_mod_means[m]) for m in modules},
    }
    with open(data_path, 'w') as fp:
        json.dump(data, fp, indent=2)
    print(f"Data  \u2192 {data_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"WT mean FE:  {wt_mean.mean():.4f}")
    print(f"SCZ mean FE: {scz_mean.mean():.4f}")
    print(f"Ratio:       {scz_mean.mean() / max(wt_mean.mean(), 1e-8):.2f}x")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
