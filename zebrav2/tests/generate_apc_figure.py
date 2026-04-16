"""
Generate the Action-Perception Cycle figure for the paper.

Four panels:
  (a) Multi-pass FE convergence — F decreases across inference passes
  (b) Adaptive blend α timeline — precision and spontaneity drive α
  (c) Spontaneity ξ dynamics — 5-factor anti-FEP signal
  (d) dF/dt drives goal switching — free-energy gradient trace

Saves to: plots/v2_paper/fig_apc.pdf and .png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'v2_paper')
os.makedirs(PLOT_DIR, exist_ok=True)

STEPS = 400
SEED = 42


def collect_trace():
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=STEPS)
    obs, _ = env.reset(seed=SEED)
    brain = ZebrafishBrainV2(device=DEVICE)
    brain.reset()

    trace = {
        'fe': [],
        'fe_per_pass': [],
        'ai_blend': [],
        'spontaneity': [],
        'fe_gradient': [],
        'ai_convergence': [],
        'goal': [],
        'precision_mean': [],
    }

    for t in range(STEPS):
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(brain.current_goal == 1,
                                0.8 if brain.current_goal == 1 else 0.0)
        inject_sensory(env)
        out = brain.step(obs, env)
        action = np.array([out['turn'], out['speed']], dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)

        trace['fe'].append(out.get('free_energy', 0.0))
        trace['ai_blend'].append(out.get('ai_blend', 0.3))
        trace['spontaneity'].append(out.get('spontaneity', 0.0))
        trace['fe_gradient'].append(out.get('fe_gradient', 0.0))
        trace['ai_convergence'].append(out.get('ai_convergence', 0.0))
        trace['goal'].append(brain.current_goal)
        trace['precision_mean'].append(
            float(brain.active_motor.column.precision.mean()))

        # Capture per-pass FE from last motor step (from active_motor)
        fpp = list(getattr(brain.active_motor, '_fe_per_pass', []))
        if fpp:
            trace['fe_per_pass'].append(fpp)

        if term or trunc:
            print(f"  [terminated at t={t}]")
            break

    return trace


def plot_figure(trace: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Action–Perception Cycle Diagnostics",
                 fontsize=13, fontweight='bold')

    # Panel (a): Multi-pass FE convergence
    ax = axes[0, 0]
    fpp = np.array(trace['fe_per_pass'])  # [T, n_passes]
    if fpp.size > 0 and fpp.ndim == 2:
        n_passes = fpp.shape[1]
        xs = np.arange(1, n_passes + 1)
        median = np.median(fpp, axis=0)
        q25 = np.percentile(fpp, 25, axis=0)
        q75 = np.percentile(fpp, 75, axis=0)
        ax.fill_between(xs, q25, q75, alpha=0.25, color='steelblue',
                        label='IQR')
        ax.plot(xs, median, 'o-', color='steelblue', linewidth=2, markersize=8,
                label='median')
        # Example individual traces
        for i in np.linspace(0, len(fpp) - 1, 5, dtype=int):
            ax.plot(xs, fpp[i], '-', color='gray', alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Inference pass")
    ax.set_ylabel("Free energy F")
    ax.set_title("(a) Iterative inference reduces surprise")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(xs if fpp.size > 0 else [1, 2, 3])

    # Panel (b): Adaptive blend timeline + precision
    ax = axes[0, 1]
    T = len(trace['ai_blend'])
    ts = np.arange(T)
    ax.plot(ts, trace['ai_blend'], color='darkorange', linewidth=1.5,
            label=r'$\alpha_{AI}$')
    ax.axhline(0.3, color='gray', linestyle='--', linewidth=0.8,
               label=r'baseline $\alpha=0.3$')
    ax.axhline(0.1, color='red', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.axhline(0.8, color='red', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time step")
    ax.set_ylabel(r'$\alpha_{AI}$')
    ax.set_title(r"(b) Adaptive blend $\alpha = 0.3 + 0.4\bar\pi - 0.3\xi$")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(ts, trace['precision_mean'], color='navy', linewidth=1,
             alpha=0.6, label=r'$\bar\pi$')
    ax2.set_ylabel(r'mean precision $\bar\pi$', color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')

    # Panel (c): Spontaneity dynamics
    ax = axes[1, 0]
    ax.plot(ts, trace['spontaneity'], color='crimson', linewidth=1.5,
            label=r'$\xi$')
    ax.fill_between(ts, 0, trace['spontaneity'],
                    where=np.array(trace['spontaneity']) > 0.3,
                    color='red', alpha=0.2, label=r'$\xi > 0.3$ (noise on)')
    ax.axhline(0.3, color='red', linestyle='--', linewidth=0.8)
    ax.axhline(0.05, color='gray', linestyle=':', linewidth=0.6,
               label='baseline (neural noise)')
    ax.set_xlabel("Time step")
    ax.set_ylabel(r'spontaneity $\xi$')
    ax.set_title("(c) Anti-FEP spontaneity: habenula, boredom, DA, social, noise")
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Panel (d): FE gradient + goal
    ax = axes[1, 1]
    ax.plot(ts, trace['fe_gradient'], color='purple', linewidth=1.2,
            label=r'$dF/dt$')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(0.05, color='purple', linestyle='--', linewidth=0.8,
               label=r'goal-switch threshold (0.05)')
    ax.set_xlabel("Time step")
    ax.set_ylabel(r'$dF/dt$')
    ax.set_title("(d) FE gradient drives goal selection")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # Overlay goal bands at bottom
    goals = np.array(trace['goal'])
    goal_colors = ['#4caf50', '#f44336', '#2196f3', '#ff9800']
    goal_names = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']
    ax_goal = ax.twinx()
    ax_goal.set_yticks([])
    y_goal = np.zeros_like(goals, dtype=float) - 0.02
    for g in range(4):
        mask = goals == g
        if mask.any():
            ax_goal.fill_between(ts, -0.05, -0.01,
                                 where=mask, color=goal_colors[g], alpha=0.7,
                                 step='mid')
    ax_goal.set_ylim(-0.05, 1.0)

    plt.tight_layout()
    out_pdf = os.path.join(PLOT_DIR, 'fig_apc.pdf')
    out_png = os.path.join(PLOT_DIR, 'fig_apc.png')
    plt.savefig(out_pdf, bbox_inches='tight', dpi=150)
    plt.savefig(out_png, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  → saved: {out_pdf}")
    print(f"  → saved: {out_png}")


def print_summary(trace):
    fpp = np.array(trace['fe_per_pass'])
    print("\n==== APC summary ====")
    print(f"  Trace length: {len(trace['fe'])} steps")
    if fpp.size > 0 and fpp.ndim == 2:
        mean_reduction = (fpp[:, 0] - fpp[:, -1]).mean()
        print(f"  Mean F reduction pass1→pass{fpp.shape[1]}: {mean_reduction:.4f}")
        print(f"  Mean F pass1: {fpp[:, 0].mean():.4f}")
        print(f"  Mean F final: {fpp[:, -1].mean():.4f}")
    print(f"  Mean α_AI: {np.mean(trace['ai_blend']):.3f}  "
          f"(range {np.min(trace['ai_blend']):.2f}–{np.max(trace['ai_blend']):.2f})")
    print(f"  Mean spontaneity: {np.mean(trace['spontaneity']):.3f}  "
          f"(max {np.max(trace['spontaneity']):.2f})")
    print(f"  |dF/dt| mean: {np.mean(np.abs(trace['fe_gradient'])):.4f}")
    steps_above_thr = np.sum(np.array(trace['spontaneity']) > 0.3)
    print(f"  Spontaneity > 0.3 in {steps_above_thr} steps "
          f"({100 * steps_above_thr / len(trace['spontaneity']):.1f}%)")


if __name__ == '__main__':
    print("Collecting APC trace...")
    trace = collect_trace()
    print_summary(trace)
    print("\nPlotting...")
    plot_figure(trace)
    print("Done.")
