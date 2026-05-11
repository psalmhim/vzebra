#!/usr/bin/env python3
"""
Generate fig_apc.pdf — Action-Perception Cycle diagnostics figure.
Runs brain for 400 steps with ckpt_best.pt, plots 4 panels:
  (a) K=3 inference pass free-energy convergence
  (b) Adaptive blend α_AI vs motor precision
  (c) Spontaneity ξ
  (d) dF/dt (free-energy gradient) + goal timeline

Run from project root:
  .venv/bin/python plots/paper/make_fig_apc.py
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 0.9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

GOAL_COLORS = ['#4878d0', '#d65f5f', '#35978f', '#762a83']
GOAL_NAMES  = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']

# ---------------------------------------------------------------------------
# Run episode
# ---------------------------------------------------------------------------
from zebrav2.engine.checkpoint import CheckpointManager
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav2.spec import DEVICE
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

CKPT = 'zebrav2/checkpoints/ckpt_best.pt'
T    = 400
SEED = 42

brain = ZebrafishBrainV2(device=DEVICE)
_cm = CheckpointManager()
if os.path.exists(CKPT):
    _cm.load(brain, CKPT)
    print(f'Loaded {CKPT}')

env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=3, max_steps=T)
obs, _ = env.reset(seed=SEED)
brain.reset()

logs = {
    'fe_pass1': [], 'fe_pass2': [], 'fe_pass3': [],   # K=3 per-pass FE
    'ai_blend': [],
    'precision': [],    # mean motor precision (proxy: ai_convergence as PE reduction)
    'spontaneity': [],
    'fe_gradient': [],
    'goal': [],
    'total_fe': [],
}

for t in range(T):
    if hasattr(env, 'set_flee_active'):
        env.set_flee_active(brain.current_goal == 1, 0.8 if brain.current_goal == 1 else 0.0)
    inject_sensory(env)
    out = brain.step(obs, {})
    action = np.array([out['turn'], out['speed']], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)

    # K=3 pass FE
    fp = out.get('ai_fe_per_pass', [])
    logs['fe_pass1'].append(fp[0] if len(fp) > 0 else out.get('ai_free_energy', 0.0))
    logs['fe_pass2'].append(fp[1] if len(fp) > 1 else out.get('ai_free_energy', 0.0))
    logs['fe_pass3'].append(fp[2] if len(fp) > 2 else out.get('ai_free_energy', 0.0))

    logs['ai_blend'].append(float(out.get('ai_blend', 0.3)))
    logs['precision'].append(float(out.get('ai_convergence', 0.0)))
    logs['spontaneity'].append(float(out.get('spontaneity', 0.0)))
    logs['fe_gradient'].append(float(out.get('fe_gradient', 0.0)))
    logs['goal'].append(int(out.get('goal', 0)))
    logs['total_fe'].append(float(out.get('total_free_energy', 0.0)))

    if term or trunc:
        print(f'Episode ended at step {t+1}')
        break

steps = len(logs['goal'])
t_ax  = np.arange(steps)
print(f'Collected {steps} steps')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(7.0, 5.5), dpi=300, facecolor='white')
gs = GridSpec(4, 1, figure=fig,
              left=0.10, right=0.97, top=0.95, bottom=0.08,
              hspace=0.55)

axes = [fig.add_subplot(gs[i]) for i in range(4)]
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def panel_label(ax, letter):
    ax.text(-0.07, 1.08, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left')

# Goal color band helper
def add_goal_bands(ax, goals, alpha=0.12):
    for i, g in enumerate(goals):
        ax.axvspan(i - 0.5, i + 0.5, color=GOAL_COLORS[g], alpha=alpha, linewidth=0)

# ── (a) K=3 inference pass FE convergence ──────────────────────────────
ax = axes[0]
panel_label(ax, 'a')
add_goal_bands(ax, logs['goal'], alpha=0.10)

fp1 = np.array(logs['fe_pass1'])
fp2 = np.array(logs['fe_pass2'])
fp3 = np.array(logs['fe_pass3'])

# Stack as [T, 3]; compute median and IQR across passes per step
fp_stack = np.stack([fp1, fp2, fp3], axis=1)
fp_med   = np.median(fp_stack, axis=1)
fp_q1    = np.percentile(fp_stack, 25, axis=1)
fp_q3    = np.percentile(fp_stack, 75, axis=1)

# Grey traces for individual passes
for p_arr, alpha in zip([fp1, fp2, fp3], [0.30, 0.30, 0.30]):
    ax.plot(t_ax, p_arr, color='#bbbbbb', linewidth=0.5, alpha=alpha)

ax.fill_between(t_ax, fp_q1, fp_q3, color='#888888', alpha=0.25, label='IQR')
ax.plot(t_ax, fp_med, color='#333333', linewidth=1.1, label='Median $F$')

delta_fe = float(np.mean(fp1 - fp3))
ax.set_ylabel('Free energy $F$', fontsize=6.5)
ax.set_title(f'Iterative inference convergence (K=3 passes, mean $\\Delta F={delta_fe:.3f}$/step)',
             fontsize=6, pad=2)
ax.legend(fontsize=5.5, loc='upper right', frameon=False)
ax.set_xlim(0, steps - 1)

# ── (b) Adaptive blend α_AI + motor precision ──────────────────────────
ax = axes[1]
panel_label(ax, 'b')
add_goal_bands(ax, logs['goal'], alpha=0.10)

blend = np.array(logs['ai_blend'])
prec  = np.array(logs['precision'])
# Normalise precision to [0,1] for overlay
prec_n = (prec - prec.min()) / (prec.max() - prec.min() + 1e-8)

ax2b = ax.twinx()
ax2b.plot(t_ax, prec_n, color='#2166ac', linewidth=0.8, alpha=0.7, label=r'$\bar{\pi}$ (norm.)')
ax2b.set_ylabel(r'Motor precision $\bar{\pi}$', fontsize=6, color='#2166ac')
ax2b.tick_params(axis='y', labelcolor='#2166ac', labelsize=5.5)
ax2b.spines['top'].set_visible(False)

ax.plot(t_ax, blend, color='#d6604d', linewidth=1.1, label=r'$\alpha_\mathrm{AI}$')
ax.axhline(0.3, color='#aaaaaa', linewidth=0.7, linestyle='--', label='baseline $\\alpha=0.3$')
ax.set_ylabel(r'Blend $\alpha_\mathrm{AI}$', fontsize=6.5)
ax.set_ylim(0, 1)
ax.legend(fontsize=5.5, loc='upper left', frameon=False)
ax.set_xlim(0, steps - 1)

# ── (c) Spontaneity ξ ──────────────────────────────────────────────────
ax = axes[2]
panel_label(ax, 'c')
add_goal_bands(ax, logs['goal'], alpha=0.10)

spont = np.array(logs['spontaneity'])
ax.plot(t_ax, spont, color='#c44e52', linewidth=0.9)
ax.axhline(0.3, color='#c44e52', linewidth=0.7, linestyle='--', alpha=0.6)
ax.fill_between(t_ax, 0.3, np.where(spont > 0.3, spont, 0.3),
                color='#c44e52', alpha=0.20, label='threshold 0.3')
ax.set_ylabel(r'Spontaneity $\xi$', fontsize=6.5)
ax.set_ylim(-0.02, max(0.6, spont.max() + 0.05))
ax.legend(fontsize=5.5, loc='upper right', frameon=False)
ax.set_xlim(0, steps - 1)

# ── (d) dF/dt + goal timeline ──────────────────────────────────────────
ax = axes[3]
panel_label(ax, 'd')
add_goal_bands(ax, logs['goal'], alpha=0.15)

feg = np.array(logs['fe_gradient'])
ax.plot(t_ax, feg, color='#762a83', linewidth=0.9, label=r'$\dot{F}=dF/dt$')
ax.axhline(0.05, color='#762a83', linewidth=0.7, linestyle='--', alpha=0.6, label='penalty threshold 0.05')
ax.fill_between(t_ax, 0.05, np.where(feg > 0.05, feg, 0.05),
                color='#762a83', alpha=0.18)
ax.set_ylabel(r'$\dot{F}$', fontsize=6.5)
ax.set_xlabel('Simulation step', fontsize=6.5)
ax.legend(fontsize=5.5, loc='upper right', frameon=False)
ax.set_xlim(0, steps - 1)

# Goal timeline legend at bottom of panel (d)
goal_handles = [mpatches.Patch(color=GOAL_COLORS[i], alpha=0.6, label=GOAL_NAMES[i])
                for i in range(4)]
ax.legend(handles=goal_handles, loc='lower right', fontsize=5,
          ncol=4, frameon=True, framealpha=0.8, edgecolor='#cccccc',
          handlelength=0.8, borderpad=0.4)

# ── Save ──
out_path = os.path.join(os.path.dirname(__file__), 'fig_apc.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
out_png = out_path.replace('.pdf', '.png')
fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
print(f'Saved: {out_png}')
