"""
Generate all paper figures for the v2 spiking zebrafish brain.

Figures:
  Fig 1: Architecture diagram (schematic — text-based)
  Fig 2: Spike raster across all layers during a behavioral episode
  Fig 3: E/I balance — power spectrum and mean firing rates per layer
  Fig 4: Ablation study — remove DA, STDP, PE feedback, measure score drop
  Fig 5: v1 vs v2 decision scenario comparison (bar chart + trajectories)
  Fig 6: Neuromodulation dynamics during flee/forage transitions
  Fig 7: Place cell theta phase precession

Run: python -m zebrav2.tests.generate_paper_figures
"""
import os
import sys
import math
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav2.spec import DEVICE

PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "paper")
os.makedirs(PLOT_DIR, exist_ok=True)

GOAL_NAMES = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
GOAL_COLORS = ["#32c850", "#dc3232", "#6496e6", "#00b4aa"]


def _run_episode(env, brain, T=300, seed=42):
    """Run one episode, collect neural traces."""
    obs, info = env.reset(seed=seed)
    brain.reset()
    np.random.seed(seed)

    traces = {
        'sfgsb_rate': [], 'sfgsd_rate': [], 'sgc_rate': [], 'so_rate': [],
        'tc_rate': [], 'trn_rate': [],
        'pals_rate': [], 'pald_rate': [],
        'd1_rate': [], 'd2_rate': [], 'gpi_rate': [], 'gate': [],
        'cea_rate': [],
        'DA': [], 'NA': [], 'HT5': [], 'ACh': [],
        'goal': [], 'turn': [], 'speed': [],
        'hunger': [], 'fatigue': [], 'stress': [],
        'free_energy': [], 'pred_error_mean': [],
        'pred_dist': [], 'amygdala': [],
        'energy': [], 'place_theta': [],
        'pos_x': [], 'pos_y': [],
    }

    for t in range(T):
        is_flee = (brain.current_goal == 1)
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(is_flee, panic_intensity=0.8 if is_flee else 0.0)
        inject_sensory(env)
        out = brain.step(obs, env)
        action = np.array([out['turn'], out['speed']], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        env._eaten_now = info.get('food_eaten_this_step', 0)

        # Collect — tectum: average L+R hemispheres; thalamus: concatenate L+R
        _sfgsb = 0.5 * (brain.tectum.sfgs_b_L.get_rate_e() + brain.tectum.sfgs_b_R.get_rate_e())
        _sfgsd = 0.5 * (brain.tectum.sfgs_d_L.get_rate_e() + brain.tectum.sfgs_d_R.get_rate_e())
        _sgc   = 0.5 * (brain.tectum.sgc_L.get_rate_e()    + brain.tectum.sgc_R.get_rate_e())
        _so    = 0.5 * (brain.tectum.so_L.get_rate_e()     + brain.tectum.so_R.get_rate_e())
        traces['sfgsb_rate'].append(_sfgsb.detach().cpu().numpy().copy())
        traces['sfgsd_rate'].append(_sfgsd.detach().cpu().numpy().copy())
        traces['sgc_rate'].append(_sgc.detach().cpu().numpy().copy())
        traces['so_rate'].append(_so.detach().cpu().numpy().copy())
        _tc  = torch.cat([brain.thalamus_L.tc_rate,  brain.thalamus_R.tc_rate])
        _trn = torch.cat([brain.thalamus_L.trn_rate, brain.thalamus_R.trn_rate])
        traces['tc_rate'].append(_tc.detach().cpu().numpy().copy())
        traces['trn_rate'].append(_trn.detach().cpu().numpy().copy())
        traces['pals_rate'].append(brain.pallium.rate_s.detach().cpu().numpy().copy())
        traces['pald_rate'].append(brain.pallium.rate_d.detach().cpu().numpy().copy())
        traces['d1_rate'].append(brain.bg.d1_rate.detach().cpu().numpy().copy())
        traces['d2_rate'].append(brain.bg.d2_rate.detach().cpu().numpy().copy())
        traces['gpi_rate'].append(float(brain.bg.gpi_rate.mean()))
        traces['gate'].append(float(brain.bg.gate.item()))
        traces['cea_rate'].append(brain.amygdala.CeA.rate.detach().cpu().numpy().copy())
        traces['DA'].append(out['DA'])
        traces['NA'].append(out['NA'])
        traces['HT5'].append(out['5HT'])
        traces['ACh'].append(out['ACh'])
        traces['goal'].append(out['goal'])
        traces['turn'].append(out['turn'])
        traces['speed'].append(out['speed'])
        traces['hunger'].append(brain.allostasis.hunger)
        traces['fatigue'].append(brain.allostasis.fatigue)
        traces['stress'].append(brain.allostasis.stress)
        traces['free_energy'].append(out['free_energy'])
        traces['pred_error_mean'].append(float((brain.pallium.pred_error ** 2).mean()))
        traces['pred_dist'].append(brain._pred_dist_gt)
        traces['amygdala'].append(brain.amygdala_alpha)
        traces['energy'].append(brain.energy)
        traces['place_theta'].append(brain.place.theta_phase)
        traces['pos_x'].append(getattr(env, 'fish_x', 400))
        traces['pos_y'].append(getattr(env, 'fish_y', 300))

        if terminated or truncated:
            break

    return traces


# =====================================================================
# Fig 2: Spike raster across layers
# =====================================================================
def fig2_spike_raster(traces):
    """Multi-layer spike raster plot."""
    T = len(traces['goal'])
    fig, axes = plt.subplots(8, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={'height_ratios': [2, 2, 1, 1, 2, 2, 1, 1]})

    layers = [
        ('SFGS-b (tectum)', 'sfgsb_rate', 'cyan'),
        ('SFGS-d (tectum)', 'sfgsd_rate', 'deepskyblue'),
        ('SGC (looming)', 'sgc_rate', 'gold'),
        ('SO (direction)', 'so_rate', 'orange'),
        ('Pallium-S', 'pals_rate', 'limegreen'),
        ('Pallium-D', 'pald_rate', 'forestgreen'),
        ('CeA (amygdala)', 'cea_rate', 'red'),
        ('D1 (BG)', 'd1_rate', 'darkorange'),
    ]

    for ax, (name, key, color) in zip(axes, layers):
        data = np.array(traces[key])  # (T, N)
        n_neurons = min(40, data.shape[1])
        step = max(1, data.shape[1] // n_neurons)
        sub = data[:, ::step][:, :n_neurons]
        # Binary raster: threshold at mean + 1 std
        thresh = sub.mean() + sub.std()
        for ni in range(sub.shape[1]):
            spike_times = np.where(sub[:, ni] > max(0.01, thresh))[0]
            ax.scatter(spike_times, np.full_like(spike_times, ni),
                       s=0.5, c=color, marker='|', linewidths=0.5)
        ax.set_ylabel(name, fontsize=7, rotation=0, labelpad=70, va='center')
        ax.set_ylim(-1, n_neurons)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Color goal background
    goals = np.array(traces['goal'])
    for ax in axes:
        for g, c in enumerate(GOAL_COLORS):
            mask = (goals == g)
            for start, end in _contiguous_ranges(mask):
                ax.axvspan(start, end, alpha=0.1, color=c)

    axes[-1].set_xlabel("Step")
    fig.suptitle("Fig 2: Multi-layer spike raster during behavioral episode",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "fig2_spike_raster.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Fig 2 saved: {path}")


def _contiguous_ranges(mask):
    ranges = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))
    return ranges


# =====================================================================
# Fig 3: E/I balance — mean firing rates per layer
# =====================================================================
def fig3_ei_balance(traces):
    T = len(traces['goal'])
    layer_names = ['SFGS-b', 'SFGS-d', 'SGC', 'SO', 'TC', 'TRN',
                   'Pal-S', 'Pal-D', 'D1', 'D2', 'CeA']
    layer_keys = ['sfgsb_rate', 'sfgsd_rate', 'sgc_rate', 'so_rate',
                  'tc_rate', 'trn_rate', 'pals_rate', 'pald_rate',
                  'd1_rate', 'd2_rate', 'cea_rate']

    means = []
    stds = []
    for key in layer_keys:
        data = np.array(traces[key])
        means.append(data.mean())
        stds.append(data.std())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of mean rates
    colors = ['cyan', 'deepskyblue', 'gold', 'orange', 'purple', 'magenta',
              'limegreen', 'forestgreen', 'darkorange', 'royalblue', 'red']
    ax1.barh(layer_names, means, xerr=stds, color=colors, alpha=0.8)
    ax1.set_xlabel("Mean firing rate")
    ax1.set_title("Mean firing rate per layer")
    ax1.invert_yaxis()

    # Time series of key layers
    for key, name, c in [('sfgsb_rate', 'SFGS-b', 'cyan'),
                          ('pals_rate', 'Pal-S', 'limegreen'),
                          ('pald_rate', 'Pal-D', 'forestgreen'),
                          ('d1_rate', 'D1', 'darkorange')]:
        data = np.array(traces[key]).mean(axis=1)
        ax2.plot(data, label=name, color=c, alpha=0.8, linewidth=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean rate")
    ax2.set_title("Layer activity over time")
    ax2.legend(fontsize=8)

    fig.suptitle("Fig 3: E/I balance — firing rates across layers",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "fig3_ei_balance.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Fig 3 saved: {path}")


# =====================================================================
# Fig 5: v1 vs v2 comparison (already generated by step29b, just copy)
# =====================================================================
def fig5_v1_vs_v2():
    src = os.path.join(PROJECT_ROOT, "plots", "v2_vs_v1_decision_scenarios.png")
    if os.path.exists(src):
        import shutil
        dst = os.path.join(PLOT_DIR, "fig5_v1_vs_v2.png")
        shutil.copy2(src, dst)
        print(f"  Fig 5 copied: {dst}")
    else:
        print("  Fig 5 SKIP: run step29b_v1_vs_v2 first")


# =====================================================================
# Fig 6: Neuromodulation dynamics during goal transitions
# =====================================================================
def fig6_neuromod(traces):
    T = len(traces['goal'])
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    # Goal
    goals = np.array(traces['goal'])
    ax = axes[0]
    for g, (name, c) in enumerate(zip(GOAL_NAMES, GOAL_COLORS)):
        mask = (goals == g).astype(float)
        ax.fill_between(range(T), g - 0.4, g + 0.4, where=mask > 0,
                         alpha=0.6, color=c, label=name)
    ax.set_ylabel("Goal")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(GOAL_NAMES, fontsize=8)
    ax.legend(loc='upper right', fontsize=7, ncol=4)

    # Neuromod
    ax = axes[1]
    ax.plot(traces['DA'], label='DA', color='gold', linewidth=0.8)
    ax.plot(traces['NA'], label='NA', color='royalblue', linewidth=0.8)
    ax.plot(traces['HT5'], label='5-HT', color='purple', linewidth=0.8)
    ax.plot(traces['ACh'], label='ACh', color='green', linewidth=0.8)
    ax.set_ylabel("Neuromod")
    ax.legend(fontsize=7, ncol=4)
    ax.set_ylim(-0.05, 1.1)

    # Allostasis
    ax = axes[2]
    ax.plot(traces['hunger'], label='Hunger', color='orange', linewidth=0.8)
    ax.plot(traces['fatigue'], label='Fatigue', color='skyblue', linewidth=0.8)
    ax.plot(traces['stress'], label='Stress', color='red', linewidth=0.8)
    ax.set_ylabel("Allostasis")
    ax.legend(fontsize=7, ncol=3)

    # Amygdala + predator dist
    ax = axes[3]
    ax.plot(traces['amygdala'], label='Amygdala', color='red', linewidth=0.8)
    ax2 = ax.twinx()
    ax2.plot(traces['pred_dist'], label='Pred dist', color='gray',
             linewidth=0.5, alpha=0.6)
    ax2.set_ylabel("Pred dist", fontsize=8)
    ax.set_ylabel("Amygdala")
    ax.legend(loc='upper left', fontsize=7)

    # Free energy + PE
    ax = axes[4]
    ax.plot(traces['free_energy'], label='Free energy', color='navy', linewidth=0.8)
    ax.plot(traces['pred_error_mean'], label='PE²', color='crimson',
            linewidth=0.6, alpha=0.7)
    ax.set_ylabel("F / PE²")
    ax.set_xlabel("Step")
    ax.legend(fontsize=7)

    fig.suptitle("Fig 6: Neuromodulation & allostatic dynamics",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "fig6_neuromod.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Fig 6 saved: {path}")


# =====================================================================
# Fig 7: Place cell activity + trajectory
# =====================================================================
def fig7_place_cells(traces):
    T = len(traces['goal'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Trajectory colored by goal
    px = np.array(traces['pos_x'])
    py = np.array(traces['pos_y'])
    goals = np.array(traces['goal'])
    for g, c in enumerate(GOAL_COLORS):
        mask = goals == g
        ax1.scatter(px[mask], py[mask], c=c, s=2, alpha=0.6,
                    label=GOAL_NAMES[g])
    ax1.plot(px[0], py[0], 'ko', markersize=8, label='Start')
    ax1.plot(px[-1], py[-1], 'ks', markersize=8, label='End')
    ax1.set_xlim(0, 800)
    ax1.set_ylim(0, 600)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title("Trajectory (colored by goal)")
    ax1.legend(fontsize=7, loc='lower right')

    # Theta phase
    ax2.plot(traces['place_theta'], color='purple', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Theta phase (rad)")
    ax2.set_title("Place cell theta oscillation")

    fig.suptitle("Fig 7: Place cell activity & navigation trajectory",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "fig7_place_cells.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Fig 7 saved: {path}")


# =====================================================================
# Fig 4: Ablation study
# =====================================================================
def fig4_ablation():
    """Run decision benchmark with ablations."""
    from zebrav2.tests.step29b_v1_vs_v2 import BrainV2Agent, _setup_scenario, \
        _run_scenario, _score_scenario, V1_SCORES

    scenarios = ["A", "B", "C", "D", "E"]
    conditions = {
        "Full v2": None,
        "No DA (fixed 0.5)": "no_da",
        "No amygdala": "no_amygdala",
        "No allostasis": "no_allostasis",
        "No world model": "no_worldmodel",
    }

    results = {}
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=300,
                                    side_panels=False)

    for cond_name, ablation in conditions.items():
        print(f"  Ablation: {cond_name}...")
        agent = BrainV2Agent()

        # Apply ablation
        if ablation == "no_da":
            agent.brain.neuromod.DA.fill_(0.5)
            orig_update = agent.brain.neuromod.update
            def _frozen_da_update(**kw):
                r = orig_update(**kw)
                agent.brain.neuromod.DA.fill_(0.5)
                return r
            agent.brain.neuromod.update = lambda **kw: _frozen_da_update(**kw)
        elif ablation == "no_amygdala":
            agent.brain.amygdala.forward = lambda **kw: 0.0
        elif ablation == "no_allostasis":
            agent.brain.allostasis.step = lambda **kw: {
                'hunger': 0, 'fatigue': 0, 'stress': 0,
                'hunger_error': 0, 'fatigue_error': 0, 'stress_error': 0,
                'urgency': 0}
            agent.brain.allostasis.get_goal_bias = lambda: np.zeros(4)
        elif ablation == "no_worldmodel":
            agent.brain.world_model.compute_efe_per_goal = lambda *a, **kw: np.zeros(4)

        scores = {}
        for sid in scenarios:
            np.random.seed(42)
            torch.manual_seed(42)
            result = _run_scenario(env, agent, sid, T=200)
            score, _ = _score_scenario(sid, result)
            scores[sid] = score
        results[cond_name] = scores

    env.close()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scenarios))
    width = 0.15
    for i, (cond, scores) in enumerate(results.items()):
        vals = [scores[s] for s in scenarios]
        offset = (i - len(conditions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=cond, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=6)

    # v1 reference line
    v1_vals = [V1_SCORES[s] for s in scenarios]
    ax.plot(x, v1_vals, 'k--', linewidth=1.5, label='v1 baseline', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Scenario {s}" for s in scenarios])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title("Fig 4: Ablation study — contribution of each module")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "fig4_ablation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Fig 4 saved: {path}")

    # Print summary
    print("\n  Ablation summary:")
    for cond, scores in results.items():
        overall = np.mean(list(scores.values()))
        print(f"    {cond:25s}: {overall:.0f}/100")


# =====================================================================
# Fig 8: Hemispheric vision system
# =====================================================================
def fig8_hemispheric_vision():
    """Copy pre-generated hemispheric vision test result into paper figures."""
    src = os.path.join(PROJECT_ROOT, "plots", "v2_hemispheric_vision.png")
    if os.path.exists(src):
        import shutil
        dst = os.path.join(PLOT_DIR, "fig8_hemispheric_vision.png")
        shutil.copy2(src, dst)
        print(f"  Fig 8 copied: {dst}")
    else:
        # Fall back: generate a simple reference panel
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        ax.text(0.5, 0.5,
                "Fig 8: Hemispheric Vision System\n\n"
                "L_tectum ← R_eye (contralateral)\n"
                "R_tectum ← L_eye (contralateral)\n\n"
                "Each hemisphere: SFGS-b / SFGS-d / SGC / SO\n"
                "Thalamus split: TC_L (150) + TC_R (150)\n"
                "See plots/v2_hemispheric_vision.png for full test results",
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        fig.suptitle("Fig 8: Hemispheric Vision System", fontsize=13, fontweight='bold')
        dst = os.path.join(PLOT_DIR, "fig8_hemispheric_vision.png")
        plt.savefig(dst, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Fig 8 saved (reference panel): {dst}")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 60)
    print("Generating paper figures for ZebrafishSNN v2")
    print("=" * 60)

    # Run episode for trace-based figures
    print("\nRunning behavioral episode for neural traces...")
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=400,
                                    side_panels=False)
    brain = ZebrafishBrainV2(device=DEVICE)
    traces = _run_episode(env, brain, T=300, seed=42)
    env.close()
    print(f"  Episode: {len(traces['goal'])} steps collected\n")

    # Generate figures
    fig2_spike_raster(traces)
    fig3_ei_balance(traces)
    fig5_v1_vs_v2()
    fig6_neuromod(traces)
    fig7_place_cells(traces)
    fig8_hemispheric_vision()

    print("\nRunning ablation study (5 conditions × 5 scenarios)...")
    fig4_ablation()

    print(f"\nAll figures saved to: {PLOT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
