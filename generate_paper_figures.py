"""
Generate individual figures for each ZebrafishBrainV2 module phase.
Saves to plots/paper/ at dpi=200.
"""
import sys
import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
OUTDIR = 'plots/paper'
os.makedirs(OUTDIR, exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────────

def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


# ============================================================
# Phase 1 — Izhikevich voltage traces (RS, FS, IB)
# ============================================================
def fig_phase1_izhikevich():
    from zebrav2.brain.neurons import IzhikevichLayer, NEURON_PARAMS
    from zebrav2.spec import DEVICE

    STEPS = 200          # 200 ms
    I_CONST = 10.0       # pA constant drive
    cell_types = ['RS', 'FS', 'IB']
    titles     = ['Regular Spiking (RS)', 'Fast Spiking (FS)', 'Intrinsic Bursting (IB)']
    colors     = ['#2264C8', '#C83232', '#1a9950']

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    t = np.arange(STEPS)

    for ax, ctype, title, col in zip(axes, cell_types, titles, colors):
        layer = IzhikevichLayer(1, ctype, DEVICE)
        layer.i_tonic.zero_()              # use only the constant drive
        vs = []
        for _ in range(STEPS):
            I = torch.tensor([I_CONST], device=DEVICE)
            layer(I)
            vs.append(layer.v[0].item())
        vs = np.array(vs)
        # mark spikes (reset events visible as negative jumps)
        ax.plot(t, vs, color=col, lw=0.8)
        # highlight spikes
        spk = np.where(np.diff(vs, prepend=vs[0]) < -40)[0]
        ax.scatter(spk, np.full_like(spk, 30, dtype=float), color=col,
                   marker='|', s=60, zorder=5)
        ax.set_ylabel('V (mV)', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylim(-90, 40)
        ax.axhline(30, ls='--', lw=0.5, color='gray')
        ax.spines[['top', 'right']].set_visible(False)

    axes[-1].set_xlabel('Time (ms)', fontsize=8)
    fig.suptitle('Phase 1 — Izhikevich Neuron Voltage Traces (I = 10 pA)', fontsize=10)
    fig.tight_layout()
    savefig(fig, 'fig_phase1_izhikevich.png')


# ============================================================
# Phase 2 — E vs I firing rates in one tectum E/I layer
# ============================================================
def fig_phase2_ei_layer():
    from zebrav2.brain.ei_layer import EILayer
    from zebrav2.spec import DEVICE, N_OT_SFGS_B

    n_total = min(N_OT_SFGS_B, 200)   # smaller slice for speed
    layer = EILayer(n_total, 'CH', DEVICE, 'SFGS-b')
    # subthreshold tonic (matches tectum setup)
    layer.E.i_tonic.fill_(-2.0)
    layer.I.i_tonic.fill_(0.5)

    STEPS = 100
    e_rates, i_rates = [], []
    rng = np.random.default_rng(42)

    for step in range(STEPS):
        # Oscillating drive: burst every 20 steps
        drive_strength = 3.0 + 4.0 * float(np.sin(step * 0.4) > 0.5)
        I_ext = torch.tensor(
            rng.normal(drive_strength, 1.0, layer.n_e).clip(-1, 10),
            dtype=torch.float32, device=DEVICE)
        r_e, r_i, _, _ = layer(I_ext, substeps=10)
        e_rates.append(r_e.mean().item())
        i_rates.append(r_i.mean().item())

    t = np.arange(STEPS)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, e_rates, color='#2264C8', lw=1.2, label='E (excitatory)')
    ax.plot(t, i_rates, color='#C83232', lw=1.2, label='I (inhibitory)')
    ax.set_xlabel('Behavioral step', fontsize=9)
    ax.set_ylabel('Mean firing rate', fontsize=9)
    ax.set_title('Phase 2 — E/I Balance: Tectum SFGS-b Layer', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    savefig(fig, 'fig_phase2_ei_layer.png')


# ============================================================
# Phase 3 — Tectum 4-layer activity heatmap (100 steps)
# ============================================================
def fig_phase3_tectum():
    from zebrav2.brain.tectum import Tectum
    from zebrav2.brain.retina import RetinaV2
    from zebrav2.spec import DEVICE

    retina = RetinaV2(DEVICE)
    tectum = Tectum(DEVICE)
    STEPS = 100
    rng = np.random.default_rng(7)
    layer_names = ['SFGS-b', 'SFGS-d', 'SGC', 'SO']

    # store per-step mean rates
    hmap = np.zeros((4, STEPS))

    for t in range(STEPS):
        # Synthetic retinal input (random with occasional looming events)
        L = torch.zeros(800, device=DEVICE)
        R = torch.zeros(800, device=DEVICE)
        if rng.random() < 0.15:          # 15% looming event
            L[:100] = torch.tensor(rng.uniform(0.3, 1.0, 100), dtype=torch.float32, device=DEVICE)
        else:
            L[:400] = torch.tensor(rng.uniform(0, 0.3, 400), dtype=torch.float32, device=DEVICE)
        R[:400] = torch.tensor(rng.uniform(0, 0.3, 400), dtype=torch.float32, device=DEVICE)
        entity_info = {'enemy': rng.uniform(0, 0.2)}
        rgc = retina(L, R, entity_info)
        out = tectum(rgc)
        hmap[0, t] = out['sfgs_b'].mean().item()
        hmap[1, t] = out['sfgs_d'].mean().item()
        hmap[2, t] = out['sgc'].mean().item()
        hmap[3, t] = out['so'].mean().item()

    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    colors_map = ['Blues', 'Oranges', 'Reds', 'Greens']
    for i, (ax, lname, cmap) in enumerate(zip(axes, layer_names, colors_map)):
        img = ax.imshow(hmap[i:i+1], aspect='auto', cmap=cmap,
                        extent=[0, STEPS, 0, 1], vmin=0, vmax=hmap[i].max()+1e-6)
        ax.set_yticks([])
        ax.set_ylabel(lname, fontsize=8, rotation=0, labelpad=45, va='center')
        plt.colorbar(img, ax=ax, shrink=0.8, pad=0.01)
        ax.spines[['top', 'right', 'left']].set_visible(False)

    axes[-1].set_xlabel('Time step', fontsize=9)
    fig.suptitle('Phase 3 — Tectum 4-Layer Activity Heatmap (100 steps)', fontsize=10)
    fig.tight_layout()
    savefig(fig, 'fig_phase3_tectum.png')


# ============================================================
# Phase 4 — Basal Ganglia: D1, D2, GPi, Gate time series
# ============================================================
def fig_phase4_bg():
    from zebrav2.brain.basal_ganglia import BasalGanglia
    from zebrav2.spec import DEVICE, N_PAL_D

    bg = BasalGanglia(DEVICE)
    STEPS = 100
    rng = np.random.default_rng(13)
    pal_d_n_e = int(0.75 * N_PAL_D)
    traces = {'D1': [], 'D2': [], 'GPi': [], 'Gate': []}
    DA_series = 0.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, STEPS))

    for t in range(STEPS):
        pal_rate = torch.tensor(
            rng.uniform(0, 0.1, N_PAL_D).astype(np.float32), device=DEVICE)
        DA = float(DA_series[t])
        out = bg(pal_rate, DA)
        traces['D1'].append(out['D1'].mean().item())
        traces['D2'].append(out['D2'].mean().item())
        traces['GPi'].append(out['GPi'].mean().item())
        traces['Gate'].append(float(out['gate']))

    t = np.arange(STEPS)
    fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    kw = dict(lw=1.1)
    axes[0].plot(t, traces['D1'],  color='#2264C8', **kw); axes[0].set_ylabel('D1 rate', fontsize=8)
    axes[1].plot(t, traces['D2'],  color='#C83232', **kw); axes[1].set_ylabel('D2 rate', fontsize=8)
    axes[2].plot(t, traces['GPi'], color='#9932CC', **kw); axes[2].set_ylabel('GPi rate', fontsize=8)
    axes[3].plot(t, traces['Gate'],color='#1a9950', **kw); axes[3].set_ylabel('Gate', fontsize=8)
    axes[3].set_ylim(0, 1)
    axes[3].set_xlabel('Time step', fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    # show DA as background shading on gate panel
    axes[3].fill_between(t, 0, DA_series * 0.5, alpha=0.15, color='orange', label='DA (scaled)')
    axes[3].legend(fontsize=7, loc='upper right')
    fig.suptitle('Phase 4 — Basal Ganglia Circuit: D1 / D2 / GPi / Gate', fontsize=10)
    fig.tight_layout()
    savefig(fig, 'fig_phase4_bg.png')


# ============================================================
# Phase 5 — Neuromodulation: DA, NA, 5-HT, ACh (100 steps)
# ============================================================
def fig_phase5_neuromod():
    from zebrav2.brain.neuromod import NeuromodSystem
    from zebrav2.spec import DEVICE

    nm = NeuromodSystem(DEVICE)
    STEPS = 100
    rng = np.random.default_rng(99)
    traces = {k: [] for k in ['DA', 'NA', '5HT', 'ACh']}

    for t in range(STEPS):
        reward = float(rng.choice([0.01, 10.0], p=[0.9, 0.1]))
        ama = float(rng.uniform(0, 0.3))
        flee = rng.random() < 0.2
        out = nm.update(reward=reward, amygdala_alpha=ama, cms=ama*0.3,
                        flee_active=flee, fatigue=0.1, circadian=0.7,
                        current_goal=2)
        traces['DA'].append(out['DA'])
        traces['NA'].append(out['NA'])
        traces['5HT'].append(out['5HT'])
        traces['ACh'].append(out['ACh'])

    t_arr = np.arange(STEPS)
    cols = {'DA': '#E06C00', 'NA': '#2264C8', '5HT': '#1a9950', 'ACh': '#9932CC'}
    labels = {'DA': 'Dopamine (DA)', 'NA': 'Noradrenaline (NA)',
              '5HT': 'Serotonin (5-HT)', 'ACh': 'Acetylcholine (ACh)'}
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in ['DA', 'NA', '5HT', 'ACh']:
        ax.plot(t_arr, traces[key], color=cols[key], lw=1.2, label=labels[key])
    ax.set_xlabel('Behavioral step', fontsize=9)
    ax.set_ylabel('Neuromodulator level', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncols=2)
    ax.set_title('Phase 5 — Four-Axis Neuromodulation over 100 Steps', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    savefig(fig, 'fig_phase5_neuromod.png')


# ============================================================
# Phase 6 — Thalamus + Pallium: TC, TRN, Pal-S E, Pal-D E
# ============================================================
def fig_phase6_thalamus_pallium():
    from zebrav2.brain.thalamus import Thalamus
    from zebrav2.brain.pallium import Pallium
    from zebrav2.spec import DEVICE, N_TC, N_TRN, N_OT_SFGS_B, N_PAL_S, N_PAL_D

    thalamus = Thalamus(DEVICE)
    pallium  = Pallium(DEVICE)
    STEPS = 100
    rng = np.random.default_rng(55)
    sfgs_b_n_e = int(0.75 * N_OT_SFGS_B)
    pal_s_n_e  = int(0.75 * N_PAL_S)

    traces = {k: [] for k in ['TC', 'TRN', 'PalS', 'PalD']}

    for t in range(STEPS):
        # Varying tectum drive
        drive = float(rng.uniform(0.0, 0.15))
        tect_rate = torch.tensor(
            rng.exponential(drive, sfgs_b_n_e).clip(0, 0.5).astype(np.float32), device=DEVICE)
        NA = float(rng.uniform(0.2, 0.6))
        tc_out = thalamus(tect_rate, pallium.rate_s, NA)

        goal_t = torch.zeros(4, device=DEVICE); goal_t[2] = 1.0  # EXPLORE
        pal_out = pallium(tc_out['TC'], goal_t, ACh_level=0.5)

        traces['TC'].append(tc_out['TC'].mean().item())
        traces['TRN'].append(tc_out['TRN'].mean().item())
        traces['PalS'].append(pal_out['rate_S'].mean().item())
        traces['PalD'].append(pal_out['rate_D'].mean().item())

    t_arr = np.arange(STEPS)
    fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    meta = [('TC',   'TC relay rate',    '#2264C8'),
            ('TRN',  'TRN gate rate',    '#C83232'),
            ('PalS', 'Pallium-S E rate', '#1a9950'),
            ('PalD', 'Pallium-D E rate', '#9932CC')]
    for ax, (key, label, col) in zip(axes, meta):
        ax.plot(t_arr, traces[key], color=col, lw=1.0)
        ax.set_ylabel(label, fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel('Behavioral step', fontsize=9)
    fig.suptitle('Phase 6 — Thalamo-Pallial Loop: TC / TRN / Pal-S / Pal-D', fontsize=10)
    fig.tight_layout()
    savefig(fig, 'fig_phase6_thalamus_pallium.png')


# ============================================================
# Phase 7 — STDP: eligibility trace magnitude and PE over time
# ============================================================
def fig_phase7_stdp():
    from zebrav2.brain.plasticity import EligibilitySTDP
    from zebrav2.spec import DEVICE

    n_pre, n_post = 64, 64
    W = torch.nn.Parameter(torch.randn(n_post, n_pre, device=DEVICE) * 0.1)
    stdp = EligibilitySTDP(W, device=DEVICE)

    STEPS = 100
    rng = np.random.default_rng(3)
    e_trace_mag, pe_series = [], []

    # Simulate: random pre/post spike trains; reward at steps 30 and 70
    for t in range(STEPS):
        pre  = torch.tensor((rng.random(n_pre)  < 0.05).astype(np.float32), device=DEVICE)
        post = torch.tensor((rng.random(n_post) < 0.05).astype(np.float32), device=DEVICE)
        stdp.update_traces(pre, post)
        # Prediction error: scalar proxy (apical - somatic mismatch)
        pe = float((post - 0.05).abs().mean().item())
        pe_series.append(pe)
        e_trace_mag.append(stdp.e_trace.abs().mean().item())
        # Consolidate at reward events
        if t in (30, 70):
            stdp.consolidate(DA=1.0, ACh=0.8)

    t_arr = np.arange(STEPS)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    ax1.plot(t_arr, e_trace_mag, color='#E06C00', lw=1.2)
    ax1.set_ylabel('|Eligibility trace|', fontsize=8)
    ax1.axvline(30, ls='--', lw=0.8, color='gray', label='DA reward')
    ax1.axvline(70, ls='--', lw=0.8, color='gray')
    ax1.legend(fontsize=7)
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.plot(t_arr, pe_series, color='#C83232', lw=1.2)
    ax2.set_ylabel('Prediction error', fontsize=8)
    ax2.set_xlabel('Time step', fontsize=9)
    ax2.axvline(30, ls='--', lw=0.8, color='gray')
    ax2.axvline(70, ls='--', lw=0.8, color='gray')
    ax2.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Phase 7 — Three-Factor STDP: Eligibility Trace and Prediction Error', fontsize=10)
    fig.tight_layout()
    savefig(fig, 'fig_phase7_stdp.png')


# ============================================================
# Phase 8 — Goal probabilities over 100 steps (4 colored bands)
# ============================================================
def fig_phase8_goals():
    """Run a lightweight stand-alone goal simulation without full brain."""
    STEPS = 100
    rng = np.random.default_rng(21)

    # Simulate EFE goal probabilities with simple rules
    goal_names = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']
    goal_cols  = ['#2264C8', '#C83232', '#1a9950', '#9932CC']
    probs_log  = np.zeros((4, STEPS))

    energy = 100.0
    G = np.array([0.25, 0.25, 0.25, 0.25])
    for t in range(STEPS):
        energy -= 0.5
        starvation = max(0, (50 - energy) / 50)
        enemy = float(rng.random() < 0.1)
        food  = float(rng.random() < 0.15)
        # Simple EFE heuristic
        G_f = -0.8 * food  - 1.5 * starvation + rng.normal(0, 0.05)
        G_flee = -0.8 * enemy + rng.normal(0, 0.05)
        G_exp  = 0.2 + rng.normal(0, 0.05)
        G_soc  = 0.25 + rng.normal(0, 0.05)
        raw = np.array([G_f, G_flee, G_exp, G_soc])
        # Softmax (lower EFE = higher probability)
        neg = -(raw - raw.min())
        exp_neg = np.exp(neg - neg.max())
        p = exp_neg / exp_neg.sum()
        probs_log[:, t] = p
        if food and p[0] > 0.3:
            energy = min(100, energy + 10)  # ate food

    fig, ax = plt.subplots(figsize=(8, 4))
    # Stacked area chart
    ax.stackplot(np.arange(STEPS), probs_log,
                 labels=goal_names, colors=goal_cols, alpha=0.75)
    ax.set_xlabel('Behavioral step', fontsize=9)
    ax.set_ylabel('Goal probability (stacked)', fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='upper right', ncols=2)
    ax.set_title('Phase 8 — Active Inference Goal Probabilities over 100 Steps', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    savefig(fig, 'fig_phase8_goals.png')


# ============================================================
# Phase 9 — Place cell activation map with trajectory
# ============================================================
def fig_phase9_place_cells():
    from zebrav2.brain.place_cells import ThetaPlaceCells
    from zebrav2.spec import DEVICE

    ARENA_W, ARENA_H = 800, 600
    place = ThetaPlaceCells(n_cells=128, arena_w=ARENA_W, arena_h=ARENA_H,
                             sigma_px=60.0, device=DEVICE)

    STEPS = 200
    rng = np.random.default_rng(77)
    # Lemniscate trajectory (figure-8 pattern)
    t_vals = np.linspace(0, 2 * np.pi, STEPS)
    traj_x = ARENA_W / 2 + 280 * np.cos(t_vals) / (1 + np.sin(t_vals)**2)
    traj_y = ARENA_H / 2 + 140 * np.sin(t_vals) * np.cos(t_vals) / (1 + np.sin(t_vals)**2)

    # Accumulate activation map
    act_map = np.zeros((ARENA_H, ARENA_W))
    for i in range(STEPS):
        px, py = float(traj_x[i]), float(traj_y[i])
        out = place(px, py, food_eaten=(rng.random() < 0.05))
        # Stamp Gaussian blobs at active cells
        rate = out['rate'].cpu().numpy()
        cx = place.cx.cpu().numpy()
        cy = place.cy.cpu().numpy()
        for j in range(place.n_cells):
            if rate[j] > 0.1:
                xi, yi = int(cx[j]), int(cy[j])
                xi = max(0, min(ARENA_W - 1, xi))
                yi = max(0, min(ARENA_H - 1, yi))
                act_map[yi, xi] += rate[j]

    # Smooth with a simple 2D Gaussian kernel (no scipy required)
    def gaussian_filter2d(arr, sigma=8, ksize=None):
        if ksize is None:
            ksize = int(4 * sigma + 1) | 1   # odd
        k = np.arange(ksize) - ksize // 2
        kernel1d = np.exp(-0.5 * (k / sigma) ** 2)
        kernel1d /= kernel1d.sum()
        # Separable convolution via numpy
        out = np.apply_along_axis(lambda r: np.convolve(r, kernel1d, mode='same'), 1, arr)
        out = np.apply_along_axis(lambda c: np.convolve(c, kernel1d, mode='same'), 0, out)
        return out
    act_map = gaussian_filter2d(act_map, sigma=8)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(act_map, origin='lower', cmap='hot',
                   extent=[0, ARENA_W, 0, ARENA_H], aspect='auto')
    ax.plot(traj_x, traj_y, 'w-', lw=0.8, alpha=0.5, label='Trajectory')
    ax.plot(traj_x[0], traj_y[0], 'go', ms=6, label='Start')
    ax.plot(traj_x[-1], traj_y[-1], 'bs', ms=6, label='End')
    plt.colorbar(im, ax=ax, label='Accumulated place cell activation')
    ax.set_xlabel('x (px)', fontsize=9)
    ax.set_ylabel('y (px)', fontsize=9)
    ax.set_title('Phase 9 — Theta-Phase Place Cell Activation Map + Trajectory', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    savefig(fig, 'fig_phase9_place_cells.png')


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print('=== Generating Phase Figures ===')
    print('Phase 1: Izhikevich traces...')
    fig_phase1_izhikevich()

    print('Phase 2: E/I layer...')
    fig_phase2_ei_layer()

    print('Phase 3: Tectum heatmap...')
    fig_phase3_tectum()

    print('Phase 4: Basal ganglia...')
    fig_phase4_bg()

    print('Phase 5: Neuromodulation...')
    fig_phase5_neuromod()

    print('Phase 6: Thalamus + Pallium...')
    fig_phase6_thalamus_pallium()

    print('Phase 7: STDP...')
    fig_phase7_stdp()

    print('Phase 8: Goal probabilities...')
    fig_phase8_goals()

    print('Phase 9: Place cells...')
    fig_phase9_place_cells()

    print('=== All figures generated ===')
    for f in sorted(os.listdir('plots/paper')):
        if f.startswith('fig_phase'):
            print(f'  plots/paper/{f}')
