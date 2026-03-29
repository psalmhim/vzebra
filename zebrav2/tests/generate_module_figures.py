"""
Generate per-module paper figures: spike rasters, tuning curves, learning curves.

Run: .venv/bin/python -u -m zebrav2.tests.generate_module_figures
"""
import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'v2_paper')
os.makedirs(PLOT_DIR, exist_ok=True)


def fig_izhikevich_cell_types():
    """Fig: All 7 Izhikevich cell type firing patterns."""
    print("  Generating Izhikevich cell types...")
    from zebrav2.brain.neurons import IzhikevichLayer, NEURON_PARAMS

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle('Izhikevich Cell Types (7 types)', fontsize=13, fontweight='bold')
    types = list(NEURON_PARAMS.keys())
    for idx, ct in enumerate(types):
        ax = axes[idx // 4, idx % 4]
        neuron = IzhikevichLayer(1, ct, DEVICE)
        vs = []
        for t in range(200):
            I = torch.tensor([8.0 if 20 < t < 180 else 0.0], device=DEVICE)
            neuron(I)
            vs.append(float(neuron.v[0]))
        ax.plot(vs, 'b-', linewidth=0.8)
        ax.set_title(f'{ct} ({NEURON_PARAMS[ct]})', fontsize=8)
        ax.set_ylim(-90, 40)
        ax.set_xlabel('ms', fontsize=7)
    axes[1, 3].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'fig_cell_types.png'), dpi=200)
    plt.close(fig)


def fig_ei_balance():
    """Fig: E/I balance in tectum layer."""
    print("  Generating E/I balance...")
    from zebrav2.brain.ei_layer import EILayer
    layer = EILayer(200, 'CH', DEVICE, 'Test-EI')
    layer.E.i_tonic.fill_(-2.0)
    layer.I.i_tonic.fill_(0.5)

    e_rates, i_rates, e_spikes, i_spikes = [], [], [], []
    for t in range(100):
        drive = torch.randn(layer.n_e, device=DEVICE) * (3.0 + 2.0 * np.sin(t * 0.1))
        rate_e, rate_i, sp_e, sp_i = layer(drive, substeps=50)
        e_rates.append(float(rate_e.mean()))
        i_rates.append(float(rate_i.mean()))
        e_spikes.append(float(sp_e.sum()))
        i_spikes.append(float(sp_i.sum()))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('E/I Balance Dynamics (200 neurons, CH type)', fontsize=13, fontweight='bold')
    axes[0].plot(e_rates, 'b-', label='E rate', linewidth=1.5)
    axes[0].plot(i_rates, 'r-', label='I rate', linewidth=1.5)
    axes[0].set_title('Mean Firing Rate')
    axes[0].legend()
    axes[1].plot(e_spikes, 'b-', alpha=0.7, label='E spikes')
    axes[1].plot(i_spikes, 'r-', alpha=0.7, label='I spikes')
    axes[1].set_title('Total Spikes per Step')
    axes[1].legend()
    axes[2].scatter(e_rates, i_rates, c=range(len(e_rates)), cmap='viridis', s=10)
    axes[2].set_xlabel('E rate'); axes[2].set_ylabel('I rate')
    axes[2].set_title('E-I Rate Correlation')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'fig_ei_balance.png'), dpi=200)
    plt.close(fig)


def fig_tectum_layers():
    """Fig: Tectum 4-layer activity profiles."""
    print("  Generating tectum layers...")
    from zebrav2.brain.tectum import Tectum
    from zebrav2.brain.retina import RetinaV2
    tect = Tectum(DEVICE)
    ret = RetinaV2(DEVICE)

    sfgs_b_rates, sfgs_d_rates, sgc_rates, so_rates = [], [], [], []
    for t in range(50):
        L = torch.rand(800, device=DEVICE) * (0.5 + 0.3 * np.sin(t * 0.2))
        R = torch.rand(800, device=DEVICE) * (0.5 + 0.3 * np.sin(t * 0.2 + 1))
        rgc = ret(L, R, {'enemy': float(0.3 * np.sin(t * 0.1))})
        out = tect(rgc)
        sfgs_b_rates.append(float(out['sfgs_b'].mean()))
        sfgs_d_rates.append(float(out['sfgs_d'].mean()))
        sgc_rates.append(float(out['sgc'].mean()))
        so_rates.append(float(out['so'].mean()))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    fig.suptitle('Tectum Layer Activity (SFGS-b, SFGS-d, SGC, SO)', fontsize=13, fontweight='bold')
    for ax, rates, name, color in zip(axes,
        [sfgs_b_rates, sfgs_d_rates, sgc_rates, so_rates],
        ['SFGS-b (CH)', 'SFGS-d (CH)', 'SGC (IB)', 'SO (RS)'],
        ['blue', 'green', 'red', 'purple']):
        ax.plot(rates, color=color, linewidth=1.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Step')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'fig_tectum_layers.png'), dpi=200)
    plt.close(fig)


def fig_neuromodulation():
    """Fig: 4-axis neuromodulation dynamics."""
    print("  Generating neuromodulation...")
    from zebrav2.brain.neuromod import NeuromodSystem
    nm = NeuromodSystem(DEVICE)
    DA_vals, NA_vals, HT5_vals, ACh_vals = [], [], [], []

    for t in range(200):
        if t < 50:
            reward, amygdala, cms, flee = 0.01, 0.0, 0.1, False
        elif t < 100:
            reward, amygdala, cms, flee = 0.0, 0.6, 0.5, True
        elif t < 150:
            reward, amygdala, cms, flee = 5.0, 0.1, 0.2, False
        else:
            reward, amygdala, cms, flee = 0.01, 0.0, 0.1, False
        out = nm.update(reward, amygdala, cms, flee, 0.2, 0.7, 0)
        DA_vals.append(out['DA']); NA_vals.append(out['NA'])
        HT5_vals.append(out['5HT']); ACh_vals.append(out['ACh'])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('4-Axis Neuromodulation Dynamics', fontsize=13, fontweight='bold')
    phases = [(0,50,'safe'),(50,100,'threat'),(100,150,'reward'),(150,200,'calm')]
    for ax, vals, name, color in zip(axes.flat,
        [DA_vals, NA_vals, HT5_vals, ACh_vals],
        ['Dopamine (DA)', 'Noradrenaline (NA)', 'Serotonin (5-HT)', 'Acetylcholine (ACh)'],
        ['#f39c12', '#2980b9', '#8e44ad', '#27ae60']):
        ax.plot(vals, color=color, linewidth=1.5)
        ax.set_title(name)
        for s, e, lbl in phases:
            ax.axvspan(s, e, alpha=0.05, color='gray')
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'fig_neuromodulation.png'), dpi=200)
    plt.close(fig)


def fig_amygdala():
    """Fig: Spiking amygdala fear response."""
    print("  Generating amygdala...")
    from zebrav2.brain.amygdala import SpikingAmygdalaV2
    amy = SpikingAmygdalaV2(device=DEVICE)
    threats = []
    for t in range(100):
        enemy_px = 20.0 if 30 < t < 60 else 0.0
        pred_dist = 50.0 if 30 < t < 60 else 300.0
        stress = 0.3 if 30 < t < 60 else 0.0
        alpha = amy(enemy_px, pred_dist, stress, 0.5 if 40 < t < 55 else 0.0)
        threats.append(alpha)

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.suptitle('Spiking Amygdala: Fear Response', fontsize=13, fontweight='bold')
    ax.plot(threats, 'r-', linewidth=2)
    ax.axvspan(30, 60, alpha=0.1, color='red', label='predator present')
    ax.set_xlabel('Step'); ax.set_ylabel('Threat Arousal')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'fig_amygdala.png'), dpi=200)
    plt.close(fig)


def fig_cerebellum():
    """Fig: Cerebellum GC sparsity + parallel fiber LTD."""
    print("  Generating cerebellum...")
    from zebrav2.brain.cerebellum import SpikingCerebellum
    cb = SpikingCerebellum(device=DEVICE)
    gc_sparse, pf_weights = [], []
    for t in range(80):
        mossy = torch.rand(128, device=DEVICE) * 0.5
        cf = max(0.0, 0.5 - t * 0.005)
        out = cb(mossy, cf, 0.5)
        gc_sparse.append(out['gc_sparsity'])
        pf_weights.append(float(cb.W_pf.data.mean()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle('Cerebellum: Sparse Coding + LTD Learning', fontsize=13, fontweight='bold')
    axes[0].plot(gc_sparse, 'b-', linewidth=1.5)
    axes[0].set_title('Granule Cell Sparsity'); axes[0].set_xlabel('Step')
    axes[1].plot(pf_weights, 'r-', linewidth=1.5)
    axes[1].set_title('Parallel Fiber Weight (LTD)'); axes[1].set_xlabel('Step')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'fig_cerebellum.png'), dpi=200)
    plt.close(fig)


def fig_interoception():
    """Fig: Heart rate + valence dynamics."""
    print("  Generating interoception...")
    from zebrav2.brain.interoception import SpikingInsularCortex
    insula = SpikingInsularCortex(device=DEVICE)
    hrs, vals, arousals = [], [], []
    for t in range(120):
        if t < 40: e, s, f, r, ta = max(20, 100-t*1.5), 0.1, t*0.005, 0.01, False
        elif t < 80: e, s, f, r, ta = max(10, 40-(t-40)*0.5), 0.7, 0.4, 0.0, True
        else: e, s, f, r, ta = min(80, 10+(t-80)*1.5), max(0.1, 0.7-(t-80)*0.015), max(0.1, 0.4-(t-80)*0.007), 0.3, False
        out = insula(e, s, f, r, ta)
        hrs.append(out['heart_rate']); vals.append(out['valence']); arousals.append(out['arousal'])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle('Interoception: Heart Rate, Valence, Arousal', fontsize=13, fontweight='bold')
    axes[0].plot(hrs, 'r-', linewidth=1.5); axes[0].set_title('Heart Rate (Hz)')
    axes[0].axvspan(40, 80, alpha=0.1, color='red')
    axes[1].plot(vals, 'purple', linewidth=1.5); axes[1].set_title('Emotional Valence')
    axes[1].axhline(0, color='k', ls='--')
    axes[2].plot(arousals, 'orange', linewidth=1.5); axes[2].set_title('Arousal')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'fig_interoception.png'), dpi=200)
    plt.close(fig)


def fig_place_cells():
    """Fig: Place cell field tiling + theta oscillation."""
    print("  Generating place cells...")
    from zebrav2.brain.place_cells import ThetaPlaceCells
    pc = ThetaPlaceCells(device=DEVICE)
    # Generate activation map
    xs = np.linspace(50, 750, 40)
    ys = np.linspace(50, 550, 30)
    act_map = np.zeros((30, 40))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            act = pc.activation(x, y)
            act_map[i, j] = float(act.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Place Cells: Spatial Tiling + Theta Phase', fontsize=13, fontweight='bold')
    im = axes[0].imshow(act_map, extent=[50,750,550,50], aspect='auto', cmap='hot')
    axes[0].set_title('Max Place Field Activation')
    axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
    plt.colorbar(im, ax=axes[0])
    # Theta phase over time
    phases = []
    for t in range(100):
        out = pc(400 + 100 * np.sin(t * 0.05), 300 + 80 * np.cos(t * 0.03))
        phases.append(out['theta_phase'])
    axes[1].plot(phases, 'b-', linewidth=1.5)
    axes[1].set_title('Theta Phase (8 Hz)'); axes[1].set_xlabel('Step')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'fig_place_cells.png'), dpi=200)
    plt.close(fig)


def fig_cpg():
    """Fig: CPG half-centre oscillation."""
    print("  Generating CPG...")
    from zebrav2.brain.spinal_cpg import SpinalCPG
    cpg = SpinalCPG(device=DEVICE)
    mLs, mRs = [], []
    for t in range(200):
        drive = min(1.0, 0.2 + t * 0.004)
        for _ in range(5):
            mL, mR, _, _, _ = cpg.step(drive, 0.3 * np.sin(t * 0.03))
        mLs.append(mL); mRs.append(mR)

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.suptitle('Spinal CPG: L/R Motor Neuron Alternation', fontsize=13, fontweight='bold')
    ax.plot(mLs, 'b-', label='Left MN', linewidth=1.5)
    ax.plot(mRs, 'r-', label='Right MN', linewidth=1.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Motor Output')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'fig_cpg.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    t0 = time.time()
    print(f"Device: {DEVICE}")
    print("Generating per-module paper figures...")

    fig_izhikevich_cell_types()
    fig_ei_balance()
    fig_tectum_layers()
    fig_neuromodulation()
    fig_amygdala()
    fig_cerebellum()
    fig_interoception()
    fig_place_cells()
    fig_cpg()

    print(f"\n9 figures saved to {PLOT_DIR}/")
    print(f"Time: {time.time()-t0:.1f}s")
