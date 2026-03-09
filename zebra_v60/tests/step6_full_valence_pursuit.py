"""
Step 6: Full Valence Pursuit with Cross-Modal Salience
Complete v13.1 equivalent: Retina → Tectum → Precision → Dopamine → BG → OT → CMS loop.

Run: python -m zebra_v60.tests.step6_full_valence_pursuit
Output: plots/v60_step6_full_valence_pursuit.png
"""
import os
import sys
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebra_v60.brain.zebrafish_snn_v60 import ZebrafishSNN_v60
from zebra_v60.brain.dopamine_v60 import DopamineSystem_v60
from zebra_v60.brain.basal_ganglia_v60 import BasalGanglia_v60
from zebra_v60.brain.optic_tectum_v60 import OpticTectum_v60
from zebra_v60.brain.thalamus_v60 import ThalamusRelay_v60
from zebra_v60.brain.device_util import get_device
from zebra_v60.world.world_env import WorldEnv
from zebra_v60.tests.step1_vision_pursuit import make_object_trajectory, compute_retinal_turn


def simulate_audio_free_energy(t, period=80):
    """Simulated auditory free energy: sinusoidal with noise."""
    return 0.3 * abs(math.sin(2 * math.pi * t / period)) + 0.05 * float(torch.randn(1).item())


def run_step6(T=300, amplitude=60.0):
    print("=" * 60)
    print("Step 6: Full Valence Pursuit (CMS)")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN_v60(device=device)
    model.reset()
    dopa_sys = DopamineSystem_v60()
    bg = BasalGanglia_v60(mode="alternating")
    ot = OpticTectum_v60()
    thal = ThalamusRelay_v60()

    world = WorldEnv(xmin=-500, xmax=500, ymin=-500, ymax=500, n_food=0)
    obj_x, obj_y = make_object_trajectory(T, amplitude=amplitude, period=120)

    fish_pos = np.array([0.0, 0.0])
    heading = 0.0

    # History
    F_hist = []
    eye_hist = []
    obj_hist = []
    rpe_hist = []
    dopa_hist = []
    bg_gate_hist = []
    valL_hist = []
    valR_hist = []
    pi_OT_hist = []
    pi_PC_hist = []
    cms_hist = []
    F_audio_hist = []

    prev_oF = None

    for t in range(T):
        world.foods = [(obj_x[t], obj_y[t])]
        effective_heading = heading + ot.eye_pos * 0.5

        with torch.no_grad():
            out = model.forward(fish_pos, effective_heading, world)

        retinal_turn = compute_retinal_turn(out)
        F_visual = model.compute_free_energy()
        F_audio = simulate_audio_free_energy(t)

        # Thalamic cross-modal salience
        cms = thal.step(F_visual, F_audio)

        # Dopamine with CMS-modulated gain
        dopa_sys.beta = thal.modulate_dopamine_gain()
        oL_mean = float(out["oL"].abs().mean())
        oR_mean = float(out["oR"].abs().mean())
        dopa, rpe, valL, valR = dopa_sys.step(F_visual, oL_mean, oR_mean)

        # Amplify valence by retinal asymmetry
        valL_eff = valL - 0.1 * retinal_turn
        valR_eff = valR + 0.1 * retinal_turn

        # BG with CMS-modulated exploration
        bg.noise = thal.modulate_bg_exploration()
        bg_gate = bg.step(valL_eff, valR_eff, dopa, rpe)

        # Optic tectum
        eye_pos = ot.step(valL_eff, valR_eff, F_visual, bg_gate, dopa)

        # Precision update with CMS modulation
        oF = out["oF"]
        if prev_oF is not None:
            error_OT = oF - prev_oF
            model.prec_OT.update_precision(error_OT)
            model.prec_PC.update_precision(torch.tensor([[F_visual]]))
            with torch.no_grad():
                model.prec_OT.gamma.data += 0.01 * (dopa - 0.5)
                model.prec_PC.gamma.data += 0.01 * (dopa - 0.5)
                # CMS modulation: high CMS reduces precision (novelty)
                cms_bias = -0.005 * min(cms, 1.0)
                model.prec_OT.gamma.data += cms_bias
                model.prec_PC.gamma.data += cms_bias
        prev_oF = oF.clone()

        F_hist.append(F_visual)
        eye_hist.append(eye_pos)
        obj_hist.append(obj_y[t] / amplitude)
        rpe_hist.append(rpe)
        dopa_hist.append(dopa)
        bg_gate_hist.append(bg_gate)
        valL_hist.append(valL_eff)
        valR_hist.append(valR_eff)
        pi_OT_hist.append(out["pi_OT"])
        pi_PC_hist.append(out["pi_PC"])
        cms_hist.append(cms)
        F_audio_hist.append(F_audio)

        if t % 50 == 0:
            print(f"  t={t:4d}  F_v={F_visual:.4f}  F_a={F_audio:.3f}  "
                  f"CMS={cms:.3f}  RPE={rpe:+.3f}  DA={dopa:.3f}  "
                  f"BG={bg_gate:+.3f}  eye={eye_pos:+.3f}")

    # Plot — 5-panel matching v13.1 format
    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

    # Panel 1: Free energy + CMS
    axes[0].plot(F_hist, label="F visual", color="steelblue")
    axes[0].plot(F_audio_hist, label="F audio", color="lightblue", alpha=0.7)
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(cms_hist, label="CMS", color="magenta", alpha=0.7, linestyle="--")
    ax0_twin.set_ylabel("CMS", color="magenta")
    axes[0].set_ylabel("Free Energy")
    axes[0].set_title("Step 6: Full Valence Pursuit with CMS")
    axes[0].legend(loc="upper left")
    ax0_twin.legend(loc="upper right")

    # Panel 2: Valence L/R
    axes[1].plot(valL_hist, label="Valence L", color="blue", alpha=0.7)
    axes[1].plot(valR_hist, label="Valence R", color="red", alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Valence L/R")
    axes[1].legend()

    # Panel 3: RPE / Dopamine
    axes[2].plot(rpe_hist, label="RPE", color="red", alpha=0.6)
    axes[2].plot(dopa_hist, label="Dopamine", color="orange")
    axes[2].axhline(0.5, color="black", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("RPE / Dopamine")
    axes[2].legend()

    # Panel 4: BG gate + Eye position
    axes[3].plot(bg_gate_hist, label="BG Gate", color="darkgreen")
    axes[3].plot(eye_hist, label="Eye (OT)", color="coral")
    axes[3].plot(obj_hist, label="Object (norm)", color="gray", alpha=0.5)
    axes[3].set_ylabel("BG Gate / Eye")
    axes[3].legend()

    # Panel 5: Precision
    axes[4].plot(pi_OT_hist, label="Precision OT", color="purple")
    axes[4].plot(pi_PC_hist, label="Precision PC", color="teal")
    axes[4].set_ylabel("Precision")
    axes[4].set_xlabel("Time step")
    axes[4].legend()

    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "plots", "v60_step6_full_valence_pursuit.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved: {save_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Steps run:      {T}")
    print(f"  Final F:        {F_hist[-1]:.4f}")
    print(f"  Mean DA:        {np.mean(dopa_hist):.3f}")
    print(f"  Mean |RPE|:     {np.mean(np.abs(rpe_hist)):.3f}")
    print(f"  Mean CMS:       {np.mean(cms_hist):.4f}")
    print(f"  Eye range:      [{min(eye_hist):.2f}, {max(eye_hist):.2f}]")
    print(f"  NaN in F:       {any(math.isnan(f) for f in F_hist)}")
    print("=" * 60)


if __name__ == "__main__":
    run_step6()
