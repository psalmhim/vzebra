# ============================================================
# SCRIPT: test_modules_dynamics.py
# AUTHOR: HJ Park & GPT-5 (PC-SNN Project)
# VERSION: v13.1b (2025-11-12)
#
# PURPOSE:
#     Multi-step dynamic stability test for PC-SNN modules.
#     Simulates a simplified closed-loop agent for 500 timesteps
#     to visualize convergence, oscillation, and precision evolution.
#
# REASON FOR UPDATE:
#     Added to automatically verify temporal stability and oscillatory
#     regimes after modularization (v13+). Generates diagnostic plots.
#
# OUTPUT:
#     plots/test_dynamics_v13_1b.png
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from modules import (
    RetinaPC, AudioPC, VisualCortexPC, WorkingMemory,
    DopamineSystem, BasalGanglia, OpticTectum, ThalamusRelay
)

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[PC-SNN] Running dynamic stability test on {device}")

    # -------------------------------
    # Initialize modules
    # -------------------------------
    retinaL, retinaR = RetinaPC(device=device), RetinaPC(device=device)
    audio = AudioPC(device=device)
    v1L, v1R = VisualCortexPC(device=device), VisualCortexPC(device=device)
    mem = WorkingMemory(device=device)
    dopa = DopamineSystem()
    bg = BasalGanglia(mode="novelty_driven")
    tectum = OpticTectum()
    thal = ThalamusRelay(device=device)

    # -------------------------------
    # Buffers
    # -------------------------------
    FvL, FvR, Fa = [], [], []
    PiL, PiR, PiA = [], [], []
    valL, valR, RPE, CMS, BG, Eye = [], [], [], [], [], []

    # -------------------------------
    # Loop
    # -------------------------------
    for t in range(500):
        # Randomized sensory input
        x = torch.rand(1, 784, device=device)
        tone = torch.rand(1, 16, device=device)

        # Retina steps
        rL, F_L, Pi_L = retinaL.step(x, rpe=dopa.rpe, valence=dopa.valL)
        rR, F_R, Pi_R = retinaR.step(torch.flip(x, dims=[1]), rpe=dopa.rpe, valence=dopa.valR)

        # Cortex and memory
        zL = v1L(rL, rR)
        zR = v1R(rR, rL)
        latent = (zL + zR) / 2
        m = mem.step(latent)

        # Audio stream
        a_pred, F_A, Pi_A = audio.step(tone, rpe=dopa.rpe, valence=(dopa.valL + dopa.valR) / 2)

        # Cross-modal salience
        cms = thal.step((F_L + F_R) / 2, F_A)

        # Dopamine update
        rewardL = 0.5 if t % 80 < 40 else -0.5
        rewardR = -rewardL
        vL_new, vR_new, rpe = dopa.update(dopa.valL, dopa.valR, rewardL, rewardR, cms=cms)

        # BG gating and eye movement
        bg_gate = bg.step(vL_new, vR_new, dopa.rpe, rpe, cms)
        eye = tectum.step(vL_new, vR_new, (F_L + F_R) / 2, bg_gate, dopa.rpe)

        # Logging
        FvL.append(F_L); FvR.append(F_R); Fa.append(F_A)
        PiL.append(Pi_L); PiR.append(Pi_R); PiA.append(Pi_A)
        valL.append(vL_new); valR.append(vR_new)
        RPE.append(rpe); CMS.append(cms); BG.append(bg_gate); Eye.append(eye)

        if t % 50 == 0:
            print(f"Step {t:03d}: Fv={F_L:.3f}/{F_R:.3f} Fa={F_A:.3f} "
                  f"ΠL={Pi_L:.3f} ΠR={Pi_R:.3f} ΠA={Pi_A:.3f} "
                  f"RPE={rpe:+.3f} CMS={cms:+.3f} BG={bg_gate:+.3f} Eye={eye:+.2f}")

    # -------------------------------
    # Visualization
    # -------------------------------
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(FvL, label="Free Energy L")
    plt.plot(FvR, label="Free Energy R")
    plt.plot(Fa, label="Audio F")
    plt.ylabel("Free Energy")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(PiL, label="ΠL")
    plt.plot(PiR, label="ΠR")
    plt.plot(PiA, label="ΠA")
    plt.plot(valL, label="Val L")
    plt.plot(valR, label="Val R")
    plt.ylabel("Precision / Valence")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(RPE, label="RPE")
    plt.plot(CMS, label="CMS")
    plt.plot(BG, label="BG Gate")
    plt.plot(Eye, label="Eye Position")
    plt.xlabel("Timestep")
    plt.ylabel("Dynamics")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/test_dynamics_v13_1b.png")
    print("[Saved] plots/test_dynamics_v13_1b.png")

if __name__ == "__main__":
    main()
