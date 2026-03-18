# ============================================================
# FILE: pcsnn_v13_2_multisensory.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v13.2 (2025-11-12)
#
# PURPOSE:
#   Multisensory coherence pursuit model in the PC-SNN hierarchy.
#   Extends the v13.1 valence pursuit agent by adding:
#     (1) Auditory predictive coding (AudioPC)
#     (2) Thalamic cross-modal salience (ThalamusRelay)
#   CMS (cross-modal salience) modulates dopamine and BG gating,
#   producing integrated audio–visual dynamics.
#
# BACKWARD COMPATIBILITY:
#   - RetinaPC, CortexMemory, DopamineSystem, BasalGanglia, OpticTectum
#     interfaces unchanged.
#   - AudioPC uses (tone, rpe, valence, precision_bias).
#   - ThalamusRelay returns scalar CMS; does not alter DopamineSystem API.
# ============================================================

import os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Import modules (must follow v13.2 module interface)
# ------------------------------------------------------------
from modules.retina_pc import RetinaPC
from modules.cortex_memory import VisualCortexPC, WorkingMemory
from modules.dopamine_system import DopamineSystem
from modules.basal_ganglia import BasalGanglia
from modules.optic_tectum import OpticTectum
from modules.audio_pc import AudioPC
from modules.thalamus_relay import ThalamusRelay


# ============================================================
# MULTISENSORY AGENT
# ============================================================
class PCSNNAI_Agent_Multi:
    def __init__(self, device="cpu", mode="alternating"):
        self.device = device

        # --- Sensory layers ---
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.audio = AudioPC(device=device)

        # --- Cortical integration and memory ---
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(device=device)

        # --- Integrative subsystems ---
        self.thalamus = ThalamusRelay()
        self.dopa = DopamineSystem()
        self.bg = BasalGanglia(mode=mode)
        self.tectum = OpticTectum()

    # --------------------------------------------------------
    def perceive_and_act(self, img, label, t):
        """
        Run one multisensory perception–action cycle.
        Returns free energies, precisions, valence states, RPE, CMS, BG, Eye.
        """

        # ====================================================
        # (1) Visual prediction (L/R retinas)
        # ====================================================
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])

        rL, FvL, PiL = self.retinaL.step(xL, rpe=self.dopa.rpe, valence=self.dopa.valL)
        rR, FvR, PiR = self.retinaR.step(xR, rpe=self.dopa.rpe, valence=self.dopa.valR)

        v1L = self.v1L(rL, rR)
        v1R = self.v1R(rR, rL)
        latent = (v1L + v1R) / 2
        mem_state = self.mem.step(latent)

        # ====================================================
        # (2) Auditory predictive coding
        # ====================================================
        tone = torch.rand(1, 16, device=self.device)
        a_pred, Fa, PiA = self.audio.step(
            tone,
            rpe=self.dopa.rpe,
            valence=(self.dopa.valL + self.dopa.valR) / 2,
            precision_bias=1.0,
        )

        # ====================================================
        # (3) Thalamic cross-modal salience (visual–audio coherence)
        # ====================================================
        CMS = self.thalamus.step(Fv=(FvL + FvR) / 2, Fa=Fa)

        # ====================================================
        # (4) Dopaminergic update
        # ====================================================
        def val_map(pred):
            if pred <= 2: return +0.5
            elif pred >= 7: return -0.5
            else: return 0.0
        rewardL, rewardR = val_map(label % 10), val_map((label + 5) % 10)

        valL, valR, rpe = self.dopa.update(
            pred_val_L=self.dopa.valL,
            pred_val_R=self.dopa.valR,
            reward_L=rewardL,
            reward_R=rewardR,
            cms=CMS,
        )

        # ====================================================
        # (5) Basal ganglia & eye control (saccade dynamics)
        # ====================================================
        bg_gate = self.bg.step(valL, valR, dopa=self.dopa.rpe, rpe=rpe, cms=CMS)
        eye = self.tectum.step(
            valL, valR, Fmean=(FvL + FvR) / 2,
            bg_gate=bg_gate, dopa=self.dopa.rpe
        )

        return (
            FvL, FvR, Fa,
            PiL, PiR, PiA,
            valL, valR, rpe,
            eye, bg_gate, self.dopa.rpe, CMS
        )


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================
def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    # --- Dataset ---
    data = datasets.MNIST("./data", train=False, download=True,
                          transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agent = PCSNNAI_Agent_Multi(device=device, mode="alternating")

    # --- Storage ---
    FvL_list, FvR_list, Fa_list = [], [], []
    PiL_list, PiR_list, PiA_list = [], [], []
    vL_list, vR_list, RPE_list, CMS_list, Eye_list, BG_list = [], [], [], [], [], []

    # ========================================================
    # SIMULATION
    # ========================================================
    for t, (img, label) in enumerate(loader):
        fL, fR, fA, piL, piR, piA, valL, valR, rpe, eye, bg, dopa, cms = \
            agent.perceive_and_act(img, label.item(), t)

        if t % 50 == 0:
            print(
                f"Step {t:03d}: "
                f"Fv={fL:.3f}/{fR:.3f} Fa={fA:.3f} "
                f"ΠL={piL:.3f} ΠR={piR:.3f} ΠA={piA:.3f} "
                f"vL={valL:+.2f} vR={valR:+.2f} "
                f"RPE={rpe:+.3f} CMS={cms:+.3f} "
                f"BG={bg:+.2f} Eye={eye:+.2f}"
            )

        # Store metrics
        FvL_list.append(fL); FvR_list.append(fR); Fa_list.append(fA)
        PiL_list.append(piL); PiR_list.append(piR); PiA_list.append(piA)
        vL_list.append(valL); vR_list.append(valR)
        RPE_list.append(rpe); CMS_list.append(cms)
        Eye_list.append(eye); BG_list.append(bg)

        if t >= 300:
            break

    # ========================================================
    # PLOT RESULTS
    # ========================================================
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(FvL_list, label="Fv Left")
    plt.plot(FvR_list, label="Fv Right")
    plt.plot(Fa_list, "--", label="Fa Audio")
    plt.legend(); plt.ylabel("Free Energy")

    plt.subplot(3, 1, 2)
    plt.plot(PiL_list, label="ΠvL"); plt.plot(PiR_list, label="ΠvR"); plt.plot(PiA_list, label="Πa")
    plt.plot(vL_list, label="Val L"); plt.plot(vR_list, label="Val R")
    plt.legend(); plt.ylabel("Precision / Valence")

    plt.subplot(3, 1, 3)
    plt.plot(RPE_list, label="RPE")
    plt.plot(CMS_list, label="CMS")
    plt.plot(Eye_list, label="Eye")
    plt.plot(BG_list, label="BG Gate")
    plt.legend(); plt.xlabel("Step")

    plt.tight_layout()
    fname = "plots/v13_2_multisensory.png"
    plt.savefig(fname)
    print(f"[Saved] {fname}")


if __name__ == "__main__":
    main()
