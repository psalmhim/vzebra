# ============================================================
# FILE: pcsnn_v13_3_bodycoherence.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v13.3 (2025-11-12)
#
# PURPOSE:
#     Adds proprioceptive predictive coding (BodyPC)
#     and tri-modal thalamic integration (ThalamusPC)
#     to the PC-SNN multisensory system.
#
# OUTPUT:
#     plots/v13_3_bodycoherence.png
# ============================================================

import os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from modules.retina_pc import RetinaPC
from modules.cortex_memory import VisualCortexPC, WorkingMemory
from modules.dopamine_system import DopamineSystem
from modules.basal_ganglia import BasalGanglia
from modules.optic_tectum import OpticTectum
from modules.audio_pc import AudioPC
from modules.body_pc import BodyPC
from modules.thalamus_pc import ThalamusPC


class PCSNNAI_Agent_Body:
    def __init__(self, device="cpu", mode="alternating"):
        self.device = device
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.audio = AudioPC(device=device)
        self.body = BodyPC(device=device)
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(device=device)
        self.thalamus = ThalamusPC()
        self.dopa = DopamineSystem()
        self.bg = BasalGanglia(mode=mode)
        self.tectum = OpticTectum()

    def perceive_and_act(self, img, label, t):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        rL, FvL, PiL = self.retinaL.step(xL, rpe=self.dopa.rpe, valence=self.dopa.valL)
        rR, FvR, PiR = self.retinaR.step(xR, rpe=self.dopa.rpe, valence=self.dopa.valR)
        v1L = self.v1L(rL, rR)
        v1R = self.v1R(rR, rL)
        latent = (v1L + v1R) / 2
        self.mem.step(latent)

        tone = torch.rand(1, 16, device=self.device)
        a_pred, Fa, PiA = self.audio.step(tone, rpe=self.dopa.rpe, valence=0.0)

        body_signal = torch.rand(1, 8, device=self.device)
        b_pred, Fb, PiB = self.body.step(body_signal, rpe=self.dopa.rpe, valence=0.0)

        CMS_total = self.thalamus.step((FvL + FvR) / 2, Fa, Fb)

        valL, valR, rpe = self.dopa.update(self.dopa.valL, self.dopa.valR, 0.0, 0.0, CMS_total)
        bg_gate = self.bg.step(valL, valR, dopa=self.dopa.rpe, cms=CMS_total)
        eye = self.tectum.step(valL, valR, Fmean=(FvL + FvR) / 2, bg_gate=bg_gate)

        return FvL, FvR, Fa, Fb, PiL, PiR, PiA, PiB, valL, valR, rpe, CMS_total, bg_gate, eye


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)
    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)
    agent = PCSNNAI_Agent_Body(device=device)

    FvL_list, FvR_list, Fa_list, Fb_list, PiL_list, PiR_list, PiA_list, PiB_list = [], [], [], [], [], [], [], []
    vL_list, vR_list, RPE_list, CMS_list, BG_list, Eye_list = [], [], [], [], [], []

    for t, (img, label) in enumerate(loader):
        fL, fR, fA, fB, piL, piR, piA, piB, vL, vR, rpe, cms, bg, eye = agent.perceive_and_act(img, label.item(), t)
        if t % 50 == 0:
            print(f"Step {t:03d}: Fv={fL:.3f}/{fR:.3f} Fa={fA:.3f} Fb={fB:.3f} ΠL={piL:.3f} ΠR={piR:.3f} ΠA={piA:.3f} ΠB={piB:.3f} vL={vL:+.2f} vR={vR:+.2f} RPE={rpe:+.3f} CMS={cms:+.3f} BG={bg:+.2f} Eye={eye:+.2f}")
        FvL_list.append(fL); FvR_list.append(fR); Fa_list.append(fA); Fb_list.append(fB)
        PiL_list.append(piL); PiR_list.append(piR); PiA_list.append(piA); PiB_list.append(piB)
        vL_list.append(vL); vR_list.append(vR); RPE_list.append(rpe); CMS_list.append(cms)
        BG_list.append(bg); Eye_list.append(eye)
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(FvL_list, label="FvL"); plt.plot(FvR_list, label="FvR")
    plt.plot(Fa_list, label="Fa"); plt.plot(Fb_list, label="Fb"); plt.legend(); plt.ylabel("Free Energy")
    plt.subplot(3,1,2)
    plt.plot(PiL_list, label="ΠvL"); plt.plot(PiR_list, label="ΠvR"); plt.plot(PiA_list, label="Πa"); plt.plot(PiB_list, label="Πb")
    plt.legend(); plt.ylabel("Precision")
    plt.subplot(3,1,3)
    plt.plot(RPE_list, label="RPE"); plt.plot(CMS_list, label="CMS"); plt.plot(BG_list, label="BG"); plt.plot(Eye_list, label="Eye")
    plt.legend(); plt.xlabel("Step"); plt.tight_layout()
    fname = "plots/v13_3_bodycoherence.png"; plt.savefig(fname)
    print(f"[Saved] {fname}")

if __name__ == "__main__":
    main()

