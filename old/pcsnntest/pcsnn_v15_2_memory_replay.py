# ============================================================
# FILE: pcsnn_v15_2_memory_replay.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v15.2 (2025-11-15)
#
# PURPOSE:
#     Demonstrates dopamine-triggered memory replay and consolidation.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules.retina_pc import RetinaPC
from modules.audio_pc import AudioPC
from modules.visual_cortex_pc import VisualCortexPC
from modules.working_memory import WorkingMemory
from modules.thalamus_relay import ThalamusRelay
from modules.dopamine_system import DopamineSystem
from modules.basal_ganglia import BasalGanglia
from modules.policy_field import GoalPolicyField as PolicyField

# ------------------------------------------------------------
class PCSNNAI_Agent:
    def __init__(self, device="cpu"):
        self.device = device

        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.audio = AudioPC(device=device)
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(n_latent=64, n_goal=3, device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy_field = PolicyField(device=device)
        self.bg = BasalGanglia(mode="exploratory")

    # --------------------------------------------------------
    def perceive_and_act(self, img):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        tone = torch.randn(1, 16, device=self.device)
        body = torch.randn(1, 8, device=self.device) * 0.05

        rL, F_L, _ = self.retinaL.step(xL)
        rR, F_R, _ = self.retinaR.step(xR)
        a_pred, F_A, _ = self.audio.step(tone)
        zL, _ = self.v1L(rL, rR)
        zR, _ = self.v1R(rR, rL)
        z_mean = 0.5 * (zL + zR)

        Fv, Fa, Fb = (F_L + F_R)/2, F_A, abs(body.mean().item())*0.01
        F_total = (Fv + Fa + Fb)/3

        cms = self.thal.step(Fv, Fa, Fb)
        valL, valR, rpe, dopa_out = self.dopa.step(Fv, Fa, Fb, cms)
        motive, gain_state, choice, gvec, td = self.policy_field.step(dopa_out, rpe, F_total, z_mean.mean().item(), cms)
        mem_state, alpha_eff = self.mem.step(z_mean, dopa_out, cms, gvec, gain_state, rpe)
        eye = self.bg.step(valL, valR, dopa_out + 0.3*motive, rpe, cms)

        return Fv, Fa, Fb, rpe, dopa_out, motive, gain_state, cms, eye, choice, mem_state.mean().item(), alpha_eff

# ------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agent = PCSNNAI_Agent(device=device)
    Fv_hist, Dopa_hist, Mot_hist, CMS_hist, Eye_hist, Choice_hist, Mem_hist, Alpha_hist = [], [], [], [], [], [], [], []

    for t, (img, _) in enumerate(loader):
        Fv, Fa, Fb, rpe, dopa, motive, gain, cms, eye, choice, mem_mean, alpha_eff = agent.perceive_and_act(img)

        if t % 50 == 0:
            print(f"Step {t:03d}: Fv={Fv:.3f} RPE={rpe:+.3f} Dopa={dopa:+.3f} Mot={motive:+.3f} "
                  f"CMS={cms:+.3f} Eye={eye:+.2f} Choice={choice} Mem={mem_mean:+.3f}")

        Fv_hist.append(Fv); Dopa_hist.append(dopa); Mot_hist.append(motive)
        CMS_hist.append(cms); Eye_hist.append(eye); Choice_hist.append(choice)
        Mem_hist.append(mem_mean); Alpha_hist.append(alpha_eff)

        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1); plt.plot(Fv_hist,label="Visual Free-Energy"); plt.legend()
    plt.subplot(4,1,2); plt.plot(Dopa_hist,label="Dopa"); plt.plot(Mot_hist,label="Mot"); plt.legend()
    plt.subplot(4,1,3); plt.plot(Mem_hist,label="Memory"); plt.plot(Alpha_hist,label="α_eff"); plt.legend()
    plt.subplot(4,1,4); plt.plot(Choice_hist,label="Choice"); plt.yticks([0,1,2],["Left","Stay","Right"]); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v15_2_memory_replay.png")
    print("[Saved] plots/v15_2_memory_replay.png")

if __name__ == "__main__":
    main()

