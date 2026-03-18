# ============================================================
# FILE: pcsnn_v14_1_meta_gain_memory.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v14.1 (2025-11-15)
#
# PURPOSE:
#     Adds adaptive motivational gain influencing memory
#     consolidation: high dopamine/motivation → longer retention.
#
# KEY COMPONENTS:
#     - RetinaPC / AudioPC : sensory predictive coding
#     - VisualCortexPC     : visual latent integration
#     - WorkingMemory      : adaptive retention via gain_state
#     - ThalamusRelay      : multimodal novelty relay (CMS)
#     - DopamineSystem     : RPE and dopamine output
#     - PolicyField        : adaptive gain computation
#     - BasalGanglia       : action/motor gating
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
from modules.policy_field import PolicyField
from modules.basal_ganglia import BasalGanglia


# ============================================================
# EXTENDED POLICY FIELD: Adaptive Gain
# ============================================================

import torch
from modules.module_template import BaseModule

class PolicyFieldAdaptive(BaseModule):
    """
    Adaptive motivational field that learns internal gain_state
    depending on dopamine and motive strength.
    """

    def __init__(self, tau=0.9, gain=0.5, device="cpu"):
        super().__init__(device=device)
        self.tau = tau
        self.base_gain = gain
        self.motive = 0.0
        self.gain_state = 1.0

    def step(self, dopa, rpe, F_total):
        target = self.base_gain * (dopa - 0.5) - 0.1 * F_total
        noise = 0.02 * torch.randn(1).item()
        self.motive = self.tau * self.motive + (1 - self.tau) * target + noise
        self.motive = float(torch.clamp(torch.tensor(self.motive), -1.0, 1.0))

        # Adaptive gain update: increases with dopaminergic and motivational intensity
        delta = 0.1 * (abs(dopa - 0.5) + abs(self.motive))
        self.gain_state = 0.9 * self.gain_state + 0.1 * (1.0 + delta)
        return self.motive, self.gain_state

    def reset_state(self):
        self.motive = 0.0
        self.gain_state = 1.0


# ============================================================
# MAIN AGENT
# ============================================================

class PCSNNAI_Agent:
    def __init__(self, device="cpu"):
        self.device = device

        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.audio = AudioPC(device=device)
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(n_latent=64, device=device)

        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy_field = PolicyFieldAdaptive(device=device)
        self.bg = BasalGanglia(mode="exploratory")

    # --------------------------------------------------------
    def perceive_and_act(self, img):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        tone = torch.randn(1, 16, device=self.device)
        body = torch.randn(1, 8, device=self.device) * 0.05

        # Visual & auditory predictive coding
        rL, F_L, Pi_L = self.retinaL.step(xL)
        rR, F_R, Pi_R = self.retinaR.step(xR)
        a_pred, F_A, Pi_A = self.audio.step(tone)

        # Visual cortical latents
        zL, _ = self.v1L(rL, rR)
        zR, _ = self.v1R(rR, rL)
        z_mean = 0.5 * (zL + zR)

        # Cross-modal salience (novelty)
        Fv = (F_L + F_R) / 2
        Fa = F_A
        Fb = abs(body.mean().item()) * 0.01
        F_total = (Fv + Fa + Fb) / 3
        cms = self.thal.step(Fv, Fa, Fb)

        # Dopaminergic system
        valL, valR, rpe, dopa_out = self.dopa.step(Fv, Fa, Fb, cms)

        # Policy field (motivation + adaptive gain)
        motive, gain_state = self.policy_field.step(dopa_out, rpe, F_total)

        # Adaptive memory consolidation
        mem_state, alpha_eff = self.mem.step(z_mean, dopa=dopa_out, cms=cms, gain_state=gain_state)

        # Motor gating
        eye = self.bg.step(valL, valR, dopa_out + 0.3 * motive, rpe, cms)

        return Fv, Fa, Fb, rpe, dopa_out, motive, gain_state, alpha_eff, cms, eye


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agent = PCSNNAI_Agent(device=device)

    Fv_hist, Fa_hist, Fb_hist, RPE_hist, Dopa_hist, Mot_hist = [], [], [], [], [], []
    Gain_hist, Alpha_hist, CMS_hist, Eye_hist = [], [], [], []

    for t, (img, label) in enumerate(loader):
        Fv, Fa, Fb, rpe, dopa, motive, gain_state, alpha_eff, cms, eye = agent.perceive_and_act(img)

        if t % 50 == 0:
            print(f"Step {t:03d}: Fv={Fv:.3f} Fa={Fa:.3f} Fb={Fb:.3f} "
                  f"RPE={rpe:+.3f} Dopa={dopa:+.3f} Mot={motive:+.3f} "
                  f"Gain={gain_state:+.3f} α_eff={alpha_eff:.3f} "
                  f"CMS={cms:+.3f} Eye={eye:+.2f}")

        Fv_hist.append(Fv); Fa_hist.append(Fa); Fb_hist.append(Fb)
        RPE_hist.append(rpe); Dopa_hist.append(dopa); Mot_hist.append(motive)
        Gain_hist.append(gain_state); Alpha_hist.append(alpha_eff)
        CMS_hist.append(cms); Eye_hist.append(eye)

        if t >= 300: break

    # --------------------------------------------------------
    # Plot results
    plt.figure(figsize=(10,10))
    plt.subplot(5,1,1); plt.plot(Fv_hist,label="Fv");plt.plot(Fa_hist,label="Fa");plt.legend()
    plt.subplot(5,1,2); plt.plot(Dopa_hist,label="Dopa");plt.plot(Mot_hist,label="Mot");plt.legend()
    plt.subplot(5,1,3); plt.plot(Gain_hist,label="Gain");plt.plot(Alpha_hist,label="α_eff");plt.legend()
    plt.subplot(5,1,4); plt.plot(CMS_hist,label="CMS");plt.plot(Eye_hist,label="Eye");plt.legend()
    plt.subplot(5,1,5); plt.plot(RPE_hist,label="RPE");plt.legend()
    plt.tight_layout(); plt.savefig("plots/v14_1_meta_gain_memory.png")
    print("[Saved] plots/v14_1_meta_gain_memory.png")


if __name__ == "__main__":
    main()
