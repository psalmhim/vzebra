# ============================================================
# FILE: pcsnn_v13_5_policy_motivation.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# DATE: 2025-11-14
#
# PURPOSE:
#     Extend the tri-modal PC-SNN agent (visual + auditory + body)
#     with a motivational / policy control loop. Dopamine (mode="policy")
#     now signals expected free-energy reduction and drives
#     basal ganglia–like motor selection.
#
# MAJOR ADDITIONS:
#     - PolicyField module: simplified PFC-like motivation vector.
#     - DopamineSystem(mode="policy") → outputs motivational bias.
#     - BG step integrates motivation to guide Eye trajectory.
#
# BIOLOGICAL ANALOGUE:
#     Prefrontal–Striatal–Dopaminergic loop (goal-directed action)
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import modular components
from modules.retina_pc import RetinaPC
from modules.audio_pc import AudioPC
from modules.cortex_memory import VisualCortexPC, WorkingMemory
from modules.thalamus_relay import ThalamusRelay
from modules.dopamine_system import DopamineSystem
from modules.basal_ganglia import BasalGanglia


# ============================================================
# POLICY FIELD (simplified PFC motivational field)
# ============================================================
class PolicyField:
    """
    Generates a motivational bias vector based on recent dopamine
    activity and expected sensory free-energy reduction.
    """

    def __init__(self, tau=0.9, gain=0.5, device="cpu"):
        self.device = device
        self.motive = 0.0
        self.tau = tau
        self.gain = gain

    def step(self, dopa, rpe, F_total):
        # Temporal integration of motivation = dopamine - free-energy
        target = self.gain * (dopa - 0.5) - 0.1 * F_total
        self.motive = self.tau * self.motive + (1 - self.tau) * target + 0.02 * torch.randn(1).item()
        self.motive = float(torch.clamp(torch.tensor(self.motive), -1.0, 1.0))
        return self.motive


# ============================================================
# FULL AGENT
# ============================================================
class PCSNNAI_Agent:
    def __init__(self, device="cpu"):
        self.device = device

        # sensory subsystems
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.audio = AudioPC(device=device)

        # cortical integrator
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(device=device)

        # multimodal integration and motivation
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.bg = BasalGanglia(mode="exploratory")
        self.policy = PolicyField(device=device)

        # simple recognizers (for valence mapping)
        self.recognizerL = torch.nn.Linear(64, 10, device=device)
        self.recognizerR = torch.nn.Linear(64, 10, device=device)

    # ------------------------------------------------------------
    def perceive_and_act(self, img, label):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        tone = torch.randn(1, 16, device=self.device)  # random tone input
        body = torch.randn(1, 8, device=self.device) * 0.05

        # Retina processing
        rL, F_L, Pi_L = self.retinaL.step(xL)
        rR, F_R, Pi_R = self.retinaR.step(xR)

        # Auditory predictive coding
        a_pred, F_A, Pi_A = self.audio.step(tone)

        # Cortical fusion
        vL = self.v1L(rL, rR)
        vR = self.v1R(rR, rL)
        mem_state = self.mem.step((vL + vR) / 2)

        # Free-energy combination
        Fv = (F_L + F_R) / 2
        Fa = F_A
        Fb = abs(body.mean().item()) * 0.01
        F_total = (Fv + Fa + Fb) / 3

        # Cross-modal salience
        cms = self.thal.step(Fv, Fa, Fb)

        # Dopamine policy mode: returns motivational RPE + valences
        valL, valR, rpe, dopa_out = self.dopa.step(Fv, Fa, Fb, cms)

        # Policy field integrates motivation
        motive = self.policy.step(dopa_out, rpe, F_total)

        # Basal ganglia gate with motivational modulation
        eye = self.bg.step(valL=valL, valR=valR, dopa=dopa_out + 0.3 * motive,
                           rpe=rpe, cms=cms)

        return Fv, Fa, Fb, Pi_L, Pi_R, Pi_A, rpe, cms, dopa_out, motive, eye


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
    Fv_hist, Fa_hist, Fb_hist, RPE_hist, Dopa_hist, Mot_hist, Eye_hist = [], [], [], [], [], [], []

    for t, (img, label) in enumerate(loader):
        Fv, Fa, Fb, PiL, PiR, PiA, rpe, cms, dopa, motive, eye = agent.perceive_and_act(img, label.item())
        if t % 50 == 0:
            print(f"Step {t:03d}: Fv={Fv:.3f} Fa={Fa:.3f} Fb={Fb:.3f} RPE={rpe:+.3f} "
                  f"CMS={cms:+.3f} Dopa={dopa:+.3f} Mot={motive:+.3f} Eye={eye:+.2f}")
        Fv_hist.append(Fv); Fa_hist.append(Fa); Fb_hist.append(Fb)
        RPE_hist.append(rpe); Dopa_hist.append(dopa); Mot_hist.append(motive); Eye_hist.append(eye)
        if t >= 300: break

    # --- plotting ---
    plt.figure(figsize=(10,7))
    plt.subplot(3,1,1)
    plt.plot(Fv_hist, label="Fv"); plt.plot(Fa_hist, label="Fa"); plt.plot(Fb_hist, label="Fb")
    plt.legend(); plt.ylabel("Free Energy")

    plt.subplot(3,1,2)
    plt.plot(Dopa_hist, label="Dopa"); plt.plot(RPE_hist, label="RPE"); plt.plot(Mot_hist, label="Motivation")
    plt.legend(); plt.ylabel("Motivational Dynamics")

    plt.subplot(3,1,3)
    plt.plot(Eye_hist, label="Eye")
    plt.legend(); plt.xlabel("Step")

    plt.tight_layout()
    plt.savefig("plots/v13_5_policy_motivation.png")
    print("[Saved] plots/v13_5_policy_motivation.png")


if __name__ == "__main__":
    main()

