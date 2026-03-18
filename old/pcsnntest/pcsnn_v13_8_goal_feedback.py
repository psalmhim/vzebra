# ============================================================
# FILE: pcsnn_v13_8_goal_feedback.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v13.8 (2025-11-14)
#
# PURPOSE:
#     Adds goal feedback to working memory.
#     Working memory now integrates current goal vector
#     (Left, Stay, Right) to stabilize context-dependent visual state.
#
# DEPENDENCIES:
#     policy_field.py (PolicyField, GoalPolicy)
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- core modules ---
from modules.retina_pc import RetinaPC
from modules.audio_pc import AudioPC
from modules.working_memory import WorkingMemory
from modules.visual_cortex_pc import VisualCortexPC
from modules.thalamus_relay import ThalamusRelay
from modules.dopamine_system import DopamineSystem
from modules.basal_ganglia import BasalGanglia
from modules.policy_field import PolicyField, GoalPolicy


# ============================================================
# Predictive-Coding SNN Agent with Goal Feedback
# ============================================================

class PCSNNAI_Agent:
    def __init__(self, device="cpu"):
        self.device = device

        # Sensory modules
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.audio = AudioPC(device=device)

        # Cortical and memory modules
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(n_latent=64, device=device)

        # Motivation and coordination modules
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy_field = PolicyField(device=device)
        self.goal_policy = GoalPolicy(n_goals=3, device=device)
        self.bg = BasalGanglia(mode="exploratory")

    # --------------------------------------------------------
    def perceive_and_act(self, img, label):
        # --- sensory inputs ---
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        tone = torch.randn(1, 16, device=self.device)
        body = torch.randn(1, 8, device=self.device) * 0.05

        # --- retina & auditory ---
        rL, F_L, Pi_L = self.retinaL.step(xL)
        rR, F_R, Pi_R = self.retinaR.step(xR)
        a_pred, F_A, Pi_A = self.audio.step(tone)

        # --- visual cortex predictive coding ---
        vL = self.v1L(rL, rR)
        vR = self.v1R(rR, rL)
        zL, errL = vL
        zR, errR = vR
        z_mean = 0.5 * (zL + zR)

        # --- motivational & goal-selection stage ---
        Fv = (F_L + F_R) / 2
        Fa = F_A
        Fb = abs(body.mean().item()) * 0.01
        F_total = (Fv + Fa + Fb) / 3

        cms = self.thal.step(Fv, Fa, Fb)
        valL, valR, rpe, dopa_out = self.dopa.step(Fv, Fa, Fb, cms)
        motive = self.policy_field.step(dopa_out, rpe, F_total)

        # discrete goal decision with persistence
        choice, prob = self.goal_policy.step(dopa_out, motive, cms)

        # convert to one-hot goal vector and update working memory
        goal_vec = torch.zeros(1, 3, device=self.device)
        goal_vec[0, choice] = 1.0
        mem_state = self.mem.step(z_mean, goal_vec=goal_vec)

        # --- basal ganglia action gating ---
        eye = self.bg.step(valL, valR, dopa_out + 0.3 * motive, rpe, cms)

        return Fv, Fa, Fb, rpe, dopa_out, motive, cms, eye, choice


# ============================================================
# Main Simulation
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agent = PCSNNAI_Agent(device=device)

    # logging buffers
    Fv_hist, Fa_hist, Fb_hist, RPE_hist, Dopa_hist, Mot_hist, CMS_hist, Eye_hist, Choice_hist = \
        [], [], [], [], [], [], [], [], []

    for t, (img, label) in enumerate(loader):
        Fv, Fa, Fb, rpe, dopa, motive, cms, eye, choice = agent.perceive_and_act(img, label.item())

        if t % 50 == 0:
            print(f"Step {t:03d}: Fv={Fv:.3f} Fa={Fa:.3f} Fb={Fb:.3f} "
                  f"RPE={rpe:+.3f} Dopa={dopa:+.3f} Mot={motive:+.3f} "
                  f"CMS={cms:+.3f} Eye={eye:+.2f} Choice={choice}")

        Fv_hist.append(Fv); Fa_hist.append(Fa); Fb_hist.append(Fb)
        RPE_hist.append(rpe); Dopa_hist.append(dopa); Mot_hist.append(motive)
        CMS_hist.append(cms); Eye_hist.append(eye); Choice_hist.append(choice)

        if t >= 300:
            break

    # --- visualization ---
    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1)
    plt.plot(Fv_hist,label="Fv"); plt.plot(Fa_hist,label="Fa"); plt.legend()

    plt.subplot(4,1,2)
    plt.plot(Dopa_hist,label="Dopa"); plt.plot(Mot_hist,label="Mot"); plt.legend()

    plt.subplot(4,1,3)
    plt.plot(CMS_hist,label="CMS"); plt.plot(Eye_hist,label="Eye"); plt.legend()

    plt.subplot(4,1,4)
    plt.plot(Choice_hist,label="Choice")
    plt.yticks([0,1,2],["Left","Stay","Right"])
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/v13_8_goal_feedback.png")
    print("[Saved] plots/v13_8_goal_feedback.png")


# ============================================================
if __name__ == "__main__":
    main()
