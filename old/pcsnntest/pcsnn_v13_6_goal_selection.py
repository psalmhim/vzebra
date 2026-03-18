# ============================================================
# FILE: pcsnn_v13_6_goal_selection.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# DATE: 2025-11-14
#
# PURPOSE:
#     Extend the v13.5 policy-motivation system with a discrete
#     Goal-Selection layer. The agent now samples actions from a
#     softmax policy influenced by dopamine, motivation, and novelty.
#
# BIOLOGICAL ANALOGUE:
#     Prefrontal–Striatal action selection (goal competition).
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# === modules ===
from modules.retina_pc import RetinaPC
from modules.audio_pc import AudioPC
from modules.cortex_memory import VisualCortexPC, WorkingMemory
from modules.thalamus_relay import ThalamusRelay
from modules.dopamine_system import DopamineSystem
from modules.basal_ganglia import BasalGanglia


# ============================================================
# GOAL POLICY (explicit goal competition)
# ============================================================
class GoalPolicy:
    """
    Maintains a small set of candidate goals (Left, Right, Stay)
    and samples actions according to dopamine- and motivation-
    modulated softmax probabilities.
    """

    def __init__(self, n_goals=3, tau=0.2, gain=1.5, device="cpu"):
        self.device = device
        self.n_goals = n_goals
        self.tau = tau        # temporal smoothing
        self.gain = gain      # dopamine scaling
        self.pref = torch.zeros(n_goals, device=device)
        self.last_choice = 1  # start at "Stay"

    def step(self, dopa, motive, cms):
        # dopamine & motivation jointly bias preference
        drive = self.gain * (dopa - 0.5) + motive - 0.2 * cms
        noise = 0.05 * torch.randn(self.n_goals, device=self.device)
        self.pref = self.tau * self.pref + (1 - self.tau) * (drive + noise)

        prob = torch.softmax(self.pref, dim=0)
        choice = torch.multinomial(prob, 1).item()
        self.last_choice = choice
        return choice, prob.detach().cpu().numpy()

    def interpret_choice(self):
        return ["Left", "Stay", "Right"][self.last_choice]


# ============================================================
# POLICY FIELD (motivational integration, unchanged)
# ============================================================
class PolicyField:
    def __init__(self, tau=0.9, gain=0.5, device="cpu"):
        self.device = device
        self.motive = 0.0
        self.tau = tau
        self.gain = gain

    def step(self, dopa, rpe, F_total):
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

        # sensory and cortical
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.audio = AudioPC(device=device)
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(device=device)

        # motivational system
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy_field = PolicyField(device=device)
        self.goal_policy = GoalPolicy(device=device)
        self.bg = BasalGanglia(mode="exploratory")

    # --------------------------------------------------------
    def perceive_and_act(self, img, label):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        tone = torch.randn(1, 16, device=self.device)
        body = torch.randn(1, 8, device=self.device) * 0.05

        rL, F_L, Pi_L = self.retinaL.step(xL)
        rR, F_R, Pi_R = self.retinaR.step(xR)
        a_pred, F_A, Pi_A = self.audio.step(tone)

        vL = self.v1L(rL, rR)
        vR = self.v1R(rR, rL)
        mem_state = self.mem.step((vL + vR) / 2)

        Fv = (F_L + F_R) / 2
        Fa = F_A
        Fb = abs(body.mean().item()) * 0.01
        F_total = (Fv + Fa + Fb) / 3

        cms = self.thal.step(Fv, Fa, Fb)
        valL, valR, rpe, dopa_out = self.dopa.step(Fv, Fa, Fb, cms)
        motive = self.policy_field.step(dopa_out, rpe, F_total)

        # === goal selection ===
        choice, probs = self.goal_policy.step(dopa_out, motive, cms)

        # map discrete choice → BG modulation
        if choice == 0:   # Left
            eye_bias = -0.6
        elif choice == 1: # Stay
            eye_bias = 0.0
        else:             # Right
            eye_bias = +0.6

        eye = self.bg.step(valL, valR, dopa_out + 0.3 * motive, rpe, cms)
        eye = 0.7 * eye + 0.3 * eye_bias

        return Fv, Fa, Fb, rpe, dopa_out, motive, cms, eye, choice, probs


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

    Fv_hist, Fa_hist, Fb_hist, RPE_hist, Dopa_hist, Mot_hist, CMS_hist, Eye_hist = [], [], [], [], [], [], [], []
    Choice_hist = []

    for t, (img, label) in enumerate(loader):
        Fv, Fa, Fb, rpe, dopa, motive, cms, eye, choice, probs = agent.perceive_and_act(img, label.item())

        if t % 50 == 0:
            print(f"Step {t:03d}: Fv={Fv:.3f} Fa={Fa:.3f} Fb={Fb:.3f} "
                  f"RPE={rpe:+.3f} Dopa={dopa:+.3f} Mot={motive:+.3f} "
                  f"CMS={cms:+.3f} Eye={eye:+.2f} Choice={choice}")

        Fv_hist.append(Fv); Fa_hist.append(Fa); Fb_hist.append(Fb)
        RPE_hist.append(rpe); Dopa_hist.append(dopa); Mot_hist.append(motive)
        CMS_hist.append(cms); Eye_hist.append(eye); Choice_hist.append(choice)

        if t >= 300: break

    # === plotting ===
    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1)
    plt.plot(Fv_hist, label="Fv"); plt.plot(Fa_hist, label="Fa"); plt.plot(Fb_hist, label="Fb")
    plt.legend(); plt.ylabel("Free Energy")

    plt.subplot(4,1,2)
    plt.plot(Dopa_hist, label="Dopa"); plt.plot(RPE_hist, label="RPE"); plt.plot(Mot_hist, label="Mot")
    plt.legend(); plt.ylabel("Motivation")

    plt.subplot(4,1,3)
    plt.plot(CMS_hist, label="CMS"); plt.plot(Eye_hist, label="Eye")
    plt.legend(); plt.ylabel("Salience / Eye")

    plt.subplot(4,1,4)
    plt.plot(Choice_hist, label="Choice")
    plt.yticks([0,1,2], ["Left","Stay","Right"])
    plt.legend(); plt.xlabel("Step")

    plt.tight_layout()
    plt.savefig("plots/v13_6_goal_selection.png")
    print("[Saved] plots/v13_6_goal_selection.png")


if __name__ == "__main__":
    main()

