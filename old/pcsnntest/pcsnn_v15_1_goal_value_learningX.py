# ============================================================
# FILE: pcsnn_v15_1b_goal_value_learning.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v15.1b (2025-11-15)
#
# PURPOSE:
#     Extends adaptive gain learning (meta-gain from PolicyField)
#     and connects it with WorkingMemory retention (α_eff).
#     Plots full dynamic traces of dopamine, motivation, gain,
#     TD-error, CMS, eye movement, and memory.
#
# CHANGELOG:
#     - Fixed GPU→CPU tensor conversion for Matplotlib.
#     - Added α_eff visualization (effective memory retention).
#     - Preserves goal-conditioned PC-SNN architecture (v14+).
# ============================================================

import os, torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules.retina_pc import RetinaPC
from modules.audio_pc import AudioPC
from modules.working_memory import WorkingMemory
from modules.visual_cortex_pc import VisualCortexPC
from modules.thalamus_relay import ThalamusRelay
from modules.dopamine_system import DopamineSystem
from modules.policy_field import GoalPolicy, GoalPolicyField as PolicyField
from modules.basal_ganglia import BasalGanglia


# ============================================================
# AGENT DEFINITION
# ============================================================
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
        self.policy = GoalPolicy(n_goals=3, device=device)
        self.bg = BasalGanglia(mode="exploratory")

    # --------------------------------------------------------
    def perceive_and_act(self, img):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        tone = torch.randn(1, 16, device=self.device)
        body = torch.randn(1, 8, device=self.device) * 0.05

        rL, F_L, Pi_L = self.retinaL.step(xL)
        rR, F_R, Pi_R = self.retinaR.step(xR)
        a_pred, F_A, Pi_A = self.audio.step(tone)

        (zL, _), (zR, _) = self.v1L(rL, rR), self.v1R(rR, rL)
        z_mean = 0.5 * (zL + zR)

        Fv = (F_L + F_R) / 2
        Fa = F_A
        Fb = abs(body.mean().item()) * 0.01
        F_total = (Fv + Fa + Fb) / 3
        cms = self.thal.step(Fv, Fa, Fb)

        valL, valR, rpe, dopa_out = self.dopa.step(Fv, Fa, Fb, cms)

        # --- new policy call with memory feedback ---
        motive, gain_state, choice, goal_vec, td_error = self.policy_field.step(
            dopa_out, rpe, F_total, mem_mean=self.mem.m.mean().item(), cms=cms
        )

        # --- memory update with adaptive gain ---
        mem_state, alpha_eff = self.mem.step(
            z_mean, dopa=dopa_out, cms=cms, goal_vec=goal_vec, gain_state=gain_state
        )

        eye = self.bg.step(valL, valR, dopa_out + 0.3 * motive, rpe, cms)

        return (
            Fv, Fa, Fb, rpe, dopa_out, motive, gain_state,
            cms, eye, choice, td_error, mem_state.mean().item(), alpha_eff
        )



# ============================================================
# MAIN SIMULATION
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)
    agent = PCSNNAI_Agent(device=device)

    # Histories
    Fv_hist, Fa_hist, Fb_hist = [], [], []
    RPE_hist, Dopa_hist, Mot_hist, Gain_hist, TD_hist = [], [], [], [], []
    CMS_hist, Eye_hist, Choice_hist, Mem_hist, Alpha_hist = [], [], [], [], []

    for t, (img, label) in enumerate(loader):
        Fv, Fa, Fb, rpe, dopa, motive, gain, cms, eye, choice, td, mem_mean,alpha_eff = agent.perceive_and_act(img)

        if t % 50 == 0:
            print(
                f"Step {t:03d}: Fv={Fv:.3f} Fa={Fa:.3f} Fb={Fb:.3f} "
                f"RPE={rpe:+.3f} Dopa={dopa:+.3f} Mot={motive:+.3f} "
                f"Gain={gain:+.3f} TD={td:+.3f} CMS={cms:+.3f} "
                f"Eye={eye:+.2f} Choice={choice} Mem={mem_mean:+.3f}"
            )

        # Convert all to Python floats
        def f(x): return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)
        Fv_hist.append(f(Fv)); Fa_hist.append(f(Fa)); Fb_hist.append(f(Fb))
        RPE_hist.append(f(rpe)); Dopa_hist.append(f(dopa)); Mot_hist.append(f(motive))
        Gain_hist.append(f(gain)); TD_hist.append(f(td))
        CMS_hist.append(f(cms)); Eye_hist.append(f(eye)); Choice_hist.append(f(choice))
        Mem_hist.append(f(mem_mean)); Alpha_hist.append(f(alpha_eff))

        if t >= 300:
            break

    # --- Ensure lists are pure floats for Matplotlib ---
    to_float = lambda x: float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)
    Gain_hist = [to_float(v) for v in Gain_hist]
    TD_hist = [to_float(v) for v in TD_hist]

    # --- Plot ---
    plt.figure(figsize=(10, 11))
    plt.subplot(5, 1, 1)
    plt.plot(Fv_hist, label="Fv"); plt.plot(Fa_hist, label="Fa"); plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(Dopa_hist, label="Dopa"); plt.plot(Mot_hist, label="Mot"); plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(Gain_hist, label="Gain"); plt.plot(TD_hist, label="TD-error"); plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(CMS_hist, label="CMS"); plt.plot(Eye_hist, label="Eye"); plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(Mem_hist, label="Mem"); plt.plot(Alpha_hist, label="α_eff"); plt.legend()

    plt.tight_layout()
    plt.savefig("plots/v15_1b_goal_value_learning.png")
    print("[Saved] plots/v15_1b_goal_value_learning.png")


# ============================================================
if __name__ == "__main__":
    main()
