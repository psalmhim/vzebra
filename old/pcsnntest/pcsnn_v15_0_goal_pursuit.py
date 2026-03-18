# ============================================================
# FILE: pcsnn_v15_0_goal_pursuit.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v15.0 (2025-11-15)
#
# PURPOSE:
#     Integrates motivational gain + memory feedback with
#     discrete goal pursuit (GoalPolicyField).
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
from modules.policy_field import GoalPolicyField

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
        self.mem = WorkingMemory(n_latent=64, n_goal=3, device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy = GoalPolicyField(n_goals=3, device=device)
        self.bg = BasalGanglia(mode="exploratory")

    # --------------------------------------------------------
    def perceive_and_act(self, img):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        tone = torch.randn(1, 16, device=self.device)
        body = torch.randn(1, 8, device=self.device) * 0.05

        # Retina & audio
        rL, F_L, Pi_L = self.retinaL.step(xL)
        rR, F_R, Pi_R = self.retinaR.step(xR)
        a_pred, F_A, Pi_A = self.audio.step(tone)
        Fv, Fa, Fb = (F_L + F_R) / 2, F_A, abs(body.mean().item()) * 0.01
        F_total = (Fv + Fa + Fb) / 3

        # Visual latent and salience
        zL, _ = self.v1L(rL, rR)
        zR, _ = self.v1R(rR, rL)
        z_mean = 0.5 * (zL + zR)
        cms = self.thal.step(Fv, Fa, Fb)

        # Dopaminergic update
        valL, valR, rpe, dopa_out = self.dopa.step(Fv, Fa, Fb, cms)
        mem_mean = float(self.mem.m.abs().mean())

        # Policy + goal inference
        motive, gain_state, choice, gvec = self.policy.step(dopa_out, rpe, F_total, mem_mean, cms)

        # Goal-conditioned memory update
        mem_state = self.mem.step(z_mean, dopa=dopa_out, cms=cms, gain_state=gain_state, goal_vec=gvec)

        # Motor output
        eye = self.bg.step(valL, valR, dopa_out + 0.3 * motive, rpe, cms)

        return Fv, Fa, Fb, rpe, dopa_out, motive, gain_state, cms, eye, choice, mem_state.mean().item()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agent = PCSNNAI_Agent(device=device)

    Fv_hist, Fa_hist, Fb_hist, RPE_hist, Dopa_hist, Mot_hist, Gain_hist, CMS_hist, Eye_hist, Choice_hist, Mem_hist = ([] for _ in range(11))

    for t, (img, label) in enumerate(loader):
        Fv, Fa, Fb, rpe, dopa, motive, gain, cms, eye, choice, mem_mean = agent.perceive_and_act(img)

        if t % 50 == 0:
            print(f"Step {t:03d}: Fv={Fv:.3f} Fa={Fa:.3f} Fb={Fb:.3f} "
                  f"RPE={rpe:+.3f} Dopa={dopa:+.3f} Mot={motive:+.3f} "
                  f"Gain={gain:+.3f} CMS={cms:+.3f} Eye={eye:+.2f} "
                  f"Choice={choice} MemMean={mem_mean:+.3f}")

        Fv_hist.append(Fv); Fa_hist.append(Fa); Fb_hist.append(Fb)
        RPE_hist.append(rpe); Dopa_hist.append(dopa); Mot_hist.append(motive)
        Gain_hist.append(gain); CMS_hist.append(cms); Eye_hist.append(eye)
        Choice_hist.append(choice); Mem_hist.append(mem_mean)

        if t >= 300: break

    plt.figure(figsize=(10,10))
    plt.subplot(5,1,1); plt.plot(Fv_hist,label="Fv");plt.plot(Fa_hist,label="Fa");plt.legend()
    plt.subplot(5,1,2); plt.plot(Dopa_hist,label="Dopa");plt.plot(Mot_hist,label="Mot");plt.legend()
    plt.subplot(5,1,3); plt.plot(Gain_hist,label="Gain");plt.plot(Mem_hist,label="‖Memory‖");plt.legend()
    plt.subplot(5,1,4); plt.plot(CMS_hist,label="CMS");plt.plot(Eye_hist,label="Eye");plt.legend()
    plt.subplot(5,1,5); plt.plot(Choice_hist,label="GoalChoice");plt.yticks([0,1,2],["Left","Stay","Right"]);plt.legend()
    plt.tight_layout(); plt.savefig("plots/v15_0_goal_pursuit.png")
    print("[Saved] plots/v15_0_goal_pursuit.png")


if __name__ == "__main__":
    main()
