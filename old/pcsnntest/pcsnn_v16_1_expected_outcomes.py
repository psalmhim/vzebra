# ============================================================
# FILE: pcsnn_v16_1_expected_outcomes.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v16.1 (2025-11-16)
#
# PURPOSE:
#     Demonstrates expected outcome–based Active Inference.
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
from modules.outcome_predictor import OutcomePredictor
from modules.policy_field import ActivePolicyField


class ActiveInferenceAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.audio = AudioPC(device=device)
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(n_latent=64, n_goal=3, device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(device=device)
        self.outcome = OutcomePredictor(n_goals=3, device=device)
        self.policy = ActivePolicyField(n_goals=3, device=device)
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
        cms = self.thal.step(Fv, Fa, Fb)
        rpe, dopa, efe, prec = self.dopa.step(Fv, Fa, Fb, cms)

        # Observed outcomes
        obs_reward = torch.tensor([1/(1+Fv+Fa+Fb)]*3, device=self.device)
        obs_unc = torch.tensor([Fv+Fa+Fb]*3, device=self.device)

        # Predict expected outcomes
        reward_exp, unc_exp = self.outcome.predict()

        # Active inference: compute EFE and policy posterior
        efe_vec, post, choice, gvec, conf = self.policy.step(
            reward_exp, unc_exp, obs_reward, obs_unc, cms
        )

        # Update expected outcome model
        rpe_r, rpe_u = self.outcome.update(choice, obs_reward[choice], obs_unc[choice])

        mem_state, _ = self.mem.step(z_mean, dopa, cms, gvec, gain_state=conf, rpe=rpe)
        eye = self.bg.step(efe_vec[0], efe_vec[1], dopa, rpe, cms)

        return Fv, Fa, Fb, efe_vec.mean().item(), dopa, conf, cms, eye, choice


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)
    agent = ActiveInferenceAgent(device=device)

    EFE_hist, Dopa_hist, Conf_hist, Choice_hist, CMS_hist = [], [], [], [], []

    for t, (img, _) in enumerate(loader):
        Fv, Fa, Fb, efe, dopa, conf, cms, eye, choice = agent.perceive_and_act(img)
        if t % 50 == 0:
            print(f"Step {t:03d}: EFE={efe:+.3f} Dopa={dopa:+.3f} Conf={conf:+.3f} CMS={cms:+.3f} Choice={choice}")
        EFE_hist.append(efe); Dopa_hist.append(dopa); Conf_hist.append(conf)
        Choice_hist.append(choice); CMS_hist.append(cms)
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1); plt.plot(EFE_hist,label="EFE"); plt.legend()
    plt.subplot(4,1,2); plt.plot(Dopa_hist,label="Dopamine"); plt.legend()
    plt.subplot(4,1,3); plt.plot(Conf_hist,label="Confidence"); plt.legend()
    plt.subplot(4,1,4); plt.plot(Choice_hist,label="Choice"); plt.yticks([0,1,2],["Left","Stay","Right"]); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v16_1_expected_outcomes.png")
    print("[Saved] plots/v16_1_expected_outcomes.png")

if __name__ == "__main__":
    main()

