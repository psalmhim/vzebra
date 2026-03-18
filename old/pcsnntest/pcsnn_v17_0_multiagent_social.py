# ============================================================
# FILE: pcsnn_v17_0_multiagent_social.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v17.0 (2025-11-17)
#
# PURPOSE:
#     Simulates two coupled Active Inference agents
#     minimizing joint expected free energy under social coupling.
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
from modules.outcome_predictor import OutcomePredictor
from modules.policy_field import ActivePolicyField
from modules.social_field import SocialField
from modules.basal_ganglia import BasalGanglia


class SocialInferenceAgent:
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

    def perceive_and_act(self, img, peer_efe, social_gain=0.3):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])
        tone = torch.randn(1, 16, device=self.device)

        rL, F_L, _ = self.retinaL.step(xL)
        rR, F_R, _ = self.retinaR.step(xR)
        a_pred, F_A, _ = self.audio.step(tone)
        zL, _ = self.v1L(rL, rR)
        zR, _ = self.v1R(rR, rL)
        z_mean = 0.5 * (zL + zR)
        Fv, Fa, Fb = (F_L + F_R)/2, F_A, 0.01 * abs(tone.mean().item())
        cms = self.thal.step(Fv, Fa, Fb)

        rpe, dopa, efe_self, prec = self.dopa.step(Fv, Fa, Fb, cms)
        obs_reward = torch.tensor([1/(1+Fv+Fa+Fb)]*3, device=self.device)
        obs_unc = torch.tensor([Fv+Fa+Fb]*3, device=self.device)

        reward_exp, unc_exp = self.outcome.predict()
        efe_vec, post, choice, gvec, conf = self.policy.step(
            reward_exp, unc_exp, obs_reward, obs_unc, cms
        )

        # simple social adjustment: mix peer EFE
        efe_vec = efe_vec + social_gain * (peer_efe - efe_vec.mean())

        rpe_r, rpe_u = self.outcome.update(choice, obs_reward[choice], obs_unc[choice])
        mem_state, _ = self.mem.step(z_mean, dopa, cms, gvec, gain_state=conf)
        eye = self.bg.step(efe_vec[0], efe_vec[1], dopa, rpe, cms)

        return efe_vec.mean().item(), dopa, conf, cms, eye, choice


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agentA = SocialInferenceAgent(device=device)
    agentB = SocialInferenceAgent(device=device)
    social = SocialField(n_agents=2, cooperative=True, device=device)

    efeA_hist, efeB_hist, social_hist, prec_hist = [], [], [], []

    for t, (img, _) in enumerate(loader):
        efeA, _, confA, cmsA, _, _ = agentA.perceive_and_act(img, peer_efe=0)
        efeB, _, confB, cmsB, _, _ = agentB.perceive_and_act(img, peer_efe=efeA)

        coupled, valence, prec = social.step([efeA, efeB])
        efeA_c, efeB_c = coupled[0].item(), coupled[1].item()

        efeA_hist.append(efeA_c); efeB_hist.append(efeB_c)
        social_hist.append(valence); prec_hist.append(prec)

        if t % 50 == 0:
            print(f"Step {t:03d}: EFE_A={efeA_c:+.3f} EFE_B={efeB_c:+.3f} "
                  f"Social={valence:+.3f} Prec={prec:+.3f}")
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1); plt.plot(efeA_hist,label="Agent A EFE"); plt.plot(efeB_hist,label="Agent B EFE"); plt.legend()
    plt.subplot(3,1,2); plt.plot(social_hist,label="Social Valence"); plt.legend()
    plt.subplot(3,1,3); plt.plot(prec_hist,label="Precision Coupling"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v17_0_multiagent_social.png")
    print("[Saved] plots/v17_0_multiagent_social.png")

if __name__ == "__main__":
    main()
