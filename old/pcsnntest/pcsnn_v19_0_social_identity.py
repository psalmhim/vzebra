# ============================================================
# FILE: pcsnn_v19_0_social_identity.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v19.0 (2025-11-19)
#
# PURPOSE:
#     Simulates identity-based polarization and selective
#     precision coupling across multi-agent Active Inference agents.
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
from modules.basal_ganglia import BasalGanglia
from modules.social_identity_field import SocialIdentityField


class IdentityAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.retina = RetinaPC(device=device)
        self.audio = AudioPC(device=device)
        self.v1 = VisualCortexPC(device=device)
        self.mem = WorkingMemory(n_latent=64, n_goal=3, device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(device=device)
        self.outcome = OutcomePredictor(n_goals=3, device=device)
        self.policy = ActivePolicyField(n_goals=3, device=device)
        self.bg = BasalGanglia(mode="exploratory")

    def perceive_and_act(self, img):
        x = img.view(1, -1).to(self.device)
        tone = torch.randn(1, 16, device=self.device)
        r, Fv, _ = self.retina.step(x)
        a_pred, Fa, _ = self.audio.step(tone)
        z, _ = self.v1(r, r)
        Fb = 0.01 * abs(tone.mean().item())
        cms = self.thal.step(Fv, Fa, Fb)
        rpe, dopa, efe_self, prec = self.dopa.step(Fv, Fa, Fb, cms)
        reward_exp, unc_exp = self.outcome.predict()
        obs_reward = torch.tensor([1/(1+Fv+Fa+Fb)]*3, device=self.device)
        obs_unc = torch.tensor([Fv+Fa+Fb]*3, device=self.device)
        efe_vec, post, choice, gvec, conf = self.policy.step(reward_exp, unc_exp, obs_reward, obs_unc, cms)
        self.outcome.update(choice, obs_reward[choice], obs_unc[choice])
        self.mem.step(z, dopa, cms, gvec, gain_state=conf)
        return efe_self, conf


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    N = 6
    agents = [IdentityAgent(device=device) for _ in range(N)]
    identity_field = SocialIdentityField(n_agents=N, n_groups=2, device=device)

    efe_hist, pol_hist, coh_hist = [], [], []

    for t, (img, _) in enumerate(loader):
        efe_list, conf_list = [], []
        for ag in agents:
            efe, conf = ag.perceive_and_act(img)
            efe_list.append(efe)
            conf_list.append(conf)

        efe_coupled, pol, coh, ids = identity_field.step(efe_list, conf_list)
        efe_hist.append(sum(efe_coupled)/N)
        pol_hist.append(pol); coh_hist.append(coh)

        if t % 50 == 0:
            print(f"Step {t:03d}: Mean EFE={efe_hist[-1]:+.3f} "
                  f"Polarization={pol:+.3f} Cohesion={coh:+.3f} Groups={ids}")
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1); plt.plot(efe_hist,label="Mean EFE"); plt.legend()
    plt.subplot(3,1,2); plt.plot(pol_hist,label="Polarization Index"); plt.legend()
    plt.subplot(3,1,3); plt.plot(coh_hist,label="Group Cohesion"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v19_0_social_identity.png")
    print("[Saved] plots/v19_0_social_identity.png")

if __name__ == "__main__":
    main()
