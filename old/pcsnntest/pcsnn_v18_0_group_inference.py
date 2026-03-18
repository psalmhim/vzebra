# ============================================================
# FILE: pcsnn_v18_0_group_inference.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v18.0 (2025-11-18)
#
# PURPOSE:
#     Simulates multi-agent active inference with collective
#     precision coupling and emergent leadership.
# ============================================================

import torch, os, random
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
from modules.group_field import GroupField
from modules.basal_ganglia import BasalGanglia


class GroupAgent:
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
        eye = self.bg.step(efe_vec[0], efe_vec[1], dopa, rpe, cms)
        return efe_self, conf, eye


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    N = 5  # number of agents
    agents = [GroupAgent(device=device) for _ in range(N)]
    group = GroupField(n_agents=N, coop_gain=0.4, leader_gain=0.3, device=device)

    efe_hist, prec_hist, leader_hist, val_hist = [], [], [], []

    for t, (img, _) in enumerate(loader):
        efe_list, conf_list = [], []
        for ag in agents:
            efe, conf, _ = ag.perceive_and_act(img)
            efe_list.append(efe)
            conf_list.append(conf)

        efe_coupled, prec, valence, leader = group.step(efe_list, conf_list)
        efe_hist.append(sum(efe_coupled)/len(efe_coupled))
        prec_hist.append(prec); val_hist.append(valence); leader_hist.append(leader)

        if t % 50 == 0:
            print(f"Step {t:03d}: Group EFE={efe_hist[-1]:+.3f} Prec={prec:+.3f} "
                  f"Val={valence:+.3f} Leader={leader}")
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1); plt.plot(efe_hist,label="Mean EFE"); plt.legend()
    plt.subplot(4,1,2); plt.plot(prec_hist,label="Group Precision"); plt.legend()
    plt.subplot(4,1,3); plt.plot(val_hist,label="Group Valence"); plt.legend()
    plt.subplot(4,1,4); plt.plot(leader_hist,label="Leader Index"); plt.yticks(range(5)); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v18_0_group_inference.png")
    print("[Saved] plots/v18_0_group_inference.png")

if __name__ == "__main__":
    main()

