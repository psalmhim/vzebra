# ============================================================
# FILE: pcsnn_v21_0_deep_temporal_inference.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v21.0 (2025-11-21)
#
# PURPOSE:
#     Integrates sensory, policy, meta, and social EFE levels
#     into a unified deep temporal Active Inference loop.
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
from modules.group_field import GroupField
from modules.meta_precision_field import MetaPrecisionField
from modules.deep_inference_field import DeepInferenceField


class DeepAgent:
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
        return efe_self, efe_vec.mean().item(), conf


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    N = 5
    agents = [DeepAgent(device=device) for _ in range(N)]
    group = GroupField(n_agents=N, device=device)
    meta = MetaPrecisionField(n_agents=N, device=device)
    deep = DeepInferenceField(device=device)

    efe_hist, meta_val_hist, trust_hist, intent_hist = [], [], [], []

    for t, (img, _) in enumerate(loader):
        efe_sensory_list, efe_policy_list, conf_list = [], [], []
        for ag in agents:
            efe_sensory, efe_policy, conf = ag.perceive_and_act(img)
            efe_sensory_list.append(efe_sensory)
            efe_policy_list.append(efe_policy)
            conf_list.append(conf)

        efe_coupled, group_prec, group_val, leader = group.step(efe_sensory_list, conf_list)
        meta_valence, trust_matrix, trust_index = meta.step(conf_list)
        efe_total, intent = deep.step(
            efe_sensory=sum(efe_coupled)/N,
            efe_policy=sum(efe_policy_list)/N,
            meta_valence=sum(meta_valence)/N,
            trust_index=trust_index
        )

        efe_hist.append(efe_total)
        meta_val_hist.append(sum(meta_valence)/N)
        trust_hist.append(trust_index)
        intent_hist.append(intent)

        if t % 50 == 0:
            print(f"Step {t:03d}: DeepEFE={efe_total:+.3f} Intent={intent:+.3f} "
                  f"Trust={trust_index:+.3f} MetaVal={meta_val_hist[-1]:+.3f}")
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1); plt.plot(efe_hist,label="Deep EFE"); plt.legend()
    plt.subplot(4,1,2); plt.plot(meta_val_hist,label="Mean Meta-Valence"); plt.legend()
    plt.subplot(4,1,3); plt.plot(trust_hist,label="Collective Trust"); plt.legend()
    plt.subplot(4,1,4); plt.plot(intent_hist,label="Action Intent"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v21_0_deep_temporal_inference.png")
    print("[Saved] plots/v21_0_deep_temporal_inference.png")

if __name__ == "__main__":
    main()

