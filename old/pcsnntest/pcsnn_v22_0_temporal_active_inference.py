# ============================================================
# FILE: pcsnn_v22_0_temporal_active_inference.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v22.0 (2025-11-22)
#
# PURPOSE:
#     Full temporal active inference with multi-level integration.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules.visual_cortex_pc import VisualCortexPC
from modules.working_memory import WorkingMemory
from modules.thalamus_relay import ThalamusRelay
from modules.dopamine_system import DopamineSystem
from modules.policy_field import ActivePolicyField
from modules.meta_precision_field import MetaPrecisionField
from modules.deep_inference_field import DeepInferenceField
from modules.temporal_inference_field import TemporalInferenceField


class TemporalAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.v1 = VisualCortexPC(n_in=784, n_latent=64, device=device)
        self.mem = WorkingMemory(n_latent=64, n_goal=3, device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(device=device)
        self.policy = ActivePolicyField(n_goals=3, device=device)
        self.meta = MetaPrecisionField(n_agents=1, device=device)
        self.deep = DeepInferenceField(device=device)
        self.temporal = TemporalInferenceField(n_state=64, n_policy=3, T=3, device=device)

    def perceive_and_predict(self, img):
        x = img.view(1, -1).to(self.device)
        z, _ = self.v1(x, x)
        Fv = torch.norm(z)
        cms = self.thal.step(Fv, 0.0, 0.0)
        rpe, dopa, efe_sens, prec = self.dopa.step(Fv, 0.0, 0.0, cms)
        motive, gain, choice, gvec, td = self.policy.step(dopa, rpe, Fv, z.mean(), cms)
        gain_val = float(gain) if isinstance(gain, torch.Tensor) else gain
        self.mem.step(z, float(dopa), cms, gvec, gain_state=gain_val)
        _, _, trust_index = self.meta.step([gain])
        efe_total, intent = self.deep.step(Fv, efe_sens, motive, trust_index)
        act, efe_mean, q_pi = self.temporal.step(z.squeeze(), dopa_gain=gain_val)
        return Fv, efe_total, efe_mean, gain, intent, act


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agent = TemporalAgent(device=device)

    efe_hist, intent_hist, act_hist, gain_hist = [], [], [], []
    def to_float(x):
        return x.item() if torch.is_tensor(x) else float(x)
    for t, (img, _) in enumerate(loader):
        Fv, efe_total, efe_mean, gain, intent, act = agent.perceive_and_predict(img)
        efe_hist.append(to_float(efe_total))
        intent_hist.append(to_float(intent))
        act_hist.append(to_float(act))
        gain_hist.append(to_float(gain))

        if t % 50 == 0:
            print(f"Step {t:03d}: EFE_total={to_float(efe_total):+.3f} EFE_mean={to_float(efe_mean):+.3f} "
                  f"Gain={to_float(gain):+.3f} Intent={to_float(intent):+.3f} Action={to_float(act)}")
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1); plt.plot(efe_hist,label="Total EFE"); plt.legend()
    plt.subplot(4,1,2); plt.plot(gain_hist,label="Gain"); plt.legend()
    plt.subplot(4,1,3); plt.plot(intent_hist,label="Intent"); plt.legend()
    plt.subplot(4,1,4); plt.plot(act_hist,label="Policy Choice"); plt.yticks([0,1,2],["Left","Stay","Right"]); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v22_0_temporal_active_inference.png")
    print("[Saved] plots/v22_0_temporal_active_inference.png")

if __name__ == "__main__":
    main()

