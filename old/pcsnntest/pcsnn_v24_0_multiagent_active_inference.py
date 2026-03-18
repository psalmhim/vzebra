# ============================================================
# FILE: pcsnn_v24_0_multiagent_active_inference.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v24.0 (2025-11-24)
#
# PURPOSE:
#     Multi-agent hierarchical active inference (MA-HTAIF)
#     demonstrating collective coherence and social trust dynamics.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules.visual_cortex_pc import VisualCortexPC
from modules.dopamine_system import DopamineSystem
from modules.multiagent_inference_field import MultiAgentInferenceField


class MultiAgentSociety:
    def __init__(self, n_agents=5, device="cpu"):
        self.device = device
        self.n_agents = n_agents
        self.v1 = [VisualCortexPC(n_in=784, n_latent=64, device=device)for _ in range(n_agents)]
        self.dopa = [DopamineSystem(device=device) for _ in range(n_agents)]
        self.field = MultiAgentInferenceField(n_agents=n_agents, device=device)

    def collective_perceive_and_infer(self, imgs):
        states, dopa_gains = [], []
        for i in range(self.n_agents):
            x = imgs[i].view(1, -1).to(self.device)
            z, _ = self.v1[i](x, x)
            Fv = torch.norm(z)
            rpe, dopa, _, _ = self.dopa[i].step(Fv, 0.0, 0.0, 0.0)
            states.append(z.squeeze())
            dopa_gains.append(dopa)
        return self.field.collective_step(states, dopa_gains)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=5, shuffle=True)

    society = MultiAgentSociety(n_agents=5, device=device)
    EFE_hist, Trust_hist, Coh_hist, Intent_hist = [], [], [], []

    for t, (imgs, _) in enumerate(loader):
        result = society.collective_perceive_and_infer(imgs)
        EFE_hist.append(result["EFE_mean"])
        Trust_hist.append(result["trust"])
        Coh_hist.append(result["coherence"])
        Intent_hist.append(result["intent_mean"])

        if t % 50 == 0:
            print(f"Step {t:03d}: EFE={result['EFE_mean']:+.3f}, "
                  f"Trust={result['trust']:+.3f}, Coherence={result['coherence']:+.3f}, "
                  f"Intent={result['intent_mean']:+.3f}")
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1); plt.plot(EFE_hist,label="Group EFE"); plt.legend()
    plt.subplot(4,1,2); plt.plot(Trust_hist,label="Trust"); plt.legend()
    plt.subplot(4,1,3); plt.plot(Coh_hist,label="Collective Coherence"); plt.legend()
    plt.subplot(4,1,4); plt.plot(Intent_hist,label="Mean Intent"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v24_0_multiagent_active_inference.png")
    print("[Saved] plots/v24_0_multiagent_active_inference.png")


if __name__ == "__main__":
    main()

