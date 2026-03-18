# ============================================================
# FILE: pcsnn_v26_0_social_meta_inference.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v26.0 (2025-11-26)
#
# PURPOSE:
#     Demonstrates meta-social inference and emergent
#     collective intelligence among cooperative agents.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules.visual_cortex_pc import VisualCortexPC
from modules.dopamine_system import DopamineSystem
from modules.social_meta_inference_field import SocialMetaInferenceField


class CollectiveMind:
    def __init__(self, n_agents=5, device="cpu"):
        self.device = device
        self.n_agents = n_agents
        self.v1 = [VisualCortexPC(n_in=784, n_latent=64, device=device) for _ in range(n_agents)]
        self.dopa = [DopamineSystem(device=device) for _ in range(n_agents)]
        self.meta_field = SocialMetaInferenceField(n_agents=n_agents, device=device)

    def step(self, imgs):
        states, dopa_gains = [], []
        for i in range(self.n_agents):
            x = imgs[i].view(1, -1).to(self.device)
            z, _ = self.v1[i](x, x)
            Fv = torch.norm(z)
            rpe, dopa, _, _ = self.dopa[i].step(Fv, 0.0, 0.0, 0.0)
            states.append(z.squeeze())
            dopa_gains.append(dopa)
        return self.meta_field.collective_meta_step(states, dopa_gains)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=5, shuffle=True)

    collective = CollectiveMind(n_agents=5, device=device)
    EFE_hist, Trust_hist, Coh_hist, Emp_hist, Meta_hist, Cult_hist = [], [], [], [], [], []

    for t, (imgs, _) in enumerate(loader):
        result = collective.step(imgs)
        EFE_hist.append(result["EFE"])
        Trust_hist.append(result["trust"])
        Coh_hist.append(result["coherence"])
        Emp_hist.append(result["empathy"])
        Meta_hist.append(result["meta_coherence"])
        Cult_hist.append(result["culture_strength"])

        if t % 50 == 0:
            print(f"Step {t:03d}: EFE={result['EFE']:+.3f}, Trust={result['trust']:+.3f}, "
                  f"Empathy={result['empathy']:+.3f}, MetaCoherence={result['meta_coherence']:+.3f}, "
                  f"Culture={result['culture_strength']:+.3f}")

        if t >= 300:
            break

    plt.figure(figsize=(10,8))
    plt.subplot(5,1,1); plt.plot(EFE_hist,label="Group EFE"); plt.legend()
    plt.subplot(5,1,2); plt.plot(Emp_hist,label="Empathy"); plt.plot(Meta_hist,label="Meta-Coherence"); plt.legend()
    plt.subplot(5,1,3); plt.plot(Coh_hist,label="Collective Coherence"); plt.legend()
    plt.subplot(5,1,4); plt.plot(Trust_hist,label="Trust"); plt.legend()
    plt.subplot(5,1,5); plt.plot(Cult_hist,label="Cultural Strength"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v26_0_social_meta_inference.png")
    print("[Saved] plots/v26_0_social_meta_inference.png")


if __name__ == "__main__":
    main()

