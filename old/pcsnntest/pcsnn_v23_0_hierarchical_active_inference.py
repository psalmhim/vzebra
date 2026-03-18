# ============================================================
# FILE: pcsnn_v23_0_hierarchical_active_inference.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v23.0 (2025-11-23)
#
# PURPOSE:
#     Multi-timescale Hierarchical Temporal Active Inference (HTAIF)
#     integrating sensory, goal, and narrative levels.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from modules.visual_cortex_pc import VisualCortexPC
from modules.dopamine_system import DopamineSystem
from modules.hierarchical_temporal_field import HierarchicalTemporalField


class HierarchicalAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.v1 = VisualCortexPC(n_in=784, n_latent=64, device=device)
        self.dopa = DopamineSystem(device=device)
        self.htaif = HierarchicalTemporalField(n_state=64, device=device)

    def perceive_and_infer(self, img):
        x = img.view(1, -1).to(self.device)
        z, _ = self.v1(x, x)
        Fv = torch.norm(z)
        rpe, dopa, efe_sens, prec = self.dopa.step(Fv, 0.0, 0.0, 0.0)
        return self.htaif.step(z.squeeze(), dopa_gain=dopa)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agent = HierarchicalAgent(device=device)
    EFE_fast, EFE_mid, EFE_slow, EFE_total, intent, coherence = [], [], [], [], [], []

    for t, (img, _) in enumerate(loader):
        result = agent.perceive_and_infer(img)
        EFE_fast.append(result["EFE_fast"])
        EFE_mid.append(result["EFE_mid"])
        EFE_slow.append(result["EFE_slow"])
        EFE_total.append(result["EFE_total"])
        intent.append(result["intent"])
        coherence.append(result["coherence"])

        if t % 50 == 0:
            print(f"Step {t:03d}: Total EFE={result['EFE_total']:+.3f}, "
                  f"Intent={result['intent']:+.3f}, Coherence={result['coherence']:+.3f}, "
                  f"MetaVal={result['meta_val']:+.3f}, Trust={result['trust']:+.3f}")
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1); plt.plot(EFE_total,label="Total EFE"); plt.legend()
    plt.subplot(4,1,2); plt.plot(EFE_fast,label="Fast"); plt.plot(EFE_mid,label="Mid"); plt.plot(EFE_slow,label="Slow"); plt.legend()
    plt.subplot(4,1,3); plt.plot(intent,label="Intent"); plt.legend()
    plt.subplot(4,1,4); plt.plot(coherence,label="Coherence"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v23_0_hierarchical_active_inference.png")
    print("[Saved] plots/v23_0_hierarchical_active_inference.png")


if __name__ == "__main__":
    main()
