# ============================================================
# FILE: pcsnn_v33_0_hierarchical_temporal_inference.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v33.0 (2025-12-03)
#
# PURPOSE:
#     Two socially coupled agents with hierarchical temporal
#     inference — including meta-memory and slow precision feedback.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from modules.free_energy_engine import FreeEnergyEngine
from modules.dopamine_system import DopamineSystem
from modules.policy_field import GoalPolicyField
from modules.thalamus_relay import ThalamusRelay
from modules.interoceptive_system import InteroceptiveField
from modules.metabolic_system import MetabolicSystem
from modules.social_coupling import SocialCoupling
from modules.meta_memory import MetaMemory


class HierarchicalAgent:
    def __init__(self, name, device="cpu"):
        self.name = name
        self.device = device
        self.engine = FreeEnergyEngine(device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy = GoalPolicyField(device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.intero = InteroceptiveField(device=device)
        self.metabolic = MetabolicSystem(device=device)
        self.meta = MetaMemory(device=device)
        self.F = 0.0

    def step(self, Fv, Fa, Fb, meta_prior=None):
        pred = self.engine.predict()[:, :3]
        obs = torch.tensor([[Fv, Fa, Fb]], device=self.device)
        F_val, err = self.engine.update_beliefs(obs, pred)
        Pi_mean = self.engine.update_precision(err)
        cms = self.thal.step(Fv, Fa, Fb)

        # integrate meta-level priors into dopamine baseline
        base_dopa = meta_prior["dopa_baseline"] if meta_prior else 0.5
        valL, valR, rpe, dopa = self.dopa.step(Fv, Fa, Fb, cms)
        dopa = 0.5 * dopa + 0.5 * base_dopa  # meta-stabilized dopamine

        motive, gain, choice, _, td = self.policy.step(dopa, rpe, Fv, 0.1)
        hr, rate, co2, intero_state = self.intero.step(dopa, cms, motive)
        glucose, temp, fatigue, cort, energy = self.metabolic.step(motive, dopa, intero_state)

        self.F = 0.9 * self.F + 0.1 * (F_val + intero_state + abs(1 - energy))
        return {
            "Name": self.name,
            "F": self.F,
            "RPE": rpe,
            "Dopa": dopa,
            "Mot": motive,
            "Energy": energy,
            "Fatigue": fatigue,
            "Homeo": 1 - (abs(1 - energy) + intero_state) / 2,
        }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("plots", exist_ok=True)
    print("Using device:", device)

    A = HierarchicalAgent("AgentA", device=device)
    B = HierarchicalAgent("AgentB", device=device)
    social = SocialCoupling(device=device)

    hist = {"SharedR": [], "Sync": [], "MetaStab": [], "DopaA": [], "DopaB": []}

    for t in range(600):
        FvA, FvB = torch.rand(1).item() * 0.5, torch.rand(1).item() * 0.5
        Fa, Fb = 0.1, 0.05

        meta_prior_A = A.meta.get_meta_prior()
        meta_prior_B = B.meta.get_meta_prior()

        outA = A.step(FvA, Fa, Fb, meta_prior_B)
        outB = B.step(FvB, Fa, Fb, meta_prior_A)

        sharedR, emp, sync = social.step(outA, outB)

        memA, stabA, baseA = A.meta.step(outA["Energy"], outA["Mot"], sync)
        memB, stabB, baseB = B.meta.step(outB["Energy"], outB["Mot"], sync)

        hist["SharedR"].append(sharedR); hist["Sync"].append(sync)
        hist["MetaStab"].append((stabA + stabB)/2)
        hist["DopaA"].append(outA["Dopa"]); hist["DopaB"].append(outB["Dopa"])

        if t % 100 == 0:
            print(f"Step {t:03d}: SharedR={sharedR:+.3f}, Sync={sync:+.3f}, "
                  f"MetaStab={(stabA+stabB)/2:+.3f}, BaseDopa={(baseA+baseB)/2:+.3f}")

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1); plt.plot(hist["SharedR"],label="Shared Reward"); plt.plot(hist["Sync"],label="Synchrony"); plt.legend()
    plt.subplot(3,1,2); plt.plot(hist["MetaStab"],label="Meta Stability"); plt.legend()
    plt.subplot(3,1,3); plt.plot(hist["DopaA"],label="Dopa A"); plt.plot(hist["DopaB"],label="Dopa B"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v33_0_hierarchical_temporal_inference.png")
    print("[Saved] plots/v33_0_hierarchical_temporal_inference.png")


if __name__ == "__main__":
    main()

