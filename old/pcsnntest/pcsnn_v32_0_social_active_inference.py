# ============================================================
# FILE: pcsnn_v32_0_social_active_inference.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v32.0 (2025-12-02)
#
# PURPOSE:
#     Two embodied agents coupled via social active inference.
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


# ============================================================
# SINGLE ACTIVE INFERENCE AGENT (SIMPLIFIED FOR SOCIAL)
# ============================================================

class MiniActiveAgent:
    def __init__(self, name, device="cpu"):
        self.name = name
        self.device = device
        self.engine = FreeEnergyEngine(device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy = GoalPolicyField(device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.intero = InteroceptiveField(device=device)
        self.metabolic = MetabolicSystem(device=device)
        self.F = 0.0

    def step(self, Fv, Fa, Fb):
        pred = self.engine.predict()[:, :3]
        obs = torch.tensor([[Fv, Fa, Fb]], device=self.device)
        F_val, err = self.engine.update_beliefs(obs, pred)
        Pi_mean = self.engine.update_precision(err)
        cms = self.thal.step(Fv, Fa, Fb)
        valL, valR, rpe, dopa = self.dopa.step(Fv, Fa, Fb, cms)
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


# ============================================================
# SOCIAL ACTIVE INFERENCE SIMULATION
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("plots", exist_ok=True)
    print("Using device:", device)

    A = MiniActiveAgent("AgentA", device=device)
    B = MiniActiveAgent("AgentB", device=device)
    social = SocialCoupling(device=device)

    hist = {"FA": [], "FB": [], "SharedR": [], "Empathy": [], "Sync": [], "FA_F": [], "FB_F": []}

    for t in range(400):
        # environmental inputs with small random asymmetry
        FvA, FvB = torch.rand(1).item() * 0.4, torch.rand(1).item() * 0.4
        Fa, Fb = 0.1, 0.05

        outA = A.step(FvA, Fa, Fb)
        outB = B.step(FvB, Fa, Fb)
        sharedR, emp, sync = social.step(outA, outB)

        # apply shared reward as dopaminergic boost
        A.dopa.rpe += 0.2 * sharedR
        B.dopa.rpe += 0.2 * sharedR

        hist["FA"].append(outA["Mot"]); hist["FB"].append(outB["Mot"])
        hist["FA_F"].append(outA["F"]); hist["FB_F"].append(outB["F"])
        hist["SharedR"].append(sharedR); hist["Empathy"].append(emp); hist["Sync"].append(sync)

        if t % 50 == 0:
            print(f"Step {t:03d}: SharedR={sharedR:+.3f}, Empathy={emp:+.3f}, Sync={sync:+.3f}, "
                  f"MotA={outA['Mot']:+.3f}, MotB={outB['Mot']:+.3f}")

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1); plt.plot(hist["FA"],label="MotA"); plt.plot(hist["FB"],label="MotB"); plt.legend()
    plt.subplot(3,1,2); plt.plot(hist["SharedR"],label="Shared Reward"); plt.plot(hist["Empathy"],label="Empathy"); plt.legend()
    plt.subplot(3,1,3); plt.plot(hist["FA_F"],label="FA FreeE"); plt.plot(hist["FB_F"],label="FB FreeE"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v32_0_social_active_inference.png")
    print("[Saved] plots/v32_0_social_active_inference.png")


if __name__ == "__main__":
    main()
