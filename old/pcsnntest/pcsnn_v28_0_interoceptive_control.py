# ============================================================
# FILE: pcsnn_v28_0_interoceptive_control.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v28.0 (2025-11-28)
#
# PURPOSE:
#     Integrates interoceptive predictive control:
#     heart, breathing, and visceral feedback linked to dopamine.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from modules.dopamine_system import DopamineSystem
from modules.policy_field import GoalPolicyField
from modules.thalamus_relay import ThalamusRelay
from modules.interoceptive_system import InteroceptiveField


class EmbodiedInteroceptiveAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy = GoalPolicyField(device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.intero = InteroceptiveField(device=device)

    def step(self, Fv, Fa, Fb):
        cms = self.thal.step(Fv, Fa, Fb)
        valL, valR, rpe, dopa = self.dopa.step(Fv, Fa, Fb, cms)
        motive, gain, choice, _, td = self.policy.step(dopa, rpe, Fv, 0.1)

        # Predictive interoceptive control
        hr, rate, co2, intero_state = self.intero.step(dopa, cms, motive)
        homeo = 1.0 - intero_state  # homeostatic balance (1 = ideal)

        return {
            "RPE": rpe,
            "Dopa": dopa,
            "Mot": motive,
            "CMS": cms,
            "HR": hr,
            "Resp": rate,
            "CO2": co2,
            "Homeo": homeo
        }


def main():
    os.makedirs("plots", exist_ok=True)
    agent = EmbodiedInteroceptiveAgent(device="cuda" if torch.cuda.is_available() else "cpu")

    hist = {k: [] for k in ["RPE", "Dopa", "Mot", "CMS", "HR", "Resp", "CO2", "Homeo"]}
    for t in range(300):
        Fv, Fa, Fb = torch.rand(1).item()*0.5, torch.rand(1).item()*0.3, torch.rand(1).item()*0.2
        out = agent.step(Fv, Fa, Fb)
        for k in hist: hist[k].append(out[k])

        if t % 50 == 0:
            print(f"Step {t:03d}: HR={out['HR']:.3f}, Resp={out['Resp']:.3f}, CO2={out['CO2']:.3f}, "
                  f"Dopa={out['Dopa']:+.3f}, Mot={out['Mot']:+.3f}, Homeo={out['Homeo']:.3f}")

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1); plt.plot(hist["HR"],label="HeartRate"); plt.plot(hist["Resp"],label="BreathingRate"); plt.legend()
    plt.subplot(3,1,2); plt.plot(hist["CO2"],label="CO2 Level"); plt.plot(hist["Homeo"],label="Homeostasis"); plt.legend()
    plt.subplot(3,1,3); plt.plot(hist["Dopa"],label="Dopa"); plt.plot(hist["Mot"],label="Motive"); plt.plot(hist["CMS"],label="Salience"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v28_0_interoceptive_control.png")
    print("[Saved] plots/v28_0_interoceptive_control.png")


if __name__ == "__main__":
    main()

