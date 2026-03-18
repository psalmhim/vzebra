# ============================================================
# FILE: pcsnn_v29_0_metabolic_regulation.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v29.0 (2025-11-29)
#
# PURPOSE:
#     Integrates interoceptive and metabolic predictive control
#     to achieve full homeostatic regulation.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from modules.dopamine_system import DopamineSystem
from modules.policy_field import GoalPolicyField
from modules.thalamus_relay import ThalamusRelay
from modules.interoceptive_system import InteroceptiveField
from modules.metabolic_system import MetabolicSystem


class FullHomeostaticAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy = GoalPolicyField(device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.intero = InteroceptiveField(device=device)
        self.metabolic = MetabolicSystem(device=device)

    def step(self, Fv, Fa, Fb):
        # Sensory input integration
        cms = self.thal.step(Fv, Fa, Fb)

        # Dopamine and motivation loop
        valL, valR, rpe, dopa = self.dopa.step(Fv, Fa, Fb, cms)
        motive, gain, choice, _, td = self.policy.step(dopa, rpe, Fv, 0.1)

        # Interoceptive regulation
        hr, rate, co2, intero_state = self.intero.step(dopa, cms, motive)

        # Metabolic regulation
        glucose, temp, fatigue, cort, energy = self.metabolic.step(motive, dopa, intero_state)

        # Global homeostatic value (balance index)
        homeo_total = (energy + (1.0 - intero_state)) / 2
        return {
            "RPE": rpe,
            "Dopa": dopa,
            "Mot": motive,
            "CMS": cms,
            "HR": hr,
            "Resp": rate,
            "Fatigue": fatigue,
            "Glucose": glucose,
            "Temp": temp,
            "Cort": cort,
            "Energy": energy,
            "HomeoTotal": homeo_total,
        }


def main():
    os.makedirs("plots", exist_ok=True)
    agent = FullHomeostaticAgent(device="cuda" if torch.cuda.is_available() else "cpu")

    hist = {k: [] for k in [
        "RPE", "Dopa", "Mot", "CMS", "HR", "Resp",
        "Fatigue", "Glucose", "Temp", "Cort", "Energy", "HomeoTotal"
    ]}

    for t in range(400):
        Fv, Fa, Fb = torch.rand(1).item()*0.5, torch.rand(1).item()*0.3, torch.rand(1).item()*0.2
        out = agent.step(Fv, Fa, Fb)
        for k in hist: hist[k].append(out[k])

        if t % 50 == 0:
            print(f"Step {t:03d}: Glu={out['Glucose']:.3f}, Temp={out['Temp']:.3f}, "
                  f"Fat={out['Fatigue']:.3f}, Cort={out['Cort']:.3f}, Energy={out['Energy']:.3f}, "
                  f"Homeo={out['HomeoTotal']:.3f}")

    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1); plt.plot(hist["Glucose"],label="Glucose"); plt.plot(hist["Temp"],label="Temp"); plt.legend()
    plt.subplot(3,1,2); plt.plot(hist["Fatigue"],label="Fatigue"); plt.plot(hist["Cort"],label="Cortisol"); plt.legend()
    plt.subplot(3,1,3); plt.plot(hist["Energy"],label="Energy"); plt.plot(hist["HomeoTotal"],label="Homeostatic Total"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v29_0_metabolic_regulation.png")
    print("[Saved] plots/v29_0_metabolic_regulation.png")


if __name__ == "__main__":
    main()

