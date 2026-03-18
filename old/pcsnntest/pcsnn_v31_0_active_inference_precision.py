# ============================================================
# FILE: pcsnn_v31_0_active_inference_precision.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v31.0 (2025-12-01)
#
# PURPOSE:
#     Full precision-weighted active inference organism.
#     Performs belief & action updates via ∂F/∂s descent.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from modules.dopamine_system import DopamineSystem
from modules.policy_field import GoalPolicyField
from modules.thalamus_relay import ThalamusRelay
from modules.interoceptive_system import InteroceptiveField
from modules.metabolic_system import MetabolicSystem
from modules.free_energy_engine import FreeEnergyEngine


class ActiveInferencePrecisionAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.engine = FreeEnergyEngine(device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy = GoalPolicyField(device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.intero = InteroceptiveField(device=device)
        self.metabolic = MetabolicSystem(device=device)
        self.F = 0.0

    def step(self, obs_v, obs_a, obs_b):
        # Predict sensory causes from current beliefs
        pred = self.engine.predict()[:, :3]  # simple 3-dim projection
        obs = torch.tensor([[obs_v, obs_a, obs_b]], device=self.device)
        F_val, err = self.engine.update_beliefs(obs, pred)
        Pi_mean = self.engine.update_precision(err)

        # Update midbrain systems with inferred errors
        cms = self.thal.step(obs_v, obs_a, obs_b)
        valL, valR, rpe, dopa = self.dopa.step(obs_v, obs_a, obs_b, cms)
        motive, gain, choice, _, td = self.policy.step(dopa, rpe, obs_v, 0.1)

        # Interoceptive + metabolic updates
        hr, rate, co2, intero_state = self.intero.step(dopa, cms, motive)
        glucose, temp, fatigue, cort, energy = self.metabolic.step(motive, dopa, intero_state)

        # Global free energy
        E_intero = intero_state
        E_metabolic = abs(1 - energy)
        survival_penalty = 1 - (energy + (1 - intero_state)) / 2
        total_F = F_val + E_intero + E_metabolic + survival_penalty - 0.3 * gain
        self.F = 0.9 * self.F + 0.1 * total_F

        return {
            "F": self.F,
            "RPE": rpe,
            "Dopa": dopa,
            "Pi": Pi_mean,
            "Mot": motive,
            "Energy": energy,
            "Fatigue": fatigue,
            "Homeo": 1 - survival_penalty,
        }


def main():
    os.makedirs("plots", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    agent = ActiveInferencePrecisionAgent(device=device)

    hist = {k: [] for k in ["F", "Dopa", "Mot", "Pi", "Energy", "Fatigue", "Homeo"]}

    for t in range(500):
        Fv, Fa, Fb = torch.rand(1).item() * 0.5, torch.rand(1).item() * 0.3, torch.rand(1).item() * 0.2
        out = agent.step(Fv, Fa, Fb)
        for k in hist: hist[k].append(out[k])

        if t % 50 == 0:
            print(f"Step {t:03d}: F={out['F']:.3f}, Pi={out['Pi']:.3f}, "
                  f"Dopa={out['Dopa']:+.3f}, Mot={out['Mot']:+.3f}, "
                  f"Energy={out['Energy']:.3f}, Fatigue={out['Fatigue']:.3f}")

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1); plt.plot(hist["F"],label="Free Energy"); plt.legend()
    plt.subplot(3,1,2); plt.plot(hist["Pi"],label="Precision"); plt.plot(hist["Dopa"],label="Dopamine"); plt.legend()
    plt.subplot(3,1,3); plt.plot(hist["Energy"],label="Energy"); plt.plot(hist["Homeo"],label="Homeostasis"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v31_0_active_inference_precision.png")
    print("[Saved] plots/v31_0_active_inference_precision.png")


if __name__ == "__main__":
    main()
