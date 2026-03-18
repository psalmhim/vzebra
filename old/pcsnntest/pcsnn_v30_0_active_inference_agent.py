# ============================================================
# FILE: pcsnn_v30_0_active_inference_agent.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v30.0 (2025-11-30)
#
# PURPOSE:
#     Fully unified brain–body active inference agent integrating
#     cortical, dopaminergic, interoceptive, metabolic, and motor systems
#     under free-energy minimization.
# ============================================================

import torch, os
import matplotlib.pyplot as plt

# --- core subsystems ---
from modules.visual_cortex_pc import VisualCortexPC
from modules.audio_pc import AudioPC
from modules.thalamus_relay import ThalamusRelay
from modules.dopamine_system import DopamineSystem
from modules.policy_field import GoalPolicyField
from modules.interoceptive_system import InteroceptiveField
from modules.metabolic_system import MetabolicSystem
from modules.motor_eye import MotorEye
from modules.motor_tail import MotorTail


# ============================================================
# FULL EMBODIED ACTIVE INFERENCE AGENT
# ============================================================

class ActiveInferenceAgent:
    def __init__(self, device="cpu"):
        self.device = device

        # Sensory layers
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.audio = AudioPC(device=device)

        # Midbrain and motivational systems
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy = GoalPolicyField(device=device)

        # Body systems
        self.intero = InteroceptiveField(device=device)
        self.metabolic = MetabolicSystem(device=device)
        self.eye = MotorEye(device=device)
        self.tail = MotorTail(device=device)

        # Global scalar: Free energy
        self.F = 0.0

    # ------------------------------------------------------------
    def step(self, sensory_input):
        """Perform one perception–action–homeostasis iteration."""
        # === sensory encoding ===
        x = sensory_input.to(self.device)
        rL, _, _ = self.v1L(x)
        rR, _, _ = self.v1R(torch.flip(x, dims=[1]))
        tone = torch.randn(1, 16, device=self.device)
        _, _, _ = self.audio.step(tone)

        Fv = torch.mean(torch.abs(rL - rR)).item()
        Fa = torch.mean(torch.abs(tone)).item()
        Fb = 0.05 * torch.rand(1).item()

        # === salience, dopamine, and policy ===
        cms = self.thal.step(Fv, Fa, Fb)
        valL, valR, rpe, dopa = self.dopa.step(Fv, Fa, Fb, cms)
        motive, gain, choice, goal_vec, td = self.policy.step(dopa, rpe, Fv, 0.1)

        # === body–brain loops ===
        hr, rate, co2, intero_state = self.intero.step(dopa, cms, motive)
        glucose, temp, fatigue, cort, energy = self.metabolic.step(motive, dopa, intero_state)
        eye_pos, _ = self.eye.step(cms, motive)
        tail_pos, _ = self.tail.step(motive, 0.0)

        # === global free-energy computation ===
        E_sensory = abs(Fv) + abs(Fa)
        E_intero = intero_state
        E_metabolic = abs(1 - energy)
        survival_penalty = (1 - (energy + (1 - intero_state)) / 2)
        value_gain = gain * (0.5 + dopa)
        self.F = 0.9 * self.F + 0.1 * (E_sensory + E_intero + E_metabolic + survival_penalty - value_gain)

        return {
            "F": self.F,
            "RPE": rpe,
            "Dopa": dopa,
            "Mot": motive,
            "CMS": cms,
            "HR": hr,
            "Resp": rate,
            "Glucose": glucose,
            "Fatigue": fatigue,
            "Energy": energy,
            "Eye": eye_pos,
            "Tail": tail_pos,
            "Choice": choice,
            "Homeo": 1.0 - survival_penalty,
        }


# ============================================================
# SIMULATION ENTRY POINT
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("plots", exist_ok=True)
    print("Using device:", device)

    agent = ActiveInferenceAgent(device=device)
    hist = {k: [] for k in [
        "F", "Dopa", "Mot", "HR", "Resp", "Glucose",
        "Fatigue", "Energy", "Eye", "Tail", "Homeo"
    ]}

    for t in range(500):
        sensory_input = torch.rand(1, 64)
        out = agent.step(sensory_input)
        for k in hist: hist[k].append(out[k])

        if t % 50 == 0:
            print(f"Step {t:03d}: F={out['F']:.3f}, Dopa={out['Dopa']:+.3f}, "
                  f"Mot={out['Mot']:+.3f}, HR={out['HR']:.3f}, Glu={out['Glucose']:.3f}, "
                  f"Fat={out['Fatigue']:.3f}, Eye={out['Eye']:+.2f}, Homeo={out['Homeo']:.3f}")

    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1); plt.plot(hist["F"],label="Free Energy"); plt.legend()
    plt.subplot(3,1,2); plt.plot(hist["Dopa"],label="Dopamine"); plt.plot(hist["Mot"],label="Motive"); plt.legend()
    plt.subplot(3,1,3); plt.plot(hist["Homeo"],label="Homeostasis"); plt.plot(hist["Energy"],label="Energy"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v30_0_active_inference_agent.png")
    print("[Saved] plots/v30_0_active_inference_agent.png")


if __name__ == "__main__":
    main()
