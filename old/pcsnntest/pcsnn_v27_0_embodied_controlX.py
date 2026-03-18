# ============================================================
# FILE: pcsnn_v27_0_embodied_control.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v27.0 (2025-11-27)
#
# PURPOSE:
#     Integrates brain–body control:
#     eye movement + tail locomotion driven by cortical motivation.
# ============================================================

import torch, os
import matplotlib.pyplot as plt
from modules.motor_eye import MotorEye
from modules.motor_tail import MotorTail
from modules.policy_field import GoalPolicyField
from modules.dopamine_system import DopamineSystem
from modules.thalamus_relay import ThalamusRelay

class EmbodiedAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.dopa = DopamineSystem(mode="policy", device=device)
        self.policy = GoalPolicyField(device=device)
        self.thal = ThalamusRelay(mode="tri", device=device)
        self.eye = MotorEye(device=device)
        self.tail = MotorTail(device=device)
        self.proprio = 0.0

    def step(self, Fv, Fa, Fb):
        cms = self.thal.step(Fv, Fa, Fb)
        valL, valR, rpe, dopa = self.dopa.step(Fv, Fa, Fb, cms)
        motive, gain, choice, _, td = self.policy.step(dopa, rpe, Fv, 0.1)

        # Body control
        eye_pos, eye_vel = self.eye.step(cms, motive)
        tail_pos, amp = self.tail.step(motive, self.proprio)
        self.proprio = 0.9 * self.proprio + 0.1 * tail_pos

        return {
            "RPE": rpe,
            "Dopa": dopa,
            "Mot": motive,
            "EyePos": eye_pos,
            "TailPos": tail_pos,
            "CMS": cms
        }

def main():
    os.makedirs("plots", exist_ok=True)
    agent = EmbodiedAgent(device="cuda" if torch.cuda.is_available() else "cpu")

    hist = {"RPE": [], "Dopa": [], "Mot": [], "Eye": [], "Tail": [], "CMS": []}
    for t in range(300):
        Fv, Fa, Fb = torch.rand(1).item()*0.5, torch.rand(1).item()*0.3, torch.rand(1).item()*0.2
        out = agent.step(Fv, Fa, Fb)
        for k in hist: hist[k].append(out[k])

        if t % 50 == 0:
            print(f"Step {t:03d}: RPE={out['RPE']:+.3f}, Dopa={out['Dopa']:+.3f}, "
                  f"Mot={out['Mot']:+.3f}, Eye={out['EyePos']:+.3f}, Tail={out['TailPos']:+.3f}, CMS={out['CMS']:+.3f}")

    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1); plt.plot(hist["RPE"],label="RPE"); plt.plot(hist["Dopa"],label="Dopa"); plt.legend()
    plt.subplot(3,1,2); plt.plot(hist["Eye"],label="EyePos"); plt.plot(hist["Tail"],label="TailPos"); plt.legend()
    plt.subplot(3,1,3); plt.plot(hist["CMS"],label="Salience (CMS)"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/v27_0_embodied_control.png")
    print("[Saved] plots/v27_0_embodied_control.png")

if __name__ == "__main__":
    main()
