# ============================================================
# MODULE: motor_tail.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v27.0 (2025-11-27)
#
# PURPOSE:
#     Predictive locomotor (tail) controller for propulsion and
#     direction, integrating dopaminergic motivation and proprioception.
# ============================================================

import torch
from modules.module_template import BaseModule

class MotorTail(BaseModule):
    """
    Predictive tail oscillation controller (locomotion).
    Couples central pattern generator (CPG) with motivational drive.
    """

    def __init__(self, freq_base=2.0, gain=0.5, inertia=0.8, device="cpu"):
        super().__init__(device=device)
        self.phase = 0.0
        self.amp = 0.0
        self.freq = freq_base
        self.gain = gain
        self.inertia = inertia

    def step(self, motive, proprio):
        # Adaptive amplitude from motive, feedback from proprioception
        amp_target = self.gain * (motive - proprio)
        self.amp = self.inertia * self.amp + 0.2 * amp_target

        # Oscillate (simplified sine-wave tail movement)
        self.phase += self.freq * 0.1
        tail_pos = self.amp * torch.sin(torch.tensor(self.phase))

        return float(tail_pos), float(self.amp)

    # --------------------------------------------------------
    # Action primitives required by ZebrafishAgent
    # --------------------------------------------------------
    def approach_signal(self):
        # Swim gently forward
        return {"tail": +0.3, "mode": "approach"}

    def escape_signal(self):
        # Burst escape — strong negative amplitude
        return {"tail": -1.0, "mode": "escape"}

    def neutral_signal(self):
        # No movement
        return {"tail": 0.0, "mode": "neutral"}
