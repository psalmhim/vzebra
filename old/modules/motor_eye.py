# ============================================================
# MODULE: motor_eye.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v27.0 (2025-11-27)
#
# PURPOSE:
#     Predictive oculomotor control — eye saccade and vergence.
#     Driven by cortical salience (CMS) and dopaminergic drive.
# ============================================================

import torch
from modules.module_template import BaseModule

class MotorEye(BaseModule):
    """
    Predictive eye movement controller:
    generates smooth pursuit or saccadic movements
    based on expected salience and motivation.
    """

    def __init__(self, inertia=0.9, gain=0.5, device="cpu"):
        super().__init__(device=device)
        self.pos = 0.0
        self.vel = 0.0
        self.inertia = inertia
        self.gain = gain
        self.noise = 0.02

    def step(self, salience, motive):
        # Desired position is proportional to salience & motive
        target = self.gain * (salience + 0.3 * motive)
        error = target - self.pos

        # Predictive update with inertia
        self.vel = self.inertia * self.vel + 0.1 * error
        self.pos += self.vel + self.noise * torch.randn(1).item()

        # Clamp position to plausible range
        self.pos = float(torch.clamp(torch.tensor(self.pos), -1.0, 1.0))
        return self.pos, self.vel
