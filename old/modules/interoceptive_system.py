# ============================================================
# MODULE: interoceptive_system.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v28.0 (2025-11-28)
#
# PURPOSE:
#     Implements interoceptive predictive control:
#     heart rate, breathing, and visceral feedback loop.
# ============================================================

import torch
from modules.module_template import BaseModule


class HeartSystem(BaseModule):
    """
    Predictive heart rate controller with dopaminergic modulation.
    """

    def __init__(self, baseline_hr=1.0, gain=0.3, inertia=0.9, device="cpu"):
        super().__init__(device=device)
        self.hr = baseline_hr
        self.baseline = baseline_hr
        self.gain = gain
        self.inertia = inertia

    def step(self, dopa, cms):
        # Expected HR from arousal & salience
        target_hr = self.baseline + self.gain * (dopa + cms - 0.5)
        error = target_hr - self.hr
        self.hr = self.inertia * self.hr + 0.1 * error
        return float(self.hr)


class BreathingSystem(BaseModule):
    """
    Predictive respiratory rhythm controller.
    """

    def __init__(self, base_rate=1.0, co2_gain=0.2, device="cpu"):
        super().__init__(device=device)
        self.rate = base_rate
        self.co2 = 0.5  # internal CO₂ estimate
        self.co2_gain = co2_gain

    def step(self, motive, hr):
        # CO₂ rises with HR and falls with breathing
        self.co2 += 0.01 * (hr - self.rate)
        self.co2 = float(torch.clamp(torch.tensor(self.co2), 0.0, 1.0))

        # Adjust breathing rate to minimize CO₂ prediction error
        target = 0.8 * motive + self.co2_gain * (self.co2 - 0.5)
        self.rate = 0.9 * self.rate + 0.1 * (1.0 + target)
        return float(self.rate), self.co2


class InteroceptiveField(BaseModule):
    """
    Integrates heart + breathing into a unified interoceptive feedback loop.
    """

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.heart = HeartSystem(device=device)
        self.breath = BreathingSystem(device=device)
        self.interoceptive_state = 0.0

    def step(self, dopa, cms, motive):
        hr = self.heart.step(dopa, cms)
        rate, co2 = self.breath.step(motive, hr)

        # Interoceptive summary: deviation from baseline equilibrium
        intero_error = (abs(hr - 1.0) + abs(rate - 1.0) + abs(co2 - 0.5)) / 3
        self.interoceptive_state = 0.9 * self.interoceptive_state + 0.1 * intero_error
        return hr, rate, co2, self.interoceptive_state
