# ============================================================
# MODULE: metabolic_system.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v29.0 (2025-11-29)
#
# PURPOSE:
#     Predictive metabolic & hormonal control for energy balance,
#     fatigue, and thermal homeostasis.
# ============================================================

import torch
from modules.module_template import BaseModule


class MetabolicSystem(BaseModule):
    """
    Predictive energy management and hormonal homeostasis.
    """

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        # Core states
        self.glucose = 1.0   # 1.0 = optimal blood sugar
        self.temp = 1.0      # normalized body temperature
        self.fatigue = 0.0   # 0 = rested
        self.cortisol = 0.5  # stress hormone
        self.energy_balance = 1.0

        # Gains and constants
        self.lr_glucose = 0.05
        self.lr_temp = 0.02
        self.lr_fatigue = 0.01
        self.lr_cort = 0.005

    # ------------------------------------------------------------
    def step(self, motive, dopa, intero_state):
        """
        Predictive update of metabolic variables.
        motive: motivational drive (increases consumption)
        dopa: dopaminergic tone (increases energy mobilization)
        intero_state: internal stability from interoceptive feedback
        """
        # Energy consumption ∝ motive and dopamine
        consumption = 0.1 + 0.3 * abs(motive) + 0.2 * max(0, dopa - 0.5)
        production = 0.15 + 0.2 * (1.0 - self.cortisol)

        # Update glucose balance
        delta_glucose = production - consumption
        self.glucose = 0.99 * self.glucose + self.lr_glucose * delta_glucose
        self.glucose = float(torch.clamp(torch.tensor(self.glucose), 0.5, 1.5))

        # Temperature drift with energy usage
        temp_target = 1.0 + 0.2 * (self.glucose - 1.0) - 0.1 * intero_state
        self.temp = 0.98 * self.temp + self.lr_temp * temp_target
        self.temp = float(torch.clamp(torch.tensor(self.temp), 0.8, 1.2))

        # Fatigue integrates metabolic imbalance
        fatigue_target = 0.1 * (1.0 - self.glucose) + 0.05 * abs(motive)
        self.fatigue = 0.99 * self.fatigue + self.lr_fatigue * fatigue_target
        self.fatigue = float(torch.clamp(torch.tensor(self.fatigue), 0.0, 1.0))

        # Cortisol (stress) rises with interoceptive instability
        cort_target = 0.5 + 0.5 * intero_state
        self.cortisol = 0.99 * self.cortisol + self.lr_cort * (cort_target - self.cortisol)
        self.cortisol = float(torch.clamp(torch.tensor(self.cortisol), 0.2, 1.2))

        # Overall energy balance
        self.energy_balance = float(
            torch.clamp(torch.tensor(self.glucose - 0.5 * self.fatigue - 0.3 * self.cortisol), 0.0, 2.0)
        )

        return self.glucose, self.temp, self.fatigue, self.cortisol, self.energy_balance

