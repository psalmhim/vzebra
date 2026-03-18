# ============================================================
# MODULE: dopamine_tectal_system.py  (FINAL FIXED)
# AUTHOR: H.J. Park & GPT-5
# VERSION: v17.3 (2025-12-06)
#
# PURPOSE:
#     Fast tectal dopamine subsystem
#     UNIFIED: returns (da_fast, rpe_fast, surprise)
#
# FIXED:
#     • Removed torch.tensor(…) warnings
#     • Removed broadcasting shape bug
#     • All dopamine states kept as Python floats
# ============================================================

import torch
from modules.module_template import BaseModule
import math

class DopamineTectalSystem(BaseModule):

    def __init__(self, device="cpu", mode="tri",
                 alpha=0.1, beta=0.4, noise=0.02):
        super().__init__(device=device)
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.noise = noise

        self.expected = 0.0
        self.rpe = 0.0
        self.dopa = 0.5
        self.val = {"L": 0.0, "R": 0.0, "B": 0.0}

    # --------------------------------------------------------
    def step(self, Fv, Fa, Fb=None, cms=0.0):
        """
        Returns:
            da_fast, rpe_fast, surprise
        """
        Fv = float(Fv)
        Fa = float(Fa)
        Fb = float(Fb) if Fb is not None else 0.0
        cms = float(cms)

        # Pseudo-reward
        reward_L = float(torch.exp(-abs(torch.tensor(Fv))))
        reward_R = float(torch.exp(-abs(torch.tensor(Fa))))
        reward_B = float(torch.exp(-abs(torch.tensor(Fb))))

        pred_L = self.val["L"]
        pred_R = self.val["R"]
        pred_B = self.val["B"]

        # ---------------- Mode-specific ----------------
        if self.mode == "simple":
            self.rpe = 0.5 * ((reward_L - pred_L) + (reward_R - pred_R))

        elif self.mode == "dual":
            rL = reward_L - pred_L
            rR = reward_R - pred_R
            self.rpe = 0.5 * (rL + rR)
            self.val["L"] += self.alpha * rL
            self.val["R"] += self.alpha * rR

        elif self.mode == "tri":
            rL = reward_L - pred_L
            rR = reward_R - pred_R
            rB = reward_B - pred_B
            self.rpe = (rL + rR + rB) / 3.0 + 0.2 * cms
            self.val["L"] += self.alpha * (rL + 0.1 * cms)
            self.val["R"] += self.alpha * (rR + 0.1 * cms)
            self.val["B"] += self.alpha * (rB + 0.1 * cms)

        elif self.mode == "policy":
            rL = reward_L - pred_L
            rR = reward_R - pred_R
            rB = reward_B - pred_B
            self.rpe = (rL + rR + rB) / 3.0 + 0.3 * cms
            motivation = torch.tanh(torch.tensor(self.rpe)).item()
            for k in self.val.keys():
                self.val[k] += self.alpha * (motivation + 0.1 * cms)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # ---------------- Dopamine Update ----------------
        self.expected += self.alpha * (self.rpe - self.expected)

        # safe scalar dopamine update (no tensors)
        x = self.beta * self.rpe + self.dopa
        x = 1 / (1 + math.exp(-x)) if abs(x) < 60 else (1.0 if x > 0 else 0.0)

        # add noise (scalar)
        x += self.noise * float(torch.randn(()))

        # clamp
        self.dopa = max(0.0, min(1.0, x))

        # Surprise
        surprise = abs(self.rpe)

        return self.dopa, self.rpe, surprise

    # --------------------------------------------------------
    def reset_state(self):
        self.expected = 0.0
        self.rpe = 0.0
        self.dopa = 0.5
        self.val = {"L": 0.0, "R": 0.0, "B": 0.0}
