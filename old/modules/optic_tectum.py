# ============================================================
# MODULE: optic_tectum.py
# AUTHOR: HJ Park & GPT-5
# VERSION: v13.2 (2025-12-06 FIXED FOR V55 AGENT)
#
# PURPOSE:
#     Midbrain gaze control integrating:
#       - prey vs predator salience (L/R)
#       - free-energy (Fmean)
#       - dopamine modulation
#       - basal ganglia gating (converted from motor dict)
#
# NOTE:
#     This version is 100% API-compatible with ZebrafishAgent v55.
# ============================================================

import torch

class OpticTectum:
    def __init__(self, mode="predictive"):
        self.mode = mode
        self.eye_pos = 0.0
        self.eye_vel = 0.0

        self.center_pull = 0.15        # homeostatic centripetal force
        self.bg_gain = 0.35            # gating influence on gaze

    # ------------------------------------------------------------
    def _parse_bg_gate(self, bg_gate):
        """
        Accept BG or motor outputs in any format:
            - float
            - tensor
            - dict {"tail": value, "mode": "..."}
        Convert to a scalar gating value.
        """
        if isinstance(bg_gate, dict):
            return float(bg_gate.get("tail", 0.0))
        if isinstance(bg_gate, torch.Tensor):
            return float(bg_gate.item())
        return float(bg_gate)

    # ------------------------------------------------------------
    def step(self, valL, valR, Fmean, bg_gate, dopa):
        """
        Inputs:
            valL: left salience (prey_prob)
            valR: right salience (pred_prob)
            Fmean: visual free energy (float)
            bg_gate: motor/basal ganglia output of ANY format
            dopa: fast dopamine (0–1)

        Output:
            eye_pos (float)
        """

        bg_val = self._parse_bg_gate(bg_gate)

        # --------------------------------------------------------
        # VISUAL salience competition
        # --------------------------------------------------------
        salience_drive = torch.tanh(
            torch.tensor(1.5 * (valR - valL))
        ).item()

        # --------------------------------------------------------
        # FREE-ENERGY penalty
        # --------------------------------------------------------
        efe_penalty = -0.3 * Fmean

        # --------------------------------------------------------
        # Basal ganglia gate → influences vergence
        # --------------------------------------------------------
        gate_term = self.bg_gain * bg_val

        # --------------------------------------------------------
        # Total tectal drive
        # --------------------------------------------------------
        drive = salience_drive + efe_penalty + gate_term

        # Momentum update
        self.eye_vel = 0.85 * self.eye_vel + 0.15 * drive

        # Recoring (dopamine ↓ ⇒ stronger central pull)
        extra_center = 0.05 if dopa < 0.4 else 0.0
        self.eye_pos += self.eye_vel - (self.center_pull + extra_center) * self.eye_pos

        # Boundary bounce
        if abs(self.eye_pos) > 0.8:
            self.eye_vel *= -0.7

        # Clamp
        self.eye_pos = float(torch.clamp(torch.tensor(self.eye_pos), -1.0, 1.0))

        return self.eye_pos

    # ------------------------------------------------------------
    def predict(self):
        return self.eye_pos
