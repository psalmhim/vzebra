# ============================================================
# MODULE: thalamus_relay.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v13.3 (2025-11-13)
#
# PURPOSE:
#     Unified thalamic relay module handling visual–auditory–body
#     free-energy integration and novelty detection.
#
# UPDATE SUMMARY:
#     • Merged all previous versions into a single expandable class.
#     • Modes: "dual" (V+A), "tri" (V+A+B), "novelty" (ΔF + stochastic β).
#     • Compatible with BaseModule interface.
#     • Backward-compatible with v13.2 CMS calculation.
#
# BIOLOGICAL ANALOGUE:
#     Tectum → Thalamus → Dopaminergic Midbrain / BG
# ============================================================

import torch
from modules.module_template import BaseModule


class ThalamusRelay(BaseModule):
    """Cross-modal salience (CMS) and novelty relay."""

    def __init__(self, alpha=0.4, beta=0.2, device="cpu", mode="dual"):
        super().__init__(device=device)
        self.mode = mode          # dual | tri | novelty
        self.alpha = alpha        # temporal smoothing
        self.beta = beta          # stochastic burst gain
        self.prev = {"Fv": 0.0, "Fa": 0.0, "Fb": 0.0}
        self.cms = 0.0

    def step(self, Fv, Fa, Fb=None):
        """
        Compute cross-modal salience.
        CMS = EMA[ mean(|ΔFi|) + cross-difference ] × (1 + β · ξ)
        """
        Fv, Fa = float(Fv), float(Fa)
        Fb = float(Fb) if Fb is not None else 0.0

        # --- novelty components ---
        dFv = abs(Fv - self.prev["Fv"])
        dFa = abs(Fa - self.prev["Fa"])
        dFb = abs(Fb - self.prev["Fb"]) if self.mode != "dual" else 0.0

        # --- cross-modal differences ---
        if self.mode == "dual":
            cross = abs(Fv - Fa)
            raw_sal = 0.5 * (dFv + dFa) + 0.5 * cross

        elif self.mode == "tri":
            cross = (abs(Fv - Fa) + abs(Fv - Fb) + abs(Fa - Fb)) / 3.0
            raw_sal = (dFv + dFa + dFb) / 3.0 + cross / 2.0

        elif self.mode == "novelty":
            cross = (abs(Fv - Fa) + abs(Fv - Fb) + abs(Fa - Fb)) / 3.0
            raw_sal = 0.5 * (dFv + dFa + dFb) + 0.5 * cross

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # --- stochastic burst and temporal smoothing ---
        noise = 1 + self.beta * torch.randn(1, device=self.device).item()
        self.cms = (1 - self.alpha) * self.cms + self.alpha * raw_sal * noise

        # --- clamp and update state ---
        self.cms = float(torch.clamp(torch.tensor(self.cms), -0.5, 0.5))
        self.prev.update({"Fv": Fv, "Fa": Fa, "Fb": Fb})
        return self.cms

    def reset_state(self):
        self.prev = {"Fv": 0.0, "Fa": 0.0, "Fb": 0.0}
        self.cms = 0.0

    def debug_state(self):
        return {
            "mode": self.mode,
            "cms": self.cms,
            "ΔFv": abs(self.prev["Fv"]),
            "ΔFa": abs(self.prev["Fa"]),
            "ΔFb": abs(self.prev["Fb"]),
        }
