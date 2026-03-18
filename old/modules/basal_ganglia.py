# ============================================================
# MODULE: basal_ganglia.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v13.3 (2025-11-13)
#
# PURPOSE:
#     Unified basal ganglia controller for action gating, oscillation,
#     and exploratory modulation under dopaminergic and CMS inputs.
#
# REASON FOR UPDATE:
#     • Merged v13.1b (alternation/novelty/exploratory dynamics)
#       with v13.3 additive architecture (mode-controlled BaseModule).
#     • Supports continuous transition from oscillatory gating
#       to policy-driven motivation control.
#
# KEY FEATURES:
#     - Internal momentum + leak dynamics
#     - Alternation timer for periodic saccade switching
#     - Novelty-driven reversals (CMS-triggered)
#     - Exploratory stochastic flipping (RPE-dependent)
#     - Optional continuous “policy” drive for later PFC integration
#
# MODES:
#     "alternating"    : periodic switching (saccade rhythm)
#     "novelty"        : CMS-driven reversal bursts
#     "exploratory"    : RPE-weighted stochastic switching
#     "policy"         : continuous motivational drive (v14+)
#
# DEPENDENCIES:
#     from modules.module_template import BaseModule
# ============================================================

import torch
from modules.module_template import BaseModule


class BasalGanglia(BaseModule):
    """Unified basal-ganglia module with multiple motor control modes."""

    def __init__(
        self,
        mode="alternating",
        alt_period=80,
        leak=0.06,
        noise=0.05,
        device="cpu"
    ):
        super().__init__(device=device)
        self.mode = mode
        self.alt_period = alt_period
        self.leak = leak
        self.noise = noise

        # internal dynamic states
        self.state = (torch.rand(1).item() - 0.5) * 0.2
        self.momentum = 0.0
        self.alt_timer = 0
        self.max_state = 1.0

        # for policy mode (future)
        self.bias = 0.0

    # ------------------------------------------------------------
    def step(self, valL, valR, dopa, rpe, cms=0.0):
        """Compute basal-ganglia gating / oscillation."""
        drive = (valR - valL) + 0.4 * (dopa - 0.5)
        self.alt_timer += 1

        # Mode-specific modulation
        if self.mode == "alternating":
            # periodic reversal
            if self.alt_timer % self.alt_period == 0:
                drive *= -1.0

        elif self.mode == "novelty":
            # cross-modal salience triggers flips
            if cms > 0.05:
                drive *= -1.0

        elif self.mode == "exploratory":
            # stochastic flipping scaled by RPE magnitude
            flip_prob = torch.sigmoid(torch.as_tensor(abs(rpe) * 6.0)).item()
            if torch.rand(1).item() < flip_prob:
                drive *= -1.0

        elif self.mode == "policy":
            # continuous motivational bias from dopamine/RPE
            drive += 0.2 * rpe + 0.1 * cms + self.bias

        # Internal dynamics (momentum, damping, leak)
        self.momentum = 0.7 * self.momentum + 0.3 * drive
        self.state += self.momentum
        self.state *= 0.98  # natural decay
        self.state -= self.leak * self.state
        self.state += self.noise * torch.randn(1).item()

        # Clamp and nonlinear activation
        self.state = float(torch.clamp(torch.tensor(self.state), -self.max_state, self.max_state))
        output = float(torch.tanh(torch.tensor(1.5 * self.state)))
        return output

    # ------------------------------------------------------------
    def reset_state(self):
        self.state = (torch.rand(1).item() - 0.5) * 0.2
        self.momentum = 0.0
        self.alt_timer = 0
        self.bias = 0.0

    # ------------------------------------------------------------
    def debug_state(self):
        return {
            "mode": self.mode,
            "state": float(self.state),
            "momentum": float(self.momentum),
            "alt_timer": int(self.alt_timer),
        }
