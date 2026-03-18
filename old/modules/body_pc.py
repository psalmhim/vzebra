# ============================================================
# MODULE: body_pc.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v13.4 (2025-11-14)
#
# PURPOSE:
#     Implements proprioceptive (body-state) predictive-coding subsystem.
#     Handles error correction, gain adaptation, and cross-modal biasing
#     from visual/auditory salience (CMS) and dopamine (valence/reward).
#
# REASON FOR UPDATE:
#     • Extends tri-modal integration (V+A+B) for v13.4 experiments.
#     • Mirrors AudioPC learning rule with proprioceptive interpretation.
#     • Provides “reflexive” vs “adaptive” modes for flexible body control.
#
# MODES:
#     "reflexive"  : minimal PC dynamics (immediate correction)
#     "adaptive"   : dopaminergic + CMS modulation (learning behavior)
#
# BIOLOGICAL ANALOGUE:
#     Spinal–somatosensory predictive loop and cerebellar forward model.
#
# DEPENDENCIES:
#     from modules.module_template import BaseModule
# ============================================================

import torch
import torch.nn as nn
from modules.module_template import BaseModule


class BodyPC(BaseModule):
    """Proprioceptive predictive-coding module for body-state regulation."""

    def __init__(self, n_in=8, n_out=8, device="cpu", mode="adaptive"):
        super().__init__(device=device)
        self.mode = mode
        self.device = device

        # Input–output mapping
        self.W = nn.Parameter(0.01 * torch.randn(n_in, n_out, device=device))
        self.Pi = torch.ones(1, n_out, device=device) * 0.1
        self.prev_pred = torch.zeros(1, n_out, device=device)

        # Learning dynamics
        self.lr_base = 0.002
        self.norm_damp = 1.0
        self.cms_gain = 0.3
        self.dopa_gain = 0.2

    # ------------------------------------------------------------
    def step(self, proprio, rpe=0.0, cms=0.0, valence=0.0):
        """
        One predictive update for proprioceptive signal.

        Args:
            proprio : proprioceptive or somatosensory input (1×n_in tensor)
            rpe : dopamine-based reward prediction error
            cms : cross-modal salience (from thalamus)
            valence : signed affective value (−1..1)
        Returns:
            pred, F_val, Pi_mean
        """
        x = proprio.view(1, -1).to(self.device)
        pred = x @ self.W
        err = pred - self.prev_pred
        self.prev_pred = pred.detach()

        # Free energy (prediction error)
        F_val = 0.5 * (err ** 2).mean()

        # Precision-weighted learning rate
        lr = self.lr_base * (0.5 / (1 + self.Pi.mean()))

        if self.mode == "adaptive":
            # Dopaminergic & salience modulation
            mod_dopa = torch.exp(-torch.abs(torch.tensor(rpe + valence, device=self.device)))
            mod_cms = 1.0 + self.cms_gain * cms
            mod = mod_dopa * mod_cms
        else:
            mod = 1.0

        # Weight update
        dW = lr * (x.T @ err) * mod
        self.W.data += dW
        self.W.data /= (1.0 + self.norm_damp * self.W.data.abs().mean())

        # Precision update
        if self.mode == "adaptive":
            self.Pi += 0.003 * (err.abs().mean() - self.Pi)
            self.Pi = torch.clamp(self.Pi, 0.01, 0.5)

        return pred.detach(), F_val.item(), self.Pi.mean().item()

    # ------------------------------------------------------------
    def reset_state(self):
        self.prev_pred.zero_()

    def debug_state(self):
        return {
            "mode": self.mode,
            "Pi_mean": float(self.Pi.mean()),
            "W_mean": float(self.W.mean()),
            "F_prev": float(self.prev_pred.mean()),
        }

