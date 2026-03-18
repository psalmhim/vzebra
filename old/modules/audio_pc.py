# ============================================================
# MODULE: audio_pc.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v13.3 (2025-11-13)
#
# PURPOSE:
#     Unified auditory predictive-coding module with mode control.
#     Extends v13.2 functionality while maintaining backward
#     compatibility for previous experiments.
#
# UPDATE SUMMARY:
#     - Added mode/state flag: {"standard", "multisensory", "hierarchical"}
#     - Added optional contextual input and CMS modulation hooks.
#     - Preserves all v13.2 equations when mode="standard".
#
# ============================================================

import torch
import torch.nn as nn
from modules.module_template import BaseModule


class AudioPC(BaseModule):
    """
    Predictive-coding auditory module (unified version).
    Supports dopaminergic modulation, precision adaptation,
    and optional multisensory or contextual inputs.
    """

    def __init__(self, n_in=16, n_out=8, device="cpu", mode="standard"):
        super().__init__(device=device)
        self.mode = mode  # standard | multisensory | hierarchical

        # Core auditory weights
        self.W = nn.Parameter(0.01 * torch.randn(n_in, n_out, device=device))

        # Optional context weights (used only in hierarchical mode)
        self.W_ctx = nn.Parameter(0.01 * torch.randn(n_out, n_out, device=device))
        self.use_ctx = mode in ["hierarchical"]

        # Precision units
        self.Pi = torch.ones(1, n_out, device=device) * 0.1

        # Memory of previous prediction
        self.prev_pred = torch.zeros(1, n_out, device=device)

        # Learning & normalization parameters
        self.lr_base = 0.002
        self.norm_damp = 1.0

    # ------------------------------------------------------------
    def step(
        self,
        tone,
        rpe=0.0,
        valence=0.0,
        precision_bias=1.0,
        ctx=None,
        cms_gain=1.0,
        **kwargs
    ):
        """
        One predictive-coding update step.

        Args:
            tone : input auditory signal (1×n_in tensor)
            rpe : dopamine reward prediction error
            valence : affective value
            precision_bias : bias term from global salience
            ctx : optional contextual (higher-level) input
            cms_gain : cross-modal gain modulation
        """
        x = tone.view(1, -1).to(self.device)

        # Optional hierarchical context contribution
        if self.use_ctx and ctx is not None:
            ctx_term = ctx.view(1, -1).to(self.device) @ self.W_ctx
            x = x + 0.2 * ctx_term  # small contextual bias

        # Forward prediction and error
        pred = x @ self.W
        err = pred - self.prev_pred
        self.prev_pred = pred.detach()

        # Free energy (mean squared prediction error)
        F_val = 0.5 * (err ** 2).mean()

        # Dopamine/valence modulation
        mod = torch.exp(-torch.abs(torch.tensor(rpe + valence, device=self.device)))

        # Adaptive learning rate scaled by precision
        lr = self.lr_base * (0.5 / (1 + self.Pi.mean()))

        # Hebbian-like update weighted by modulation and precision bias
        dW = lr * (x.T @ err) * mod * precision_bias * cms_gain
        self.W.data += dW
        self.W.data /= (1.0 + self.norm_damp * self.W.data.abs().mean())

        # Precision adaptation (only active for advanced modes)
        if self.mode in ["standard", "multisensory", "hierarchical"]:
            self.Pi += 0.003 * (err.abs().mean() * precision_bias - self.Pi)
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
            "W_std": float(self.W.std()),
        }
