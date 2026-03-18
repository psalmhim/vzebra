# ============================================================
# MODULE: visual_cortex_pc.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v13.8 (2025-11-14)
#
# PURPOSE:
#     Extends v13.3 by adding optional goal-conditioned top-down
#     modulation (ctx_goal).  Allows cortical latent to integrate
#     both contextual (working memory) and motivational (goal)
#     feedback signals.
#
# COMPATIBILITY:
#     Fully backward-compatible with v13.3.
# ============================================================

# ============================================================
# MODULE: visual_cortex_pc.py   (with CortexMemory integration)
# VERSION: v14.0 (2025-11-17)
# ============================================================

import torch
import torch.nn as nn
from modules.module_template import BaseModule


class VisualCortexPC(BaseModule):
    """Predictive-coding visual layer with context and slow-template integration."""

    def __init__(self, n_in=64, n_latent=64, device="cpu",
                 mode="contextual",
                 ctx_gain=0.1,
                 goal_gain=0.2,
                 template_gain=0.15,
                 cross_gain=0.2,
                 beta=0.1):

        super().__init__(device=device)
        self.mode = mode
        self.device = device

        # Parameters
        self.ctx_gain = ctx_gain
        self.goal_gain = goal_gain
        self.template_gain = template_gain
        self.cross_gain = cross_gain
        self.beta = beta

        # Encoder / decoder weights
        self.W_enc = nn.Parameter(0.01 * torch.randn(n_in, n_latent, device=device))
        self.W_dec = nn.Parameter(0.01 * torch.randn(n_latent, n_in, device=device))

    # ------------------------------------------------------------
    def forward(self,
                r_this,
                r_other=None,
                ctx=None,
                ctx_goal=None,
                template=None,
                audio_latent=None):
        """
        Predictive-coding cortical layer with:
            - interhemispheric fusion (r_this, r_other)
            - fast working-memory context (ctx)
            - goal-driven modulation (ctx_goal)
            - slow cortical template memory (template)
        """

        # Interhemispheric fusion
        if r_other is not None:
            fused = 0.5 * (r_this + r_other)
            if self.mode == "competitive":
                inhib = self.cross_gain * r_other.mean(0, keepdim=True)
                fused = fused - inhib
        else:
            fused = r_this

        # Encode latent
        z = fused @ self.W_enc

        # Integrate fast working memory
        if ctx is not None:
            z = z + self.ctx_gain * ctx

        # Integrate goal modulation
        if ctx_goal is not None:
            z = z + self.goal_gain * ctx_goal

        # Integrate slow cortical template memory
        if template is not None:
            z = z + self.template_gain * template

        # Audio latent integration (multimodal)
        if audio_latent is not None:
            z = z + self.audio_gain * audio_latent    

        # Decode and compute error
        recon = z @ self.W_dec
        err = fused - recon

        # Hebbian updates
        self.W_enc.data += self.beta * (fused.T @ z) * 1e-4
        self.W_dec.data += self.beta * (z.T @ fused) * 1e-4

        # Normalize for stability
        self.W_enc.data /= (1.0 + self.W_enc.data.abs().mean())
        self.W_dec.data /= (1.0 + self.W_dec.data.abs().mean())

        return z.detach(), err.detach()

    # ------------------------------------------------------------
    def debug_state(self):
        return {
            "mode": self.mode,
            "W_enc_mean": float(self.W_enc.mean()),
            "W_dec_mean": float(self.W_dec.mean()),
        }
