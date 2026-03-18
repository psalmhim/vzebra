# ============================================================
# MODULE: cortex_memory.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v16.0 (2025-11-17)
#
# PURPOSE:
#     Slow-timescale cortical consolidation memory.
#     Stores stable latent templates, integrates repeated patterns,
#     and provides long-term priors to VisualCortexPC and
#     TemporalInferenceField.
#
# FEATURES:
#     • Very slow Hebbian consolidation (cortical timescale)
#     • Optional dopamine-boosted consolidation
#     • Persistent memory trace ("template")
#     • Separate from fast WorkingMemory (short-term context)
#     • Robust normalization for long-term stability
#
# BIOLOGICAL ANALOGUE:
#     Ventral / dorsal cortical stream (V4, IT, PPC),
#     slow synaptic consolidation, stable feature priors,
#     hierarchical predictive memory.
# ============================================================

import torch
import torch.nn as nn
from modules.module_template import BaseModule


class CortexMemory(BaseModule):
    """Slow cortical consolidation for stable latent templates."""

    def __init__(self,
                 n_latent=64,
                 device="cpu",
                 lr=0.0005,            # slow cortical learning rate
                 decay=0.995,          # slow moving-average decay
                 dopa_boost=0.1,       # DA modulation of consolidation
                 mode="slow"):
        super().__init__(device=device)

        self.mode = mode
        self.n_latent = n_latent
        self.decay = decay
        self.lr = lr
        self.dopa_boost = dopa_boost

        # Slow cortical template memory
        self.template = torch.zeros(1, n_latent, device=device)

        # Slow recurrent “cortical consolidation” weights
        self.W = nn.Parameter(0.01 * torch.randn(n_latent, n_latent, device=device))

    # ------------------------------------------------------------
    def step(self, z_t, dopa=0.0):
        """
        Slow consolidation step.
        z_t  : current latent from VisualCortexPC / WM
        dopa : slow dopamine (hierarchical signal) can accelerate consolidation
        """
        z_t = z_t.to(self.device)

        # Optional DA-boosted decay (more DA → faster consolidation)
        if self.mode in ["slow", "dopa"]:
            decay_eff = self.decay - self.dopa_boost * float(dopa)
            decay_eff = max(0.90, min(decay_eff, 0.999))  # keep in stable range
        else:
            decay_eff = self.decay

        # Slow update of template memory
        # This is a cortical timescale exponential average
        projected = z_t @ self.W
        self.template = decay_eff * self.template + (1 - decay_eff) * projected

        # Hebbian-like consolidation of cortical weights
        self.W.data += self.lr * (z_t.T @ self.template)
        self.W.data /= (1.0 + self.W.data.abs().mean())  # normalize for stability

        return self.template.detach()

    # ------------------------------------------------------------
    def get_prior(self):
        """Return the slow cortical template as a prior for V1/V2 or TI."""
        return self.template.detach()

    # ------------------------------------------------------------
    def reset_state(self):
        self.template.zero_()

    # ------------------------------------------------------------
    def debug_state(self):
        return {
            "mode": self.mode,
            "template_mean": float(self.template.mean()),
            "template_std": float(self.template.std()),
            "W_mean": float(self.W.mean()),
        }

