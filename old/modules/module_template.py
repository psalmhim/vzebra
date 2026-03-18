# ============================================================
# MODULE TEMPLATE — PC-SNN Architecture Standard
# AUTHOR: H.J. Park & GPT-5
# VERSION: v13.2 (2025-11-12)
#
# PURPOSE:
#   Defines the structural and documentation standards for all
#   modules in the Predictive Coding Spiking Neural Network (PC-SNN)
#   architecture (v13 → v20+).
#
#   Each module represents a predictive-coding subsystem that:
#       - Receives sensory or latent input
#       - Generates predictions
#       - Computes prediction error (Free Energy, F)
#       - Updates internal states (weights, precision, memory)
#       - Returns (pred, F, Pi) or equivalent summary outputs
#
# COMPATIBILITY RULES:
#   1. Backward-compatible class and method signatures.
#   2. Consistent return order and tensor shapes.
#   3. New arguments must have safe defaults (no refactoring required).
#   4. All modules must implement a .step() method with identical format.
# ============================================================

import torch
import torch.nn as nn

class ModuleTemplate(nn.Module):
    """
    TEMPLATE CLASS — replace 'ModuleTemplate' with actual module name
    (e.g., RetinaPC, AudioPC, ThalamusPC, DopamineSystem, etc.)

    Each predictive module models local inference:
        x  →  prediction (μ)
        μ  →  error ε = x - μ
        F  =  0.5 * ε²  (free energy)
    and precision-weighted plasticity under dopaminergic and contextual control.
    """

    def __init__(
        self,
        n_in=16,
        n_out=16,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        # ---- Core parameters ----
        self.W = nn.Parameter(0.01 * torch.randn(n_in, n_out, device=device))
        self.Pi = torch.ones(1, n_out, device=device) * 0.1
        self.prev_pred = torch.zeros(1, n_out, device=device)
        self.device = device

        # ---- Optional extensions ----
        # Use kwargs.get() to preserve backward compatibility
        self.tau = kwargs.get("tau", 0.9)             # memory decay
        self.alpha = kwargs.get("alpha", 0.002)       # learning rate scale
        self.precision_bias = kwargs.get("precision_bias", 1.0)
        self.valence = kwargs.get("valence", 0.0)

        # ---- Internal state bookkeeping ----
        self.version = kwargs.get("version", "v13.2")
        self.name = kwargs.get("name", self.__class__.__name__)

    # ============================================================
    # STEP FUNCTION STANDARD
    # ============================================================
    def step(
        self,
        x,
        rpe: float = 0.0,
        valence: float = 0.0,
        precision_bias: float = 1.0,
        **kwargs
    ):
        """
        Perform one predictive-coding update step.

        Arguments
        ----------
        x : torch.Tensor
            Input vector or latent representation.
        rpe : float
            Reward prediction error or dopaminergic signal.
        valence : float
            Local motivational valence (positive/negative bias).
        precision_bias : float
            Contextual precision scaling (e.g., thalamic coherence).
        kwargs : dict
            Optional arguments for specialized behavior (ignored by default).

        Returns
        -------
        pred : torch.Tensor
            Current prediction (μ).
        F_val : float
            Scalar free-energy term (prediction error magnitude).
        Pi_val : float
            Mean precision after update.
        """
        # Convert input
        x = x.view(1, -1).to(self.device)

        # Prediction and error
        pred = x @ self.W
        err = pred - self.prev_pred
        self.prev_pred = pred.detach()

        # Free energy
        F_val = 0.5 * (err ** 2).mean()

        # Dopamine/valence modulation
        mod = torch.exp(-torch.abs(torch.tensor(rpe + valence, device=self.device)))
        lr = self.alpha * (0.5 / (1 + self.Pi.mean()))
        dW = lr * (x.T @ err) * mod * precision_bias
        self.W.data += dW
        self.W.data /= (1.0 + self.W.data.abs().mean())  # normalize weights

        # Precision update
        self.Pi += 0.003 * (err.abs().mean() * precision_bias - self.Pi)
        self.Pi = torch.clamp(self.Pi, 0.01, 0.5)

        return pred.detach(), F_val.item(), self.Pi.mean().item()

    # ============================================================
    # RESET FUNCTION STANDARD
    # ============================================================
    def reset_state(self):
        """Reset persistent states to initial values."""
        self.prev_pred.zero_()
        self.Pi[:] = 0.1

    # ============================================================
    # DEBUGGING / LOGGING STANDARD
    # ============================================================
    def debug_state(self):
        """Return internal statistics for monitoring."""
        return {
            "module": self.name,
            "version": self.version,
            "mean_W": float(self.W.mean()),
            "mean_Pi": float(self.Pi.mean()),
            "device": str(self.device),
        }

# Alias for backward/forward compatibility
BaseModule = ModuleTemplate
# ============================================================
# END OF TEMPLATE
# ============================================================
