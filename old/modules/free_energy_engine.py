# ============================================================
# MODULE: free_energy_engine.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v31.0 (2025-12-01)
#
# PURPOSE:
#     Implements continuous precision-weighted free-energy
#     computation and gradient descent for belief & action updates.
# ============================================================

import torch
from modules.module_template import BaseModule


class FreeEnergyEngine(BaseModule):
    """
    Precision-weighted active inference core.
    Maintains modality precisions (Π) and belief state s.
    """

    def __init__(self, mode, n_state=16, device="cpu"):
        super().__init__(device=device)
        self.mode = mode
        self.s = torch.zeros(1, n_state, device=device)  # latent belief state
        self.Pi_s = torch.ones(1, n_state, device=device) * 1.0  # sensory precision
        self.lr_belief = 0.05
        self.lr_precision = 0.01
        self.decay = 0.99

    # ------------------------------------------------------------
    def compute_free_energy(self, obs, pred):
        """
        Compute total precision-weighted free energy.
        """
        err = obs - pred
        F = torch.sum(self.Pi_s * (err ** 2))
        return F, err

    # ------------------------------------------------------------
    def update_beliefs(self, obs, pred):
        F, err = self.compute_free_energy(obs, pred)
        grad_s = 2 * self.Pi_s * (pred - obs)
        self.s = self.s - self.lr_belief * grad_s
        return F.item(), err.detach()

    # ------------------------------------------------------------
    def update_precision(self, err):
        """
        Adapt precisions based on recent variance.
        """
        var_est = torch.var(err) + 1e-6
        self.Pi_s = self.decay * self.Pi_s + self.lr_precision * (1.0 / var_est)
        return self.Pi_s.mean().item()

    # ------------------------------------------------------------
    def predict(self):
        return torch.sigmoid(self.s)  # internal prediction signal

    # ------------------------------------------------------------
    def reset_state(self):
        self.s.zero_()
        self.Pi_s.fill_(1.0)


