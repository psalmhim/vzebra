# ============================================================
# MODULE: retina_pc.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v20.1 (Warning-free)
# ============================================================

import torch
import torch.nn as nn
from modules.module_template import BaseModule


class RetinaPC(BaseModule):
    """Retina-level predictive coding encoder."""

    def __init__(self, n_in=2048, latent_dim=64, mode="adaptive", device="cpu"):
        super().__init__(device=device)
        self.mode = mode
        self.n_in = n_in
        self.latent_dim = latent_dim

        self.W = nn.Parameter(0.01 * torch.randn(n_in, latent_dim, device=device))
        self.prev_pred = torch.zeros(1, latent_dim, device=device)
        self.Pi = torch.ones(1, device=device) * 0.2
        self.lr_base = 0.001

    # ------------------------------------------------------------
        # ------------------------------------------------------------
    def step(self, ret):
        """
        ret = dict from RetinaRenderer:
             { "ON": tensor(1,N), "OFF": tensor(1,N) }
        """

        # 1. Build retinal feature vector
        x = torch.cat([ret["ON"], ret["OFF"]], dim=1).to(self.device)   # (1, n_in)
        n_incoming = x.shape[1]

        # If incoming retinal dimension doesn't match weight matrix → re-init safely
        if self.W.shape[0] != n_incoming:
            print(f"[RetinaPC] Adjusting input dimension: {self.W.shape[0]} → {n_incoming}")
            self.W = nn.Parameter(0.01 * torch.randn(n_incoming, self.latent_dim, device=self.device))

        # 2. Predict generative latent state
        pred = x @ self.W                                               # (1, latent_dim)

        # 3. Prediction error
        err = pred - self.prev_pred                                     # (1, latent_dim)
        self.prev_pred = pred.detach()

        # 4. Free Energy = precision * error^2
        F_val = float(0.5 * self.Pi.detach() * (err.detach()**2).mean())

        # 5. Precision adaptation (if enabled)
        if self.mode in ["adaptive", "dopamine"]:
            self.Pi += 0.01 * (err.abs().mean().detach() - self.Pi)
            self.Pi = torch.clamp(self.Pi, 0.05, 2.0)

        # 6. Hebbian generative model update W
        lr = self.lr_base * float(self.Pi)
        dW = lr * (x.T @ err.detach())
        self.W.data += dW
        self.W.data /= (1.0 + self.W.data.abs().mean())

        return pred.detach(), F_val, float(self.Pi)


    # ------------------------------------------------------------
    def reset_state(self):
        self.prev_pred.zero_()
        self.Pi[:] = 0.2
