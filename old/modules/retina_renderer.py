# ============================================================
# MODULE: retina_renderer.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v55.1 (2025-12-05)
# ============================================================

import torch
import math


class RetinaRenderer:
    def __init__(self, n_features=32, sigma_center=0.25, sigma_surround=0.5, device="cpu"):
        self.n = n_features
        self.device = device

        # 1D visual arc [-1 .. 1]
        self.u = torch.linspace(-1, 1, n_features, device=device).view(1, -1)

        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround

    # ------------------------------------------------------------
    def render(self, dx, dy, size):
        """
        dx, dy : relative position
        size   : proxy for retinal angular area
        """

        # angle of object
        angle = math.atan2(dy, dx)                     # -pi..pi
        az = angle / math.pi                           # -1..1
        az = max(-1.0, min(1.0, az))                   # clamp

        az = torch.tensor(az, dtype=torch.float32, device=self.device)

        # --- Center receptive field ----------------------------
        center = torch.exp(-0.5 * ((self.u - az) ** 2) / (self.sigma_center ** 2))

        # --- Surround receptive field ---------------------------
        surround = torch.exp(-0.5 * ((self.u - az) ** 2) / (self.sigma_surround ** 2))

        # ON: center - surround (LoG)
        ON_raw = center - 0.5 * surround
        ON_raw = torch.relu(ON_raw)

        # OFF: surround - center
        OFF_raw = surround - 0.5 * center
        OFF_raw = torch.relu(OFF_raw)

        # size modulation
        # small → ON dominant, large → OFF dominant (loom)
        ON = ON_raw * (1.0 / (1.0 + size))
        OFF = OFF_raw * min(1.0, size / 10.0)

        return {
            "ON": ON,
            "OFF": OFF
        }
