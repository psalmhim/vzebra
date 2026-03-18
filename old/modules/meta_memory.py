# ============================================================
# MODULE: meta_memory.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v33.0 (2025-12-03)
#
# PURPOSE:
#     Implements slow-timescale meta-memory and temporal
#     precision modulation for hierarchical active inference.
# ============================================================

import torch
from modules.module_template import BaseModule

class MetaMemory(BaseModule):
    """
    Slow-timescale meta-memory integrating:
    - metabolic energy
    - motivational value (slow DA)
    - social synchrony
    Computes:
    - meta-precision (stability)
    - dopamine baseline drift
    """

    def __init__(self, mode, tau_slow=0.99, device="cpu"):
        super().__init__(device=device)
        self.mode = mode
        self.tau_slow = tau_slow

        self.memory = {"energy": 0.5, "mot": 0.0, "sync": 0.0}
        self.stability = 1.0
        self.baseline_dopa = 0.5

    # --------------------------------------------------------
    def step(self, energy, mot, sync):

        # Convert all inputs to floats (handles tensors & floats)
        energy = float(energy)
        mot = float(mot)
        sync = float(sync)

        # Long-timescale integration
        self.memory["energy"] = (
            self.tau_slow * self.memory["energy"]
            + (1 - self.tau_slow) * energy
        )
        self.memory["mot"] = (
            self.tau_slow * self.memory["mot"]
            + (1 - self.tau_slow) * mot
        )
        self.memory["sync"] = (
            self.tau_slow * self.memory["sync"]
            + (1 - self.tau_slow) * sync
        )

        # Meta-stability → meta-precision
        stability = (
            1.0
            - abs(self.memory["energy"] - energy)
            - abs(self.memory["mot"] - mot)
            - 0.5 * abs(self.memory["sync"] - sync)
        )

        self.stability = float(
            torch.clamp(torch.tensor(stability), 0.0, 1.0)
        )

        # Social synchrony → dopamine baseline
        target = 0.5 + 0.5 * self.memory["sync"]
        self.baseline_dopa = 0.9 * self.baseline_dopa + 0.1 * target

        return self.memory, self.stability, self.baseline_dopa

    # --------------------------------------------------------
    def get_meta_prior(self):
        """Return hierarchical priors."""
        return {
            "energy_prior": self.memory["energy"],
            "mot_prior": self.memory["mot"],
            "precision_gain": self.stability,
            "dopa_baseline": self.baseline_dopa,
        }

    # --------------------------------------------------------
    def reset_state(self):
        self.memory = {"energy": 0.5, "mot": 0.0, "sync": 0.0}
        self.stability = 1.0
        self.baseline_dopa = 0.5

