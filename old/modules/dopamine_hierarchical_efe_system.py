# ============================================================
# MODULE: dopamine_hierarchical_efe_system.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v17.0 (2025-11-17)
#
# PURPOSE:
#     Slow hierarchical dopamine system implementing:
#         • expected free energy (EFE) estimation
#         • hierarchical uncertainty tracking
#         • precision modulation
#
# BIOLOGICAL ANALOGUE:
#     VTA → Striatum/Pallium dopamine,
#     policy-level precision and motivation.
# ============================================================

import torch
from modules.module_template import BaseModule


class DopamineHierarchicalEFESystem(BaseModule):
    """Hierarchical dopaminergic subsystem for EFE-based precision modulation."""

    def __init__(self, device="cpu", lambda_uncertainty=0.3):
        super().__init__(device=device)
        self.lambda_uncertainty = lambda_uncertainty

        self.rpe = 0.0
        self.dopa = 0.5
        self.precision_est = 0.5

    # --------------------------------------------------------
    def step(self, Fv, Fa, Fb, cms, precision_inputs=None):
        """
        Compute expected free energy (EFE) and hierarchical dopamine.
        Returns:
            (rpe_hier, dopa_hier, efe, precision_est)
        """
        F_mean = float((Fv + Fa + Fb) / 3)

        # hierarchical precision estimate
        new_precision = 1.0 / (1.0 + F_mean)
        self.precision_est = 0.9 * self.precision_est + 0.1 * new_precision

        # components of EFE
        uncertainty_term = self.lambda_uncertainty * (1 - self.precision_est)
        novelty_term = 0.1 * abs(cms)

        efe = F_mean + uncertainty_term + novelty_term

        # dopamine = negative EFE
        self.rpe = -efe
        self.dopa = float(torch.sigmoid(torch.tensor(self.rpe + 0.5)))

        return self.rpe, self.dopa, efe, self.precision_est

    # --------------------------------------------------------
    def reset_state(self):
        self.rpe = 0.0
        self.dopa = 0.5
        self.precision_est = 0.5

    # --------------------------------------------------------
    def debug_state(self):
        return {
            "rpe_hier": float(self.rpe),
            "dopa_hier": float(self.dopa),
            "precision_est": float(self.precision_est),
        }

