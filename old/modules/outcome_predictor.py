# ============================================================
# MODULE: outcome_predictor.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v16.1 (2025-11-16)
#
# PURPOSE:
#     Predicts expected reward and uncertainty per goal.
#     Used by Active Inference system to compute EFE.
# ============================================================

import torch
from modules.module_template import BaseModule

class OutcomePredictor(BaseModule):
    """Expected outcome predictor for each goal."""

    def __init__(self, n_goals=3, lr=0.05, device="cpu"):
        super().__init__(device=device)
        self.n_goals = n_goals
        self.lr = lr
        self.reward_exp = torch.zeros(n_goals, device=device)
        self.uncertainty_exp = torch.ones(n_goals, device=device) * 0.5

    # --------------------------------------------------------
    def update(self, goal_idx, reward_obs, uncertainty_obs):
        """Update expected outcomes using a delta rule."""
        g = int(goal_idx)
        rpe_r = reward_obs - self.reward_exp[g]
        rpe_u = uncertainty_obs - self.uncertainty_exp[g]
        self.reward_exp[g] += self.lr * rpe_r
        self.uncertainty_exp[g] += self.lr * rpe_u
        return float(rpe_r), float(rpe_u)

    # --------------------------------------------------------
    def predict(self):
        """Return current expected outcomes for all goals."""
        return self.reward_exp.clone(), self.uncertainty_exp.clone()

