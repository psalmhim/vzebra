# ============================================================
# MODULE: empathy_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v17.1 (2025-11-17)
#
# PURPOSE:
#     Adds hierarchical empathy inference: each agent predicts
#     its partner’s belief (EFE, confidence, goal) and uses it
#     to modulate its own dopaminergic gain and precision.
# ============================================================

import torch
from modules.module_template import BaseModule

class EmpathyField(BaseModule):
    """
    Hierarchical empathy module for inferring peer belief states.
    Each agent tracks an estimate of its partner’s expected free energy,
    confidence, and inferred goal.
    """

    def __init__(self, n_goals=3, lr=0.05, device="cpu"):
        super().__init__(device=device)
        self.n_goals = n_goals
        self.lr = lr
        self.peer_efe_est = 0.5
        self.peer_conf_est = 0.5
        self.goal_belief = torch.ones(n_goals, device=device) / n_goals
        self.empathic_valence = 0.0

    # --------------------------------------------------------
    def step(self, peer_efe_obs, peer_conf_obs, peer_choice):
        """
        Update internal belief about peer’s mental state.
        """
        # EFE and confidence estimation update
        self.peer_efe_est += self.lr * (peer_efe_obs - self.peer_efe_est)
        self.peer_conf_est += self.lr * (peer_conf_obs - self.peer_conf_est)

        # goal belief update (soft categorical)
        goal_vec = torch.zeros(self.n_goals, device=self.device)
        goal_vec[int(peer_choice) % self.n_goals] = 1.0
        self.goal_belief = 0.9 * self.goal_belief + 0.1 * goal_vec

        # compute empathic valence
        # if peer is confident and has low EFE → positive empathy
        self.empathic_valence = float(self.peer_conf_est - self.peer_efe_est)

        return self.empathic_valence, self.goal_belief.clone(), self.peer_efe_est, self.peer_conf_est

