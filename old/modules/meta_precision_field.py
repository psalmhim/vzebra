# ============================================================
# MODULE: meta_precision_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v20.0 (2025-11-20)
#
# PURPOSE:
#     Implements meta-level precision monitoring and
#     collective trust inference across agents.
# ============================================================

import torch
from modules.module_template import BaseModule


class MetaPrecisionField(BaseModule):
    """
    Estimates the volatility of an agent's precision (meta-confidence)
    and computes trust weighting toward others' beliefs.
    """

    def __init__(self, mode, n_agents=5, lr=0.1, trust_tau=0.9, trust_gain=0.4, device="cpu"):
        super().__init__(device=device)
        self.mode = mode
        self.n_agents = n_agents
        self.lr = lr
        self.trust_tau = trust_tau
        self.trust_gain = trust_gain

        self.self_meta_var = torch.zeros(n_agents, device=device)
        self.trust_matrix = torch.ones(n_agents, n_agents, device=device) * 0.5
        self.meta_valence = torch.zeros(n_agents, device=device)

    # --------------------------------------------------------
    def step(self, conf_list):
        conf = torch.tensor(conf_list, device=self.device)
        dconf = torch.diff(conf, prepend=conf[:1])  # temporal derivative approximation
        self.self_meta_var = (1 - self.lr) * self.self_meta_var + self.lr * (dconf ** 2)

        # Compute meta-valence: stable precision = positive valence
        self.meta_valence = torch.exp(-5 * self.self_meta_var) - 0.5

        # Trust weighting: agents with lower meta-variance are trusted more
        trust_base = torch.exp(-torch.abs(self.self_meta_var.unsqueeze(1) - self.self_meta_var.unsqueeze(0)))
        self.trust_matrix = self.trust_tau * self.trust_matrix + (1 - self.trust_tau) * trust_base

        # Collective trust index (mean mutual confidence)
        trust_index = float(self.trust_matrix.mean())

        return self.meta_valence.tolist(), self.trust_matrix.clone(), trust_index

