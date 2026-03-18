# ============================================================
# MODULE: social_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v17.0 (2025-11-17)
#
# PURPOSE:
#     Implements social coupling between multiple Active Inference agents.
#     Each agent adjusts its precision and expected free energy
#     in response to peers' behavior.
# ============================================================

import torch
from modules.module_template import BaseModule

class SocialField(BaseModule):
    """
    Social coupling between agents' expected free energies.
    Supports cooperative (positive) or competitive (negative) coupling.
    """

    def __init__(self, n_agents=2, coupling_strength=0.3, cooperative=True, device="cpu"):
        super().__init__(device=device)
        self.n_agents = n_agents
        self.coupling_strength = coupling_strength
        self.sign = 1.0 if cooperative else -1.0
        self.precision_coupling = torch.ones(n_agents, device=device) * 0.5
        self.social_valence = 0.0

    # --------------------------------------------------------
    def step(self, efe_list):
        """
        efe_list: tensor or list of agents' expected free energies.
        Returns socially coupled EFE and valence term.
        """
        efe = torch.tensor(efe_list, device=self.device, dtype=torch.float32)
        mean_efe = efe.mean()
        diff = efe - mean_efe

        # social coupling term: penalize or reward divergence
        social_term = self.sign * self.coupling_strength * (diff ** 2)
        coupled_efe = efe + social_term

        # compute social valence (alignment measure)
        self.social_valence = -self.sign * diff.mean().item()
        # update shared precision (social synchrony)
        self.precision_coupling = 0.9 * self.precision_coupling + 0.1 * torch.sigmoid(-diff.abs())

        return coupled_efe, self.social_valence, self.precision_coupling.mean().item()
