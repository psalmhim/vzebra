# ============================================================
# MODULE: group_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v18.0 (2025-11-18)
#
# PURPOSE:
#     Implements multi-agent collective precision coupling and
#     emergent leadership based on shared expected free energy.
# ============================================================

import torch
from modules.module_template import BaseModule

class GroupField(BaseModule):
    """
    Collective precision and leadership emergence among N agents.
    """

    def __init__(self, n_agents=5, coop_gain=0.4, leader_gain=0.2, device="cpu"):
        super().__init__(device=device)
        self.n_agents = n_agents
        self.coop_gain = coop_gain
        self.leader_gain = leader_gain
        self.group_precision = torch.ones(1, device=device) * 0.5
        self.leader_index = 0
        self.group_valence = 0.0

    # --------------------------------------------------------
    def step(self, efe_list, conf_list):
        efe = torch.tensor(efe_list, device=self.device)
        conf = torch.tensor(conf_list, device=self.device)

        # Group average EFE and confidence
        efe_mean = efe.mean()
        conf_mean = conf.mean()

        # Leader = agent with highest (confidence / EFE)
        leader_score = conf / (efe + 1e-5)
        self.leader_index = torch.argmax(leader_score).item()

        # Group precision synchrony
        precision_coupling = 1.0 / (1.0 + (efe - efe_mean).abs())
        self.group_precision = 0.9 * self.group_precision + 0.1 * precision_coupling.mean()

        # Group-level valence (alignment measure)
        self.group_valence = float(conf_mean - efe_mean)

        # Adjust individual EFE through collective coupling
        efe_coupled = efe - self.coop_gain * (efe - efe_mean)
        efe_coupled[self.leader_index] -= self.leader_gain * (conf[self.leader_index] - conf_mean)

        return efe_coupled.tolist(), self.group_precision.item(), self.group_valence, self.leader_index

