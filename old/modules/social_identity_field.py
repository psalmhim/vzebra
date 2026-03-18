# ============================================================
# MODULE: social_identity_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v19.0 (2025-11-19)
#
# PURPOSE:
#     Models selective in-group/out-group precision coupling and
#     emergent social polarization.
# ============================================================

import torch
from modules.module_template import BaseModule


class SocialIdentityField(BaseModule):
    """
    Extends GroupField to include identity-based selective coupling.
    Each agent has an identity vector that biases coupling strength.
    """

    def __init__(self, n_agents=6, n_groups=2, base_coupling=0.3,
                 bias_gain=0.4, memory_rate=0.9, device="cpu"):
        super().__init__(device=device)
        self.n_agents = n_agents
        self.n_groups = n_groups
        self.base_coupling = base_coupling
        self.bias_gain = bias_gain
        self.memory_rate = memory_rate

        # Assign random group identities (0 or 1)
        self.identities = torch.randint(0, n_groups, (n_agents,), device=device)
        # Memory matrix of past alignment (bias history)
        self.bias_memory = torch.zeros((n_agents, n_agents), device=device)
        self.polarization_index = 0.0

    # --------------------------------------------------------
    def step(self, efe_list, conf_list):
        efe = torch.tensor(efe_list, device=self.device)
        conf = torch.tensor(conf_list, device=self.device)

        efe_mean = efe.mean()
        conf_mean = conf.mean()

        # Identity-based coupling matrix (+1 for same group, −1 for different)
        id_matrix = (self.identities.unsqueeze(1) == self.identities.unsqueeze(0)).float()
        id_matrix = 2 * id_matrix - 1  # convert to +1 / −1

        # Dynamic coupling update (cooperation vs. competition)
        coupling = self.base_coupling * id_matrix + self.bias_gain * self.bias_memory
        efe_diffs = efe.unsqueeze(0) - efe.unsqueeze(1)

        # Bias memory update (Hebbian: consistent alignment strengthens bond)
        align = -efe_diffs.sign() * id_matrix
        self.bias_memory = self.memory_rate * self.bias_memory + (1 - self.memory_rate) * align

        # Polarization index: between-group EFE gap
        group_means = [efe[self.identities == g].mean() for g in range(self.n_groups)]
        if len(group_means) == 2:
            self.polarization_index = float(abs(group_means[0] - group_means[1]))

        # Apply selective coupling
        efe_coupled = efe + torch.sum(coupling * efe_diffs, dim=1) / self.n_agents
        efe_coupled = torch.clamp(efe_coupled, min=0.0)

        # Social cohesion measure (in-group alignment)
        cohesion = float((id_matrix * (-efe_diffs.abs())).mean())

        return efe_coupled.tolist(), self.polarization_index, cohesion, self.identities.tolist()

