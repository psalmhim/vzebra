# ============================================================
# MODULE: social_coupling.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v32.0 (2025-12-02)
#
# PURPOSE:
#     Implements cross-agent coupling for social active inference.
#     Agents share precision-weighted belief and dopamine states
#     to minimize *shared free energy*.
# ============================================================

import torch
from modules.module_template import BaseModule


class SocialCoupling(BaseModule):
    """
    Coupling between two active inference agents.
    - Shared precision (synchrony)
    - Dopamine resonance (social reward)
    - Empathy field (mutual prediction)
    """

    def __init__(self, lambda_sync=0.3, lambda_empathy=0.2, device="cpu"):
        super().__init__(device=device)
        self.lambda_sync = lambda_sync
        self.lambda_empathy = lambda_empathy

    def step(self, agentA, agentB):
        """
        Compute bidirectional coupling between agents.
        agentA, agentB : dict outputs from each agent
        """
        dopa_sync = 1.0 - abs(agentA["Dopa"] - agentB["Dopa"])
        motive_align = 1.0 - abs(agentA["Mot"] - agentB["Mot"])
        energy_align = 1.0 - abs(agentA["Energy"] - agentB["Energy"])

        empathy = self.lambda_empathy * (motive_align + energy_align) / 2
        synchrony = self.lambda_sync * dopa_sync

        shared_reward = synchrony + empathy
        return shared_reward, empathy, synchrony

