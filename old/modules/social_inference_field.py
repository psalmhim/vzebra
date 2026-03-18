# ============================================================
# MODULE: social_inference_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v25.0 (2025-11-25)
#
# PURPOSE:
#     Adds communication, empathy, and cooperation to the
#     Multi-Agent Hierarchical Active Inference (MA-HTAIF) framework.
# ============================================================

import torch
import torch.nn.functional as F
from modules.module_template import BaseModule
from modules.multiagent_inference_field import MultiAgentInferenceField


class SocialInferenceField(BaseModule):
    """
    Multi-agent social cognition extension.
    - Adds belief exchange (communication)
    - Empathy inference (predict others’ EFE/posteriors)
    - Shared goal priors and cooperation field
    """

    def __init__(self, n_agents=5, n_state=64, n_policy=3, device="cpu"):
        super().__init__(device=device)
        self.field = MultiAgentInferenceField(n_agents=n_agents, n_state=n_state, n_policy=n_policy, device=device)
        self.n_agents = n_agents
        self.comm_precision = torch.ones(n_agents, n_agents, device=device)
        self.shared_goal = torch.zeros(n_state, device=device)
        self.empathy_lr = 0.05
        self.coop_gain = 0.3
        self.comm_decay = 0.98

    # --------------------------------------------------------
    def communicate(self, q_beliefs):
        """
        Agents exchange compressed beliefs (policy posteriors).
        Communication precision determines how strongly messages
        affect receiver’s posterior.
        """
        n = len(q_beliefs)
        q_matrix = torch.stack(q_beliefs)
        mean_belief = q_matrix.mean(dim=0)
        for i in range(n):
            q_beliefs[i] = (1 - self.comm_precision[i].mean()) * q_beliefs[i] + \
                           self.comm_precision[i].mean() * mean_belief
        self.comm_precision *= self.comm_decay
        return q_beliefs

    # --------------------------------------------------------
    def empathic_inference(self, efe_self, efe_others):
        """
        Infer others’ affective/goal states by minimizing EFE difference.
        """
        diff = torch.tensor(efe_self - efe_others, device=self.device)
        empathy = torch.exp(-diff.abs())
        self.comm_precision += self.empathy_lr * (empathy - self.comm_precision)
        self.comm_precision = torch.clamp(self.comm_precision, 0.1, 1.0)
        return empathy.mean().item()

    # --------------------------------------------------------
    def cooperative_step(self, states, dopa_gains):
        """
        Perform full social active inference cycle with:
        - Communication
        - Empathy update
        - Cooperative goal adjustment
        """
        # Step 1: Individual inference
        result = self.field.collective_step(states, dopa_gains)

        # Step 2: Communication (belief exchange)
        q_beliefs = [agent.L2.q_pi for agent in self.field.agents]
        q_beliefs = self.communicate(q_beliefs)

        # Step 3: Empathy (predict others’ free energy)
        efe_list = [result["EFE_mean"] for _ in range(self.n_agents)]
        empathy_score = self.empathic_inference(result["EFE_mean"], sum(efe_list)/len(efe_list))

        # Step 4: Shared goal modulation
        coop_term = self.coop_gain * empathy_score * self.field.group_coherence
        self.shared_goal += coop_term * torch.randn_like(self.shared_goal)
        self.shared_goal = torch.tanh(self.shared_goal)

        # Step 5: Update group-level coherence
        group_eff = result["EFE_mean"] - coop_term

        return {
            "EFE": group_eff,
            "trust": result["trust"],
            "coherence": result["coherence"],
            "empathy": empathy_score,
            "intent": result["intent_mean"],
        }

