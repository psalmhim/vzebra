# ============================================================
# MODULE: multiagent_inference_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v24.0 (2025-11-24)
#
# PURPOSE:
#     Implements multi-agent hierarchical active inference (MA-HTAIF)
#     with shared world models, trust-weighted belief exchange,
#     and emergent collective coherence.
# ============================================================

import torch
from modules.module_template import BaseModule
from modules.hierarchical_temporal_field import HierarchicalTemporalField


class MultiAgentInferenceField(BaseModule):
    """
    Multi-agent extension of Hierarchical Temporal Active Inference.
    Each agent runs its own HTAIF but exchanges policy/narrative posteriors
    with others, modulated by dynamic trust weights.
    """

    def __init__(self, n_agents=5, n_state=64, n_policy=3, device="cpu"):
        super().__init__(device=device)
        self.n_agents = n_agents
        self.agents = [HierarchicalTemporalField(n_state=n_state, n_policy=n_policy, device=device)
                       for _ in range(n_agents)]
        self.trust_matrix = torch.ones(n_agents, n_agents, device=device)
        self.trust_decay = 0.98
        self.trust_lr = 0.02
        self.group_coherence = 1.0

    # --------------------------------------------------------
    def update_trust(self, efe_list):
        """Update inter-agent trust based on EFE similarity."""
        efe_tensor = torch.tensor(efe_list, device=self.device)
        diff = torch.cdist(efe_tensor.unsqueeze(1), efe_tensor.unsqueeze(1), p=1)
        delta = torch.exp(-diff)
        self.trust_matrix = self.trust_decay * self.trust_matrix + self.trust_lr * delta
        self.trust_matrix.fill_diagonal_(1.0)
        return self.trust_matrix

    # --------------------------------------------------------
    def collective_step(self, states, dopa_gains):
        """Perform one collective inference cycle for all agents."""
        efe_fast_list, efe_mid_list, efe_slow_list = [], [], []
        coherence_list, intent_list = [], []

        # Each agent infers locally
        for i, agent in enumerate(self.agents):
            result = agent.step(states[i], dopa_gains[i])
            efe_fast_list.append(result["EFE_fast"])
            efe_mid_list.append(result["EFE_mid"])
            efe_slow_list.append(result["EFE_slow"])
            coherence_list.append(result["coherence"])
            intent_list.append(result["intent"])

        # Trust update based on EFE alignment
        efe_means = [(a + b + c) / 3 for a, b, c in zip(efe_fast_list, efe_mid_list, efe_slow_list)]
        self.update_trust(efe_means)

        # Compute collective coherence (social alignment)
        mean_coherence = sum(coherence_list) / len(coherence_list)
        trust_coherence = self.trust_matrix.mean().item()
        self.group_coherence = 0.5 * mean_coherence + 0.5 * trust_coherence

        # Return aggregated group metrics
        return {
            "EFE_mean": sum(efe_means) / len(efe_means),
            "trust": trust_coherence,
            "coherence": self.group_coherence,
            "intent_mean": sum(intent_list) / len(intent_list),
        }
