# ============================================================
# MODULE: social_meta_inference_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v26.0 (2025-11-26)
#
# PURPOSE:
#     Extends SocialInferenceField with meta-social inference:
#     agents model others’ belief models and update collective
#     cultural priors through recursive inference.
# ============================================================

import torch
from modules.module_template import BaseModule
from modules.social_inference_field import SocialInferenceField


class SocialMetaInferenceField(BaseModule):
    """
    Meta-social active inference field:
    Each agent models not only others’ states (level-1 empathy)
    but also others’ models (level-2 meta-empathy).
    """

    def __init__(self, n_agents=5, n_state=64, n_policy=3, device="cpu"):
        super().__init__(device=device)
        self.social_field = SocialInferenceField(n_agents=n_agents, n_state=n_state, n_policy=n_policy, device=device)
        self.n_agents = n_agents

        # meta-representations: agents’ beliefs about others’ models
        self.meta_beliefs = torch.zeros(n_agents, n_agents, n_policy, device=device)
        self.meta_precision = torch.ones(n_agents, n_agents, device=device)
        self.cultural_prior = torch.zeros(n_state, device=device)
        self.lr_meta = 0.05
        self.lr_culture = 0.01

    # --------------------------------------------------------
    def infer_others_models(self, efe_self, efe_matrix):
        """
        Estimate how well each agent predicts others’ free energies.
        Updates meta-precision and belief alignment.
        """
        efe_tensor = torch.tensor(efe_matrix, device=self.device)
        diff = (efe_tensor.unsqueeze(0) - efe_tensor.unsqueeze(1)).abs()
        sim = torch.exp(-diff)
        self.meta_precision = 0.9 * self.meta_precision + 0.1 * sim.mean(dim=2)
        self.meta_precision = torch.clamp(self.meta_precision, 0.1, 1.0)
        return self.meta_precision.mean().item()

    # --------------------------------------------------------
    def update_cultural_prior(self, shared_goal, empathy, meta_coherence):
        """
        Slowly consolidate shared priors across agents (culture).
        """
        delta_culture = self.lr_culture * (empathy * meta_coherence) * shared_goal
        self.cultural_prior = 0.99 * self.cultural_prior + delta_culture
        self.cultural_prior = torch.tanh(self.cultural_prior)
        return float(self.cultural_prior.abs().mean())

    # --------------------------------------------------------
    def collective_meta_step(self, states, dopa_gains):
        """
        Full multi-agent meta-inference cycle:
        - Cooperative social inference (from v25)
        - Meta-social inference (belief about others’ models)
        - Cultural prior consolidation
        """
        # Step 1: base social inference
        base_result = self.social_field.cooperative_step(states, dopa_gains)
        efe = base_result["EFE"]

        # Step 2: meta-social inference (agents model others)
        efe_matrix = [[efe + torch.randn(1).item() * 0.05 for _ in range(self.n_agents)]
                      for _ in range(self.n_agents)]
        meta_coherence = self.infer_others_models(efe, efe_matrix)

        # Step 3: update shared cultural prior
        culture_strength = self.update_cultural_prior(
            self.social_field.shared_goal,
            base_result["empathy"],
            meta_coherence
        )

        return {
            "EFE": efe,
            "trust": base_result["trust"],
            "coherence": base_result["coherence"],
            "empathy": base_result["empathy"],
            "meta_coherence": meta_coherence,
            "culture_strength": culture_strength,
        }

