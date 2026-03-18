# ============================================================
# MODULE: deep_inference_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v21.0 (2025-11-21)
#
# PURPOSE:
#     Integrates multi-level EFE terms (sensory, policy,
#     meta-confidence, and social alignment) into one deep
#     temporal Active Inference control signal.
# ============================================================

import torch
from modules.module_template import BaseModule


class DeepInferenceField(BaseModule):
    """
    Computes integrated Expected Free Energy (EFE) over multiple
    hierarchical levels of inference.
    """

    def __init__(self, w_sensory=1.0, w_policy=0.8, w_meta=0.6, w_social=0.5,
                 eta=0.1, device="cpu"):
        super().__init__(device=device)
        self.w_sensory = w_sensory
        self.w_policy = w_policy
        self.w_meta = w_meta
        self.w_social = w_social
        self.eta = eta
        self.integrated_EFE = 0.0
        self.action_intent = 0.0

    # --------------------------------------------------------
    def step(self, efe_sensory, efe_policy, meta_valence, trust_index):
        # Weighted combination of EFE terms
        total_efe = (
            self.w_sensory * efe_sensory +
            self.w_policy * efe_policy -
            self.w_meta * meta_valence -
            self.w_social * trust_index
        )

        # Smooth integration over time
        self.integrated_EFE = (1 - self.eta) * self.integrated_EFE + self.eta * total_efe

        # Compute “intentional action” strength: how strongly the agent commits
        self.action_intent = torch.tanh(torch.tensor(-total_efe)).item()

        return self.integrated_EFE, self.action_intent
