# ============================================================
# MODULE: hierarchical_temporal_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v23.0 (2025-11-23)
#
# PURPOSE:
#     Implements multi-layer Hierarchical Temporal Active Inference (HTAIF)
#     with nested Expected Free Energy loops across fast, mid, and slow scales.
# ============================================================

import torch
from modules.module_template import BaseModule
from modules.temporal_inference_field import TemporalInferenceField
from modules.deep_inference_field import DeepInferenceField
from modules.meta_precision_field import MetaPrecisionField


class HierarchicalTemporalField(BaseModule):
    """
    Multi-scale temporal active inference hierarchy.
    - Layer 1: Fast sensory–motor loop (short horizon)
    - Layer 2: Mid-level goal–policy loop (medium horizon)
    - Layer 3: Slow narrative–identity loop (long horizon)
    """

    def __init__(self, n_state=64, n_policy=3, device="cpu"):
        super().__init__(device=device)

        # Layers
        self.L1 = TemporalInferenceField(n_state=n_state, n_policy=n_policy, T=2, device=device)
        self.L2 = TemporalInferenceField(n_state=n_state, n_policy=n_policy, T=5, device=device)
        self.L3 = TemporalInferenceField(n_state=n_state, n_policy=n_policy, T=10, device=device)

        # Integrators
        self.deep = DeepInferenceField(device=device)
        self.meta = MetaPrecisionField(n_agents=1, device=device)

        # Hierarchical precision coupling
        self.precision_scale = [1.0, 0.7, 0.5]
        self.global_coherence = 1.0

    # --------------------------------------------------------
    def step(self, s_t, dopa_gain=1.0):
        """
        Perform hierarchical active inference update across three temporal scales.
        """
        # Fast sensory–motor
        act_fast, efe_fast, q_fast = self.L1.step(s_t, dopa_gain)
        # Mid-level policy
        act_mid, efe_mid, q_mid = self.L2.step(s_t, dopa_gain * 0.8)
        # Slow narrative–identity
        act_slow, efe_slow, q_slow = self.L3.step(s_t, dopa_gain * 0.6)

        # Compute hierarchical coupling (meta-inference)
        meta_val, _, trust_index = self.meta.step([float(q_fast.mean()), float(q_mid.mean()), float(q_slow.mean())])
        efe_total, intent = self.deep.step(
            efe_sensory=efe_fast,
            efe_policy=efe_mid,
            meta_valence=sum(meta_val)/3,
            trust_index=trust_index
        )

        # Global coherence = alignment of hierarchical posteriors
        coherence = (torch.cosine_similarity(q_fast, q_mid, dim=0) + 
                     torch.cosine_similarity(q_mid, q_slow, dim=0)) / 2
        self.global_coherence = float(coherence.mean())

        return {
            "EFE_fast": efe_fast,
            "EFE_mid": efe_mid,
            "EFE_slow": efe_slow,
            "EFE_total": efe_total,
            "intent": intent,
            "coherence": self.global_coherence,
            "meta_val": sum(meta_val)/3,
            "trust": trust_index,
            "actions": (act_fast, act_mid, act_slow)
        }

