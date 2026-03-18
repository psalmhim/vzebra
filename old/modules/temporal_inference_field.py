# ============================================================
# MODULE: temporal_inference_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v22.0 (2025-11-22)
#
# PURPOSE:
#     Implements deep temporal active inference — full generative
#     p(s,a,π) model, EFE decomposition, and policy sampling.
# ============================================================

import torch
import torch.nn.functional as F
from modules.module_template import BaseModule


class TemporalInferenceField(BaseModule):
    """
    Full temporal active inference controller.
    Predicts next sensory state under candidate policies,
    computes Expected Free Energy (EFE) = risk + ambiguity,
    and samples policy to minimize it.
    """

    def __init__(self, mode, n_state=64, n_policy=3, T=3, beta=1.0, gamma=0.9, device="cpu"):
        super().__init__(device=device)
        self.mode = mode
        self.n_state = n_state
        self.n_policy = n_policy
        self.T = T
        self.beta = beta
        self.gamma = gamma

        # Generative parameters
        self.A = torch.randn(n_state, n_policy, device=device) * 0.01  # state→outcome mapping
        self.B = torch.randn(n_state, n_state, n_policy, device=device) * 0.01  # transition
        self.C = torch.zeros(n_state, device=device)  # preferred outcomes (goal prior)

        # Internal posterior over policies
        self.q_pi = torch.ones(n_policy, device=device) / n_policy
        self.last_action = 1
        self.efe_trace = []

    # --------------------------------------------------------
    def predict_future(self, s_t):
        """Predict T-step future outcomes for each policy."""
        pred_states = []
        for p in range(self.n_policy):
            s = s_t.clone()
            states = [s]
            for _ in range(self.T):
                s = torch.tanh(self.B[:, :, p] @ s)
                states.append(s)
            pred_states.append(torch.stack(states))
        return torch.stack(pred_states)  # (policy × T × state)

    # --------------------------------------------------------
    def compute_EFE(self, pred_states):
        """Compute Expected Free Energy = risk + ambiguity."""
        EFE = []
        for p in range(self.n_policy):
            s_pred = pred_states[p, -1]
            risk = F.mse_loss(s_pred, self.C)  # deviation from preferred outcomes
            ambiguity = s_pred.var() * 0.1
            efe = risk + ambiguity
            EFE.append(efe)
        return torch.tensor(EFE, device=self.device)

    # --------------------------------------------------------
    def update_posterior(self, EFE, dopa_gain=1.0):
        """Update posterior q(π|s,a) using precision-weighted RPE."""
        precision = self.beta * dopa_gain
        log_post = -precision * EFE
        self.q_pi = F.softmax(log_post, dim=0)
        choice = torch.multinomial(self.q_pi, 1).item()
        self.last_action = choice
        self.efe_trace.append(float(EFE.mean()))
        return choice, EFE.mean().item()

    # --------------------------------------------------------
    def step(self, s_t, dopa_gain=1.0):
        pred_states = self.predict_future(s_t)
        EFE = self.compute_EFE(pred_states)
        choice, efe_mean = self.update_posterior(EFE, dopa_gain)
        return choice, efe_mean, self.q_pi.detach()

