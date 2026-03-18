# ============================================================
# MODULE: working_memory.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v15.2 (2025-11-15)
#
# PURPOSE:
#     Adds dopamine-triggered replay and consolidation.
#     When dopamine bursts occur, the module strengthens
#     recent latent traces (short-term replay).
#
# UPDATE SUMMARY:
#     • Adds replay buffer and replay() method
#     • Replay triggered by dopamine > threshold or |RPE| small
#     • Consolidation reinforces m (memory) and W_rec weights
# ============================================================

import torch
import torch.nn as nn
from modules.module_template import BaseModule


class WorkingMemory(BaseModule):
    """Goal-conditioned working memory with dopaminergic replay."""

    def __init__(self, n_latent=64, n_goal=None, device="cpu", mode="contextual"):
        super().__init__(device=device)
        self.mode = mode
        self.n_latent = n_latent
        self.n_goal = n_goal or 0

        # Memory trace
        self.m = torch.zeros(1, n_latent, device=device)

        # Parameters
        self.alpha = 0.1
        self.beta = 0.05
        self.dopa_gain = 0.2
        self.cms_gain = 0.2
        self.goal_gain = 0.3
        self.replay_gain = 0.15
        self.gamma = 2.0

        # Replay buffer
        self.replay_buf = []
        self.replay_len = 5
        self.replay_threshold = 0.7  # dopamine threshold for replay

        # Recurrent weights
        self.W_rec = nn.Parameter(0.01 * torch.randn(n_latent, n_latent, device=device))

    # ------------------------------------------------------------
    def step(self, z_t, dopa=0.0, cms=0.0, goal_vec=None, gain_state=1.0, rpe=0.0, audio_latent=None):
        """Update memory and trigger replay when dopamine surge occurs."""

        if audio_latent is not None:
            z_t = z_t + 0.1 * audio_latent

        z_t = z_t.to(self.device)

        # Recurrent term
        rec_term = self.m @ self.W_rec if self.mode in ["recurrent", "contextual"] else 0.0

        # Bias terms
        bias = 0.0
        if self.mode == "contextual":
            bias = self.dopa_gain * dopa + self.cms_gain * cms

        # Goal term
        goal_term = 0.0
        if self.n_goal and goal_vec is not None:
            g = goal_vec.to(self.device)
            goal_term = self.goal_gain * g.mean()

        # Adaptive retention coefficient
        alpha_eff = self.alpha / (1.0 + self.gamma * gain_state)
        alpha_eff = float(torch.clamp(torch.tensor(alpha_eff), 0.01, 0.2))

        # Exponential integration
        self.m = (1 - alpha_eff) * self.m + alpha_eff * (z_t + self.beta * rec_term + bias + goal_term)

        # Store latent for possible replay
        self.replay_buf.append(z_t.detach().clone())
        if len(self.replay_buf) > self.replay_len:
            self.replay_buf.pop(0)

        # Trigger replay if dopamine high or RPE near zero
        if (float(dopa) > self.replay_threshold) or (abs(float(rpe)) < 0.05):
            self.replay()

        return self.m.detach(), alpha_eff

    # ------------------------------------------------------------
    def replay(self):
        """Replay recent latent patterns to reinforce memory."""
        if not self.replay_buf:
            return
        for z in self.replay_buf[-3:]:
            self.m = (1 - self.replay_gain) * self.m + self.replay_gain * z
            # optional weight reinforcement
            self.W_rec.data += 1e-4 * (z.T @ self.m)
        self.W_rec.data /= (1.0 + self.W_rec.data.abs().mean())

    # ------------------------------------------------------------
    def reset_state(self):
        self.m.zero_()
        self.replay_buf.clear()

    # ------------------------------------------------------------
    def debug_state(self):
        return {
            "mode": self.mode,
            "mean_mem": float(self.m.mean()),
            "std_mem": float(self.m.std()),
            "replay_size": len(self.replay_buf),
        }
