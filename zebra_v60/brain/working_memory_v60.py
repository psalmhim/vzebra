"""
Goal-Conditioned Working Memory for Active Inference.

Maintains a latent memory trace that integrates classification, goal state,
and neuromodulatory signals. Provides temporal context for policy selection.

Pure numpy — no torch dependency.
"""
import numpy as np


class WorkingMemory_v60:
    """Working memory with goal-conditioned integration and dopamine modulation."""

    def __init__(self, n_latent=None, n_goals=4, buffer_len=20):
        # Auto-size latent dim: 5(cls) + n_goals(goal) + 5(scalars) + n_goals(trace)
        if n_latent is None:
            n_latent = 10 + 2 * n_goals
        self.n_latent = n_latent
        self.n_goals = n_goals
        self.buffer_len = buffer_len

        self.alpha = 0.1   # base retention rate
        self.gamma = 2.0   # gain-state denominator scaling

        self.m = np.zeros(n_latent)       # memory trace
        self.goal_trace = np.zeros(n_goals)  # decaying goal history
        self.cls_buffer = []               # recent classification buffer

    def step(self, cls_probs, goal_vec, dopa, cms, F_visual, pi_OT, pi_PC,
             gain_state=1.0):
        """Update working memory with current observations.

        Args:
            cls_probs: array [5] — classifier probabilities
            goal_vec: array [n_goals] — one-hot active goal
            dopa: float — dopamine level (0–1)
            cms: float — cross-modal surprise
            F_visual: float — visual free energy
            pi_OT: float — OT precision
            pi_PC: float — PC precision
            gain_state: float — overall gain modulation

        Returns:
            m_out: array [n_latent] — current memory state
            alpha_eff: float — effective retention rate (for diagnostics)
            cls_summary: array [5] — mean classification over buffer
        """
        # Update goal trace (exponential decay)
        self.goal_trace = 0.9 * self.goal_trace + 0.1 * goal_vec

        # Construct latent vector z_t
        # Layout: [cls(5), goal(n_goals), scalars(5), goal_trace(n_goals)]
        ng = self.n_goals
        z_t = np.zeros(self.n_latent)
        z_t[0:5] = cls_probs                  # classification
        z_t[5:5 + ng] = goal_vec               # active goal
        s = 5 + ng
        z_t[s] = dopa                          # dopamine
        z_t[s + 1] = cms                       # cross-modal surprise
        z_t[s + 2] = F_visual                  # free energy
        z_t[s + 3] = pi_OT                    # OT precision
        z_t[s + 4] = pi_PC                    # PC precision
        z_t[s + 5:s + 5 + ng] = self.goal_trace  # decaying goal history

        # Adaptive retention rate
        alpha_eff = self.alpha / (1.0 + self.gamma * gain_state)
        alpha_eff = np.clip(alpha_eff, 0.01, 0.2)

        # Dopamine modulates retention (high dopa = stronger memory)
        dopa_mod = 1.0 + 0.2 * (dopa - 0.5)

        # CMS modulates decay (high surprise = faster update)
        effective_alpha = alpha_eff * (1.0 + 0.3 * cms)
        effective_alpha = np.clip(effective_alpha, 0.01, 0.3)

        # Memory update
        self.m = (1.0 - effective_alpha) * self.m * dopa_mod + effective_alpha * z_t

        # Prevent unbounded growth
        norm = np.linalg.norm(self.m)
        if norm > 5.0:
            self.m = self.m * (5.0 / norm)

        # Classification buffer for temporal smoothing
        self.cls_buffer.append(cls_probs.copy())
        if len(self.cls_buffer) > self.buffer_len:
            self.cls_buffer.pop(0)

        cls_summary = np.mean(self.cls_buffer, axis=0)

        return self.m.copy(), alpha_eff, cls_summary

    def get_mean(self):
        """Scalar summary of memory activation for policy selection."""
        return float(np.abs(self.m).mean())

    def reset(self):
        self.m = np.zeros(self.n_latent)
        self.goal_trace = np.zeros(self.n_goals)
        self.cls_buffer = []
