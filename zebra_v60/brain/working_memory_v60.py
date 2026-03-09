"""
Goal-Conditioned Working Memory for Active Inference.

Maintains a latent memory trace that integrates classification, goal state,
and neuromodulatory signals. Provides temporal context for policy selection.

Includes SpikingWorkingMemory (Level 2 SNN-up): recurrent TwoComp circuit
that sustains activity through self-excitation, modeling PFC persistent firing.
"""
import numpy as np
import torch
import torch.nn as nn


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


class SpikingWorkingMemory(nn.Module):
    """Recurrent spiking working memory (Level 2 SNN-up).

    Models PFC persistent firing via recurrent self-excitation.
    Activity sustains through W_rec when input is consistent,
    and decays naturally when input is withdrawn.

    Wraps the original WorkingMemory_v60 interface: accepts the same
    arguments in step(), returns the same (m_out, alpha_eff, cls_summary).
    """

    def __init__(self, n_latent=None, n_goals=4, buffer_len=20, device="cpu"):
        super().__init__()
        # Auto-size input dim: 5(cls) + n_goals(goal) + 5(scalars) + n_goals(trace)
        if n_latent is None:
            n_latent = 10 + 2 * n_goals
        self.n_input = n_latent
        self.n_latent = n_latent
        self.n_goals = n_goals
        self.buffer_len = buffer_len
        self.device = device

        # Input projection (leaky integrator)
        self.W_in = nn.Parameter(
            0.02 * torch.randn(n_latent, n_latent, device=device))

        # Recurrent self-excitation → persistent firing
        self.W_rec = nn.Parameter(
            0.3 * torch.eye(n_latent, device=device)
            + 0.01 * torch.randn(n_latent, n_latent, device=device))

        # Membrane state
        self.v = torch.zeros(1, n_latent, device=device)
        self.tau = 0.85  # slower decay = longer memory retention

        # Hebbian learning parameters
        self.eta_in = 0.0005    # W_in learning rate
        self.eta_rec = 0.0003   # W_rec learning rate
        self.decay = 0.9999     # weight decay (prevents unbounded growth)
        self._prev_v = torch.zeros(1, n_latent, device=device)

        # Goal trace (kept as numpy for compatibility)
        self.goal_trace = np.zeros(n_goals)
        self.cls_buffer = []

        self.to(device)

    def step(self, cls_probs, goal_vec, dopa, cms, F_visual, pi_OT, pi_PC,
             gain_state=1.0):
        """Update spiking working memory with current observations.

        Same interface as WorkingMemory_v60.step().
        """
        # Update goal trace (exponential decay)
        self.goal_trace = 0.9 * self.goal_trace + 0.1 * goal_vec

        # Construct input vector z_t (same layout as original)
        ng = self.n_goals
        z_np = np.zeros(self.n_latent)
        z_np[0:5] = cls_probs
        z_np[5:5 + ng] = goal_vec
        s = 5 + ng
        z_np[s] = dopa
        z_np[s + 1] = cms
        z_np[s + 2] = F_visual
        z_np[s + 3] = pi_OT
        z_np[s + 4] = pi_PC
        z_np[s + 5:s + 5 + ng] = self.goal_trace

        z_t = torch.tensor(z_np, dtype=torch.float32,
                           device=self.device).unsqueeze(0)

        # Dopamine-modulated time constant (high dopa = stronger retention)
        tau_eff = self.tau + 0.1 * (dopa - 0.5)
        tau_eff = max(0.7, min(0.95, tau_eff))

        # CMS modulates update rate (high surprise = faster integration)
        update_rate = (1 - tau_eff) * (1.0 + 0.3 * cms)
        update_rate = max(0.02, min(0.3, update_rate))

        # Store for Hebbian learning
        self._prev_v = self.v.detach().clone()
        self._last_z_t = z_t.detach().clone()

        # Recurrent spiking dynamics: v = tau*v + (1-tau)*(input + recurrent)
        drive = z_t @ self.W_in
        recurrent = self.v @ self.W_rec
        self.v = tau_eff * self.v + update_rate * (drive + recurrent)

        # Clamp to prevent runaway (analogous to saturation)
        self.v = torch.clamp(self.v, -5.0, 5.0)

        # Convert to numpy for interface compatibility
        m_out = self.v[0].detach().cpu().numpy().copy()

        # Alpha_eff diagnostic (effective update rate)
        alpha_eff = float(update_rate)

        # Classification buffer for temporal smoothing
        self.cls_buffer.append(cls_probs.copy())
        if len(self.cls_buffer) > self.buffer_len:
            self.cls_buffer.pop(0)
        cls_summary = np.mean(self.cls_buffer, axis=0)

        return m_out, alpha_eff, cls_summary

    @property
    def m(self):
        """Numpy view of memory state (compatibility with WorkingMemory_v60)."""
        return self.v[0].detach().cpu().numpy()

    @torch.no_grad()
    def learn(self, rpe, dopa):
        """Dopamine-gated Hebbian update for W_in and W_rec.

        W_in learns which input features drive useful memory states.
        W_rec learns which co-activation patterns should persist.

        Only updates when RPE magnitude is significant (surprise-driven
        consolidation). Dopamine gates the sign: positive RPE strengthens
        active connections, negative RPE weakens them.

        Args:
            rpe: float — reward prediction error from dopamine system
            dopa: float — dopamine level [0, 1]
        """
        if abs(rpe) < 0.05:
            return

        # Dopamine modulation: dopa > 0.5 → potentiate, < 0.5 → depress
        dopa_mod = 2.0 * (dopa - 0.5)  # range [-1, 1]
        signal = rpe * dopa_mod

        # W_in: Hebbian on (input, current state) — strengthens input
        # channels that drove useful memory activations
        # v shape [1, n], _prev_z shape [1, n] (stored from step)
        if hasattr(self, '_last_z_t') and self._last_z_t is not None:
            dW_in = self.eta_in * signal * (self._last_z_t.t() @ self.v)
            self.W_in.data += dW_in.clamp(-0.01, 0.01)

        # W_rec: Hebbian on (prev_v, current_v) — strengthens temporal
        # co-activation patterns (attractor formation)
        dW_rec = self.eta_rec * signal * (self._prev_v.t() @ self.v)
        self.W_rec.data += dW_rec.clamp(-0.01, 0.01)

        # Weight decay to prevent unbounded growth
        self.W_in.data *= self.decay
        self.W_rec.data *= self.decay

    def get_mean(self):
        """Scalar summary of memory activation for policy selection."""
        return float(self.v.abs().mean().item())

    def get_saveable_state(self):
        """Return learned weights for checkpoint persistence."""
        return {
            "W_in": self.W_in.data.cpu().clone(),
            "W_rec": self.W_rec.data.cpu().clone(),
        }

    def load_saveable_state(self, state):
        """Restore learned weights from checkpoint."""
        self.W_in.data = state["W_in"].to(self.device)
        self.W_rec.data = state["W_rec"].to(self.device)

    def reset(self):
        self.v = torch.zeros(1, self.n_latent, device=self.device)
        self._prev_v = torch.zeros(1, self.n_latent, device=self.device)
        self._last_z_t = None
        self.goal_trace = np.zeros(self.n_goals)
        self.cls_buffer = []
