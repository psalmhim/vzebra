"""
RL Critic for AIF Parameter Tuning (Step 15).

Two components:
- RLCritic_v60: Small value network V(s) trained via TD(0) from gym rewards.
- EFEWeightAdapter: Maintains learnable perturbations on EFE coefficients,
  updated by TD error. AIF always selects the goal; the critic only nudges
  the coefficients. Mirrors the biological prefrontal-dopaminergic loop.

RLCritic_v60 uses torch; EFEWeightAdapter is pure numpy.
"""
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RLCritic_v60 — lightweight value network + ring-buffer replay
# ---------------------------------------------------------------------------
class RLCritic_v60(nn.Module):
    """TD(0) value critic over working-memory state (16-dim)."""

    def __init__(self, state_dim=16, hidden=32, gamma=0.99, lr=3e-3,
                 buffer_size=256, batch_size=32, grad_clip=1.0,
                 device="cpu"):
        super().__init__()
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.grad_clip = grad_clip
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # Isolated RNG so critic sampling doesn't perturb environment RNG
        self._rng = np.random.RandomState(12345)

        # Ring buffer
        self._buf_s = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._buf_r = np.zeros(buffer_size, dtype=np.float32)
        self._buf_ns = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._buf_done = np.zeros(buffer_size, dtype=np.float32)
        self._buf_idx = 0
        self._buf_full = False

        # Stats
        self._update_count = 0
        self._last_td_error = 0.0

    # -- public API --

    def predict(self, state_np):
        """Return V(s) as a float (no grad)."""
        with torch.no_grad():
            s = torch.from_numpy(
                np.asarray(state_np, dtype=np.float32)
            ).unsqueeze(0).to(self.device)
            return float(self.net(s).item())

    def push(self, state, reward, next_state, done):
        """Add a transition to the ring buffer."""
        idx = self._buf_idx % self.buffer_size
        self._buf_s[idx] = np.asarray(state, dtype=np.float32)
        self._buf_r[idx] = reward
        self._buf_ns[idx] = np.asarray(next_state, dtype=np.float32)
        self._buf_done[idx] = float(done)
        self._buf_idx += 1
        if self._buf_idx >= self.buffer_size:
            self._buf_full = True

    def update(self):
        """Sample mini-batch, compute TD(0) loss, one Adam step.

        Returns:
            mean_td_error (float) or None if buffer too small.
        """
        n = self.buffer_size if self._buf_full else (self._buf_idx % self.buffer_size)
        if n < self.batch_size:
            return None

        indices = self._rng.randint(0, n, size=self.batch_size)
        s = torch.from_numpy(self._buf_s[indices]).to(self.device)
        r = torch.from_numpy(self._buf_r[indices]).to(self.device)
        ns = torch.from_numpy(self._buf_ns[indices]).to(self.device)
        d = torch.from_numpy(self._buf_done[indices]).to(self.device)

        v_s = self.net(s).squeeze(-1)
        with torch.no_grad():
            v_ns = self.net(ns).squeeze(-1)
            target = r + self.gamma * v_ns * (1.0 - d)

        td_error = target - v_s
        loss = (td_error ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._update_count += 1
        self._last_td_error = float(td_error.mean().item())
        return self._last_td_error

    def get_stats(self):
        n = self.buffer_size if self._buf_full else (self._buf_idx % self.buffer_size)
        return {
            "update_count": self._update_count,
            "last_td_error": self._last_td_error,
            "buffer_size": n,
        }

    def reset(self):
        """Clear replay buffer but keep learned weights."""
        self._rng = np.random.RandomState(12345)
        self._buf_idx = 0
        self._buf_full = False
        self._update_count = 0
        self._last_td_error = 0.0


# ---------------------------------------------------------------------------
# EFEWeightAdapter — learnable perturbations on EFE coefficients (pure numpy)
# ---------------------------------------------------------------------------

# Hand-tuned prior weights (from goal_policy_v60 compute_efe)
PRIOR_FORAGE = np.array([0.20, -0.80, 0.20, 0.15], dtype=np.float64)
PRIOR_FLEE = np.array([0.10, -0.80, 0.40], dtype=np.float64)
PRIOR_EXPLORE = np.array([0.30, -0.50, 0.10, 0.20], dtype=np.float64)


class EFEWeightAdapter:
    """Maintains learnable perturbations dw on EFE coefficients.

    The three EFE equations are rewritten as linear in feature vectors:
        G_forage  = (PRIOR_f + dw_f) . [uncertainty, p_food, (0.5-dopa), 1.0]
        G_flee    = (PRIOR_l + dw_l) . [cms, p_enemy, 1.0]
        G_explore = (PRIOR_e + dw_e) . [uncertainty, (p_nothing+p_environ), cms, 1.0]

    Sign convention: negative weights in PRIOR encode attraction (e.g. -0.80
    for p_food means food lowers forage EFE). Features are always positive.

    Update rule (applied only when confidence <= 0.95):
        dw_g -= eta * td_error * pi_g * phi_g     (positive RPE lowers EFE)
        dw_g -= lambda_reg * dw_g                  (L2 pull toward prior)
        dw_g  = clip(dw_g, -delta_max, +delta_max) (hard clamp)
    """

    def __init__(self, eta_w=0.10, lambda_reg=0.002, delta_max=0.5,
                 warmup_steps=120):
        self.eta_w = eta_w
        self.lambda_reg = lambda_reg
        self.delta_max = delta_max
        self._warmup_steps = warmup_steps
        self._step_count = 0

        # Perturbations (initialised at zero = pure AIF at start)
        self.dw_forage = np.zeros_like(PRIOR_FORAGE)
        self.dw_flee = np.zeros_like(PRIOR_FLEE)
        self.dw_explore = np.zeros_like(PRIOR_EXPLORE)

        # Cached feature vectors from last compute_efe call
        self._phi_forage = None
        self._phi_flee = None
        self._phi_explore = None

    # -- EFE computation (drop-in replacement for GoalPolicy compute_efe) --

    def compute_efe(self, cls_probs, pi_OT, pi_PC, dopa, cms,
                    rpe=None, F_visual=None, mem_mean=None):
        """Compute EFE using prior + perturbation weights.

        Args match GoalPolicy_v60.compute_efe signature; rpe, F_visual,
        mem_mean are accepted but unused (adapter uses a simpler feature set).

        Returns:
            efe: array [3] — EFE per goal (lower = preferred)
        """
        p_nothing = cls_probs[0]
        p_food = cls_probs[1]
        p_enemy = cls_probs[2]
        p_environ = cls_probs[4]
        uncertainty = 1.0 - 0.5 * (pi_OT + pi_PC)

        # Feature vectors (positive; sign is in the prior weights)
        self._phi_forage = np.array(
            [uncertainty, p_food, (0.5 - dopa), 1.0])
        self._phi_flee = np.array(
            [cms, p_enemy, 1.0])
        self._phi_explore = np.array(
            [uncertainty, (p_nothing + p_environ), cms, 1.0])

        w_f = PRIOR_FORAGE + self.dw_forage
        w_l = PRIOR_FLEE + self.dw_flee
        w_e = PRIOR_EXPLORE + self.dw_explore

        g_forage = float(w_f @ self._phi_forage)
        g_flee = float(w_l @ self._phi_flee)
        g_explore = float(w_e @ self._phi_explore)

        return np.array([g_forage, g_flee, g_explore])

    # -- Weight update --

    def update(self, td_error, posterior, confidence, shortcut_active):
        """Update perturbation weights using TD error signal.

        Args:
            td_error: float — TD error from critic
            posterior: array [3] — goal posterior probabilities
            confidence: float — meta-precision (0–1)
            shortcut_active: bool — whether habit shortcut fired
        """
        self._step_count += 1

        # Gate: warm-up period lets the critic learn a stable baseline
        if self._step_count < self._warmup_steps:
            return

        # Gate: only adapt when AIF is not fully confident
        if confidence > 0.95:
            return

        if self._phi_forage is None:
            return

        eta = self.eta_w
        lam = self.lambda_reg
        dmax = self.delta_max

        # Policy-weighted gradient: positive TD error (better-than-expected)
        # should LOWER EFE for the active goal (reinforce it).
        self.dw_forage -= eta * td_error * posterior[0] * self._phi_forage
        self.dw_flee -= eta * td_error * posterior[1] * self._phi_flee
        self.dw_explore -= eta * td_error * posterior[2] * self._phi_explore

        # L2 regularization toward prior (zero perturbation)
        self.dw_forage -= lam * self.dw_forage
        self.dw_flee -= lam * self.dw_flee
        self.dw_explore -= lam * self.dw_explore

        # Hard clamp
        self.dw_forage = np.clip(self.dw_forage, -dmax, dmax)
        self.dw_flee = np.clip(self.dw_flee, -dmax, dmax)
        self.dw_explore = np.clip(self.dw_explore, -dmax, dmax)

    def get_deltas(self):
        """Return current perturbation vectors."""
        return {
            "forage": self.dw_forage.copy(),
            "flee": self.dw_flee.copy(),
            "explore": self.dw_explore.copy(),
        }

    def get_delta_norms(self):
        """Return L2 norms of perturbation vectors per goal."""
        return np.array([
            np.linalg.norm(self.dw_forage),
            np.linalg.norm(self.dw_flee),
            np.linalg.norm(self.dw_explore),
        ])

    def get_max_abs_delta(self):
        """Return the largest absolute perturbation across all weights."""
        return max(
            np.abs(self.dw_forage).max(),
            np.abs(self.dw_flee).max(),
            np.abs(self.dw_explore).max(),
        )

    def reset(self):
        """Reset perturbations to zero (pure AIF)."""
        self._step_count = 0
        self.dw_forage = np.zeros_like(PRIOR_FORAGE)
        self.dw_flee = np.zeros_like(PRIOR_FLEE)
        self.dw_explore = np.zeros_like(PRIOR_EXPLORE)
        self._phi_forage = None
        self._phi_flee = None
        self._phi_explore = None
