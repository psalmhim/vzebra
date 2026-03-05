"""
Hippocampal Place Cell Network for Spatial Navigation (Step 17).

128 RBF place cells tiling the 800x600 gym pixel arena. Supports:
  - Adaptive cell allocation (new cells for novel locations)
  - Path integration (dead reckoning) with visual correction
  - Sharp-wave ripple replay (offline consolidation)
  - EFE-style planning over goal-directed trajectories

Data structures are pure numpy for efficiency. Planning mirrors the
VAE planner API: simulate 3 goal trajectories, query place cells for
food/risk/novelty, score with pragmatic + epistemic formula.

~128 cells at ~50px spacing provide ~15% arena coverage per field
(sigma=60), consistent with zebrafish Dl spatial coding data.
"""
import numpy as np


class PlaceCellNetwork:
    """128-cell hippocampal place field network in gym pixel space."""

    def __init__(self, n_cells=128, arena_w=800, arena_h=600,
                 sigma_init=60.0, match_threshold=40.0,
                 ema_rate=0.1, drift_rate=0.02,
                 path_integration_gain=0.8,
                 replay_interval=50, replay_length=10,
                 warmup_steps=100, max_blend=0.3):
        self.n_cells = n_cells
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.sigma_init = sigma_init
        self.match_threshold = match_threshold
        self.ema_rate = ema_rate
        self.drift_rate = drift_rate
        self.path_integration_gain = path_integration_gain
        self.replay_interval = replay_interval
        self.replay_length = replay_length
        self.warmup_steps = warmup_steps
        self.max_blend = max_blend

        # Place cell fields
        self.centroids = np.zeros((n_cells, 2), dtype=np.float32)
        self.sigma = np.full(n_cells, sigma_init, dtype=np.float32)
        self.food_rate = np.zeros(n_cells, dtype=np.float32)
        self.risk = np.zeros(n_cells, dtype=np.float32)
        self.visit_count = np.zeros(n_cells, dtype=np.float32)
        self.n_allocated = 0

        # Path integration state
        self._pi_pos = np.array([400.0, 300.0], dtype=np.float32)
        self._pi_heading = 0.0

        # Trajectory buffer for replay
        self._trajectory = []

        # Step counter
        self._step_count = 0

        # Diagnostics
        self._last_pi_error = 0.0
        self._last_G_plan = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # RBF activation
    # ------------------------------------------------------------------

    def _rbf_activations(self, pos):
        """Compute RBF activations for all allocated cells at pos [2].

        Returns:
            activations: numpy [n_cells] — Gaussian response per cell
        """
        if self.n_allocated == 0:
            return np.zeros(self.n_cells, dtype=np.float32)
        active = self.centroids[:self.n_allocated]
        diff = active - pos[np.newaxis, :]
        dist_sq = np.sum(diff ** 2, axis=1)
        sigma_sq = self.sigma[:self.n_allocated] ** 2
        w = np.exp(-dist_sq / (2.0 * sigma_sq))
        full_w = np.zeros(self.n_cells, dtype=np.float32)
        full_w[:self.n_allocated] = w
        return full_w

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, pos):
        """Query place cell map at position [2] → (food_rate, risk).

        Returns:
            food_rate: float — expected food availability
            risk: float — expected danger level
        """
        w = self._rbf_activations(pos)
        w_sum = w.sum() + 1e-8
        food = float(np.dot(w, self.food_rate) / w_sum)
        risk_val = float(np.dot(w, self.risk) / w_sum)
        return food, risk_val

    def query_epistemic(self, pos):
        """Query epistemic signals at position [2].

        Returns:
            novelty: float [0, 1] — high when location is unvisited
            confidence: float [0, 1] — max RBF weight (spatial familiarity)
        """
        if self.n_allocated == 0:
            return 1.0, 0.0
        w = self._rbf_activations(pos)
        w_sum = w.sum() + 1e-8
        raw_familiarity = float(np.dot(w, self.visit_count) / w_sum)
        familiarity = raw_familiarity / (raw_familiarity + 5.0)
        novelty = 1.0 - familiarity
        max_w = float(w[:self.n_allocated].max())
        return novelty, max_w

    # ------------------------------------------------------------------
    # Update (allocate / EMA update)
    # ------------------------------------------------------------------

    def update(self, pos, food_signal, risk_signal):
        """Update place cell map with observation at pos [2].

        Allocates a new cell if no existing cell is within match_threshold.
        Recycles least-visited cell when full.
        """
        pos = np.asarray(pos, dtype=np.float32)
        self._step_count += 1

        # Store in trajectory for replay
        self._trajectory.append({
            "pos": pos.copy(),
            "food": float(food_signal),
            "risk": float(risk_signal),
        })
        if len(self._trajectory) > 500:
            self._trajectory = self._trajectory[-500:]

        # Find closest allocated cell
        if self.n_allocated > 0:
            active = self.centroids[:self.n_allocated]
            dists = np.linalg.norm(active - pos[np.newaxis, :], axis=1)
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]
        else:
            best_dist = float('inf')
            best_idx = 0

        if best_dist > self.match_threshold:
            # Allocate new cell or recycle
            if self.n_allocated < self.n_cells:
                idx = self.n_allocated
                self.n_allocated += 1
            else:
                idx = int(np.argmin(self.visit_count))
            self.centroids[idx] = pos.copy()
            self.sigma[idx] = self.sigma_init
            self.food_rate[idx] = float(food_signal)
            self.risk[idx] = float(risk_signal)
            self.visit_count[idx] = 1.0
        else:
            # EMA update existing cell
            idx = best_idx
            alpha = self.ema_rate
            self.food_rate[idx] += alpha * (food_signal - self.food_rate[idx])
            self.risk[idx] += alpha * (risk_signal - self.risk[idx])
            # Centroid drift toward observation
            self.centroids[idx] += self.drift_rate * (pos - self.centroids[idx])
            self.visit_count[idx] += 1.0

    # ------------------------------------------------------------------
    # Path integration
    # ------------------------------------------------------------------

    def path_integrate(self, turn_rate, speed):
        """Dead reckoning update using motor efference copy.

        Args:
            turn_rate: float — angular velocity (rad-ish)
            speed: float — forward speed [0, 1]
        """
        gain = self.path_integration_gain
        self._pi_heading += turn_rate * gain
        # Convert speed to pixel displacement (~5 px/step at speed=1)
        dx = np.cos(self._pi_heading) * speed * 5.0 * gain
        dy = np.sin(self._pi_heading) * speed * 5.0 * gain
        self._pi_pos[0] += dx
        self._pi_pos[1] += dy
        # Clamp to arena
        self._pi_pos[0] = np.clip(self._pi_pos[0], 0, self.arena_w)
        self._pi_pos[1] = np.clip(self._pi_pos[1], 0, self.arena_h)

    def correct_path_integration(self, visual_pos, weight=0.3):
        """Correct path-integrated position using visual landmark fix.

        Args:
            visual_pos: numpy [2] — actual gym pixel position
            weight: float — correction weight (0=trust PI, 1=trust visual)
        """
        visual_pos = np.asarray(visual_pos, dtype=np.float32)
        error = np.linalg.norm(self._pi_pos - visual_pos)
        self._last_pi_error = float(error)
        self._pi_pos += weight * (visual_pos - self._pi_pos)

    # ------------------------------------------------------------------
    # Replay (sharp-wave ripple analog)
    # ------------------------------------------------------------------

    def replay(self):
        """Replay recent trajectory entries with attenuated learning.

        Simulates sharp-wave ripple consolidation by re-updating
        place cells along recent trajectory with 0.5 attenuation.
        """
        if len(self._trajectory) < 2:
            return
        replay_entries = self._trajectory[-self.replay_length:]
        attenuation = 0.5
        for entry in replay_entries:
            # Attenuated update (don't allocate new cells during replay)
            pos = entry["pos"]
            if self.n_allocated > 0:
                active = self.centroids[:self.n_allocated]
                dists = np.linalg.norm(active - pos[np.newaxis, :], axis=1)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] < self.match_threshold * 1.5:
                    alpha = self.ema_rate * attenuation
                    self.food_rate[best_idx] += alpha * (
                        entry["food"] - self.food_rate[best_idx])
                    self.risk[best_idx] += alpha * (
                        entry["risk"] - self.risk[best_idx])

    def replay_extended(self, length=None, attenuation=0.3):
        """Extended replay for sleep consolidation.

        Longer replay with lower attenuation for deeper memory consolidation.
        Defaults to 2x normal replay length with 0.3 attenuation (vs 0.5).

        Args:
            length: int or None — replay entries (default: 2 * replay_length)
            attenuation: float — learning rate multiplier (lower = gentler)
        """
        if len(self._trajectory) < 2:
            return
        if length is None:
            length = self.replay_length * 2
        replay_entries = self._trajectory[-length:]
        for entry in replay_entries:
            pos = entry["pos"]
            if self.n_allocated > 0:
                active = self.centroids[:self.n_allocated]
                dists = np.linalg.norm(active - pos[np.newaxis, :], axis=1)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] < self.match_threshold * 1.5:
                    alpha = self.ema_rate * attenuation
                    self.food_rate[best_idx] += alpha * (
                        entry["food"] - self.food_rate[best_idx])
                    self.risk[best_idx] += alpha * (
                        entry["risk"] - self.risk[best_idx])

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _get_blend_weight(self):
        """0 during warmup, ramps to max_blend after warmup_steps."""
        if self._step_count < self.warmup_steps:
            return 0.0
        elapsed = self._step_count - self.warmup_steps
        ramp = 100  # steps to ramp to full blend
        t = min(1.0, elapsed / max(1, ramp))
        return self.max_blend * t

    def plan(self, current_pos, last_action, dopa, cls_probs):
        """Simulate 3 goal trajectories, return G_plan[3].

        Mirrors VAE planner: simulate 3 steps per goal in pixel space,
        query place cells for food/risk/novelty, score with
        pragmatic + epistemic formula.

        Args:
            current_pos: numpy [2] — gym pixel position
            last_action: numpy [2] — [turn_rate, speed]
            dopa: float — dopamine level
            cls_probs: numpy [5] — classifier output

        Returns:
            G_plan: numpy [3] — planning bonus per goal (lower = better)
        """
        blend = self._get_blend_weight()
        if blend < 1e-6:
            self._last_G_plan = np.zeros(3, dtype=np.float32)
            return self._last_G_plan

        current_pos = np.asarray(current_pos, dtype=np.float32)
        turn = float(last_action[0]) if len(last_action) > 0 else 0.0
        speed = float(last_action[1]) if len(last_action) > 1 else 0.5

        # Simulate 3 goal trajectories (3 steps each)
        plan_horizon = 3
        heading = self._pi_heading

        # Stereotyped action patterns per goal
        # FORAGE: continue forward
        # FLEE: sharp turn, accelerate
        # EXPLORE: weave slowly
        goal_actions = [
            [(turn * 0.3, min(1.0, speed * 1.1))] * plan_horizon,
            [(0.8 if turn >= 0 else -0.8, min(1.0, speed * 1.4))] * plan_horizon,
            [(0.3 * ((-1) ** s), 0.6) for s in range(plan_horizon)],
        ]

        epistemic_weights = np.array([-0.15, -0.05, -0.5], dtype=np.float32)
        G = np.zeros(3, dtype=np.float32)

        for gi in range(3):
            sim_pos = current_pos.copy()
            sim_heading = heading
            total_food = 0.0
            total_risk = 0.0
            total_novelty = 0.0

            for step in range(plan_horizon):
                act_turn, act_speed = goal_actions[gi][step]
                sim_heading += act_turn * 0.5
                sim_pos[0] += np.cos(sim_heading) * act_speed * 5.0
                sim_pos[1] += np.sin(sim_heading) * act_speed * 5.0
                sim_pos[0] = np.clip(sim_pos[0], 0, self.arena_w)
                sim_pos[1] = np.clip(sim_pos[1], 0, self.arena_h)

                food_r, risk_r = self.query(sim_pos)
                novelty, _ = self.query_epistemic(sim_pos)
                total_food += food_r
                total_risk += risk_r
                total_novelty += novelty

            avg_novelty = total_novelty / plan_horizon

            # Pragmatic score (lower = more attractive)
            if gi == 0:  # FORAGE
                pragmatic = -0.6 * total_food + 0.3 * total_risk
            elif gi == 1:  # FLEE
                pragmatic = 0.8 * total_risk - 0.1 * total_food
            else:  # EXPLORE
                pragmatic = -0.3 * total_food + 0.2 * total_risk

            G[gi] = pragmatic + epistemic_weights[gi] * avg_novelty

        G_plan = G * blend
        self._last_G_plan = G_plan.copy()
        return G_plan

    # ------------------------------------------------------------------
    # Diagnostics & reset
    # ------------------------------------------------------------------

    def get_diagnostics(self):
        """Return monitoring dict."""
        return {
            "n_allocated": self.n_allocated,
            "pi_error": self._last_pi_error,
            "pi_pos": self._pi_pos.copy(),
            "G_plan": self._last_G_plan.copy(),
            "blend_weight": self._get_blend_weight(),
            "step": self._step_count,
            "visit_mean": float(self.visit_count[:max(1, self.n_allocated)].mean()),
            "trajectory_len": len(self._trajectory),
        }

    def get_saveable_state(self):
        """Return learned place fields for checkpoint."""
        n = self.n_allocated
        return {
            "centroids": self.centroids[:n].copy(),
            "sigma": self.sigma[:n].copy(),
            "food_rate": self.food_rate[:n].copy(),
            "risk": self.risk[:n].copy(),
            "visit_count": self.visit_count[:n].copy(),
            "n_allocated": n,
        }

    def load_saveable_state(self, state):
        """Restore learned place fields."""
        n = state["n_allocated"]
        self.centroids[:] = 0.0
        self.sigma[:] = self.sigma_init
        self.food_rate[:] = 0.0
        self.risk[:] = 0.0
        self.visit_count[:] = 0.0
        self.centroids[:n] = state["centroids"]
        self.sigma[:n] = state["sigma"]
        self.food_rate[:n] = state["food_rate"]
        self.risk[:n] = state["risk"]
        self.visit_count[:n] = state["visit_count"]
        self.n_allocated = n

    def reset_episode(self):
        """Clear transient state but keep learned place fields."""
        self._pi_pos = np.array([400.0, 300.0], dtype=np.float32)
        self._pi_heading = 0.0
        self._trajectory = []
        self._step_count = 0
        self._last_pi_error = 0.0
        self._last_G_plan = np.zeros(3, dtype=np.float32)

    def reset(self):
        """Clear all state."""
        self.centroids[:] = 0.0
        self.sigma[:] = self.sigma_init
        self.food_rate[:] = 0.0
        self.risk[:] = 0.0
        self.visit_count[:] = 0.0
        self.n_allocated = 0
        self._pi_pos = np.array([400.0, 300.0], dtype=np.float32)
        self._pi_heading = 0.0
        self._trajectory = []
        self._step_count = 0
        self._last_pi_error = 0.0
        self._last_G_plan = np.zeros(3, dtype=np.float32)
