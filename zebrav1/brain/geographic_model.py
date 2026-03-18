"""
Geographic World Model — hippocampal-like spatial map (Step 31).

Grid-based occupancy map for obstacles plus food density heatmap,
both learned from retinal observations. Epistemic uncertainty
(inverse visit frequency) drives early exploration.

Wraps existing PlaceCellNetwork and adds:
  - Obstacle occupancy grid (40×30 cells, 20px resolution)
  - Food density heatmap with temporal decay
  - Epistemic value map → exploration bonus for goal selection
  - Persistence across episodes (learned spatial knowledge)

Neuroscience basis: zebrafish dorsolateral pallium (Dl) — mammalian
hippocampus homolog — encodes allocentric spatial maps.  Obstacle
occupancy mirrors boundary-vector cells.  Epistemic foraging follows
information-seeking in zebrafish larvae (Dreosti et al. 2015).

Pure numpy — no torch.
"""
import math
import numpy as np


class GeographicModel:
    """Grid-based spatial model: obstacle map + food density + exploration.

    Args:
        arena_w, arena_h: arena dimensions in gym pixels
        grid_w, grid_h: grid resolution (cells)
        decay_halflife: food density temporal decay (steps)
    """

    def __init__(self, arena_w=800, arena_h=600,
                 grid_w=40, grid_h=30, decay_halflife=200):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_w = arena_w / grid_w  # 20 px
        self.cell_h = arena_h / grid_h  # 20 px
        self._decay_rate = 0.5 ** (1.0 / decay_halflife)

        # Obstacle occupancy belief: 0.5 = uncertain, 0→clear, 1→blocked
        self.obstacle_belief = np.full((grid_h, grid_w), 0.5,
                                       dtype=np.float32)
        # Observation count per cell (drives confidence)
        self.obs_count = np.zeros((grid_h, grid_w), dtype=np.float32)

        # Food density (EMA of food pixel observations)
        self.food_density = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.food_last_seen = np.full((grid_h, grid_w), -1,
                                      dtype=np.int32)

        # Epistemic value: 1/sqrt(visit_count + 1)
        self.visit_count = np.zeros((grid_h, grid_w), dtype=np.float32)

        self._step_count = 0

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _pos_to_grid(self, pos):
        """Gym pixel position → grid (gx, gy), clamped."""
        gx = int(np.clip(pos[0] / self.cell_w, 0, self.grid_w - 1))
        gy = int(np.clip(pos[1] / self.cell_h, 0, self.grid_h - 1))
        return gx, gy

    def _grid_to_pos(self, gx, gy):
        """Grid cell centre → gym pixel position."""
        return ((gx + 0.5) * self.cell_w, (gy + 0.5) * self.cell_h)

    # ------------------------------------------------------------------
    # Update from retinal observation
    # ------------------------------------------------------------------

    def update(self, pos, retinal_features, heading, step):
        """Update geographic beliefs from current retinal observation.

        Args:
            pos: array-like [2] — gym pixel position of fish
            retinal_features: dict from _extract_retinal_features()
            heading: float — fish heading (radians)
            step: int — current step
        """
        self._step_count = step
        gx, gy = self._pos_to_grid(pos)

        # Mark current cell visited
        self.visit_count[gy, gx] += 1.0

        # Update viewing cone (±45°, up to 5 cells ahead)
        self._update_viewing_cone(pos, heading, retinal_features, step)

    def _update_viewing_cone(self, pos, heading, rf, step):
        """Bayesian update of cells in the fish's viewing cone.

        Uses lateralised retinal signals: obstacle_px_L updates cells
        to the LEFT of heading, obstacle_px_R updates cells to the RIGHT.
        """
        obs_L = rf.get("obstacle_px_L", 0.0)
        obs_R = rf.get("obstacle_px_R", 0.0)
        obs_total = obs_L + obs_R
        food_total = rf.get("food_px_total", 0.0)

        # Viewing cone: 5 cells deep, ±45° (3 cells wide at max depth)
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        for depth in range(1, 6):
            for lateral in range(-depth, depth + 1):
                # Position in front of fish
                dx = cos_h * depth * self.cell_w - sin_h * lateral * self.cell_w
                dy = sin_h * depth * self.cell_h + cos_h * lateral * self.cell_h
                cx = pos[0] + dx
                cy = pos[1] + dy
                gx, gy = self._pos_to_grid([cx, cy])

                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    # Observation weight: closer = more reliable
                    w = 1.0 / (depth * depth)

                    # Lateralised obstacle belief: L pixels → left cells,
                    # R pixels → right cells, centre gets both
                    if lateral < 0:
                        obs_signal = obs_L
                    elif lateral > 0:
                        obs_signal = obs_R
                    else:
                        obs_signal = obs_total * 0.5

                    if obs_signal > 5:
                        self.obstacle_belief[gy, gx] += w * 0.20
                    elif obs_total < 2 and depth <= 3:
                        # Clear view → decrease belief
                        self.obstacle_belief[gy, gx] -= w * 0.10

                    self.obstacle_belief[gy, gx] = np.clip(
                        self.obstacle_belief[gy, gx], 0.02, 0.98)
                    self.obs_count[gy, gx] += w

                    # Food density update (near cells only)
                    if depth <= 3 and food_total > 0:
                        food_signal = food_total / 40.0  # normalise
                        alpha = min(0.3, w * 0.2)
                        self.food_density[gy, gx] = (
                            (1 - alpha) * self.food_density[gy, gx]
                            + alpha * food_signal)
                        self.food_last_seen[gy, gx] = step

                    # Visit proximity: nearby cells get fractional visits
                    self.visit_count[gy, gx] += w * 0.3

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_obstacle(self, pos):
        """Query obstacle belief at position. Returns [0, 1]."""
        gx, gy = self._pos_to_grid(pos)
        return float(self.obstacle_belief[gy, gx])

    def query_food_density(self, pos, current_step=None):
        """Query food density with temporal decay."""
        gx, gy = self._pos_to_grid(pos)
        density = float(self.food_density[gy, gx])
        if current_step is not None and self.food_last_seen[gy, gx] >= 0:
            age = current_step - self.food_last_seen[gy, gx]
            density *= self._decay_rate ** age
        return density

    def get_exploration_bonus(self, pos):
        """Epistemic value: high for unvisited regions. Returns [0, 1].

        Visit counts normalised by expected observation rate (~5 per step
        from viewing cone) to prevent premature decay.
        """
        gx, gy = self._pos_to_grid(pos)
        return 1.0 / math.sqrt(self.visit_count[gy, gx] / 5.0 + 1.0)

    def get_exploration_map(self):
        """Full epistemic value map [grid_h, grid_w]."""
        return 1.0 / np.sqrt(self.visit_count / 5.0 + 1.0)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def suggest_exploration_target(self, current_pos, heading,
                                   n_candidates=8):
        """Return highest-epistemic-value target in a fan of directions.

        Returns:
            (target_pos, epistemic_value): best exploration target
        """
        best_val = -1.0
        best_pos = None
        look_dist = 4 * self.cell_w  # 80 px ahead

        for i in range(n_candidates):
            angle = heading + (i - n_candidates / 2) * (math.pi / n_candidates)
            tx = current_pos[0] + look_dist * math.cos(angle)
            ty = current_pos[1] + look_dist * math.sin(angle)
            tx = np.clip(tx, 0, self.arena_w - 1)
            ty = np.clip(ty, 0, self.arena_h - 1)

            ep = self.get_exploration_bonus([tx, ty])
            obs = self.query_obstacle([tx, ty])
            # Penalise obstacle-heavy targets
            val = ep * (1.0 - obs)

            if val > best_val:
                best_val = val
                best_pos = (tx, ty)

        return best_pos, best_val

    def plan_geographic(self, current_pos, heading, n_samples=6):
        """Geographic planning bonus for 4 goals.

        Returns:
            G_geo: numpy [4] — EFE bonus per goal (lower = better)
        """
        # Sample positions in forward arc
        look_dist = 3 * self.cell_w
        food_ahead = 0.0
        obs_ahead = 0.0
        epist_ahead = 0.0

        for i in range(n_samples):
            angle = heading + (i - n_samples / 2) * 0.15
            tx = current_pos[0] + look_dist * math.cos(angle)
            ty = current_pos[1] + look_dist * math.sin(angle)
            food_ahead += self.query_food_density([tx, ty], self._step_count)
            obs_ahead += self.query_obstacle([tx, ty])
            epist_ahead += self.get_exploration_bonus([tx, ty])

        food_ahead /= n_samples
        obs_ahead /= n_samples
        epist_ahead /= n_samples

        # FORAGE: lower EFE if food-rich ahead
        g_forage = -0.3 * food_ahead + 0.2 * obs_ahead

        # FLEE: lower EFE if clear escape routes (low obstacle)
        g_flee = 0.3 * obs_ahead

        # EXPLORE: lower EFE if high epistemic value ahead
        g_explore = -0.4 * epist_ahead

        # SOCIAL: neutral geographic influence
        g_social = 0.0

        return np.array([g_forage, g_flee, g_explore, g_social],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def get_saveable_state(self):
        return {
            "obstacle_belief": self.obstacle_belief.copy(),
            "obs_count": self.obs_count.copy(),
            "food_density": self.food_density.copy(),
            "food_last_seen": self.food_last_seen.copy(),
            "visit_count": self.visit_count.copy(),
        }

    def load_saveable_state(self, state):
        self.obstacle_belief[:] = state["obstacle_belief"]
        self.obs_count[:] = state["obs_count"]
        self.food_density[:] = state["food_density"]
        self.food_last_seen[:] = state["food_last_seen"]
        self.visit_count[:] = state["visit_count"]

    def reset_episode(self):
        """Keep learned spatial maps, reset transient state."""
        # Food density decays across episodes (food re-spawns)
        self.food_density *= 0.3
        self.food_last_seen[:] = -1

    def reset(self):
        """Full reset — clear all learned knowledge."""
        self.obstacle_belief[:] = 0.5
        self.obs_count[:] = 0.0
        self.food_density[:] = 0.0
        self.food_last_seen[:] = -1
        self.visit_count[:] = 0.0
        self._step_count = 0

    def get_diagnostics(self):
        explored_frac = float((self.visit_count > 0).sum()) / (
            self.grid_w * self.grid_h)
        return {
            "explored_frac": explored_frac,
            "mean_obstacle_belief": float(self.obstacle_belief.mean()),
            "mean_food_density": float(self.food_density.mean()),
            "step_count": self._step_count,
        }
