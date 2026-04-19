"""
Predator place cells: spatial memory for hunting strategy.

64 Gaussian place fields tile the arena. Each cell tracks:
  - prey_density: how often prey are seen when the cell is active
  - catch_rate: successful catches near this cell
  - failure_rate: failed hunts near this cell
  - visit_count: how often the predator has been here

Used for:
  - Patrol targeting (go where prey tend to be)
  - Ambush site selection (high prey density + high catch rate)
  - Hunting bonus (knowledge of local prey patterns)
"""
import numpy as np


class PredatorPlaceCells:
    def __init__(self, n_cells=64, arena_w=800, arena_h=600, sigma=80.0, seed=42):
        self.n_cells = n_cells
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.sigma = sigma
        self.sigma2 = 2.0 * sigma * sigma

        rng = np.random.RandomState(seed)
        self.cx = rng.uniform(0, arena_w, size=n_cells).astype(np.float32)
        self.cy = rng.uniform(0, arena_h, size=n_cells).astype(np.float32)

        # Per-cell learned values
        self.prey_density = np.zeros(n_cells, dtype=np.float32)
        self.catch_rate = np.zeros(n_cells, dtype=np.float32)
        self.failure_rate = np.zeros(n_cells, dtype=np.float32)
        self.visit_count = np.zeros(n_cells, dtype=np.float32)

    def activation(self, x, y):
        """Gaussian place field activation for position (x, y)."""
        dx = self.cx - x
        dy = self.cy - y
        dist2 = dx * dx + dy * dy
        return np.exp(-dist2 / self.sigma2)

    def update(self, pred_x, pred_y, visible_prey_positions,
               hunt_success=False, hunt_failure=False):
        """Update place cell memory from current observation.

        pred_x, pred_y: predator position
        visible_prey_positions: list of (x, y) tuples for visible prey
        hunt_success: True if a catch just happened here
        hunt_failure: True if a hunt just failed here
        """
        act = self.activation(pred_x, pred_y)
        active_mask = act > 0.1
        alpha = 0.05

        # Update visit count
        self.visit_count[active_mask] += 1.0

        # Update prey density from visible prey
        if visible_prey_positions:
            for px, py in visible_prey_positions:
                prey_act = self.activation(px, py)
                self.prey_density += alpha * prey_act
        # Slow decay everywhere
        self.prey_density *= 0.998

        # Catch / failure signals reinforce active cells
        if hunt_success:
            self.catch_rate[active_mask] = (
                (1.0 - alpha) * self.catch_rate[active_mask] + alpha
            )
        if hunt_failure:
            self.failure_rate[active_mask] = (
                (1.0 - alpha) * self.failure_rate[active_mask] + alpha
            )

        # Slow decay on catch/failure rates
        self.catch_rate *= 0.999
        self.failure_rate *= 0.999

    def get_patrol_target(self, hunger=0.0):
        """Return (x, y) patrol target biased toward prey-rich areas.

        hunger: 0 = full, 1 = starving.
        When hungry, weight prey_density more; when full, explore more.
        """
        # Score each cell: prey density weighted by hunger, novelty for low visit
        novelty = 1.0 / (1.0 + self.visit_count * 0.01)
        score = (
            (0.3 + 0.5 * hunger) * self.prey_density
            + (0.3 - 0.2 * hunger) * novelty
            + 0.2 * self.catch_rate
            - 0.1 * self.failure_rate
        )
        # Softmax-style selection (temperature-scaled)
        score = score - score.max()
        weights = np.exp(score * 3.0)
        weights /= weights.sum() + 1e-8
        idx = np.random.choice(self.n_cells, p=weights)

        # Add jitter so patrol isn't exactly on the centroid
        jitter_x = np.random.normal(0, 40)
        jitter_y = np.random.normal(0, 40)
        x = float(np.clip(self.cx[idx] + jitter_x, 50, self.arena_w - 50))
        y = float(np.clip(self.cy[idx] + jitter_y, 50, self.arena_h - 50))
        return (x, y)

    def get_ambush_site(self):
        """Return (x, y) of best ambush location: high prey density + catch rate."""
        score = (
            0.5 * self.prey_density
            + 0.4 * self.catch_rate
            - 0.2 * self.failure_rate
        )
        idx = int(np.argmax(score))
        jitter_x = np.random.normal(0, 20)
        jitter_y = np.random.normal(0, 20)
        x = float(np.clip(self.cx[idx] + jitter_x, 30, self.arena_w - 30))
        y = float(np.clip(self.cy[idx] + jitter_y, 30, self.arena_h - 30))
        return (x, y)

    def get_hunting_bonus(self, pred_x, pred_y):
        """Return hunting bonuses based on place cell knowledge at current pos.

        Returns dict with:
          hunt_bonus: how much local knowledge helps hunting (0-1)
          ambush_quality: how good this spot is for ambushing (0-1)
        """
        act = self.activation(pred_x, pred_y)
        # Weighted prey density at current location
        local_prey = float(np.dot(act, self.prey_density))
        local_catch = float(np.dot(act, self.catch_rate))
        local_failure = float(np.dot(act, self.failure_rate))

        hunt_bonus = float(np.clip(local_prey * 0.5 + local_catch * 0.5, 0, 1))
        ambush_quality = float(np.clip(
            local_prey * 0.6 + local_catch * 0.3 - local_failure * 0.2, 0, 1
        ))
        return {
            'hunt_bonus': hunt_bonus,
            'ambush_quality': ambush_quality,
        }

    def reset(self):
        """Clear all learned spatial memory."""
        self.prey_density[:] = 0
        self.catch_rate[:] = 0
        self.failure_rate[:] = 0
        self.visit_count[:] = 0
