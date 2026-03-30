"""
Geographic model: spatial map of arena with food patches and risk zones.

Maintains a discretized grid map of the arena with:
  - Food density history (where food has been found)
  - Risk zones (where predator has been encountered)
  - Exploration coverage (visited vs unvisited cells)

Biases EFE toward known-good foraging regions and away from risk zones.
Complements place cells with explicit spatial reasoning.

Architecture:
  Grid: 10x8 cells covering 800x600 arena (80x75 per cell)
  Each cell stores: food_score, risk_score, visit_count, last_visit
"""
import numpy as np


class GeographicModel:
    def __init__(self, arena_w=800, arena_h=600, grid_cols=10, grid_rows=8):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.cell_w = arena_w / grid_cols
        self.cell_h = arena_h / grid_rows

        self.food_score = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        self.risk_score = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        self.visit_count = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        self.last_visit = np.full((grid_rows, grid_cols), -100, dtype=np.int32)
        self._step = 0

    def _pos_to_cell(self, x, y):
        col = max(0, min(self.grid_cols - 1, int(x / self.cell_w)))
        row = max(0, min(self.grid_rows - 1, int(y / self.cell_h)))
        return row, col

    def update(self, fish_x, fish_y, food_eaten, pred_dist, pred_x=None, pred_y=None):
        """Update map based on fish's current experience."""
        self._step += 1
        row, col = self._pos_to_cell(fish_x, fish_y)

        # Visit tracking
        self.visit_count[row, col] += 1
        self.last_visit[row, col] = self._step

        # Food: boost cell where food was eaten
        if food_eaten > 0:
            self.food_score[row, col] += 0.5
            # Also boost neighboring cells (food patch)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r2, c2 = row + dr, col + dc
                    if 0 <= r2 < self.grid_rows and 0 <= c2 < self.grid_cols:
                        self.food_score[r2, c2] += 0.1

        # Risk: mark cells near predator
        if pred_x is not None and pred_y is not None and pred_dist < 200:
            pr, pc = self._pos_to_cell(pred_x, pred_y)
            risk_intensity = max(0, 1.0 - pred_dist / 200.0)
            self.risk_score[pr, pc] += risk_intensity * 0.3
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r2, c2 = pr + dr, pc + dc
                    if 0 <= r2 < self.grid_rows and 0 <= c2 < self.grid_cols:
                        self.risk_score[r2, c2] += risk_intensity * 0.1

        # Decay
        self.food_score *= 0.999
        self.risk_score *= 0.995

    def get_forage_direction(self, fish_x, fish_y, fish_heading):
        """Returns angular bias toward best known food patch."""
        row, col = self._pos_to_cell(fish_x, fish_y)
        best_score = -1.0
        best_angle = 0.0

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # Score: food value - risk, weighted by recency
                recency = max(0, 1.0 - (self._step - self.last_visit[r, c]) / 500.0)
                score = self.food_score[r, c] - 0.5 * self.risk_score[r, c]
                score *= (0.3 + 0.7 * recency)  # prefer recently visited
                if score > best_score and (r != row or c != col):
                    best_score = score
                    cx = (c + 0.5) * self.cell_w
                    cy = (r + 0.5) * self.cell_h
                    best_angle = np.arctan2(cy - fish_y, cx - fish_x)

        if best_score > 0.01:
            rel = np.arctan2(np.sin(best_angle - fish_heading),
                             np.cos(best_angle - fish_heading))
            return float(rel), float(best_score)
        return 0.0, 0.0

    def get_exploration_target(self, fish_x, fish_y, fish_heading):
        """Returns direction toward least-visited cell."""
        min_visits = float('inf')
        best_angle = 0.0
        row, col = self._pos_to_cell(fish_x, fish_y)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                dist = abs(r - row) + abs(c - col)
                if dist < 1:
                    continue
                # Prefer nearby unvisited cells
                score = self.visit_count[r, c] + dist * 0.5
                if score < min_visits:
                    min_visits = score
                    cx = (c + 0.5) * self.cell_w
                    cy = (r + 0.5) * self.cell_h
                    best_angle = np.arctan2(cy - fish_y, cx - fish_x)

        rel = np.arctan2(np.sin(best_angle - fish_heading),
                         np.cos(best_angle - fish_heading))
        coverage = float(np.mean(self.visit_count > 0))
        return float(rel), coverage

    def get_efe_bias(self, fish_x, fish_y):
        """EFE biases from geographic knowledge."""
        row, col = self._pos_to_cell(fish_x, fish_y)
        local_food = float(self.food_score[max(0,row-1):min(self.grid_rows,row+2),
                                           max(0,col-1):min(self.grid_cols,col+2)].mean())
        local_risk = float(self.risk_score[max(0,row-1):min(self.grid_rows,row+2),
                                           max(0,col-1):min(self.grid_cols,col+2)].mean())
        coverage = float(np.mean(self.visit_count > 0))
        return {
            'forage_bias': -local_food * 0.2,     # food nearby → forage more
            'flee_bias': -local_risk * 0.15,       # risk nearby → flee more
            'explore_bias': -(1.0 - coverage) * 0.1,  # low coverage → explore
        }

    def reset(self):
        self.food_score[:] = 0
        self.risk_score[:] = 0
        self.visit_count[:] = 0
        self.last_visit[:] = -100
        self._step = 0
