"""
Lightweight internal world model for EFE planning.

Predicts future states under each goal policy:
  - Energy trajectory (metabolic cost model)
  - Threat level trajectory (predator model extrapolation)
  - Food availability (place cell memory)

Feeds EFE with expected outcomes per policy, enabling
model-based goal selection beyond reactive heuristics.
"""
import math
import numpy as np


class InternalWorldModel:
    """Predicts future state under each goal for EFE computation."""

    def __init__(self, horizon=10):
        self.horizon = horizon
        # Metabolic costs per goal (energy/step)
        self.metabolic = {0: -0.3, 1: -0.5, 2: -0.15, 3: -0.1}  # forage, flee, explore, social
        # Expected food gain per step when foraging (updated from experience)
        self.food_gain_rate = 0.2
        self._food_gain_ema = 0.2

    def predict_energy(self, current_energy: float, goal: int) -> float:
        """Predict energy at horizon under goal policy."""
        cost = self.metabolic.get(goal, -0.2)
        gain = self.food_gain_rate if goal == 0 else 0.0
        return max(0.0, current_energy + (cost + gain) * self.horizon)

    def predict_threat(self, pred_model, fish_pos, goal: int) -> float:
        """Predict average threat level over horizon under goal policy."""
        # Flee reduces threat, forage/explore may increase it
        base_threat = pred_model.get_threat_level(fish_pos)
        if goal == 1:  # FLEE
            return base_threat * 0.3  # expect threat reduction
        elif goal == 0:  # FORAGE — may move toward food near predator
            return base_threat * 1.2
        return base_threat

    def compute_efe_per_goal(self, energy: float, pred_model, fish_pos,
                             place_cells_bonus: dict, allostasis) -> np.ndarray:
        """
        Compute expected free energy for each goal.
        Lower EFE = more preferred.
        Returns: (4,) array [forage, flee, explore, social]
        """
        efe = np.zeros(4, dtype=np.float32)

        for g in range(4):
            # Pragmatic value: predicted energy outcome
            pred_e = self.predict_energy(energy, g)
            energy_cost = max(0.0, 50.0 - pred_e) / 50.0  # penalize low energy

            # Threat cost
            threat = self.predict_threat(pred_model, fish_pos, g)

            # Epistemic value: information gain (explore has high epistemic value)
            epistemic = 0.3 if g == 2 else 0.0

            # Combine: EFE = pragmatic_cost - epistemic_value
            efe[g] = 0.5 * energy_cost + 0.4 * threat - 0.2 * epistemic

        return efe

    def update_food_gain(self, eaten: bool):
        """Update expected food gain rate from experience."""
        obs = 1.0 if eaten else 0.0
        self._food_gain_ema = 0.95 * self._food_gain_ema + 0.05 * obs
        self.food_gain_rate = self._food_gain_ema

    def reset(self):
        self._food_gain_ema = 0.2
        self.food_gain_rate = 0.2
