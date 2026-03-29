"""
Allostatic interoception: hunger, fatigue, stress tracking.

Computes allostatic errors (predicted - setpoint) that bias EFE goal selection.
  hunger  → bias FORAGE
  stress  → bias FLEE
  fatigue → bias EXPLORE (rest)

Also provides: speed cap, dopamine gain modulation, state context.
"""
import numpy as np


class AllostaticRegulator:
    def __init__(self):
        self.hunger_setpoint = 0.3
        self.fatigue_setpoint = 0.2
        self.stress_setpoint = 0.1
        self.trend_alpha = 0.05
        self.prior_strength = 0.15

        self.hunger = 0.0
        self.fatigue = 0.0
        self.stress = 0.0
        self.hunger_predicted = 0.0
        self.fatigue_predicted = 0.0
        self.stress_predicted = 0.0
        self.hunger_error = 0.0
        self.fatigue_error = 0.0
        self.stress_error = 0.0
        self.urgency = 0.0

    def step(self, energy: float, speed: float, pred_dist: float) -> dict:
        # Hunger
        self.hunger = max(0.0, min(1.0, 1.0 - energy / 100.0))
        self.hunger_predicted += self.trend_alpha * (self.hunger - self.hunger_predicted)
        self.hunger_error = self.hunger_predicted - self.hunger_setpoint

        # Fatigue
        if speed > 0.6:
            self.fatigue += 0.0015 * speed
        else:
            self.fatigue -= 0.05
        self.fatigue = max(0.0, min(1.0, self.fatigue))
        self.fatigue_predicted += self.trend_alpha * (self.fatigue - self.fatigue_predicted)
        self.fatigue_error = self.fatigue_predicted - self.fatigue_setpoint

        # Stress
        if pred_dist < 300.0:
            proximity = max(0.0, 1.0 - pred_dist / 300.0)
            self.stress += 0.15 * proximity
        self.stress *= 0.92
        self.stress = max(0.0, min(1.0, self.stress))
        self.stress_predicted += self.trend_alpha * (self.stress - self.stress_predicted)
        self.stress_error = self.stress_predicted - self.stress_setpoint

        self.urgency = max(abs(self.hunger_error), abs(self.fatigue_error),
                           abs(self.stress_error))
        return {
            'hunger': self.hunger, 'fatigue': self.fatigue, 'stress': self.stress,
            'hunger_error': self.hunger_error, 'fatigue_error': self.fatigue_error,
            'stress_error': self.stress_error, 'urgency': self.urgency,
        }

    def get_goal_bias(self) -> np.ndarray:
        """EFE bias [forage, flee, explore, social]. Lower = more attractive."""
        s = self.prior_strength
        hunger_u = max(0.0, self.hunger_error)
        if self.hunger > 0.75:
            hunger_u *= 1.0 + 5.0 * (self.hunger - 0.6) / 0.4
        elif self.hunger > 0.6:
            hunger_u *= 1.0 + 3.0 * (self.hunger - 0.6) / 0.4
        critical = 1.0 + max(0.0, self.hunger - 0.7) * 3.0
        return np.array([
            -s * hunger_u * critical,
            -s * max(0.0, self.stress_error),
            -s * max(0.0, self.fatigue_error),
            -s * max(0.0, 0.3 - self.stress - self.fatigue),
        ], dtype=np.float32)

    def get_speed_cap(self) -> float:
        if self.fatigue > 0.7:
            excess = (self.fatigue - 0.7) / 0.3
            return max(0.4, 1.0 - 0.6 * excess)
        return 1.0

    def reset(self):
        self.hunger = 0.0
        self.fatigue = 0.0
        self.stress = 0.0
        self.hunger_predicted = 0.0
        self.fatigue_predicted = 0.0
        self.stress_predicted = 0.0
        self.hunger_error = 0.0
        self.fatigue_error = 0.0
        self.stress_error = 0.0
        self.urgency = 0.0
