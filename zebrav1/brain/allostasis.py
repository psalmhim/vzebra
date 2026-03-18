"""
Allostatic Interoception Module (Step 18).

Predictive tracking of 3 internal variables — hunger, fatigue, stress —
with homeostatic setpoints. Computes allostatic errors (predicted - setpoint)
that bias goal selection via EFE modulation.

  Hunger:  1 - energy/100, with EMA trend prediction (~20 step window)
  Fatigue: +0.01*speed when active, -0.02 at rest (speed < 0.3)
  Stress:  +0.3*(1 - pred_dist/100) near predator, *0.95 decay

Each variable feeds into goal policy as additive EFE bias:
  hunger  → bias FORAGE (lower EFE for foraging)
  stress  → bias FLEE   (lower EFE for fleeing)
  fatigue → bias EXPLORE (lower EFE for slow exploration / rest)

Also provides:
  - Dopamine gain modulation (stress dampens dopa sensitivity)
  - Fatigue-based speed cap (limits max speed when fatigued)
  - State context extension (3 extra dims for VAE: hunger, fatigue, stress)

Pure numpy — no torch dependency.
"""
import numpy as np


class AllostaticRegulator:
    """Predictive interoception with 3 allostatic variables."""

    def __init__(self,
                 hunger_setpoint=0.3, hunger_trend_alpha=0.05,
                 fatigue_gain=0.0015, fatigue_recovery=0.05,
                 fatigue_rest_threshold=0.6, fatigue_setpoint=0.2,
                 stress_gain=0.15, stress_decay=0.92,
                 stress_pred_threshold=300.0, stress_setpoint=0.1,
                 prior_strength=0.15, dopa_mod_strength=0.08,
                 precision_mod_strength=0.1, speed_cap_fatigue=0.7):
        # Hunger parameters
        self.hunger_setpoint = hunger_setpoint
        self.hunger_trend_alpha = hunger_trend_alpha

        # Fatigue parameters
        self.fatigue_gain = fatigue_gain
        self.fatigue_recovery = fatigue_recovery
        self.fatigue_rest_threshold = fatigue_rest_threshold
        self.fatigue_setpoint = fatigue_setpoint

        # Stress parameters
        self.stress_gain = stress_gain
        self.stress_decay = stress_decay
        self.stress_pred_threshold = stress_pred_threshold
        self.stress_setpoint = stress_setpoint

        # Modulation strengths
        self.prior_strength = prior_strength
        self.dopa_mod_strength = dopa_mod_strength
        self.precision_mod_strength = precision_mod_strength
        self.speed_cap_fatigue = speed_cap_fatigue

        # Internal state
        self.hunger = 0.0
        self.fatigue = 0.0
        self.stress = 0.0

        # Predicted values (EMA-based trend)
        self.hunger_predicted = 0.0
        self.fatigue_predicted = 0.0
        self.stress_predicted = 0.0

        # Allostatic errors (predicted - setpoint)
        self.hunger_error = 0.0
        self.fatigue_error = 0.0
        self.stress_error = 0.0

        # Urgency: overall allostatic drive
        self.urgency = 0.0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, energy, speed, pred_dist, p_colleague=0.0):
        """Update all 3 allostatic variables and compute errors.

        Args:
            energy: float — current energy level [0, 100]
            speed: float — current movement speed [0, 1]
            pred_dist: float — distance to predator in pixels
            p_colleague: float — colleague detection probability [0, 1]

        Returns:
            dict with all variables, errors, and urgency
        """
        # --- Hunger ---
        self.hunger = max(0.0, min(1.0, 1.0 - energy / 100.0))
        # EMA trend prediction
        self.hunger_predicted += self.hunger_trend_alpha * (
            self.hunger - self.hunger_predicted)
        self.hunger_error = self.hunger_predicted - self.hunger_setpoint

        # --- Fatigue ---
        if speed > self.fatigue_rest_threshold:
            self.fatigue += self.fatigue_gain * speed
        else:
            self.fatigue -= self.fatigue_recovery
        self.fatigue = max(0.0, min(1.0, self.fatigue))
        self.fatigue_predicted += self.hunger_trend_alpha * (
            self.fatigue - self.fatigue_predicted)
        self.fatigue_error = self.fatigue_predicted - self.fatigue_setpoint

        # --- Stress ---
        if pred_dist < self.stress_pred_threshold:
            proximity = max(0.0, 1.0 - pred_dist / self.stress_pred_threshold)
            self.stress += self.stress_gain * proximity
        # Social comfort: colleague presence reduces stress by up to 10%
        social_comfort = min(0.10, p_colleague * 0.15)
        self.stress *= (self.stress_decay - social_comfort)
        self.stress = max(0.0, min(1.0, self.stress))
        self.stress_predicted += self.hunger_trend_alpha * (
            self.stress - self.stress_predicted)
        self.stress_error = self.stress_predicted - self.stress_setpoint

        # --- Urgency ---
        self.urgency = max(
            abs(self.hunger_error),
            abs(self.fatigue_error),
            abs(self.stress_error))

        return {
            "hunger": self.hunger,
            "fatigue": self.fatigue,
            "stress": self.stress,
            "hunger_predicted": self.hunger_predicted,
            "fatigue_predicted": self.fatigue_predicted,
            "stress_predicted": self.stress_predicted,
            "hunger_error": self.hunger_error,
            "fatigue_error": self.fatigue_error,
            "stress_error": self.stress_error,
            "urgency": self.urgency,
        }

    # ------------------------------------------------------------------
    # Goal prior bias
    # ------------------------------------------------------------------

    def get_goal_prior_bias(self):
        """Compute additive EFE bias from allostatic errors → numpy[4].

        Returns:
            bias: numpy [4] — [FORAGE_bias, FLEE_bias, EXPLORE_bias, SOCIAL_bias]
                Lower value = more attractive (EFE convention)
        """
        strength = self.prior_strength
        # Hunger → bias FORAGE (make foraging more attractive)
        # Nonlinear urgency: hunger bias grows sharply below energy=40
        hunger_urgency = max(0.0, self.hunger_error)
        if self.hunger > 0.75:  # energy < 25 — critical starvation zone
            hunger_urgency *= 1.0 + 5.0 * (self.hunger - 0.6) / 0.4
        elif self.hunger > 0.6:  # energy < 40
            hunger_urgency *= 1.0 + 3.0 * (self.hunger - 0.6) / 0.4
        # Extra forage drive at critical energy (orexigenic override)
        critical_boost = 1.0 + max(0.0, self.hunger - 0.7) * 3.0
        forage_bias = -strength * hunger_urgency * critical_boost
        # Stress → bias FLEE (make fleeing more attractive)
        flee_bias = -strength * max(0.0, self.stress_error)
        # Fatigue → bias EXPLORE (slow exploration = rest)
        explore_bias = -strength * max(0.0, self.fatigue_error)
        # Low stress + low fatigue → bias SOCIAL (safe to socialize)
        social_bias = -strength * max(0.0, 0.3 - self.stress - self.fatigue)

        return np.array([forage_bias, flee_bias, explore_bias, social_bias],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    # Preferred outcome params (Step 25 — for EFE engine)
    # ------------------------------------------------------------------

    def get_preferred_outcome_params(self):
        """Return allostatic state for EFE preferred outcome modulation.

        Returns:
            dict with hunger, fatigue, stress, and their errors
        """
        return {
            "hunger": self.hunger,
            "fatigue": self.fatigue,
            "stress": self.stress,
            "hunger_error": self.hunger_error,
            "fatigue_error": self.fatigue_error,
            "stress_error": self.stress_error,
        }

    # ------------------------------------------------------------------
    # Neuromodulation
    # ------------------------------------------------------------------

    def modulate_dopamine_gain(self, base_gain):
        """Stress dampens dopamine sensitivity.

        Args:
            base_gain: float — current dopamine beta parameter

        Returns:
            modulated_gain: float — adjusted beta
        """
        # High stress → reduce dopa gain (defensive mode)
        damping = 1.0 - self.dopa_mod_strength * self.stress
        return base_gain * max(0.5, damping)

    def get_speed_cap(self):
        """Fatigue limits maximum speed when fatigue > 0.7.

        Returns:
            speed_cap: float [0, 1] — maximum allowed speed
        """
        if self.fatigue > self.speed_cap_fatigue:
            # Linear decrease from 1.0 at threshold to 0.4 at fatigue=1.0
            excess = (self.fatigue - self.speed_cap_fatigue) / (
                1.0 - self.speed_cap_fatigue + 1e-8)
            return max(0.4, 1.0 - 0.6 * excess)
        return 1.0

    # ------------------------------------------------------------------
    # State context extension
    # ------------------------------------------------------------------

    def get_state_ctx_extension(self):
        """Return 3 extra state context dims for VAE.

        Returns:
            numpy [3] — [hunger, fatigue, stress] all in [0, 1]
        """
        return np.array([self.hunger, self.fatigue, self.stress],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    # Diagnostics & reset
    # ------------------------------------------------------------------

    def get_diagnostics(self):
        """Return monitoring dict."""
        return {
            "hunger": self.hunger,
            "fatigue": self.fatigue,
            "stress": self.stress,
            "hunger_error": self.hunger_error,
            "fatigue_error": self.fatigue_error,
            "stress_error": self.stress_error,
            "urgency": self.urgency,
            "speed_cap": self.get_speed_cap(),
            "goal_bias": self.get_goal_prior_bias().tolist(),
        }

    def reset(self):
        """Reset all internal state."""
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
