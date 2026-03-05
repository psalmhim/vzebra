"""
Active Inference Policy Selection via Expected Free Energy (EFE).

Maintains three goal states (FORAGE, FLEE, EXPLORE) and selects between
them by minimizing EFE. Persistence timer prevents goal flickering.
Emergency override for high-confidence enemy detection.

Pure numpy — no torch dependency.
"""
import numpy as np

GOAL_FORAGE = 0
GOAL_FLEE = 1
GOAL_EXPLORE = 2
GOAL_NAMES = ["FORAGE", "FLEE", "EXPLORE"]


class GoalPolicy_v60:
    """EFE-based policy selection with persistence and emergency override."""

    def __init__(self, n_goals=3, beta=2.0, persist_steps=8,
                 weight_adapter=None):
        self.n_goals = n_goals
        self.beta = beta
        self.persist_steps = persist_steps
        self.weight_adapter = weight_adapter

        self.timer = 0
        self.last_choice = GOAL_EXPLORE
        self.efe_smooth = np.zeros(n_goals)  # exponential moving average
        self._alpha = 0.7  # EFE smoothing factor (higher = faster adaptation)
        self._external_efe_bonus = np.zeros(n_goals)

    def set_plan_bonus(self, bonus):
        """Set external EFE bonus from VAE planner (Step 16)."""
        self._external_efe_bonus = np.asarray(bonus, dtype=np.float64)

    def compute_efe(self, cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual,
                    mem_mean):
        """Compute Expected Free Energy for each goal.

        Args:
            cls_probs: array [5] — softmax classifier output
                       [nothing, food, enemy, colleague, environment]
            pi_OT: float — optic tectum precision (0–1)
            pi_PC: float — predictive coding precision (0–1)
            dopa: float — dopamine level (0–1)
            rpe: float — reward prediction error
            cms: float — cross-modal surprise
            F_visual: float — visual free energy
            mem_mean: float — working memory mean activation

        Returns:
            efe: array [3] — EFE per goal (lower = preferred)
        """
        # Delegate to weight adapter if present (Step 15 RL critic)
        if self.weight_adapter is not None:
            return self.weight_adapter.compute_efe(
                cls_probs, pi_OT, pi_PC, dopa, cms,
                rpe=rpe, F_visual=F_visual, mem_mean=mem_mean)

        p_nothing = cls_probs[0]
        p_food = cls_probs[1]
        p_enemy = cls_probs[2]
        p_environ = cls_probs[4]

        uncertainty = 1.0 - 0.5 * (pi_OT + pi_PC)

        # FORAGE: attractive when food is visible
        g_forage = (0.2 * uncertainty
                    - 0.8 * p_food
                    + 0.2 * (0.5 - dopa)
                    + 0.15)

        # FLEE: attractive when enemy is detected
        # Moderate baseline (+0.2) so FLEE activates more readily
        g_flee = (0.1 * cms
                  - 0.8 * p_enemy
                  + 0.2)

        # EXPLORE: attractive when seeing nothing/environment
        g_explore = (0.3 * uncertainty
                     - 0.5 * (p_nothing + p_environ)
                     + 0.1 * cms
                     + 0.2)

        return np.array([g_forage, g_flee, g_explore])

    def step(self, cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual,
             mem_mean, hunger=0.0, hunger_error=0.0):
        """Select goal by minimizing smoothed EFE.

        Returns:
            choice: int — selected goal index
            goal_vec: array [3] — one-hot goal vector
            posterior: array [3] — policy posterior probabilities
            confidence: float — entropy-based meta-precision (0–1)
            efe_vec: array [3] — raw EFE values
        """
        efe_raw = self.compute_efe(cls_probs, pi_OT, pi_PC, dopa, rpe, cms,
                                   F_visual, mem_mean)

        # Apply external planning bonus (Step 16 VAE planner)
        efe_raw = efe_raw + self._external_efe_bonus

        # Light smoothing for EFE history (used for diagnostics/plots)
        self.efe_smooth = ((1 - self._alpha) * self.efe_smooth
                           + self._alpha * efe_raw)

        # Use raw EFE for selection — the persistence timer provides stability.
        # Smoothing the selection signal causes excessive goal stickiness.
        efe_rel = efe_raw - efe_raw.min()

        # Policy posterior: softmax(-beta * G)
        logits = -self.beta * efe_rel
        logits = logits - logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        posterior = exp_logits / (exp_logits.sum() + 1e-12)

        # Emergency override: if p_enemy > 0.5, force FLEE
        p_enemy = cls_probs[2]
        p_food = cls_probs[1]
        if p_enemy > 0.5:
            choice = GOAL_FLEE
            self.last_choice = choice
            self.timer = 0
        # Emergency FORAGE: critically low energy with declining trend
        elif hunger > 0.7 and hunger_error > 0.3:
            choice = GOAL_FORAGE
            self.last_choice = choice
            self.timer = 0
        elif (self.last_choice == GOAL_FLEE and p_enemy < 0.15
              and self.timer >= 3):
            # Early exit from FLEE when no enemy detected
            choice = int(np.argmax(posterior))
            self.last_choice = choice
            self.timer = 0
        elif (self.last_choice == GOAL_FORAGE and p_food < 0.15
              and self.timer >= 4):
            # Early exit from FORAGE when no food detected
            choice = int(np.argmax(posterior))
            self.last_choice = choice
            self.timer = 0
        elif self.timer < self.persist_steps:
            # Persist current goal
            self.timer += 1
            choice = self.last_choice
        else:
            # Re-select: pick goal with highest posterior (lowest EFE)
            choice = int(np.argmax(posterior))
            self.last_choice = choice
            self.timer = 0

        # Goal vector (one-hot)
        goal_vec = np.zeros(self.n_goals)
        goal_vec[choice] = 1.0

        # Confidence: entropy-based meta-precision
        entropy = -np.sum(posterior * np.log(posterior + 1e-12))
        max_entropy = np.log(self.n_goals)
        confidence = 1.0 - entropy / max_entropy

        return choice, goal_vec, posterior, confidence, self.efe_smooth.copy()

    def reset(self):
        self.timer = 0
        self.last_choice = GOAL_EXPLORE
        self.efe_smooth = np.zeros(self.n_goals)
        self._external_efe_bonus = np.zeros(self.n_goals)


def goal_to_behavior(active_goal, cls_probs, posterior, confidence):
    """Convert active goal to behavioral modulation parameters.

    Args:
        active_goal: int — GOAL_FORAGE, GOAL_FLEE, or GOAL_EXPLORE
        cls_probs: array [5] — classifier probabilities
        posterior: array [3] — goal posterior
        confidence: float — meta-precision (0–1)

    Returns:
        approach_gain: float — positive=approach, negative=flee
        speed_mod: float — speed multiplier
        explore_mod: float — BG exploration multiplier
        turn_strategy: str — description for logging
    """
    p_food = cls_probs[1]
    p_enemy = cls_probs[2]

    if active_goal == GOAL_FORAGE:
        approach_gain = 1.0 + 0.5 * p_food * confidence
        speed_mod = 1.0 + 0.3 * p_food
        explore_mod = 0.5
        turn_strategy = "forage-approach"

    elif active_goal == GOAL_FLEE:
        approach_gain = -(0.8 + 0.5 * p_enemy * confidence)
        speed_mod = 1.3 + 0.4 * p_enemy
        explore_mod = 0.3
        turn_strategy = "flee-avoid"

    else:  # GOAL_EXPLORE
        approach_gain = 0.3
        speed_mod = 0.9
        explore_mod = 1.5 + 0.5 * (1 - confidence)
        turn_strategy = "explore-search"

    return approach_gain, speed_mod, explore_mod, turn_strategy
