"""
Habit Shortcut Module for v1.

Learns stimulus→goal associations from repeated selections. When confidence
is high and RPE is stable, bypasses full EFE deliberation.

Pure numpy — no torch dependency.
"""
import numpy as np


class HabitNetwork:
    """Learns habitual stimulus→goal mappings from repeated consistent selections."""

    def __init__(self, n_goals=3, habit_threshold=12, rpe_threshold=0.15,
                 conf_threshold=0.55, decay=0.98, n_obs_bins=8):
        self.n_goals = n_goals
        self.habit_threshold = habit_threshold
        self.rpe_threshold = rpe_threshold
        self.conf_threshold = conf_threshold
        self.decay = decay
        self.n_obs_bins = n_obs_bins

        # Habit table: bucket_tuple → np.ndarray[n_goals] (strength per goal)
        self.habit_table = {}

        # Streak tracking
        self._streak_count = 0
        self._streak_goal = -1
        self._streak_bucket = None

        # Statistics
        self.total_habit_uses = 0
        self.total_efe_uses = 0

    def _discretize(self, cls_probs):
        """Discretize food/enemy/colleague probs into bucket tuple."""
        p_food = cls_probs[1]
        p_enemy = cls_probs[2]
        p_colleague = cls_probs[3] if len(cls_probs) > 3 else 0.0
        bins = self.n_obs_bins
        b_food = int(np.clip(p_food * bins, 0, bins - 1))
        b_enemy = int(np.clip(p_enemy * bins, 0, bins - 1))
        b_colleague = int(np.clip(p_colleague * bins, 0, bins - 1))
        return (b_food, b_enemy, b_colleague)

    def step(self, cls_probs, goal_choice, confidence, rpe):
        """Process one timestep and decide whether to use habit shortcut.

        Args:
            cls_probs: array [5] — classifier probabilities
            goal_choice: int — EFE-selected goal
            confidence: float — meta-precision (0–1)
            rpe: float — reward prediction error

        Returns:
            effective_goal: int — the goal to actually use
            shortcut_active: bool — whether habit shortcut was used
        """
        bucket = self._discretize(cls_probs)

        # Decay all habit strengths
        for key in self.habit_table:
            self.habit_table[key] *= self.decay

        # Check if conditions are stable for streak building
        stable = abs(rpe) < self.rpe_threshold and confidence > self.conf_threshold

        if stable and goal_choice == self._streak_goal and bucket == self._streak_bucket:
            self._streak_count += 1
        else:
            self._streak_count = 1
            self._streak_goal = goal_choice
            self._streak_bucket = bucket

        # Consolidate habit after threshold consecutive steps
        if self._streak_count >= self.habit_threshold and stable:
            if bucket not in self.habit_table:
                self.habit_table[bucket] = np.zeros(self.n_goals)
            self.habit_table[bucket][goal_choice] += 1.0

        # Check if habit shortcut should fire
        if bucket in self.habit_table:
            strengths = self.habit_table[bucket]
            best_goal = int(np.argmax(strengths))
            if strengths[best_goal] > 3.0:
                self.total_habit_uses += 1
                return best_goal, True

        self.total_efe_uses += 1
        return goal_choice, False

    def get_habit_strength(self, cls_probs, goal):
        """Get habit strength for a specific observation-goal pair."""
        bucket = self._discretize(cls_probs)
        if bucket in self.habit_table:
            return float(self.habit_table[bucket][goal])
        return 0.0

    def get_stats(self):
        """Return usage statistics."""
        total = self.total_habit_uses + self.total_efe_uses
        return {
            "total_habit_uses": self.total_habit_uses,
            "total_efe_uses": self.total_efe_uses,
            "habit_rate": self.total_habit_uses / max(total, 1),
        }

    def get_saveable_state(self):
        """Return learned habit table for checkpoint."""
        return {
            "habit_table": {k: v.copy() for k, v in self.habit_table.items()},
            "total_habit_uses": self.total_habit_uses,
            "total_efe_uses": self.total_efe_uses,
        }

    def load_saveable_state(self, state):
        """Restore learned habit table."""
        self.habit_table = {k: v.copy() for k, v in state["habit_table"].items()}
        self.total_habit_uses = state["total_habit_uses"]
        self.total_efe_uses = state["total_efe_uses"]

    def reset_episode(self):
        """Clear episodic streaks but keep learned habits."""
        self._streak_count = 0
        self._streak_goal = -1
        self._streak_bucket = None

    def reset(self):
        self.habit_table = {}
        self._streak_count = 0
        self._streak_goal = -1
        self._streak_bucket = None
        self.total_habit_uses = 0
        self.total_efe_uses = 0
