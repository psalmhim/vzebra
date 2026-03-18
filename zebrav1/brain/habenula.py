"""
Habenula — behavioral flexibility and learned helplessness (Step 36).

The habenula computes a "disappointment" signal: when expected reward
repeatedly fails to materialize, it drives strategy switching.
Implements three functions:

  1. Anti-reward: sustained negative RPE → helplessness accumulates
  2. Strategy switching: when helplessness > threshold, force goal change
  3. Behavioral flexibility: tracks which goals have been failing
     and biases selection away from them

Neuroscience: zebrafish lateral habenula (LHb) receives reward
prediction error from the entopeduncular nucleus and projects to
raphe/VTA. Sustained negative RPE → LHb activation → suppression
of dopamine → learned helplessness (Amo et al. 2014; Okamoto et al.
2012). Asymmetric left-right habenula mediates boldness/shyness
personality differences.

Pure numpy — no torch.
"""
import numpy as np


class Habenula:
    """Habenula anti-reward and behavioral flexibility module.

    Args:
        helplessness_threshold: float — frustration level to trigger switch
        helplessness_decay: float — per-step decay of frustration
        helplessness_gain: float — per-step gain from negative RPE
        memory_horizon: int — steps to track goal failure history
    """

    def __init__(self, helplessness_threshold=0.7,
                 helplessness_decay=0.995, helplessness_gain=0.02,
                 memory_horizon=50):
        self.threshold = helplessness_threshold
        self.decay = helplessness_decay
        self.gain = helplessness_gain
        self.memory_horizon = memory_horizon

        # Frustration level per goal (accumulated negative RPE)
        self.frustration = np.zeros(4, dtype=np.float32)

        # Overall helplessness (max frustration across goals)
        self.helplessness = 0.0

        # Goal failure history: how many of last N steps in each goal
        # produced negative RPE
        self._goal_history = []  # [(goal_idx, rpe)]
        self._switch_cooldown = 0

    def step(self, current_goal, rpe, dopa):
        """Update helplessness and check for strategy switch.

        Args:
            current_goal: int — current active goal (0-3)
            rpe: float — reward prediction error
            dopa: float — dopamine level [0, 1]

        Returns:
            switch_signal: bool — True if should force goal change
            goal_bias: np.array[4] — EFE bias (positive = avoid goal)
            diag: dict
        """
        # Decay frustration
        self.frustration *= self.decay

        # Accumulate frustration from negative RPE
        if rpe < -0.05:
            self.frustration[current_goal] += self.gain * abs(rpe)

        # Positive RPE reduces frustration for current goal
        if rpe > 0.1:
            self.frustration[current_goal] *= 0.8

        # Clamp
        self.frustration = np.clip(self.frustration, 0.0, 1.0)

        # Overall helplessness
        self.helplessness = float(self.frustration.max())

        # Track goal history
        self._goal_history.append((current_goal, rpe))
        if len(self._goal_history) > self.memory_horizon:
            self._goal_history = self._goal_history[-self.memory_horizon:]

        # Strategy switch: if current goal frustration exceeds threshold
        # and cooldown expired
        switch = False
        if self._switch_cooldown > 0:
            self._switch_cooldown -= 1
        elif self.frustration[current_goal] > self.threshold:
            switch = True
            self._switch_cooldown = 15  # don't switch again too soon
            # Reset frustration for the goal we're switching FROM
            # (give it a fresh chance later)
            self.frustration[current_goal] *= 0.3

        # Goal bias: frustrated goals get positive EFE bias (avoid them)
        goal_bias = self.frustration * 0.5  # scale factor

        # Dopamine modulation: low dopa amplifies helplessness
        # (depressed state → harder to recover)
        dopa_mod = max(0.5, 2.0 * (1.0 - dopa))
        goal_bias *= dopa_mod

        return switch, goal_bias, {
            "helplessness": self.helplessness,
            "frustration": self.frustration.copy(),
            "switch_signal": switch,
            "cooldown": self._switch_cooldown,
        }

    def get_saveable_state(self):
        return {
            "frustration": self.frustration.copy(),
        }

    def load_saveable_state(self, state):
        self.frustration = state["frustration"].copy()

    def reset(self):
        self.frustration[:] = 0.0
        self.helplessness = 0.0
        self._goal_history = []
        self._switch_cooldown = 0

    def get_diagnostics(self):
        return {
            "helplessness": self.helplessness,
            "frustration": self.frustration.tolist(),
            "cooldown": self._switch_cooldown,
        }
