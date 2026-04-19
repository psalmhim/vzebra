"""
SocialMemory: tracks conspecific behavioral states and learns social inference weights.

Three learned weights updated via error-driven (precision-EMA) online learning:
  w_alarm       — amplify/dampen social alarm when conspecifics flee
  w_food_cue    — trust level for food inferred from conspecific clustering
  w_competition — penalty for crowded food patches (drives toward less-contested areas)

Update rule (precision-EMA, avoids asymmetric drift):
  - After alarm fires: accumulate "predator was close" over a 20-step horizon window.
    At horizon: update w_alarm by pulling toward EMA of alarm precision.
  - Same for food cue over a 30-step window.
  - Competition weight updated per-episode via simple fitness correlation.

Design notes:
  - `update_states()` is the only place eating/fleeing state is inferred (no duplication).
  - `get_food_cue_efe()` uses cached `_fish_states`, not raw colleagues.
  - Outcome evaluation uses a window (any hit in horizon) not a single snapshot.
"""
import math
import numpy as np


class SocialMemory:
    def __init__(self):
        # Per-fish behavioral state estimates (keyed by index in colleagues list)
        self._fish_states: dict = {}
        # Learned inference weights (init 1.0 = same as original fixed coefficients)
        self.w_alarm = 1.0         # scales shoaling alarm coefficient (0.2)
        self.w_food_cue = 1.0      # scales food-cue EFE contribution
        self.w_competition = 1.0   # scales competition crowding penalty
        self._lr = 0.05
        # Running precision estimates (EMA of hit rate) for convergent weight update
        self._alarm_precision_ema: float = 0.5   # init neutral
        self._food_precision_ema: float = 0.5
        # Alarm outcome window tracking
        self._alarm_triggered: bool = False
        self._alarm_step: int = 0
        self._alarm_horizon: int = 20
        self._alarm_hit_in_window: bool = False  # any hit during horizon window
        # Food cue outcome window tracking
        self._food_cue_taken: bool = False
        self._food_cue_step: int = 0
        self._food_cue_horizon: int = 30
        self._food_hit_in_window: bool = False
        self._step: int = 0

    # -------------------------------------------------------------------------
    # State update (call every step)
    # -------------------------------------------------------------------------

    def update_states(self, my_x: float, my_y: float, colleagues: list):
        """
        Infer each conspecific's behavioral state from kinematics.
        colleagues: list of {'x', 'y', 'heading', 'speed'}
        """
        self._step += 1
        self._fish_states = {}
        for i, c in enumerate(colleagues):
            speed = c.get('speed', 0.5)
            dist = math.sqrt((c['x'] - my_x) ** 2 + (c['y'] - my_y) ** 2) + 1e-8
            self._fish_states[i] = {
                'eating':  speed < 0.30,
                'fleeing': speed > 1.50,
                'pos':     (c['x'], c['y']),
                'speed':   speed,
                'dist':    dist,
                'heading': c.get('heading', 0.0),
            }

    # -------------------------------------------------------------------------
    # Signal accessors (used in EFE computation)
    # -------------------------------------------------------------------------

    def get_social_alarm(self, raw_alarm: float) -> float:
        """Learned-weighted alarm. Triggers alarm tracking when signal crosses threshold."""
        scaled = raw_alarm * self.w_alarm
        if scaled > 0.05:
            self.set_alarm_triggered()
        return min(1.0, scaled)

    def get_food_cue_efe(self, my_x: float, my_y: float,
                          fish_heading: float) -> float:
        """
        EFE bonus for FORAGE based on cached conspecific eating states.
        Returns a negative value (lower EFE = more preferred) when signal is strong.
        Uses _fish_states from the most recent update_states() call.
        """
        eating_positions = [s['pos'] for s in self._fish_states.values() if s['eating']]
        if not eating_positions:
            return 0.0
        cx = sum(x for x, _ in eating_positions) / len(eating_positions)
        cy = sum(y for _, y in eating_positions) / len(eating_positions)
        dist = math.sqrt((cx - my_x) ** 2 + (cy - my_y) ** 2) + 1e-8
        bearing = math.atan2(cy - my_y, cx - my_x)
        angle_diff = abs(math.atan2(math.sin(bearing - fish_heading),
                                     math.cos(bearing - fish_heading)))
        direction_factor = max(0.0, math.cos(angle_diff))  # 1 ahead, 0 behind
        proximity_factor = max(0.0, 1.0 - dist / 200.0)
        strength = proximity_factor * (0.3 + 0.7 * direction_factor) * self.w_food_cue
        if strength > 0.05:
            self.set_food_cue_taken()
        return -min(0.15, strength * 0.20)  # negative = more preferred

    def get_competition_penalty(self, my_x: float, my_y: float) -> float:
        """
        EFE cost for FORAGE when conspecifics crowd the same foraging area.
        Returns a positive value (higher EFE = less preferred).
        """
        if not self._fish_states:
            return 0.0
        nearby_foragers = sum(
            1 for s in self._fish_states.values()
            if s['dist'] < 100 and not s['fleeing']
        )
        if nearby_foragers < 2:
            return 0.0
        return min(0.12, nearby_foragers * 0.04) * self.w_competition

    # -------------------------------------------------------------------------
    # Error-driven weight updates (call every step)
    # -------------------------------------------------------------------------

    def update_alarm_outcome(self, pred_was_close: bool):
        """
        Call every step (not just when colleagues visible) to evaluate alarm predictions.
        Accumulates any predator-close event within the horizon window.
        Applies precision-EMA update at horizon end.
        """
        if not self._alarm_triggered:
            return
        # Accumulate hit within window
        if pred_was_close:
            self._alarm_hit_in_window = True
        steps_since = self._step - self._alarm_step
        if steps_since < self._alarm_horizon:
            return
        # Horizon reached: update precision EMA then pull weight toward precision
        hit = self._alarm_hit_in_window
        self._alarm_precision_ema = (0.9 * self._alarm_precision_ema
                                     + 0.1 * (1.0 if hit else 0.0))
        # Target weight: 0.2 (skeptical) to 3.0 (highly trusted), linear in precision
        target_w = 0.2 + 2.8 * self._alarm_precision_ema
        self.w_alarm += self._lr * (target_w - self.w_alarm)
        self.w_alarm = float(np.clip(self.w_alarm, 0.2, 3.0))
        self._alarm_triggered = False
        self._alarm_hit_in_window = False

    def update_food_cue_outcome(self, food_found: bool):
        """
        Call every step to evaluate food-cue predictions.
        Accumulates any food-found event within the horizon window.
        """
        if not self._food_cue_taken:
            return
        if food_found:
            self._food_hit_in_window = True
        steps_since = self._step - self._food_cue_step
        if steps_since < self._food_cue_horizon:
            return
        hit = self._food_hit_in_window
        self._food_precision_ema = (0.9 * self._food_precision_ema
                                    + 0.1 * (1.0 if hit else 0.0))
        target_w = 0.2 + 2.8 * self._food_precision_ema
        self.w_food_cue += self._lr * (target_w - self.w_food_cue)
        self.w_food_cue = float(np.clip(self.w_food_cue, 0.2, 3.0))
        self._food_cue_taken = False
        self._food_hit_in_window = False

    def set_alarm_triggered(self):
        if not self._alarm_triggered:
            self._alarm_triggered = True
            self._alarm_step = self._step
            self._alarm_hit_in_window = False

    def set_food_cue_taken(self):
        if not self._food_cue_taken:
            self._food_cue_taken = True
            self._food_cue_step = self._step
            self._food_hit_in_window = False

    # -------------------------------------------------------------------------
    # Episode reset & serialization
    # -------------------------------------------------------------------------

    def reset(self):
        """Reset per-episode tracking state. Learned weights and precision EMAs persist."""
        self._fish_states = {}
        self._alarm_triggered = False
        self._alarm_hit_in_window = False
        self._food_cue_taken = False
        self._food_hit_in_window = False
        self._step = 0

    def state_dict(self) -> dict:
        return {
            'w_alarm':               float(self.w_alarm),
            'w_food_cue':            float(self.w_food_cue),
            'w_competition':         float(self.w_competition),
            'alarm_precision_ema':   float(self._alarm_precision_ema),
            'food_precision_ema':    float(self._food_precision_ema),
        }

    def load_state_dict(self, d: dict):
        self.w_alarm       = float(d.get('w_alarm', 1.0))
        self.w_food_cue    = float(d.get('w_food_cue', 1.0))
        self.w_competition = float(d.get('w_competition', 1.0))
        self._alarm_precision_ema = float(d.get('alarm_precision_ema', 0.5))
        self._food_precision_ema  = float(d.get('food_precision_ema', 0.5))
