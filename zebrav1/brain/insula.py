"""
Insular Cortex — interoceptive awareness and emotional expression (Step 42).

The insula integrates bodily signals (heart rate, energy, fatigue, pain)
into a unified interoceptive state that modulates emotional expression
and decision-making.

Key functions:
  1. Heart rate detection → arousal level (fear, excitement)
  2. Energy state awareness → hunger urgency
  3. Emotional valence → approach (positive) vs withdraw (negative)
  4. Interoceptive prediction error → surprise from body state changes

Neuroscience: the anterior insula (homolog: zebrafish pallium) creates
a "feeling state" from interoceptive signals. Heart rate acceleration
signals threat arousal; deceleration signals safety (Craig 2009).
The insula-amygdala circuit converts bodily arousal into emotional
behavior (Critchley & Harrison 2013).

Pure numpy.
"""
import numpy as np


class Insula:
    """Interoceptive awareness from bodily signals.

    Args:
        arousal_tau: float — arousal smoothing time constant
        pe_threshold: float — interoceptive PE threshold for surprise
    """

    def __init__(self, arousal_tau=0.85, pe_threshold=0.1):
        self.arousal_tau = arousal_tau
        self.pe_threshold = pe_threshold

        # State
        self.arousal = 0.0         # overall bodily arousal [0, 1]
        self.valence = 0.0         # emotional valence [-1, 1]
        self.heart_rate = 0.3      # current HR belief
        self.predicted_hr = 0.3    # predicted HR (from efference copy)
        self.hr_pe = 0.0           # heart rate prediction error

        # Emotional state
        self.fear = 0.0
        self.excitement = 0.0
        self.calm = 1.0

    def step(self, heart_rate, energy_ratio, speed, is_fleeing,
             threat_level=0.0):
        """Update interoceptive state from bodily signals.

        Args:
            heart_rate: float [0, 1] — current heart rate
            energy_ratio: float [0, 1] — energy / max
            speed: float — current swim speed (normalised)
            is_fleeing: bool — flee goal active
            threat_level: float [0, 1] — from predator model

        Returns:
            diag: dict with interoceptive state
        """
        # Heart rate prediction error
        self.predicted_hr = 0.3 + 0.3 * speed  # expected HR from speed
        self.hr_pe = heart_rate - self.predicted_hr
        self.heart_rate = heart_rate

        # Arousal from HR: high HR = high arousal
        self.arousal = (self.arousal_tau * self.arousal
                        + (1 - self.arousal_tau) * heart_rate)

        # Emotional decomposition
        if self.hr_pe > self.pe_threshold:
            # HR higher than expected → fear/surprise
            self.fear = min(1.0, self.fear * 0.9 + 0.2 * self.hr_pe)
            self.excitement = max(0.0, self.excitement - 0.05)
        elif self.hr_pe < -self.pe_threshold:
            # HR lower than expected → calming
            self.fear = max(0.0, self.fear - 0.1)
            self.calm = min(1.0, self.calm + 0.05)
        else:
            self.fear *= 0.95
            self.excitement *= 0.95

        # Threat amplifies fear via arousal
        if threat_level > 0.3 and self.arousal > 0.5:
            self.fear = min(1.0, self.fear + 0.1 * threat_level * self.arousal)

        # Excitement from positive events (eating, approaching food)
        if energy_ratio > 0.7 and not is_fleeing:
            self.excitement = min(1.0, self.excitement + 0.02)

        # Calm when safe
        self.calm = max(0.0, 1.0 - self.fear - self.excitement)

        # Valence: positive = safe/fed, negative = threatened/hungry
        hunger_distress = max(0.0, 0.5 - energy_ratio)
        self.valence = self.excitement - self.fear - hunger_distress

        return self.get_diagnostics()

    def get_emotional_bias(self):
        """Emotional state → EFE bias for goal selection.

        Returns:
            bias: np.array[4] — [FORAGE, FLEE, EXPLORE, SOCIAL]
        """
        forage = -0.1 * max(0.0, -self.valence)  # hungry/distressed → forage
        flee = -0.15 * self.fear                   # fear → flee
        explore = -0.1 * self.calm                 # calm → explore
        social = -0.05 * self.excitement           # excited → social
        return np.array([forage, flee, explore, social], dtype=np.float32)

    def get_speed_boost(self):
        """Heart rate provides energy for sustained speed.

        High HR = more O2 supply = can maintain higher speed.
        But costs more energy (already handled in env).

        Returns:
            speed_mult: float [0.8, 1.2]
        """
        return 0.95 + 0.1 * self.heart_rate  # subtle: 0.95-1.05

    def reset(self):
        self.arousal = 0.0
        self.valence = 0.0
        self.heart_rate = 0.3
        self.predicted_hr = 0.3
        self.hr_pe = 0.0
        self.fear = 0.0
        self.excitement = 0.0
        self.calm = 1.0

    def get_diagnostics(self):
        return {
            "arousal": self.arousal,
            "valence": self.valence,
            "heart_rate": self.heart_rate,
            "hr_pe": self.hr_pe,
            "fear": self.fear,
            "excitement": self.excitement,
            "calm": self.calm,
        }
