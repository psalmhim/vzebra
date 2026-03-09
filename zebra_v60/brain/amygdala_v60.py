"""
Amygdala / Insular Fear Response Module.

Produces a fast threat_arousal signal [0, 1] from three inputs:
  1. Retinal enemy pixels — direct fast-pathway visual alarm
  2. Predator proximity — graded distance signal
  3. Allostatic stress — slower interoceptive channel

The arousal signal rises quickly (60% jump toward raw input) but decays
slowly (×0.85 per step), modelling the biological persistence of fear
after a threat disappears.

Effects integrated in brain_agent:
  - Boosts classifier p_enemy by 0.3 × threat_arousal
  - Increases allostatic stress by 0.1 × threat_arousal
  - Extends flee burst duration: 5 + int(5 × threat_arousal) steps

Pure numpy — no torch dependency.
"""


class Amygdala:
    """Minimal fear-circuit producing a persistent threat arousal signal."""

    def __init__(self, decay=0.85, rise_rate=0.6, retinal_gain=0.08,
                 proximity_range=200.0):
        self.decay = decay
        self.rise_rate = rise_rate
        self.retinal_gain = retinal_gain
        self.proximity_range = proximity_range

        self.threat_arousal = 0.0

    def step(self, enemy_pixels, pred_dist, stress, pred_facing_score=0.0):
        """Update threat arousal from multi-modal threat evidence.

        Args:
            enemy_pixels: int — total enemy-type pixels on retina
            pred_dist: float — distance to predator in pixels
            stress: float — current allostatic stress [0, 1]
            pred_facing_score: float [0, 1] — how directly the predator
                faces the fish (0=away, 1=dead-on). Amplifies threat
                when predator is nearby and looking at the fish.

        Returns:
            threat_arousal: float [0, 1]
        """
        # Fast pathway: direct retinal enemy pixel input
        retinal_threat = min(1.0, enemy_pixels * self.retinal_gain)

        # Proximity boost (saturates at proximity_range)
        proximity = max(0.0, 1.0 - pred_dist / self.proximity_range)

        # Gaze-direction threat: predator facing the fish from nearby
        gaze_threat = pred_facing_score * proximity * 0.8

        # Take the strongest threat signal
        raw = max(retinal_threat, 0.5 * proximity, stress, gaze_threat)

        # Leaky integration: rises fast, decays slower
        if raw > self.threat_arousal:
            self.threat_arousal = ((1.0 - self.rise_rate) * self.threat_arousal
                                   + self.rise_rate * raw)
        else:
            self.threat_arousal *= self.decay

        return self.threat_arousal

    def get_diagnostics(self):
        """Return monitoring dict."""
        return {"threat_arousal": self.threat_arousal}

    def reset(self):
        """Reset internal state."""
        self.threat_arousal = 0.0
