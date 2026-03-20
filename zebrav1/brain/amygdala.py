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
    """Fear circuit with episodic conditioning: near-death → lasting fear.

    After a near-death event (predator very close), the fear_baseline
    rises permanently for the episode.  This makes the fish increasingly
    cautious — each close call amplifies future threat responses.

    Neuroscience: amygdala fear conditioning creates long-term potentiation
    of threat-related synapses.  A single traumatic event can produce
    lasting hypervigilance (LeDoux 2000).
    """

    def __init__(self, decay=0.75, rise_rate=0.6, retinal_gain=0.08,
                 proximity_range=200.0):
        self.decay = decay
        self.rise_rate = rise_rate
        self.retinal_gain = retinal_gain
        self.proximity_range = proximity_range

        self.threat_arousal = 0.0
        # Episodic fear: grows after near-death, never resets within episode
        self.fear_baseline = 0.0
        self.near_death_count = 0

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

        # Episodic fear conditioning: near-death creates lasting baseline
        # Near-death = predator extremely close (proximity > 0.9) AND facing fish
        if proximity > 0.9 and pred_facing_score > 0.5:
            self.fear_baseline = min(0.3, self.fear_baseline + 0.03)
            self.near_death_count += 1

        # Add fear baseline to raw signal (more cautious after trauma)
        raw = max(raw, raw + self.fear_baseline * proximity)

        # Leaky integration: rises fast, decays slower
        # Decay floor = fear_baseline (never goes below trauma level)
        if raw > self.threat_arousal:
            self.threat_arousal = ((1.0 - self.rise_rate) * self.threat_arousal
                                   + self.rise_rate * raw)
        else:
            self.threat_arousal = max(
                self.fear_baseline * 0.3,
                self.threat_arousal * self.decay)

        return self.threat_arousal

    def get_diagnostics(self):
        return {
            "threat_arousal": self.threat_arousal,
            "fear_baseline": self.fear_baseline,
            "near_death_count": self.near_death_count,
        }

    def reset(self):
        """Reset transient state. Fear baseline persists across resets
        within an episode but clears on full reset."""
        self.threat_arousal = 0.0

    def reset_full(self):
        """Full reset including episodic fear memory."""
        self.threat_arousal = 0.0
        self.fear_baseline = 0.0
        self.near_death_count = 0
