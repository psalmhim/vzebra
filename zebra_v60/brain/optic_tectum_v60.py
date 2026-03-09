"""
Optic Tectum eye dynamics for v60.

Integrates BG gating, valence asymmetry, free energy, and dopamine
into continuous eye dynamics. Produces smooth pursuit + saccadic movements.
"""
import math


class OpticTectum_v60:
    def __init__(self, center_pull=0.18, bg_gain=0.12):
        self.eye_pos = 0.0
        self.eye_vel = 0.0
        self.center_pull = center_pull
        self.bg_gain = bg_gain
        self._saccade_cooldown = 0
        self._last_novelty = 0.0

    def trigger_saccade(self, direction, magnitude=0.6):
        """Rapid eye shift toward target. direction: -1 (left) to +1 (right)."""
        if self._saccade_cooldown > 0:
            return False
        self.eye_vel = direction * magnitude
        self.eye_pos += direction * magnitude * 0.5
        self.eye_pos = max(-1.0, min(1.0, self.eye_pos))
        self._saccade_cooldown = 8  # frames before next saccade
        return True

    def step(self, valL, valR, F_mean, bg_gate, dopa, novelty_drive=0.0):
        """
        Compute eye position from integrated signals.

        Args:
            novelty_drive: directional novelty signal (-1 left, +1 right)
                weighted by novelty strength. Biases eye toward novel regions.

        Returns:
            eye_pos: current eye position [-1, 1]
        """
        self._last_novelty = abs(novelty_drive)

        # Decrement saccade cooldown
        if self._saccade_cooldown > 0:
            self._saccade_cooldown -= 1
        # Visual salience competition
        salience_drive = math.tanh(1.5 * (valR - valL))

        # Free-energy penalty (look away from high-error regions)
        efe_penalty = -0.15 * F_mean

        # Basal ganglia gate (scaled down)
        gate_term = self.bg_gain * bg_gate

        # Total tectal drive (novelty biases gaze toward novel regions)
        drive = salience_drive + efe_penalty + gate_term + 0.25 * novelty_drive

        # Smooth momentum integration
        self.eye_vel = 0.65 * self.eye_vel + 0.35 * drive

        # Dopamine-modulated center pull: low DA → stronger centering
        dopa_center = self.center_pull * (1.0 + 0.5 * max(0, 0.5 - dopa))

        # Position update with velocity and centering
        self.eye_pos += 0.2 * self.eye_vel - dopa_center * self.eye_pos

        # Soft boundary: increasing resistance near edges
        if abs(self.eye_pos) > 0.6:
            overshoot = abs(self.eye_pos) - 0.6
            self.eye_vel -= 0.4 * overshoot * (1 if self.eye_pos > 0 else -1)

        self.eye_pos = max(-1.0, min(1.0, self.eye_pos))
        return self.eye_pos

    def reset(self):
        self.eye_pos = 0.0
        self.eye_vel = 0.0
        self._saccade_cooldown = 0
        self._last_novelty = 0.0
