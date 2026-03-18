"""
Vestibular System — gravity sensing and balance (Step 37).

Simulates otolith-based gravity detection and semicircular canal
angular velocity sensing.  Outputs body orientation estimate and
stabilisation reflexes (dorsal light response, vestibulo-ocular reflex).

Neuroscience: zebrafish larvae use utricular otoliths for gravity
sensing (Riley & Bhatt 2004).  The vestibulo-ocular reflex (VOR)
stabilises gaze during swimming via cerebellum-mediated gain control
(Bianco et al. 2012).

Pure numpy.
"""
import math
import numpy as np


class VestibularSystem:
    """Otolith + semicircular canal vestibular model.

    Args:
        gravity_gain: float — otolith sensitivity
        angular_gain: float — semicircular canal sensitivity
        vor_gain: float — vestibulo-ocular reflex gain
        tau_canal: float — semicircular canal time constant
    """

    def __init__(self, gravity_gain=1.0, angular_gain=0.8,
                 vor_gain=0.5, tau_canal=0.9):
        self.gravity_gain = gravity_gain
        self.angular_gain = angular_gain
        self.vor_gain = vor_gain
        self.tau_canal = tau_canal

        # Body state
        self.pitch = 0.0       # forward/back tilt (rad)
        self.roll = 0.0        # left/right tilt (rad)
        self.angular_vel = 0.0  # yaw angular velocity (rad/step)

        # Semicircular canal (high-pass filtered angular velocity)
        self._canal_state = 0.0

        # Stabilisation outputs
        self.pitch_correction = 0.0
        self.roll_correction = 0.0
        self.vor_eye_compensation = 0.0

    def step(self, heading_change, speed, depth_asymmetry=0.0):
        """Update vestibular state from motor/sensory signals.

        Args:
            heading_change: float — change in heading this step (rad)
            speed: float — swim speed [0, ~1.6]
            depth_asymmetry: float — left-right depth difference
                (from binocular disparity, proxy for roll)

        Returns:
            diag: dict with vestibular state
        """
        # Angular velocity from heading change (semicircular canal)
        self.angular_vel = heading_change
        self._canal_state = (self.tau_canal * self._canal_state
                             + (1 - self.tau_canal) * heading_change
                             * self.angular_gain)

        # Pitch: speed-dependent (fast swimming = nose-down tendency)
        self.pitch = -0.1 * speed + np.random.normal(0, 0.01)

        # Roll: from bilateral asymmetry + noise
        self.roll = depth_asymmetry * 0.3 + np.random.normal(0, 0.01)

        # Dorsal light response: correct pitch toward horizontal
        self.pitch_correction = -self.gravity_gain * self.pitch

        # Roll stabilisation
        self.roll_correction = -self.gravity_gain * self.roll

        # VOR: compensatory eye movement opposite to head turn
        self.vor_eye_compensation = -self.vor_gain * self._canal_state

        return self.get_diagnostics()

    def get_balance_penalty(self):
        """Return balance quality [0, 1]. 1 = perfect, 0 = tumbling."""
        tilt = math.sqrt(self.pitch ** 2 + self.roll ** 2)
        return max(0.0, 1.0 - tilt * 5.0)

    def get_speed_correction(self):
        """Reduce speed when off-balance."""
        balance = self.get_balance_penalty()
        if balance < 0.8:
            return 0.7 + 0.3 * balance
        return 1.0

    def reset(self):
        self.pitch = 0.0
        self.roll = 0.0
        self.angular_vel = 0.0
        self._canal_state = 0.0
        self.pitch_correction = 0.0
        self.roll_correction = 0.0
        self.vor_eye_compensation = 0.0

    def get_diagnostics(self):
        return {
            "pitch": self.pitch,
            "roll": self.roll,
            "angular_vel": self.angular_vel,
            "canal_state": self._canal_state,
            "balance": self.get_balance_penalty(),
            "vor_eye": self.vor_eye_compensation,
        }
