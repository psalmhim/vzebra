"""
Spinal Central Pattern Generator — oscillatory motor circuits (Step 38).

Replaces the phenomenological bout-glide-idle counter with a biologically
grounded half-centre oscillator that generates rhythmic swim patterns.
The CPG receives descending drive from the brain (speed/turn commands)
and produces patterned motor output with intrinsic frequency modulation.

Neuroscience: zebrafish spinal cord contains V0-V2 interneurons
forming a half-centre oscillator.  Swim frequency scales with
descending drive from reticulospinal neurons (Kinkhabwala et al.
2011; McLean et al. 2007).

Pure numpy.
"""
import math
import numpy as np


class SpinalCPG:
    """Half-centre oscillator for rhythmic swimming.

    Two mutually inhibiting units (left/right) produce alternating
    burst patterns.  Descending drive modulates frequency and amplitude.

    Args:
        base_freq: float — natural oscillation frequency (cycles/step)
        mutual_inhibition: float — left↔right inhibition strength
        tau: float — membrane time constant
        drive_to_freq: float — descending drive → frequency scaling
        noise_sigma: float — biological noise in motor output
    """

    def __init__(self, base_freq=0.15, mutual_inhibition=0.5,
                 tau=0.8, drive_to_freq=0.3, noise_sigma=0.02):
        self.base_freq = base_freq
        self.w_inh = mutual_inhibition
        self.tau = tau
        self.drive_to_freq = drive_to_freq
        self.noise_sigma = noise_sigma

        # Half-centre state (left, right)
        self.v_L = 0.5
        self.v_R = 0.0
        self._phase = 0.0

        # Output state
        self.motor_L = 0.0
        self.motor_R = 0.0
        self.bout_active = False
        self._cycle_count = 0

    def step(self, descending_drive, turn_bias=0.0):
        """Update CPG and produce motor output.

        Args:
            descending_drive: float [0, 1] — brain speed command
            turn_bias: float [-1, 1] — brain turn command
                positive = bias right motor (turn left)

        Returns:
            motor_L: float — left motor neuron activation
            motor_R: float — right motor neuron activation
            speed: float — effective swim speed
            turn: float — effective turn rate
            diag: dict
        """
        # Frequency modulation by descending drive
        freq = self.base_freq + self.drive_to_freq * descending_drive

        # Phase advance
        self._phase += freq
        if self._phase >= 1.0:
            self._phase -= 1.0
            self._cycle_count += 1

        # Half-centre oscillator dynamics
        # Sinusoidal drive with mutual inhibition
        drive_osc = math.sin(2 * math.pi * self._phase)
        ext_L = max(0, drive_osc) * descending_drive
        ext_R = max(0, -drive_osc) * descending_drive

        # Add turn bias (asymmetric drive)
        ext_L += max(0, turn_bias) * 0.5
        ext_R += max(0, -turn_bias) * 0.5

        # Leaky integration with mutual inhibition
        self.v_L = (self.tau * self.v_L
                    + (1 - self.tau) * (ext_L - self.w_inh * self.v_R))
        self.v_R = (self.tau * self.v_R
                    + (1 - self.tau) * (ext_R - self.w_inh * self.v_L))

        # Rectify
        self.v_L = max(0.0, min(1.0, self.v_L))
        self.v_R = max(0.0, min(1.0, self.v_R))

        # Motor output with biological noise
        self.motor_L = self.v_L + np.random.normal(0, self.noise_sigma)
        self.motor_R = self.v_R + np.random.normal(0, self.noise_sigma)
        self.motor_L = max(0.0, min(1.0, self.motor_L))
        self.motor_R = max(0.0, min(1.0, self.motor_R))

        # Bout detection: active when either side > threshold
        self.bout_active = max(self.motor_L, self.motor_R) > 0.3

        # Convert to speed and turn
        speed = (self.motor_L + self.motor_R) / 2.0
        turn = (self.motor_R - self.motor_L) * 2.0  # differential drive

        return self.motor_L, self.motor_R, speed, turn, {
            "phase": self._phase,
            "freq": freq,
            "v_L": self.v_L,
            "v_R": self.v_R,
            "bout_active": self.bout_active,
            "cycle_count": self._cycle_count,
        }

    def reset(self):
        self.v_L = 0.5
        self.v_R = 0.0
        self._phase = 0.0
        self.motor_L = 0.0
        self.motor_R = 0.0
        self.bout_active = False
        self._cycle_count = 0

    def get_diagnostics(self):
        return {
            "phase": self._phase,
            "v_L": self.v_L,
            "v_R": self.v_R,
            "motor_L": self.motor_L,
            "motor_R": self.motor_R,
            "bout_active": self.bout_active,
        }
