"""
Proprioceptive Feedback — body state sensing (Step 41).

Closes the motor control loop by providing sensory feedback about
the body's actual movement state, complementing the motor efference
copy. Detects discrepancies between commanded and actual movement
(e.g., collision with obstacle, water current, muscle fatigue).

Components:
  - Muscle spindle: actual speed and turn rate (noisy observation)
  - Golgi tendon organ: force/effort level
  - Proprioceptive prediction error: commanded - actual

Neuroscience: fish lack muscle spindles but have Rohon-Beard
mechanosensory neurons and dorsal root ganglia that detect body
bending and tail movement (Clarke et al. 1984).  Proprioceptive
feedback is critical for adaptive locomotion in zebrafish larvae
(Knafo & Bhatt 2022).

Pure numpy.
"""
import numpy as np


class ProprioceptiveSystem:
    """Body state sensing from movement feedback.

    Args:
        speed_noise: float — observation noise on speed
        turn_noise: float — observation noise on turn rate
        effort_decay: float — effort EMA decay rate
    """

    def __init__(self, speed_noise=0.05, turn_noise=0.03,
                 effort_decay=0.9):
        self.speed_noise = speed_noise
        self.turn_noise = turn_noise
        self.effort_decay = effort_decay

        # Observed state (noisy)
        self.observed_speed = 0.0
        self.observed_turn = 0.0

        # Commanded state (from brain)
        self.commanded_speed = 0.0
        self.commanded_turn = 0.0

        # Prediction errors
        self.speed_pe = 0.0  # commanded - observed
        self.turn_pe = 0.0

        # Effort level (cumulative exertion)
        self.effort = 0.0

        # Collision detection: large speed PE = unexpected deceleration
        self.collision_signal = 0.0

    def step(self, commanded_speed, commanded_turn,
             actual_speed, actual_turn):
        """Update proprioceptive state.

        Args:
            commanded_speed: float — brain's speed command
            commanded_turn: float — brain's turn command
            actual_speed: float — env's actual fish speed
            actual_turn: float — env's actual heading change

        Returns:
            diag: dict
        """
        self.commanded_speed = commanded_speed
        self.commanded_turn = commanded_turn

        # Noisy observation (sensory neuron response)
        self.observed_speed = actual_speed + np.random.normal(
            0, self.speed_noise)
        self.observed_turn = actual_turn + np.random.normal(
            0, self.turn_noise)

        # Prediction error: expected - observed
        self.speed_pe = commanded_speed - self.observed_speed
        self.turn_pe = commanded_turn - self.observed_turn

        # Effort: cumulative exertion (affects fatigue)
        self.effort = (self.effort_decay * self.effort
                       + (1 - self.effort_decay) * abs(actual_speed))

        # Collision detection: large positive speed PE
        # (commanded high speed but actual low = hit something)
        if self.speed_pe > 0.3 and commanded_speed > 0.4:
            self.collision_signal = min(1.0, self.speed_pe * 2.0)
        else:
            self.collision_signal *= 0.8  # decay

        return self.get_diagnostics()

    def get_motor_adjustment(self):
        """Suggest motor adjustment from proprioceptive error.

        Returns:
            speed_adj: float — multiplicative speed adjustment
            turn_adj: float — additive turn correction
        """
        # If we're going slower than commanded, boost
        if self.speed_pe > 0.1:
            speed_adj = 1.0 + min(0.2, self.speed_pe * 0.5)
        elif self.speed_pe < -0.1:
            speed_adj = 1.0 - min(0.2, abs(self.speed_pe) * 0.3)
        else:
            speed_adj = 1.0

        # Turn correction
        turn_adj = -0.1 * self.turn_pe

        return speed_adj, turn_adj

    def get_effort_fatigue_contribution(self):
        """Effort → fatigue rate multiplier.

        High effort = faster fatigue accumulation.

        Returns:
            fatigue_mult: float [1.0, 2.0]
        """
        return 1.0 + self.effort

    def reset(self):
        self.observed_speed = 0.0
        self.observed_turn = 0.0
        self.commanded_speed = 0.0
        self.commanded_turn = 0.0
        self.speed_pe = 0.0
        self.turn_pe = 0.0
        self.effort = 0.0
        self.collision_signal = 0.0

    def get_diagnostics(self):
        return {
            "speed_pe": self.speed_pe,
            "turn_pe": self.turn_pe,
            "effort": self.effort,
            "collision": self.collision_signal,
            "observed_speed": self.observed_speed,
        }
