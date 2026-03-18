"""
Basal Ganglia motor gating for v1.

Implements action selection via switching dynamics driven by
RPE and dopamine. Gates approach vs avoidance behavior.
Two modes: alternating (sinusoidal saccades) and exploratory (RPE-driven).
Optimized: DA-modulated saccade period, stronger exploration, bounded momentum.
"""
import math
import torch


class BasalGanglia:
    def __init__(self, mode="alternating"):
        self.mode = mode
        self.state = 0.0
        self.momentum = 0.0
        self.leak = 0.10         # reduced from 0.15 for more persistent motor programs
        self.k_gate = 0.8
        self.max_state = 0.6
        self.noise = 0.03
        self.alt_timer = 0
        self.alt_base_period = 70

    def step(self, valL, valR, dopa, rpe):
        """
        Compute basal ganglia gating signal.

        Returns:
            gate: gating signal in [-1, 1], smoothly modulated
        """
        # Basic drive: bilateral asymmetry + dopamine modulation
        drive = (valR - valL) + 0.3 * (dopa - 0.5)

        if self.mode == "alternating":
            self.alt_timer += 1
            # DA-modulated period: high DA → shorter period (faster saccades)
            period = self.alt_base_period * (1.3 - 0.3 * dopa)
            phase = 2.0 * math.pi * self.alt_timer / (2.0 * period)
            alt_signal = 0.3 * math.sin(phase)
            drive += alt_signal
        else:
            # Exploratory: RPE-triggered switches (stronger exploration)
            switch_prob = float(torch.sigmoid(torch.tensor(abs(rpe) * 5.0)))
            if torch.rand(1).item() < switch_prob * 0.12:  # was 0.05
                self.momentum *= -(0.3 + 0.3 * min(abs(rpe), 1.0))  # RPE-scaled reversal

        # Smooth momentum integration (faster response)
        self.momentum = 0.65 * self.momentum + 0.35 * drive  # was 0.8/0.2

        # Bound momentum to prevent runaway
        self.momentum = max(-0.8, min(0.8, self.momentum))

        # Apply momentum to state with damping
        self.state += 0.3 * self.momentum

        # Dopamine-gated leak: high DA reduces leak
        effective_leak = self.leak * (1.0 + 0.5 * max(0, 0.5 - dopa))
        self.state -= effective_leak * self.state

        # Exploration noise
        self.state += self.noise * float(torch.randn(1).item())

        # Saturation
        self.state = max(-self.max_state, min(self.max_state, self.state))

        # Smooth gating output
        return math.tanh(self.k_gate * self.state)

    def reset(self):
        self.state = 0.0
        self.momentum = 0.0
        self.alt_timer = 0
