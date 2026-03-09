"""
Dopamine system for v60: RPE + valence computation.

Computes reward prediction error (RPE) from visual free energy,
and bilateral valence signals that modulate precision and motor gating.
Now includes direct food reward and optimized parameters.
"""
import math
import torch


class DopamineSystem_v60:
    def __init__(self, alpha=0.05, beta=2.0, decay=0.98, noise=0.01):
        self.alpha = alpha       # value learning rate
        self.beta = beta         # sigmoid gain (was 3.0, now 2.0 for better sensitivity)
        self.decay = decay       # value decay (was 0.995, now 0.98 for faster adaptation)
        self.noise = noise       # dopamine noise (was 0.02, now 0.01)

        # Internal state
        self.expected_value = 0.0
        self.rpe = 0.0
        self.dopa = 0.5          # dopamine level [0, 1]
        self.valL = 0.0          # left valence
        self.valR = 0.0          # right valence
        self.val_decay = 0.02

    def step(self, F_visual, oL_mean=0.0, oR_mean=0.0, eaten=0):
        """
        Update dopamine system given current visual free energy and
        bilateral OT activations.

        Args:
            F_visual: scalar free energy (lower = better tracking)
            oL_mean: mean left OT activation
            oR_mean: mean right OT activation
            eaten: number of food items eaten this step (direct reward)

        Returns:
            dopa: dopamine level [0, 1]
            rpe: reward prediction error
            valL: left valence
            valR: right valence
        """
        # Pseudo-reward: high when free energy is low + food capture bonus
        reward = float(math.exp(-abs(F_visual)))
        if eaten > 0:
            reward += 0.5 * eaten  # direct food reward boost

        # RPE = reward - expected
        self.rpe = reward - self.expected_value

        # Value learning
        self.expected_value += self.alpha * self.rpe
        self.expected_value *= self.decay

        # Dopamine = sigmoid(beta * RPE)
        x = self.beta * self.rpe
        if abs(x) < 60:
            self.dopa = 1.0 / (1.0 + math.exp(-x))
        else:
            self.dopa = 1.0 if x > 0 else 0.0
        self.dopa += self.noise * float(torch.randn(1).item())
        self.dopa = max(0.0, min(1.0, self.dopa))

        # Bilateral valence from OT activation asymmetry
        rpe_L = oL_mean - self.expected_value * 0.5
        rpe_R = oR_mean - self.expected_value * 0.5
        self.valL = (1 - self.val_decay) * self.valL + 0.1 * (rpe_L - 0.5 * rpe_R)
        self.valR = (1 - self.val_decay) * self.valR + 0.1 * (rpe_R - 0.5 * rpe_L)

        # Mean-subtract to maintain bilateral balance
        mean_val = 0.5 * (self.valL + self.valR)
        self.valL -= mean_val
        self.valR -= mean_val
        self.valL = max(-0.5, min(0.5, self.valL))
        self.valR = max(-0.5, min(0.5, self.valR))

        return self.dopa, self.rpe, self.valL, self.valR

    def modulate_precision(self, pi_value, kappa=0.3):
        """Dopamine-modulated precision: Pi *= (1 + kappa * (dopa - 0.5))"""
        return pi_value * (1.0 + kappa * (self.dopa - 0.5))

    def reset(self):
        self.expected_value = 0.0
        self.rpe = 0.0
        self.dopa = 0.5
        self.valL = 0.0
        self.valR = 0.0
