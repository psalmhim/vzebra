"""
Spiking vestibular system: tilt and acceleration sensing.

Zebrafish inner ear: utricular otolith detects gravity/linear acceleration,
semicircular canals detect angular velocity.

Free Energy Principle:
  Generative model predicts expected vestibular input from motor commands.
  Prediction error drives postural correction (vestibulo-spinal reflex).
  Precision: high during active movement (efference copy available),
  low during passive perturbation (unexpected motion → high PE → correction).

Architecture:
  6 RS neurons: 2 pitch, 2 roll, 2 yaw (angular velocity)
  + prediction error computed from expected vs actual vestibular signal
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingVestibular(nn.Module):
    def __init__(self, n_neurons=6, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))
        self._prev_heading = 0.0
        self.angular_velocity = 0.0
        self.tilt = 0.0

        # FEP: two-compartment vestibular prediction (Lee et al. 2026)
        # 2 channels: turn, speed
        self.pc = TwoCompColumn(n_channels=2, n_per_ch=4, substeps=8, device=device)
        self._predicted_turn = 0.0
        self._predicted_speed = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, heading: float, speed: float, turn_rate: float,
                predicted_turn: float = None, predicted_speed: float = None) -> dict:
        """
        heading: current fish heading (rad)
        speed: current speed
        turn_rate: current turn rate (from motor command)
        predicted_turn: brain's predicted turn (efference copy, optional)
        predicted_speed: brain's predicted speed (optional)
        """
        self.angular_velocity = turn_rate
        self.tilt = min(1.0, abs(turn_rate) * speed * 0.5)  # centripetal tilt

        # --- Spiking encoding of actual vestibular input ---
        I = torch.zeros(self.n, device=self.device)
        I[0] = max(0, turn_rate) * 10.0   # yaw right
        I[1] = max(0, -turn_rate) * 10.0  # yaw left
        I[2] = speed * 5.0                # forward acceleration
        I[3] = max(0, -speed + 0.5) * 5.0 # deceleration
        I[4] = self.tilt * 8.0            # roll right
        I[5] = self.tilt * 8.0            # roll left
        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP: two-compartment vestibular prediction (Lee et al. 2026) ---
        if predicted_turn is not None:
            self._predicted_turn = predicted_turn
        if predicted_speed is not None:
            self._predicted_speed = predicted_speed

        sensory = torch.tensor([turn_rate, speed], device=self.device)
        prediction = torch.tensor([self._predicted_turn, self._predicted_speed],
                                   device=self.device)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(0.7 * pe[0] + 0.3 * pe[1])
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        # Postural correction: precision-weighted PE
        postural = -self.precision * self.prediction_error * 0.3

        return {
            'angular_velocity': self.angular_velocity,
            'tilt': self.tilt,
            'rate': float(self.rate.mean()),
            'postural_correction': postural,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self._prev_heading = 0.0
        self.angular_velocity = 0.0
        self.tilt = 0.0
        self.pc.reset()
        self._predicted_turn = 0.0
        self._predicted_speed = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
