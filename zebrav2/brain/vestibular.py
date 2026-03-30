"""
Spiking vestibular system: tilt and acceleration sensing.

Zebrafish inner ear: utricular otolith detects gravity/linear acceleration,
semicircular canals detect angular velocity.

Architecture:
  6 RS neurons: 2 pitch, 2 roll, 2 yaw (angular velocity)
  Outputs: tilt angle, angular velocity, postural correction signal
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


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

    @torch.no_grad()
    def forward(self, heading: float, speed: float, turn_rate: float) -> dict:
        self.angular_velocity = turn_rate
        self.tilt = min(1.0, abs(turn_rate) * speed * 0.5)  # centripetal tilt
        I = torch.zeros(self.n, device=self.device)
        I[0] = max(0, turn_rate) * 10.0   # yaw right
        I[1] = max(0, -turn_rate) * 10.0  # yaw left
        I[2] = speed * 5.0                # forward acceleration
        I[3] = max(0, -speed + 0.5) * 5.0 # deceleration
        I[4] = self.tilt * 8.0            # roll right
        I[5] = self.tilt * 8.0            # roll left
        for _ in range(10):  # reduced substeps
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)
        return {
            'angular_velocity': self.angular_velocity,
            'tilt': self.tilt,
            'rate': float(self.rate.mean()),
            'postural_correction': -turn_rate * 0.1,  # compensatory signal
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self._prev_heading = 0.0
        self.angular_velocity = 0.0
        self.tilt = 0.0
