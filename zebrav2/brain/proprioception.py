"""
Spiking proprioception: body position and collision detection.

Encodes fish's own body state: position, speed, heading, wall proximity.
Collision detection triggers a flash response.

Architecture:
  8 RS neurons: 2 speed, 2 heading, 2 wall proximity, 2 collision
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingProprioception(nn.Module):
    def __init__(self, n_neurons=8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))
        self._prev_x = 400.0
        self._prev_y = 300.0
        self.collision = False
        self.wall_proximity = 0.0

    @torch.no_grad()
    def forward(self, fish_x: float, fish_y: float, speed: float,
                heading: float, arena_w: int = 800, arena_h: int = 600) -> dict:
        # Speed encoding
        dx = fish_x - self._prev_x
        dy = fish_y - self._prev_y
        actual_speed = math.sqrt(dx*dx + dy*dy)
        self._prev_x, self._prev_y = fish_x, fish_y

        # Wall proximity
        margin = 50
        self.wall_proximity = max(0.0,
            max((margin - fish_x) / margin if fish_x < margin else 0.0,
                (fish_x - (arena_w - margin)) / margin if fish_x > arena_w - margin else 0.0,
                (margin - fish_y) / margin if fish_y < margin else 0.0,
                (fish_y - (arena_h - margin)) / margin if fish_y > arena_h - margin else 0.0))

        # Collision: speed expected but no movement (hit wall/rock)
        self.collision = actual_speed < 1.0 and speed > 0.5

        I = torch.zeros(self.n, device=self.device)
        I[0] = actual_speed * 5.0           # speed +
        I[1] = max(0, 3.0 - actual_speed) * 3.0  # speed - (deceleration)
        I[2] = (math.cos(heading) + 1) * 3.0  # heading east
        I[3] = (math.sin(heading) + 1) * 3.0  # heading north
        I[4] = self.wall_proximity * 12.0    # wall close L
        I[5] = self.wall_proximity * 12.0    # wall close R
        I[6] = 15.0 if self.collision else 0.0  # collision L
        I[7] = 15.0 if self.collision else 0.0  # collision R

        for _ in range(SUBSTEPS):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        return {
            'actual_speed': actual_speed,
            'wall_proximity': self.wall_proximity,
            'collision': self.collision,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self._prev_x = 400.0
        self._prev_y = 300.0
        self.collision = False
        self.wall_proximity = 0.0
