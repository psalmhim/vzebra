"""
Spiking proprioception: body position and collision detection.

Free Energy Principle:
  Proprioceptive prediction error is the CORE of active inference motor control
  (Friston 2011, "What is optimal about motor control?").

  Motor commands don't directly move the body — they set proprioceptive
  PREDICTIONS. The spinal reflex arc then acts to fulfill those predictions
  by minimizing proprioceptive PE.

  PE = predicted_body_state - actual_body_state
  Motor output = reflex_gain * PE  (classical reflex arc)

Architecture:
  8 RS neurons: 2 speed, 2 heading, 2 wall proximity, 2 collision
  + prediction error per channel (speed, heading, wall, collision)
  + precision per channel (modulated by context)
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


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

        # FEP: two-compartment proprioceptive prediction (Lee et al. 2026)
        # 4 channels: speed, heading, wall, collision
        self.pc = TwoCompColumn(n_channels=4, n_per_ch=4, substeps=8, device=device)
        self.pc.set_precision_channel(2, 0.5)   # wall: γ=0.5 → π=0.62
        self.pc.set_precision_channel(3, 1.0)   # collision: γ=1.0 → π=0.73

        self._predicted_speed = 0.0
        self._predicted_heading_delta = 0.0
        self._predicted_wall = 0.0

        self.speed_pe = 0.0
        self.heading_pe = 0.0
        self.wall_pe = 0.0
        self.collision_pe = 0.0
        self.free_energy = 0.0
        self._prev_heading = 0.0

    def set_predictions(self, predicted_speed: float = None,
                        predicted_heading_delta: float = None):
        """Set descending predictions from motor cortex (active inference)."""
        if predicted_speed is not None:
            self._predicted_speed = predicted_speed
        if predicted_heading_delta is not None:
            self._predicted_heading_delta = predicted_heading_delta

    @torch.no_grad()
    def forward(self, fish_x: float, fish_y: float, speed: float,
                heading: float, arena_w: int = 800, arena_h: int = 600) -> dict:
        # Speed encoding
        dx = fish_x - self._prev_x
        dy = fish_y - self._prev_y
        actual_speed = math.sqrt(dx*dx + dy*dy)
        self._prev_x, self._prev_y = fish_x, fish_y

        # Heading delta
        heading_delta = heading - self._prev_heading
        heading_delta = math.atan2(math.sin(heading_delta), math.cos(heading_delta))
        self._prev_heading = heading

        # Wall proximity
        margin = 50
        self.wall_proximity = max(0.0,
            max((margin - fish_x) / margin if fish_x < margin else 0.0,
                (fish_x - (arena_w - margin)) / margin if fish_x > arena_w - margin else 0.0,
                (margin - fish_y) / margin if fish_y < margin else 0.0,
                (fish_y - (arena_h - margin)) / margin if fish_y > arena_h - margin else 0.0))

        # Collision: speed expected but no movement (hit wall/rock)
        self.collision = actual_speed < 1.0 and speed > 0.5

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        I[0] = actual_speed * 5.0
        I[1] = max(0, 3.0 - actual_speed) * 3.0
        I[2] = (math.cos(heading) + 1) * 3.0
        I[3] = (math.sin(heading) + 1) * 3.0
        I[4] = self.wall_proximity * 12.0
        I[5] = self.wall_proximity * 12.0
        I[6] = 15.0 if self.collision else 0.0
        I[7] = 15.0 if self.collision else 0.0

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP: two-compartment proprioceptive prediction (Lee et al. 2026) ---
        norm_speed = min(1.0, actual_speed / 4.0)
        coll_val = 1.0 if self.collision else 0.0
        sensory = torch.tensor([norm_speed, heading_delta,
                                self.wall_proximity, coll_val],
                               device=self.device)
        prediction = torch.tensor([self._predicted_speed,
                                   self._predicted_heading_delta,
                                   self._predicted_wall, 0.0],
                                  device=self.device)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.speed_pe = float(pe[0])
        self.heading_pe = float(pe[1])
        self.wall_pe = float(pe[2])
        self.collision_pe = float(pe[3])
        self.free_energy = pc_out['free_energy']

        return {
            'actual_speed': actual_speed,
            'heading_delta': heading_delta,
            'wall_proximity': self.wall_proximity,
            'collision': self.collision,
            'rate': float(self.rate.mean()),
            # FEP outputs
            'speed_pe': self.speed_pe,
            'heading_pe': self.heading_pe,
            'wall_pe': self.wall_pe,
            'collision_pe': self.collision_pe,
            'free_energy': self.free_energy,
            'precision': pc_out['precision'].detach().cpu().tolist(),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self._prev_x = 400.0
        self._prev_y = 300.0
        self._prev_heading = 0.0
        self.collision = False
        self.wall_proximity = 0.0
        self.pc.reset()
        self._predicted_speed = 0.0
        self._predicted_heading_delta = 0.0
        self._predicted_wall = 0.0
        self.speed_pe = 0.0
        self.heading_pe = 0.0
        self.wall_pe = 0.0
        self.collision_pe = 0.0
        self.free_energy = 0.0
