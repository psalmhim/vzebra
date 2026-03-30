"""
Spiking lateral line: mechanoreceptive proximity sense.

Zebrafish lateral line neuromasts detect water displacement from nearby
moving objects. Anterior (head) and posterior (trunk/tail) lines provide
directional flow information.

Architecture:
  10 anterior neuromasts (RS) — detect frontal/lateral flow
  10 posterior neuromasts (RS) — detect caudal flow
  Outputs: proximity estimate, approach direction, flow magnitude
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingLateralLine(nn.Module):
    def __init__(self, n_anterior=10, n_posterior=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_ant = n_anterior
        self.n_post = n_posterior

        self.anterior = IzhikevichLayer(n_anterior, 'RS', device)
        self.posterior = IzhikevichLayer(n_posterior, 'RS', device)
        self.anterior.i_tonic.fill_(-2.0)
        self.posterior.i_tonic.fill_(-2.0)

        self.register_buffer('ant_rate', torch.zeros(n_anterior, device=device))
        self.register_buffer('post_rate', torch.zeros(n_posterior, device=device))

        self.proximity = 0.0
        self.flow_direction = 0.0  # -1 left, +1 right
        self.flow_magnitude = 0.0
        self.detection_range = 150.0  # pixels

    @torch.no_grad()
    def forward(self, fish_x: float, fish_y: float, fish_heading: float,
                pred_x: float, pred_y: float,
                pred_vx: float = 0.0, pred_vy: float = 0.0) -> dict:
        dx = pred_x - fish_x
        dy = pred_y - fish_y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6

        if dist > self.detection_range:
            # Beyond range — minimal activity
            I_ant = torch.zeros(self.n_ant, device=self.device)
            I_post = torch.zeros(self.n_post, device=self.device)
            self.proximity = 0.0
            self.flow_direction = 0.0
            self.flow_magnitude = 0.0
        else:
            self.proximity = max(0.0, 1.0 - dist / self.detection_range)
            # Bearing relative to fish heading
            angle_to = math.atan2(dy, dx)
            rel_angle = math.atan2(math.sin(angle_to - fish_heading),
                                   math.cos(angle_to - fish_heading))
            # Flow from predator movement
            speed = math.sqrt(pred_vx**2 + pred_vy**2)
            self.flow_magnitude = min(1.0, speed * self.proximity * 0.1)
            self.flow_direction = math.sin(rel_angle)  # left/right

            # Anterior: stronger for frontal approach
            frontal = max(0.0, math.cos(rel_angle))
            I_ant = torch.full((self.n_ant,), self.proximity * frontal * 15.0,
                               device=self.device)
            # Posterior: stronger for caudal approach
            caudal = max(0.0, -math.cos(rel_angle))
            I_post = torch.full((self.n_post,), self.proximity * caudal * 12.0,
                                device=self.device)
            # Lateral asymmetry
            if self.flow_direction > 0:  # right side
                I_ant[:self.n_ant // 2] *= 1.5
            else:
                I_ant[self.n_ant // 2:] *= 1.5

        for _ in range(10):  # reduced substeps
            self.anterior(I_ant + torch.randn(self.n_ant, device=self.device) * 0.5)
            self.posterior(I_post + torch.randn(self.n_post, device=self.device) * 0.5)

        self.ant_rate.copy_(self.anterior.rate)
        self.post_rate.copy_(self.posterior.rate)

        return {
            'proximity': self.proximity,
            'flow_direction': self.flow_direction,
            'flow_magnitude': self.flow_magnitude,
            'ant_rate': float(self.ant_rate.mean()),
            'post_rate': float(self.post_rate.mean()),
            'dist': dist if dist < self.detection_range else 999.0,
        }

    def reset(self):
        self.anterior.reset()
        self.posterior.reset()
        self.ant_rate.zero_()
        self.post_rate.zero_()
        self.proximity = 0.0
        self.flow_direction = 0.0
        self.flow_magnitude = 0.0
