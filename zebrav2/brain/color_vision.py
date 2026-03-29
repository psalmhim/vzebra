"""
Spiking color vision: UV/Blue/Green/Red cone channels.

Zebrafish have 4 cone types (UV, short/blue, medium/green, long/red)
plus rods for scotopic vision. Tetrachromatic color processing enables
object classification by spectral signature.

Architecture:
  4 × 8 RS neurons (one population per cone type)
  Color opponent channels: R-G, B-Y (UV+B vs G+R)
  Object spectral signatures:
    Food (green algae): high G, moderate R
    Predator: high R (warm body), low UV
    Conspecific: UV-reflective (zebrafish stripes)
    Rock: broadband, low UV
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingColorVision(nn.Module):
    def __init__(self, n_per_channel=8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_ch = n_per_channel

        # 4 cone populations
        self.uv_pop = IzhikevichLayer(n_per_channel, 'RS', device)
        self.blue_pop = IzhikevichLayer(n_per_channel, 'RS', device)
        self.green_pop = IzhikevichLayer(n_per_channel, 'RS', device)
        self.red_pop = IzhikevichLayer(n_per_channel, 'RS', device)
        for pop in [self.uv_pop, self.blue_pop, self.green_pop, self.red_pop]:
            pop.i_tonic.fill_(-2.0)

        self.register_buffer('uv_rate', torch.zeros(n_per_channel, device=device))
        self.register_buffer('blue_rate', torch.zeros(n_per_channel, device=device))
        self.register_buffer('green_rate', torch.zeros(n_per_channel, device=device))
        self.register_buffer('red_rate', torch.zeros(n_per_channel, device=device))

        # Spectral signature templates
        # [UV, B, G, R] normalized
        self.signatures = {
            'food': [0.1, 0.2, 0.8, 0.4],
            'enemy': [0.05, 0.1, 0.3, 0.7],
            'conspecific': [0.8, 0.5, 0.3, 0.2],
            'rock': [0.2, 0.3, 0.3, 0.3],
        }

    @torch.no_grad()
    def forward(self, retinal_type_L: torch.Tensor,
                retinal_type_R: torch.Tensor) -> dict:
        """
        retinal_type_L/R: (400,) type channel from retina (0-1 entity type values)
        Derive approximate spectral content from entity types.
        """
        # Map type values to spectral content
        type_all = torch.cat([retinal_type_L, retinal_type_R])
        food_px = (type_all > 0.7).float().sum()
        enemy_px = ((type_all - 0.5).abs() < 0.1).float().sum()
        rock_px = ((type_all - 0.75).abs() < 0.1).float().sum()
        conspe_px = ((type_all - 0.25).abs() < 0.1).float().sum()
        total = food_px + enemy_px + rock_px + conspe_px + 1e-8

        # Weighted spectral sum
        uv_drive = (food_px * 0.1 + enemy_px * 0.05 + conspe_px * 0.8 + rock_px * 0.2) / total
        blue_drive = (food_px * 0.2 + enemy_px * 0.1 + conspe_px * 0.5 + rock_px * 0.3) / total
        green_drive = (food_px * 0.8 + enemy_px * 0.3 + conspe_px * 0.3 + rock_px * 0.3) / total
        red_drive = (food_px * 0.4 + enemy_px * 0.7 + conspe_px * 0.2 + rock_px * 0.3) / total

        I_uv = torch.full((self.n_ch,), float(uv_drive) * 10.0, device=self.device)
        I_b = torch.full((self.n_ch,), float(blue_drive) * 10.0, device=self.device)
        I_g = torch.full((self.n_ch,), float(green_drive) * 10.0, device=self.device)
        I_r = torch.full((self.n_ch,), float(red_drive) * 10.0, device=self.device)

        for _ in range(SUBSTEPS):
            self.uv_pop(I_uv + torch.randn(self.n_ch, device=self.device) * 0.3)
            self.blue_pop(I_b + torch.randn(self.n_ch, device=self.device) * 0.3)
            self.green_pop(I_g + torch.randn(self.n_ch, device=self.device) * 0.3)
            self.red_pop(I_r + torch.randn(self.n_ch, device=self.device) * 0.3)

        self.uv_rate.copy_(self.uv_pop.rate)
        self.blue_rate.copy_(self.blue_pop.rate)
        self.green_rate.copy_(self.green_pop.rate)
        self.red_rate.copy_(self.red_pop.rate)

        # Color opponent channels
        rg_opponent = float(self.red_rate.mean() - self.green_rate.mean())
        by_opponent = float((self.uv_rate.mean() + self.blue_rate.mean()) / 2
                            - (self.green_rate.mean() + self.red_rate.mean()) / 2)

        return {
            'uv': float(self.uv_rate.mean()),
            'blue': float(self.blue_rate.mean()),
            'green': float(self.green_rate.mean()),
            'red': float(self.red_rate.mean()),
            'rg_opponent': rg_opponent,
            'by_opponent': by_opponent,
        }

    def reset(self):
        for pop in [self.uv_pop, self.blue_pop, self.green_pop, self.red_pop]:
            pop.reset()
        self.uv_rate.zero_()
        self.blue_rate.zero_()
        self.green_rate.zero_()
        self.red_rate.zero_()
