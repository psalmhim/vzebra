"""
Spiking olfactory system: alarm substance + food odor.

Zebrafish olfaction:
  Olfactory epithelium → Olfactory bulb → Telencephalon (Dp, Dl)

Alarm substance (Schreckstoff): released by injured conspecifics,
triggers antipredator behavior (Speedie & Gerlai 2008).

Food odor: amino acid gradients guide foraging (Friedrich & Korsching 1997).

Architecture:
  10 alarm neurons (RS) — detect alarm substance concentration
  10 food-odor neurons (RS) — detect amino acid gradient
  Outputs: alarm level, food gradient direction, odor confidence
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingOlfaction(nn.Module):
    def __init__(self, n_alarm=10, n_food=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_alarm = n_alarm
        self.n_food = n_food

        self.alarm_pop = IzhikevichLayer(n_alarm, 'RS', device)
        self.food_pop = IzhikevichLayer(n_food, 'RS', device)
        self.alarm_pop.i_tonic.fill_(-2.0)
        self.food_pop.i_tonic.fill_(-2.0)

        self.register_buffer('alarm_rate', torch.zeros(n_alarm, device=device))
        self.register_buffer('food_rate', torch.zeros(n_food, device=device))

        self.alarm_level = 0.0
        self.food_gradient_dir = 0.0  # relative angle to strongest food odor
        self.food_odor_strength = 0.0
        self.odor_range = 200.0

    @torch.no_grad()
    def forward(self, fish_x: float, fish_y: float, fish_heading: float,
                foods: list, conspecific_injured: bool = False,
                pred_dist: float = 999.0) -> dict:
        # Alarm substance: nearby predator kill or injured conspecific
        alarm_drive = 0.0
        if conspecific_injured:
            alarm_drive = 0.8
        elif pred_dist < 100:
            alarm_drive = max(0.0, 0.3 * (1.0 - pred_dist / 100.0))

        I_alarm = torch.full((self.n_alarm,), alarm_drive * 15.0, device=self.device)

        # Food odor: find nearest food, compute gradient direction
        best_dist = self.odor_range
        best_angle = 0.0
        total_odor = 0.0
        for food in foods:
            fx, fy = food[0], food[1]
            dx = fx - fish_x
            dy = fy - fish_y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6
            if dist < self.odor_range:
                odor = max(0.0, 1.0 - dist / self.odor_range)
                total_odor += odor
                if dist < best_dist:
                    best_dist = dist
                    best_angle = math.atan2(dy, dx)

        if total_odor > 0.01:
            rel_angle = math.atan2(math.sin(best_angle - fish_heading),
                                   math.cos(best_angle - fish_heading))
            self.food_gradient_dir = rel_angle
            self.food_odor_strength = min(1.0, total_odor)
        else:
            self.food_gradient_dir = 0.0
            self.food_odor_strength = 0.0

        I_food = torch.full((self.n_food,), self.food_odor_strength * 12.0,
                            device=self.device)
        # Directional encoding: left-biased neurons for left odor, etc
        if self.food_gradient_dir > 0:  # food on left (CCW)
            I_food[:self.n_food // 2] *= 1.5
        else:
            I_food[self.n_food // 2:] *= 1.5

        for _ in range(10):  # reduced substeps
            self.alarm_pop(I_alarm + torch.randn(self.n_alarm, device=self.device) * 0.3)
            self.food_pop(I_food + torch.randn(self.n_food, device=self.device) * 0.3)

        self.alarm_rate.copy_(self.alarm_pop.rate)
        self.food_rate.copy_(self.food_pop.rate)
        self.alarm_level = float(self.alarm_rate.mean()) * 5.0

        return {
            'alarm_level': self.alarm_level,
            'food_gradient_dir': self.food_gradient_dir,
            'food_odor_strength': self.food_odor_strength,
            'alarm_rate': float(self.alarm_rate.mean()),
            'food_odor_rate': float(self.food_rate.mean()),
        }

    def get_forage_bias(self) -> float:
        """Olfactory bias toward food: negative = attract."""
        return -self.food_odor_strength * 0.2

    def get_flee_bias(self) -> float:
        """Alarm substance drives flee."""
        return -self.alarm_level * 0.3

    def reset(self):
        self.alarm_pop.reset()
        self.food_pop.reset()
        self.alarm_rate.zero_()
        self.food_rate.zero_()
        self.alarm_level = 0.0
        self.food_gradient_dir = 0.0
        self.food_odor_strength = 0.0
