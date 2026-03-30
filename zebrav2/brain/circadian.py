"""
Spiking circadian clock: day/night cycle.

Zebrafish pineal gland contains intrinsic circadian oscillator
(Cahill 1996). Melatonin peaks at night, suppressing activity.

Architecture:
  6 RS neurons: 3 "day" neurons, 3 "night" neurons
  Mutual inhibition creates bistable day/night state
  External zeitgeber: light level from retina
  Outputs: circadian phase, melatonin level, activity drive
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingCircadian(nn.Module):
    def __init__(self, n_day=3, n_night=3, cycle_period=2400, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_day = n_day
        self.n_night = n_night
        self.cycle_period = cycle_period  # steps per full day/night cycle

        self.day_pop = IzhikevichLayer(n_day, 'RS', device)
        self.night_pop = IzhikevichLayer(n_night, 'RS', device)
        self.day_pop.i_tonic.fill_(2.0)  # day-active by default
        self.night_pop.i_tonic.fill_(-2.0)

        self.register_buffer('day_rate', torch.zeros(n_day, device=device))
        self.register_buffer('night_rate', torch.zeros(n_night, device=device))

        self._step = 0
        self.phase = 0.0  # 0=dawn, 0.5=dusk, 1.0=next dawn
        self.melatonin = 0.0  # 0=day, 1=night peak
        self.activity_drive = 1.0  # 1.0=full activity, 0.3=sleep pressure

    @torch.no_grad()
    def forward(self, light_level: float = 0.7) -> dict:
        self._step += 1
        self.phase = (self._step % self.cycle_period) / self.cycle_period

        # Sinusoidal drive: day neurons peak at phase=0.25, night at 0.75
        day_drive = max(0, math.cos(2 * math.pi * (self.phase - 0.25))) * 8.0
        night_drive = max(0, math.cos(2 * math.pi * (self.phase - 0.75))) * 8.0

        # Light entrainment: light boosts day, suppresses night
        day_drive += light_level * 3.0
        night_drive -= light_level * 2.0

        I_day = torch.full((self.n_day,), day_drive, device=self.device)
        I_night = torch.full((self.n_night,), max(0, night_drive), device=self.device)

        for _ in range(10):  # reduced substeps (6 neurons don't need 50ms)
            self.day_pop(I_day - float(self.night_rate.mean()) * 5.0
                         + torch.randn(self.n_day, device=self.device) * 0.3)
            self.night_pop(I_night - float(self.day_rate.mean()) * 5.0
                           + torch.randn(self.n_night, device=self.device) * 0.3)

        self.day_rate.copy_(self.day_pop.rate)
        self.night_rate.copy_(self.night_pop.rate)

        day_mean = float(self.day_rate.mean())
        night_mean = float(self.night_rate.mean())
        self.melatonin = night_mean / (day_mean + night_mean + 1e-8)
        self.activity_drive = 0.3 + 0.7 * (1.0 - self.melatonin)

        return {
            'phase': self.phase,
            'melatonin': self.melatonin,
            'activity_drive': self.activity_drive,
            'day_rate': day_mean,
            'night_rate': night_mean,
            'is_night': self.phase > 0.5,
        }

    def reset(self):
        self.day_pop.reset()
        self.night_pop.reset()
        self.day_rate.zero_()
        self.night_rate.zero_()
        self._step = 0
        self.phase = 0.0
        self.melatonin = 0.0
        self.activity_drive = 1.0
