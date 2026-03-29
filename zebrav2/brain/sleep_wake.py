"""
Spiking sleep/wake system: memory consolidation.

Zebrafish show sleep-like quiescent states with reduced responsiveness
(Zhdanova et al. 2001). During sleep, VAE/STDP consolidation replays
recent experience.

Architecture:
  4 RS neurons: 2 wake-promoting (orexin), 2 sleep-promoting (VLPO analogue)
  Flip-flop bistable switch (Saper et al. 2005)
  Sleep consolidation: replay VAE buffer + STDP homeostatic rescaling
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingSleepWake(nn.Module):
    def __init__(self, n_wake=2, n_sleep=2, device=DEVICE):
        super().__init__()
        self.device = device
        self.wake_pop = IzhikevichLayer(n_wake, 'RS', device)
        self.sleep_pop = IzhikevichLayer(n_sleep, 'RS', device)
        self.wake_pop.i_tonic.fill_(3.0)   # default awake
        self.sleep_pop.i_tonic.fill_(-2.0)

        self.register_buffer('wake_rate', torch.zeros(n_wake, device=device))
        self.register_buffer('sleep_rate', torch.zeros(n_sleep, device=device))

        self.is_sleeping = False
        self.sleep_pressure = 0.0  # accumulates with time awake
        self.sleep_debt = 0.0
        self.consolidation_done = False
        self._awake_steps = 0
        self.n_wake = n_wake
        self.n_sleep = n_sleep

    @torch.no_grad()
    def forward(self, circadian_melatonin: float = 0.0,
                arousal: float = 0.5, threat: float = 0.0) -> dict:
        self._awake_steps += 1

        # Sleep pressure accumulates (process S)
        if not self.is_sleeping:
            self.sleep_pressure += 0.001
            self.sleep_pressure = min(1.0, self.sleep_pressure)
        else:
            self.sleep_pressure -= 0.01
            self.sleep_pressure = max(0.0, self.sleep_pressure)

        # Drive: melatonin + sleep pressure promote sleep; arousal/threat promote wake
        sleep_drive = (circadian_melatonin * 8.0 + self.sleep_pressure * 6.0
                       - arousal * 5.0 - threat * 10.0)
        wake_drive = (arousal * 5.0 + threat * 10.0 + (1.0 - circadian_melatonin) * 4.0
                      - self.sleep_pressure * 3.0)

        I_wake = torch.full((self.n_wake,), max(0, wake_drive), device=self.device)
        I_sleep = torch.full((self.n_sleep,), max(0, sleep_drive), device=self.device)

        for _ in range(SUBSTEPS):
            # Flip-flop mutual inhibition
            self.wake_pop(I_wake - float(self.sleep_rate.mean()) * 8.0
                          + torch.randn(self.n_wake, device=self.device) * 0.3)
            self.sleep_pop(I_sleep - float(self.wake_rate.mean()) * 8.0
                           + torch.randn(self.n_sleep, device=self.device) * 0.3)

        self.wake_rate.copy_(self.wake_pop.rate)
        self.sleep_rate.copy_(self.sleep_pop.rate)

        wake_mean = float(self.wake_rate.mean())
        sleep_mean = float(self.sleep_rate.mean())
        self.is_sleeping = sleep_mean > wake_mean and threat < 0.2

        # Consolidation flag: true during first sleep bout
        if self.is_sleeping and not self.consolidation_done:
            self.consolidation_done = True

        return {
            'is_sleeping': self.is_sleeping,
            'sleep_pressure': self.sleep_pressure,
            'wake_rate': wake_mean,
            'sleep_rate': sleep_mean,
            'consolidation_active': self.is_sleeping,
            'responsiveness': 0.3 if self.is_sleeping else 1.0,
        }

    def reset(self):
        self.wake_pop.reset()
        self.sleep_pop.reset()
        self.wake_rate.zero_()
        self.sleep_rate.zero_()
        self.is_sleeping = False
        self.sleep_pressure = 0.0
        self.consolidation_done = False
        self._awake_steps = 0
