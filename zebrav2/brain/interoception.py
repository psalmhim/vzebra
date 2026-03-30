"""
Spiking insular cortex: interoceptive awareness.

Zebrafish homologue: dorsal pallium area Dp (insula-like function).

Three interoceptive channels encoded as spiking populations:
  Hunger neurons (10): fire rate proportional to energy deficit
  Fatigue neurons (10): fire rate proportional to accumulated effort
  Stress neurons (10): fire rate proportional to threat exposure

Output: emotional valence, heart rate modulation, arousal signal.

Heart rate model:
  Baseline ~2 Hz (zebrafish), increases with stress/effort.
  Bradycardia (freezing) during acute threat.
"""
import torch
import torch.nn as nn
import math
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingInsularCortex(nn.Module):
    def __init__(self, n_per_channel=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_ch = n_per_channel

        # Three interoceptive channels
        self.hunger_pop = IzhikevichLayer(n_per_channel, 'RS', device)
        self.fatigue_pop = IzhikevichLayer(n_per_channel, 'RS', device)
        self.stress_pop = IzhikevichLayer(n_per_channel, 'RS', device)

        # Subthreshold: need drive to fire
        self.hunger_pop.i_tonic.fill_(-3.0)
        self.fatigue_pop.i_tonic.fill_(-3.0)
        self.stress_pop.i_tonic.fill_(-3.0)

        # Valence integration neuron (4 RS — easier to activate)
        self.valence_pop = IzhikevichLayer(4, 'RS', device)
        self.valence_pop.i_tonic.fill_(0.0)

        # State
        self.register_buffer('hunger_rate', torch.zeros(n_per_channel, device=device))
        self.register_buffer('fatigue_rate', torch.zeros(n_per_channel, device=device))
        self.register_buffer('stress_rate', torch.zeros(n_per_channel, device=device))
        self.register_buffer('valence_rate', torch.zeros(4, device=device))

        # Heart rate model
        self.heart_rate = 2.0       # Hz (zebrafish baseline)
        self.heart_phase = 0.0      # oscillation phase
        self.heartbeat = False      # current beat
        self._step_count = 0

        # Emotional valence (-1 negative, +1 positive)
        self.valence = 0.0
        self.arousal = 0.0

    @torch.no_grad()
    def forward(self, energy: float, stress: float, fatigue: float,
                reward: float = 0.0, threat_acute: bool = False) -> dict:
        """
        Update interoceptive state.
        energy: 0-100
        stress, fatigue: 0-1
        reward: current reward signal
        threat_acute: True during active predator attack
        """
        self._step_count += 1

        # Interoceptive drives (proportional to deficit/excess)
        hunger_drive = max(0.0, 1.0 - energy / 100.0) * 12.0
        fatigue_drive = fatigue * 10.0
        stress_drive = stress * 12.0

        # Run spiking populations
        h_spikes = torch.zeros(self.n_ch, device=self.device)
        f_spikes = torch.zeros(self.n_ch, device=self.device)
        s_spikes = torch.zeros(self.n_ch, device=self.device)

        I_h = torch.full((self.n_ch,), hunger_drive, device=self.device)
        I_f = torch.full((self.n_ch,), fatigue_drive, device=self.device)
        I_s = torch.full((self.n_ch,), stress_drive, device=self.device)

        for _ in range(15):  # reduced substeps
            sp_h = self.hunger_pop(I_h + torch.randn(self.n_ch, device=self.device) * 0.5)
            sp_f = self.fatigue_pop(I_f + torch.randn(self.n_ch, device=self.device) * 0.5)
            sp_s = self.stress_pop(I_s + torch.randn(self.n_ch, device=self.device) * 0.5)
            h_spikes += sp_h
            f_spikes += sp_f
            s_spikes += sp_s

        self.hunger_rate.copy_(self.hunger_pop.rate)
        self.fatigue_rate.copy_(self.fatigue_pop.rate)
        self.stress_rate.copy_(self.stress_pop.rate)

        # Mean rates
        h_mean = float(self.hunger_rate.mean())
        f_mean = float(self.fatigue_rate.mean())
        s_mean = float(self.stress_rate.mean())

        # Valence integration
        # Positive: satiety + safety, Negative: hunger + stress + fatigue
        positive = max(0.0, reward * 2.0 + (1.0 - stress) * 0.3 + (energy / 100.0) * 0.2)
        negative = max(0.0, h_mean * 5.0 + s_mean * 6.0 + f_mean * 3.0 + (1.0 - energy / 100.0) * 0.3)
        I_val = torch.tensor([positive * 12.0, positive * 6.0,
                              negative * 12.0, negative * 6.0], device=self.device)
        for _ in range(15):  # reduced substeps
            self.valence_pop(I_val + torch.randn(4, device=self.device) * 0.3)
        self.valence_rate.copy_(self.valence_pop.rate)

        # Valence: positive neurons - negative neurons
        pos_rate = float(self.valence_rate[:2].mean())
        neg_rate = float(self.valence_rate[2:].mean())
        self.valence = 0.8 * self.valence + 0.2 * (pos_rate - neg_rate)
        self.valence = max(-1.0, min(1.0, self.valence))

        # Arousal: total interoceptive activity
        self.arousal = min(1.0, h_mean + f_mean + s_mean)

        # Heart rate model
        # Baseline 2 Hz, modulated by stress/effort
        hr_target = 2.0 + 3.0 * stress + 1.5 * fatigue + 1.0 * self.arousal
        # Bradycardia during acute threat (freezing response)
        if threat_acute and stress > 0.5:
            hr_target = max(1.0, hr_target * 0.5)
        # Tachycardia during flee
        if fatigue > 0.3 and stress > 0.3:
            hr_target = min(8.0, hr_target * 1.3)

        self.heart_rate = 0.9 * self.heart_rate + 0.1 * hr_target
        self.heart_rate = max(0.5, min(8.0, self.heart_rate))

        # Heart beat detection (phase oscillator)
        dt_step = 0.05  # ~50ms per behavioral step
        self.heart_phase += self.heart_rate * dt_step * 2 * math.pi
        self.heartbeat = math.sin(self.heart_phase) > 0.9

        return {
            'hunger_rate': h_mean,
            'fatigue_rate': f_mean,
            'stress_rate': s_mean,
            'valence': self.valence,
            'arousal': self.arousal,
            'heart_rate': self.heart_rate,
            'heartbeat': self.heartbeat,
        }

    def get_allostatic_bias(self) -> dict:
        """Spiking-derived bias for goal selection."""
        h = float(self.hunger_rate.mean())
        s = float(self.stress_rate.mean())
        f = float(self.fatigue_rate.mean())
        return {
            'forage_bias': -h * 0.3,    # hunger → forage
            'flee_bias': -s * 0.2,       # stress → flee
            'explore_bias': -f * 0.15,   # fatigue → rest/explore
            'social_bias': 0.0,
        }

    def reset(self):
        self.hunger_pop.reset()
        self.fatigue_pop.reset()
        self.stress_pop.reset()
        self.valence_pop.reset()
        self.hunger_rate.zero_()
        self.fatigue_rate.zero_()
        self.stress_rate.zero_()
        self.valence_rate.zero_()
        self.heart_rate = 2.0
        self.heart_phase = 0.0
        self.heartbeat = False
        self.valence = 0.0
        self.arousal = 0.0
        self._step_count = 0
