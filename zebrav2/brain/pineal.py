"""
Spiking pineal gland: direct photoreception and melatonin synthesis.

Zebrafish pineal (epiphysis):
  - Contains intrinsic photoreceptors (parapinopsin, exo-rhodopsin)
  - Directly photosensitive — no retinal input needed
  - Primary melatonin synthesis organ (aanat2 enzyme)
  - Feeds SCN-like circadian system
  - Aanat2 expression peaks at night; light suppresses melatonin

Distinct from SpikingCircadian: pineal is the organ that PRODUCES
melatonin; circadian.py models the downstream SCN-like oscillator
that uses melatonin as input.

Architecture:
  6 RS neurons: 3 photoreceptor (light-sensitive), 3 melatonin-producing
  + 2-channel TwoCompColumn (light prediction, melatonin prediction)
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingPineal(nn.Module):
    def __init__(self, n_neurons=6, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 2 channels (light level, melatonin)
        self.pc = TwoCompColumn(n_channels=2, n_per_ch=4, substeps=8, device=device)

        # State
        self.light_detected = 0.0      # pineal photoreceptor activation [0,1]
        self.melatonin_raw = 0.0       # aanat2-driven melatonin synthesis [0,1]
        self.melatonin_smoothed = 0.0  # EMA-smoothed output [0,1]
        self._mel_ema = 0.0            # internal EMA state

        # Melatonin synthesis dynamics
        self._SYNTHESIS_RATE = 0.05    # per-step increase in darkness
        self._SUPPRESSION_RATE = 0.1   # per-step decrease under light
        self._EMA_ALPHA = 0.05         # smoothing constant

        # FEP
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, light_level: float = 0.7,
                retinal_luminance: float = None) -> dict:
        """
        light_level: ambient light [0,1] (0=dark, 1=bright)
        retinal_luminance: optional additional light signal from retina
        """
        # --- Pineal photoreception ---
        # Direct photosensitivity (parapinopsin/exo-rhodopsin)
        if retinal_luminance is not None:
            self.light_detected = min(1.0, light_level * 0.6 + retinal_luminance * 0.4)
        else:
            self.light_detected = light_level

        # --- Melatonin synthesis (aanat2 pathway) ---
        # Light suppresses melatonin; darkness allows synthesis
        if self.light_detected > 0.3:
            # Light suppression: rapid melatonin decline
            self.melatonin_raw = max(0.0,
                                     self.melatonin_raw - self._SUPPRESSION_RATE * self.light_detected)
        else:
            # Dark synthesis: gradual melatonin increase
            darkness = 1.0 - self.light_detected
            self.melatonin_raw = min(1.0,
                                     self.melatonin_raw + self._SYNTHESIS_RATE * darkness)

        # Smooth output (biological melatonin changes slowly)
        self._mel_ema += self._EMA_ALPHA * (self.melatonin_raw - self._mel_ema)
        self.melatonin_smoothed = self._mel_ema

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        I[0] = self.light_detected * 15.0       # photoreceptor ON
        I[1] = self.light_detected * 10.0       # photoreceptor sustained
        I[2] = (1.0 - self.light_detected) * 8.0  # photoreceptor OFF (dark)
        I[3] = self.melatonin_raw * 12.0         # melatonin synthesis +
        I[4] = self.melatonin_raw * 8.0          # melatonin sustained
        I[5] = (1.0 - self.melatonin_raw) * 6.0  # melatonin inhibited

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        sensory = torch.tensor([self.light_detected, self.melatonin_raw],
                               device=self.device, dtype=torch.float32)
        # Prediction: expect current state to persist
        prediction = torch.tensor([self.light_detected, self._mel_ema],
                                  device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'light_detected': self.light_detected,
            'melatonin_raw': self.melatonin_raw,
            'melatonin_smoothed': self.melatonin_smoothed,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.light_detected = 0.0
        self.melatonin_raw = 0.0
        self.melatonin_smoothed = 0.0
        self._mel_ema = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
