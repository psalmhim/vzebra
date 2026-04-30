"""
Spiking auditory system: sound detection via inner ear.

Zebrafish hearing:
  - Saccule: primary auditory end-organ, otolith detects particle motion
  - Weberian ossicles: tripus/intercalarium link swim bladder to inner ear,
    amplifying pressure waves (50-4000 Hz, best sensitivity 100-800 Hz)
  - Mauthner cells: giant reticulospinal neurons for acoustic startle (C-start)

Free Energy Principle:
  Generative model predicts ambient acoustic profile.  Novel sounds
  (high PE) → startle or orienting.  Repeated sounds → habituation.
  Predator low-frequency sounds vs conspecific vocalizations are
  spectrally separated.

Architecture:
  6 RS neurons: 2 low-freq (predator/swim), 2 mid-freq (conspecific),
                2 high-freq (prey/ambient)
  + 3-channel TwoCompColumn for spectral prediction
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingAuditory(nn.Module):
    def __init__(self, n_neurons=6, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 3 frequency bands (low <200Hz, mid 200-1000Hz, high >1000Hz)
        self.pc = TwoCompColumn(n_channels=3, n_per_ch=4, substeps=8, device=device)

        # State
        self.low_freq = 0.0       # predator rumble / water displacement
        self.mid_freq = 0.0       # conspecific calls
        self.high_freq = 0.0      # prey splash / ambient
        self.acoustic_salience = 0.0
        self.startle_trigger = False   # Mauthner-cell C-start
        self.predator_acoustic = 0.0
        self.conspecific_acoustic = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

        # Temporal for habituation
        self._prev_low = 0.0
        self._prev_mid = 0.0
        self._prev_high = 0.0
        self._startle_refractory = 0   # refractory period after startle

    @torch.no_grad()
    def forward(self, predator_distance: float = 999.0,
                conspecific_distances: list = None,
                ambient_noise: float = 0.1,
                sudden_onset: float = 0.0) -> dict:
        """
        predator_distance: px to nearest predator (low-freq swim sounds)
        conspecific_distances: list of distances to nearby conspecifics
        ambient_noise: background noise level [0,1]
        sudden_onset: abrupt loud sound [0,1] (e.g. tap on tank)
        """
        if conspecific_distances is None:
            conspecific_distances = []

        # --- Compute frequency-band intensities ---
        # Low frequency: predator motion → water displacement ∝ 1/distance²
        if predator_distance < 200:
            self.low_freq = max(0, (200 - predator_distance) / 200) ** 2
        else:
            self.low_freq = 0.0

        # Mid frequency: conspecific vocalizations (closer = louder)
        if conspecific_distances:
            closest = min(conspecific_distances)
            if closest < 150:
                self.mid_freq = max(0, (150 - closest) / 150) * 0.8
            else:
                self.mid_freq = 0.0
        else:
            self.mid_freq = 0.0

        # High frequency: ambient + sudden onset
        self.high_freq = min(1.0, ambient_noise * 0.3 + sudden_onset)

        # --- Acoustic salience: spectral change from previous step ---
        delta_low = abs(self.low_freq - self._prev_low)
        delta_mid = abs(self.mid_freq - self._prev_mid)
        delta_high = abs(self.high_freq - self._prev_high)
        self.acoustic_salience = delta_low * 0.4 + delta_mid * 0.3 + delta_high * 0.3
        self._prev_low = self.low_freq
        self._prev_mid = self.mid_freq
        self._prev_high = self.high_freq

        # --- Mauthner startle: sudden loud onset → C-start escape ---
        if self._startle_refractory > 0:
            self._startle_refractory -= 1
            self.startle_trigger = False
        elif sudden_onset > 0.6 or (delta_low > 0.4 and self.low_freq > 0.5):
            self.startle_trigger = True
            self._startle_refractory = 10  # 10 steps refractory
        else:
            self.startle_trigger = False

        # Named outputs for brain integration
        self.predator_acoustic = self.low_freq
        self.conspecific_acoustic = self.mid_freq

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        I[0] = self.low_freq * 12.0    # low freq +
        I[1] = self.low_freq * 8.0     # low freq -
        I[2] = self.mid_freq * 12.0    # mid freq +
        I[3] = self.mid_freq * 8.0     # mid freq -
        I[4] = self.high_freq * 12.0   # high freq +
        I[5] = self.high_freq * 8.0    # high freq -

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        sensory = torch.tensor([self.low_freq, self.mid_freq, self.high_freq],
                               device=self.device, dtype=torch.float32)
        prediction = torch.tensor([self._prev_low, self._prev_mid, self._prev_high],
                                  device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'low_freq': self.low_freq,
            'mid_freq': self.mid_freq,
            'high_freq': self.high_freq,
            'acoustic_salience': self.acoustic_salience,
            'startle_trigger': self.startle_trigger,
            'predator_acoustic': self.predator_acoustic,
            'conspecific_acoustic': self.conspecific_acoustic,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.low_freq = 0.0
        self.mid_freq = 0.0
        self.high_freq = 0.0
        self.acoustic_salience = 0.0
        self.startle_trigger = False
        self.predator_acoustic = 0.0
        self.conspecific_acoustic = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
        self._prev_low = 0.0
        self._prev_mid = 0.0
        self._prev_high = 0.0
        self._startle_refractory = 0
