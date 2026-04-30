"""
Spiking nociception: pain and noxious-stimulus detection.

Zebrafish nociceptors:
  - TRPA1: chemical irritants (allyl isothiocyanate, acrolein)
  - TRPV1: thermal noxious stimuli (>33°C), capsaicin-like ligands
  - Rohon-Beard / trigeminal: mechanical nociception (pressure, tissue damage)

Free Energy Principle:
  Generative model predicts expected pain level from context (predator nearby,
  collision, temperature).  Unexpected pain (high PE) triggers withdrawal reflex
  and sensitisation.  Repeated mild stimuli → habituation (low PE, attenuated
  response).

Architecture:
  6 RS neurons: 2 mechanical (Rohon-Beard), 2 thermal (TRPV1), 2 chemical (TRPA1)
  + 3-channel TwoCompColumn for prediction error
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingNociception(nn.Module):
    def __init__(self, n_neurons=6, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 3 channels (mechanical, thermal, chemical)
        self.pc = TwoCompColumn(n_channels=3, n_per_ch=4, substeps=8, device=device)

        # State
        self.pain_level = 0.0          # overall nociceptive intensity [0,1]
        self.mechanical_pain = 0.0     # collision/pressure
        self.thermal_pain = 0.0        # temperature extremes
        self.chemical_pain = 0.0       # noxious chemicals
        self.withdrawal_reflex = 0.0   # motor override strength [0,1]
        self.sensitisation = 1.0       # gain factor (increases with repeated pain)
        self._habituation = 1.0        # decreases with repeated mild stimuli
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

        # Temporal state for sensitisation/habituation
        self._pain_history = []        # recent pain levels (last 20 steps)

    @torch.no_grad()
    def forward(self, collision: bool = False, wall_proximity: float = 0.0,
                predator_distance: float = 999.0, temperature: float = 25.0,
                chemical_irritant: float = 0.0,
                predicted_pain: float = None) -> dict:
        """
        collision: True if fish hit wall/object (from proprioception)
        wall_proximity: [0,1] nearness to arena boundary
        predator_distance: px distance to nearest predator (bite damage proxy)
        temperature: water temperature °C (zebrafish comfort 24-28°C)
        chemical_irritant: concentration of noxious chemical [0,1]
        predicted_pain: descending prediction of expected pain
        """
        # --- Compute nociceptive signals ---
        # Mechanical: collision + very close to wall + predator bite
        mech = 0.0
        if collision:
            mech += 0.8
        if wall_proximity > 0.7:
            mech += 0.3 * wall_proximity
        # Predator bite: damage when very close (<40px)
        if predator_distance < 40:
            mech += max(0, (40 - predator_distance) / 40) * 0.9
        mech = min(1.0, mech)
        self.mechanical_pain = mech

        # Thermal: deviation from comfort zone (24-28°C)
        # TRPV1 activates >33°C, cold nociceptors <15°C
        if temperature > 33:
            therm = min(1.0, (temperature - 33) / 10)
        elif temperature < 15:
            therm = min(1.0, (15 - temperature) / 10)
        elif temperature > 28:
            therm = (temperature - 28) / 5 * 0.3  # mild discomfort
        elif temperature < 24:
            therm = (24 - temperature) / 9 * 0.3
        else:
            therm = 0.0
        self.thermal_pain = therm

        # Chemical: direct TRPA1 activation
        self.chemical_pain = min(1.0, chemical_irritant)

        # Apply sensitisation/habituation
        raw_pain = (mech * 0.5 + therm * 0.25 + self.chemical_pain * 0.25) * self.sensitisation * self._habituation
        self.pain_level = min(1.0, raw_pain)

        # Update history for sensitisation/habituation
        self._pain_history.append(self.pain_level)
        if len(self._pain_history) > 20:
            self._pain_history.pop(0)
        # Sensitisation: repeated strong pain → increased gain
        strong_count = sum(1 for p in self._pain_history if p > 0.5)
        self.sensitisation = min(2.0, 1.0 + strong_count * 0.05)
        # Habituation: repeated mild pain → decreased response
        mild_count = sum(1 for p in self._pain_history if 0.05 < p < 0.3)
        self._habituation = max(0.5, 1.0 - mild_count * 0.025)

        # Withdrawal reflex: proportional to pain intensity
        # Immediate: sharp pain → strong withdrawal, mild → weak/no reflex
        if self.pain_level > 0.3:
            self.withdrawal_reflex = min(1.0, self.pain_level * 1.2)
        else:
            self.withdrawal_reflex = 0.0

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        I[0] = mech * 15.0          # mechanical noci +
        I[1] = mech * 10.0          # mechanical noci -
        I[2] = therm * 15.0         # thermal noci +
        I[3] = therm * 10.0         # thermal noci -
        I[4] = self.chemical_pain * 15.0  # chemical noci +
        I[5] = self.chemical_pain * 10.0  # chemical noci -

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        pred = predicted_pain if predicted_pain is not None else 0.0
        sensory = torch.tensor([mech, therm, self.chemical_pain], device=self.device, dtype=torch.float32)
        prediction = torch.tensor([pred, pred * 0.5, pred * 0.5], device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(0.5 * pe[0] + 0.25 * pe[1] + 0.25 * pe[2])
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'pain_level': self.pain_level,
            'mechanical_pain': self.mechanical_pain,
            'thermal_pain': self.thermal_pain,
            'chemical_pain': self.chemical_pain,
            'withdrawal_reflex': self.withdrawal_reflex,
            'sensitisation': self.sensitisation,
            'habituation': self._habituation,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.pain_level = 0.0
        self.mechanical_pain = 0.0
        self.thermal_pain = 0.0
        self.chemical_pain = 0.0
        self.withdrawal_reflex = 0.0
        self.sensitisation = 1.0
        self._habituation = 1.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
        self._pain_history.clear()
