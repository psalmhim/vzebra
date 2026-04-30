"""
Spiking gustatory system: taste/chemosensory evaluation.

Zebrafish gustation:
  - Taste buds on lips, oral cavity, pharyngeal arches, gill rakers
  - T1R receptors: amino acids (L-alanine, L-proline) → appetitive
  - T2R receptors: bitter compounds (quinine, denatonium) → aversive
  - Vagal lobe: primary gustatory centre (enlarged in cyprinids)

Free Energy Principle:
  Generative model predicts taste profile from visual food identification.
  Mismatch (e.g. food looks normal but tastes bitter) → high PE → spit-out
  reflex + learned aversion.

Architecture:
  6 RS neurons: 2 amino acid (appetitive), 2 bitter (aversive), 2 umami (nutrient)
  + 3-channel TwoCompColumn for taste prediction
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingGustatory(nn.Module):
    def __init__(self, n_neurons=6, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 3 taste channels (amino_acid, bitter, umami)
        self.pc = TwoCompColumn(n_channels=3, n_per_ch=4, substeps=8, device=device)

        # State
        self.amino_acid = 0.0       # appetitive (food quality)
        self.bitter = 0.0           # aversive (toxin)
        self.umami = 0.0            # nutrient richness
        self.palatability = 0.0     # net food desirability [-1, 1]
        self.spit_reflex = False    # reject response
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

        # Taste memory (conditioned aversion)
        self._aversion_history = []  # track bitter encounters

    @torch.no_grad()
    def forward(self, eating: bool = False, food_quality: float = 0.0,
                food_distance: float = 999.0, toxin_level: float = 0.0,
                predicted_taste: float = None) -> dict:
        """
        eating: True if fish is currently consuming food
        food_quality: nutritional value of food [0,1]
        food_distance: px to nearest food (taste smell at close range)
        toxin_level: concentration of toxic/bitter substance [0,1]
        predicted_taste: descending prediction of expected taste
        """
        # --- Compute taste signals ---
        # Amino acids: only active when very close to food or eating
        if eating:
            self.amino_acid = min(1.0, food_quality * 0.9 + 0.1)
            self.umami = min(1.0, food_quality * 0.7)
        elif food_distance < 30:
            # Gustatory detection at close range (dissolved amino acids)
            proximity_factor = max(0, (30 - food_distance) / 30)
            self.amino_acid = food_quality * 0.3 * proximity_factor
            self.umami = food_quality * 0.2 * proximity_factor
        else:
            self.amino_acid = 0.0
            self.umami = 0.0

        # Bitter: toxin detection (strongest when eating)
        if eating:
            self.bitter = min(1.0, toxin_level * 1.5)
        else:
            self.bitter = min(1.0, toxin_level * 0.5)

        # Palatability: net value = appetitive - aversive
        self.palatability = (self.amino_acid * 0.5 + self.umami * 0.5) - self.bitter
        self.palatability = max(-1.0, min(1.0, self.palatability))

        # Spit reflex: bitter > appetitive while eating
        self.spit_reflex = eating and self.bitter > 0.4 and self.bitter > self.amino_acid

        # Track aversion
        if self.spit_reflex:
            self._aversion_history.append(1.0)
        if len(self._aversion_history) > 10:
            self._aversion_history.pop(0)

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        I[0] = self.amino_acid * 15.0    # amino acid +
        I[1] = self.amino_acid * 8.0     # amino acid -
        I[2] = self.bitter * 15.0        # bitter +
        I[3] = self.bitter * 8.0         # bitter -
        I[4] = self.umami * 12.0         # umami +
        I[5] = self.umami * 8.0          # umami -

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        pred = predicted_taste if predicted_taste is not None else 0.0
        sensory = torch.tensor([self.amino_acid, self.bitter, self.umami],
                               device=self.device, dtype=torch.float32)
        prediction = torch.tensor([pred * 0.5, 0.0, pred * 0.5],
                                  device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'amino_acid': self.amino_acid,
            'bitter': self.bitter,
            'umami': self.umami,
            'palatability': self.palatability,
            'spit_reflex': self.spit_reflex,
            'aversion_count': len(self._aversion_history),
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.amino_acid = 0.0
        self.bitter = 0.0
        self.umami = 0.0
        self.palatability = 0.0
        self.spit_reflex = False
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
        self._aversion_history.clear()
