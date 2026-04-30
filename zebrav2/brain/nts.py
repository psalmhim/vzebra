"""
Spiking nucleus tractus solitarius: primary visceral sensory relay.

Zebrafish NTS (Nucleus Tractus Solitarius):
  - Located in caudal hindbrain (rhombomere 7)
  - Primary relay for gustatory and visceral afferent information
  - Receives taste input from cranial nerves VII, IX, X (facial, glossopharyngeal, vagus)
  - Receives vagal visceral afferents from gut, heart, swim bladder
  - Projects to hypothalamus (feeding), parabrachial (arousal), dorsal motor vagus (autonomic)
  - Integrates taste + satiety signals for meal termination
  - Baroreflex: vagal baroreceptor input -> NTS -> parasympathetic cardiac output
  - Area postrema (adjacent, lacks BBB) feeds nausea/toxin signals into NTS

Distinct from hypothalamus: NTS is the PRIMARY relay — it preprocesses
visceral/gustatory signals before hypothalamus integrates them with
homeostatic drives.

Free Energy Principle:
  NTS computes prediction error between expected and actual visceral state.
  Gustatory PE: unexpected taste -> novelty signal -> cautious ingestion.
  Visceral PE: gut distension mismatch -> satiety or nausea.

Architecture:
  10 RS neurons: 4 gustatory relay, 3 visceral afferent relay, 3 cardiorespiratory
  + 2-channel TwoCompColumn (gustatory PE, visceral PE)
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingNTS(nn.Module):
    def __init__(self, n_neurons=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)  # low spontaneous activity; driven by inputs
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 2 channels (gustatory PE, visceral PE)
        self.pc = TwoCompColumn(n_channels=2, n_per_ch=4, substeps=8, device=device)

        # State
        self.taste_relay = 0.0          # processed gustatory -> forebrain
        self.visceral_relay = 0.0       # gut status -> hypothalamus
        self.cardio_output = 0.0        # baroreflex parasympathetic signal
        self.satiety_signal = 0.0       # meal termination signal [0,1]
        self.nausea_relay = 0.0         # area postrema toxin/mismatch signal

        # Internal accumulators
        self._taste_accumulator = 0.0   # sensory-specific satiety buildup
        self._prev_taste = 0.0          # previous taste input for change detection
        self._prev_visceral = 0.0       # previous visceral input
        self._baroreceptor_baseline = 0.5  # resting blood pressure setpoint

        # Disorder-modulatable gain parameters (wildtype = 1.0)
        self._satiety_scale = 1.0       # satiety sensitivity multiplier
        self._visceral_scale = 1.0      # visceral relay gain multiplier

        # FEP
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, taste_input: float = 0.0,
                vagal_afferent: float = 0.0,
                baroreceptor: float = 0.0,
                respiratory_feedback: float = 0.0) -> dict:
        """
        taste_input: gustatory signal from cranial nerves [0,1]
        vagal_afferent: visceral afferent from gut/organs [0,1]
        baroreceptor: blood pressure sensing via aortic/carotid [0,1]
        respiratory_feedback: gill ventilation rate feedback [0,1]
        """
        # --- Gustatory relay ---
        # Taste change detection + sustained component
        delta_taste = abs(taste_input - self._prev_taste)
        self.taste_relay = min(1.0, taste_input * 0.6 + delta_taste * 0.4)
        self._prev_taste = taste_input

        # --- Sensory-specific satiety ---
        # Repeated taste input accumulates satiety (habituates to ongoing food)
        if taste_input > 0.1:
            self._taste_accumulator = min(1.0,
                                          self._taste_accumulator + taste_input * 0.02)
        else:
            # Slow decay when not eating
            self._taste_accumulator = max(0.0, self._taste_accumulator - 0.005)
        self.satiety_signal = min(1.0, (self._taste_accumulator * 0.8
                                  + vagal_afferent * 0.3) * self._satiety_scale)

        # --- Visceral afferent relay ---
        delta_visceral = abs(vagal_afferent - self._prev_visceral)
        self.visceral_relay = min(1.0, (vagal_afferent * 0.5
                                        + delta_visceral * 0.3) * self._visceral_scale)
        self._prev_visceral = vagal_afferent

        # --- Nausea relay (area postrema) ---
        # Nausea triggered by high visceral afferent + taste mismatch
        visceral_excess = max(0.0, vagal_afferent - 0.7)
        taste_visceral_mismatch = abs(taste_input - vagal_afferent)
        self.nausea_relay = min(1.0, visceral_excess * 1.5
                                + taste_visceral_mismatch * 0.2)

        # --- Cardiorespiratory (baroreflex) ---
        # Deviation from baseline BP -> parasympathetic correction
        bp_error = baroreceptor - self._baroreceptor_baseline
        # Positive error (high BP) -> increase parasympathetic (slow heart)
        # Negative error (low BP) -> decrease parasympathetic
        self.cardio_output = min(1.0, max(0.0,
                                          0.5 + bp_error * 1.5
                                          + respiratory_feedback * 0.2))

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        # Gustatory relay neurons (0-3)
        I[0] = taste_input * 12.0            # raw taste
        I[1] = delta_taste * 15.0            # taste novelty
        I[2] = self.satiety_signal * 10.0    # satiety accumulation
        I[3] = self.nausea_relay * 8.0       # nausea/aversion

        # Visceral afferent relay neurons (4-6)
        I[4] = vagal_afferent * 12.0         # gut distension
        I[5] = delta_visceral * 15.0         # visceral change
        I[6] = self.visceral_relay * 10.0    # sustained visceral

        # Cardiorespiratory neurons (7-9)
        I[7] = abs(bp_error) * 15.0          # baroreflex error
        I[8] = respiratory_feedback * 10.0   # respiratory rate
        I[9] = self.cardio_output * 8.0      # parasympathetic output

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        sensory = torch.tensor([self.taste_relay, self.visceral_relay],
                               device=self.device, dtype=torch.float32)
        prediction = torch.tensor([self._taste_accumulator * 0.5,
                                   self._baroreceptor_baseline],
                                  device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'taste_relay': self.taste_relay,
            'visceral_relay': self.visceral_relay,
            'cardio_output': self.cardio_output,
            'satiety_signal': self.satiety_signal,
            'nausea_relay': self.nausea_relay,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.taste_relay = 0.0
        self.visceral_relay = 0.0
        self.cardio_output = 0.0
        self.satiety_signal = 0.0
        self.nausea_relay = 0.0
        self._taste_accumulator = 0.0
        self._prev_taste = 0.0
        self._prev_visceral = 0.0
        self._baroreceptor_baseline = 0.5
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
