"""
Spiking area postrema: circumventricular organ for blood-borne signal detection.

Zebrafish area postrema (AP):
  - Located at the caudal end of the fourth ventricle (obex region)
  - Circumventricular organ: lacks a complete blood-brain barrier (BBB)
  - Neurons directly sense blood-borne chemicals, toxins, and metabolites
  - Chemosensory role: detects emetic toxins, blood glucose, osmolarity
  - Projects to nucleus tractus solitarius (NTS) and dorsal vagal complex
  - Triggers nausea/avoidance behavior → conditioned taste aversion

In zebrafish:
  - AP neurons express glucose transporters (GLUT) and osmosensors
  - Toxin detection drives avoidance swimming away from contaminated water
  - Connected to vagal circuits for visceral reflex coordination
  - Inflammatory cytokines (TNF-α, IL-1β) detected → sickness behavior
  - Glucose sensing: hypoglycemia → increased foraging drive

Free Energy Principle:
  AP computes chemosensory prediction errors: deviations of blood
  composition from expected homeostatic setpoints.  Toxin detection
  is a high-precision surprise signal → strong avoidance (active inference).
  Metabolic PE (glucose, osmolarity) drives allostatic behavioral corrections.

Architecture:
  8 RS neurons: 4 chemosensory (blood toxin detection), 4 emetic (nausea/avoidance)
  + 2-channel TwoCompColumn (chemosensory PE, metabolic PE)
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingAreaPostrema(nn.Module):
    def __init__(self, n_neurons=8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.5)  # quiescent baseline; fires on toxin
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 2 channels (chemosensory PE, metabolic PE)
        self.pc = TwoCompColumn(n_channels=2, n_per_ch=4, substeps=8, device=device)

        # Output state
        self.nausea_signal = 0.0     # nausea/avoidance drive [0, 1]
        self.toxin_detected = False  # boolean toxin alarm
        self.glucose_status = 0.0    # hypo(-1) / normo(0) / hyper(+1)
        self.osmotic_drive = 0.0     # thirst / water-seeking [0, 1]

        # Internal processing
        self._chemosensory_activation = 0.0  # aggregate toxin/chemical signal
        self._metabolic_deviation = 0.0      # metabolic homeostatic error
        self._nausea_accumulator = 0.0       # temporal integration of nausea

        # Setpoints
        self._glucose_setpoint = 0.5   # expected normal blood glucose
        self._osmolarity_setpoint = 0.5  # expected normal osmolarity
        self._toxin_threshold = 0.3    # toxin level triggering alarm

        # FEP state
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, blood_toxin: float = 0.0,
                blood_glucose: float = 0.5,
                blood_osmolarity: float = 0.5,
                inflammatory_signal: float = 0.0) -> dict:
        """
        blood_toxin: toxin concentration in blood [0, 1] (0=clean, 1=lethal)
        blood_glucose: blood glucose level [0, 1] (0=hypoglycemia, 1=hyperglycemia)
        blood_osmolarity: blood osmolarity [0, 1] (0=hypotonic, 1=hypertonic)
        inflammatory_signal: circulating cytokines [0, 1] (TNF-α, IL-1β)
        """
        # --- Chemosensory detection ---
        # AP neurons directly sense blood composition (no BBB)
        # Toxin detection: threshold-based alarm
        self.toxin_detected = blood_toxin > self._toxin_threshold
        self._chemosensory_activation = min(1.0,
            blood_toxin * 0.6 + inflammatory_signal * 0.3 +
            abs(blood_osmolarity - self._osmolarity_setpoint) * 0.1)

        # --- Nausea / emetic signal ---
        # Toxin → strong nausea; inflammation → moderate nausea
        # Temporal accumulation: sustained toxin exposure → increasing nausea
        raw_nausea = blood_toxin * 0.7 + inflammatory_signal * 0.2
        if self.toxin_detected:
            raw_nausea += 0.3  # alarm boost
        # Accumulate nausea with slow decay
        self._nausea_accumulator = (0.9 * self._nausea_accumulator +
                                    0.1 * raw_nausea)
        self.nausea_signal = min(1.0, max(0.0,
            self._nausea_accumulator * 1.2))

        # --- Glucose sensing ---
        # Deviation from setpoint: negative = hypoglycemia, positive = hyperglycemia
        glucose_error = blood_glucose - self._glucose_setpoint
        # Map to [-1, 1]: hypo/normo/hyper
        self.glucose_status = max(-1.0, min(1.0, glucose_error * 2.0))

        # --- Metabolic deviation ---
        # Aggregate metabolic error for allostatic drive
        glucose_dev = abs(glucose_error)
        osmolarity_dev = abs(blood_osmolarity - self._osmolarity_setpoint)
        self._metabolic_deviation = min(1.0, glucose_dev + osmolarity_dev)

        # --- Osmotic drive ---
        # Hyperosmolarity → thirst/water-seeking; hypoosmolarity → reduced intake
        osmotic_error = blood_osmolarity - self._osmolarity_setpoint
        self.osmotic_drive = min(1.0, max(0.0, osmotic_error * 2.0))

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        # Chemosensory neurons (0-3): toxin/chemical detection
        I[0] = blood_toxin * 15.0                    # toxin primary
        I[1] = inflammatory_signal * 12.0            # cytokine detection
        I[2] = self._chemosensory_activation * 10.0  # aggregate chemical
        I[3] = float(self.toxin_detected) * 12.0     # alarm burst
        # Emetic / avoidance neurons (4-7)
        I[4] = self.nausea_signal * 15.0             # nausea drive
        I[5] = abs(self.glucose_status) * 10.0       # glucose deviation
        I[6] = self.osmotic_drive * 12.0             # osmotic signal
        I[7] = self._metabolic_deviation * 8.0       # metabolic error

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        # Channel 0: chemosensory PE (toxin/inflammation vs expected clean blood)
        # Channel 1: metabolic PE (glucose + osmolarity vs setpoints)
        sensory = torch.tensor([
            self._chemosensory_activation, self._metabolic_deviation
        ], device=self.device, dtype=torch.float32)
        prediction = torch.tensor([
            0.0,  # expect no toxins/inflammation
            0.0,  # expect no metabolic deviation
        ], device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'nausea_signal': self.nausea_signal,
            'toxin_detected': self.toxin_detected,
            'glucose_status': self.glucose_status,
            'osmotic_drive': self.osmotic_drive,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.nausea_signal = 0.0
        self.toxin_detected = False
        self.glucose_status = 0.0
        self.osmotic_drive = 0.0
        self._chemosensory_activation = 0.0
        self._metabolic_deviation = 0.0
        self._nausea_accumulator = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
