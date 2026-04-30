"""
Spiking vagus nerve: bidirectional visceral afferent/efferent pathway.

Zebrafish vagus nerve (cranial nerve X):
  - Largest cranial nerve; bidirectional visceral communication
  - Afferent (sensory, ~80%): gut → nucleus tractus solitarius (NTS) → brain
    Conveys visceral state: gut distension, nutrient status, inflammation
  - Efferent (motor, ~20%): dorsal motor nucleus → gut, heart, gills
    Parasympathetic brake on heart rate, gut motility regulation
  - Cardiac branch: parasympathetic slowing of heart rate (vagal tone)
    Zebrafish have vagal innervation of the heart by 3 dpf
  - Respiratory branch: modulates gill ventilation rate

Vagal tone is a biomarker of parasympathetic activity:
  - High vagal tone: calm, resting → slow heart rate, active digestion
  - Low vagal tone (vagal withdrawal): stress → fast heart rate, gut shutdown
  - Heart rate variability (HRV) reflects vagal tone in zebrafish

Free Energy Principle:
  Vagus afferents carry interoceptive prediction errors (visceral surprise).
  Efferent signals implement active inference: adjusting visceral organs
  to match the brain's generative model of expected body state.
  Precision on vagal afferents = interoceptive attention (Craig 2002).

Architecture:
  10 RS neurons: 4 afferent (gut→brain), 4 efferent (brain→gut), 2 cardiac
  + 2-channel TwoCompColumn (visceral afferent PE, cardiac PE)
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingVagusNerve(nn.Module):
    def __init__(self, n_neurons=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)  # moderate tonic: active at rest
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 2 channels (visceral afferent PE, cardiac PE)
        self.pc = TwoCompColumn(n_channels=2, n_per_ch=4, substeps=8, device=device)

        # State variables
        self.vagal_tone = 0.5          # parasympathetic brake [0,1], 0.5=resting
        self.gut_signal = 0.0          # ascending visceral information [0,1]
        self.cardiac_output = 0.0      # heart rate modulation [0,1]
        self.respiratory_drive = 0.0   # gill ventilation drive [0,1]

        # Internal processing state
        self._afferent_activity = 0.0  # gut→brain sensory stream
        self._efferent_activity = 0.0  # brain→gut motor command
        self._cardiac_brake = 0.5      # parasympathetic cardiac slowing

        # Setpoints for interoceptive prediction
        self._hr_setpoint = 0.4        # expected resting heart rate (low)
        self._gut_setpoint = 0.3       # expected resting gut activity (moderate)

        # FEP state
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, heart_rate: float = 0.4,
                gut_state: float = 0.3,
                stress: float = 0.0,
                respiratory_rate: float = 0.3) -> dict:
        """
        heart_rate: current heart rate [0, 1] (0=bradycardia, 1=tachycardia)
        gut_state: visceral afferent signal [0, 1] (0=empty/quiet, 1=full/active)
        stress: sympathetic stress drive [0, 1]
        respiratory_rate: gill ventilation rate [0, 1]
        """
        # --- Vagal tone computation ---
        # High vagal tone when calm (low stress), low when stressed
        # Parasympathetic withdrawal under stress
        raw_tone = 1.0 - stress * 0.8
        # Smooth update: vagal tone changes slowly (autonomic inertia)
        self.vagal_tone = 0.85 * self.vagal_tone + 0.15 * max(0.0, min(1.0, raw_tone))

        # --- Afferent pathway: gut → brain ---
        # Gut afferents carry nutrient, distension, and inflammatory signals
        # Modulated by vagal tone (higher tone → better gut-brain communication)
        self._afferent_activity = gut_state * (0.5 + 0.5 * self.vagal_tone)
        self.gut_signal = min(1.0, self._afferent_activity)

        # --- Efferent pathway: brain → gut ---
        # Parasympathetic efferents promote digestion, inhibited by stress
        self._efferent_activity = self.vagal_tone * 0.7 * (1.0 - stress * 0.5)
        self._efferent_activity = min(1.0, max(0.0, self._efferent_activity))

        # --- Cardiac branch ---
        # Vagal brake slows heart rate; withdrawal lets sympathetic drive through
        self._cardiac_brake = self.vagal_tone * 0.6
        # Cardiac output: deviation from brake-modulated rate
        target_hr = heart_rate * (1.0 - self._cardiac_brake)
        self.cardiac_output = min(1.0, max(0.0, target_hr))

        # --- Respiratory branch ---
        # Vagus modulates gill ventilation; stress increases respiratory drive
        raw_resp = respiratory_rate * 0.5 + stress * 0.3 + (1.0 - self.vagal_tone) * 0.2
        self.respiratory_drive = min(1.0, max(0.0, raw_resp))

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        # Afferent neurons (0-3): gut→brain signals
        I[0] = self.gut_signal * 12.0          # gut distension
        I[1] = gut_state * 10.0                # nutrient sensing
        I[2] = self._afferent_activity * 8.0   # visceral afferent
        I[3] = (1.0 - self.vagal_tone) * 10.0  # afferent under low tone
        # Efferent neurons (4-7): brain→gut commands
        I[4] = self._efferent_activity * 12.0  # parasympathetic drive
        I[5] = self.vagal_tone * 10.0          # tonic efferent
        I[6] = self._efferent_activity * 8.0   # gut motility command
        I[7] = (1.0 - stress) * 6.0            # rest-and-digest
        # Cardiac neurons (8-9)
        I[8] = self._cardiac_brake * 15.0      # cardiac vagal brake
        I[9] = self.cardiac_output * 10.0      # cardiac output

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        # Channel 0: visceral afferent PE (gut state vs expected)
        # Channel 1: cardiac PE (heart rate vs expected)
        sensory = torch.tensor([gut_state, heart_rate], device=self.device, dtype=torch.float32)
        prediction = torch.tensor([self._gut_setpoint, self._hr_setpoint],
                                  device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'vagal_tone': self.vagal_tone,
            'gut_signal': self.gut_signal,
            'cardiac_output': self.cardiac_output,
            'respiratory_drive': self.respiratory_drive,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.vagal_tone = 0.5
        self.gut_signal = 0.0
        self.cardiac_output = 0.0
        self.respiratory_drive = 0.0
        self._afferent_activity = 0.0
        self._efferent_activity = 0.0
        self._cardiac_brake = 0.5
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
