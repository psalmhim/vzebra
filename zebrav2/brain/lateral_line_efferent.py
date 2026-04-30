"""
Spiking lateral line efferent: top-down suppression of self-generated flow.

Zebrafish lateral line efferent system:
  - Efferent neurons in rhombomeres 4-6 project to lateral line neuromasts
  - Cholinergic (ACh) efferents suppress hair cell sensitivity during swimming
  - Corollary discharge: copy of motor command predicts self-generated water flow
  - During active swimming, efferent suppression prevents hair cell saturation
  - External stimuli (predator wake, water current) still detected above suppression
  - Critical for distinguishing self-motion from external flow (reafference problem)
  - Homologous to mammalian olivocochlear efferents (auditory self-suppression)

Distinct from lateral line afferents: this module computes the SUPPRESSION
signal that gates incoming lateral line input, not the raw sensory processing.

Free Energy Principle:
  Efferent system minimizes surprise from self-generated flow.
  Flow prediction PE: predicted vs actual lateral line input during swimming.
  Suppression PE: suppression gain vs motor command magnitude mismatch.
  Low PE = accurate efference copy -> clean external flow signal.

Architecture:
  8 RS neurons: 4 corollary discharge (predict self-motion flow),
                4 suppression (gate lateral line input)
  + 2-channel TwoCompColumn (flow prediction PE, suppression PE)
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingLateralLineEfferent(nn.Module):
    def __init__(self, n_neurons=8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)  # silent at rest; driven by motor commands
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 2 channels (flow prediction PE, suppression PE)
        self.pc = TwoCompColumn(n_channels=2, n_per_ch=4, substeps=8, device=device)

        # State
        self.suppression_gain = 0.0     # how much to attenuate lateral line [0,1]
        self.predicted_flow = 0.0       # expected self-generated flow [0,1]
        self.external_flow = 0.0        # lateral_line - predicted self-flow [0,1]
        self.corollary_discharge = 0.0  # efference copy signal [0,1]

        # Internal
        self._prev_motor = 0.0          # previous motor command for derivative
        self._flow_prediction_ema = 0.0 # EMA of flow prediction for smooth tracking
        self._TAU_EMA = 0.15            # EMA smoothing (fast adaptation)
        self._suppression_scale = 1.0   # disorder-modulatable suppression efficacy

        # FEP
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, motor_command: float = 0.0,
                lateral_line_input: float = 0.0,
                swim_speed: float = 0.0,
                turn_rate: float = 0.0) -> dict:
        """
        motor_command: copy of current swim command [0,1]
        lateral_line_input: raw flow signal from lateral line hair cells [0,1]
        swim_speed: current swimming speed [0,1]
        turn_rate: current turning magnitude [0,1]
        """
        # --- Corollary discharge ---
        # Efference copy: direct copy of motor command magnitude
        # Turning also generates asymmetric flow -> include turn_rate
        self.corollary_discharge = min(1.0, motor_command * 0.7
                                       + turn_rate * 0.3)

        # --- Predicted self-generated flow ---
        # Self-flow is proportional to swim speed + motor command (with delay via EMA)
        raw_predicted = min(1.0, swim_speed * 0.6 + motor_command * 0.4)
        self._flow_prediction_ema = (self._flow_prediction_ema * (1.0 - self._TAU_EMA)
                                     + raw_predicted * self._TAU_EMA)
        self.predicted_flow = self._flow_prediction_ema

        # --- Suppression gain ---
        # More motor activity -> stronger suppression of lateral line
        # Rapid motor onset -> stronger suppression (delta term)
        # _suppression_scale: disorder-modulatable efficacy (wildtype=1.0)
        delta_motor = abs(motor_command - self._prev_motor)
        self._prev_motor = motor_command
        self.suppression_gain = min(1.0, (motor_command * 0.6
                                    + swim_speed * 0.25
                                    + delta_motor * 0.3) * self._suppression_scale)

        # --- External flow extraction ---
        # Subtract predicted self-flow (scaled by suppression) from raw input
        self_flow_estimate = self.predicted_flow * self.suppression_gain
        self.external_flow = max(0.0, min(1.0,
                                          lateral_line_input - self_flow_estimate))

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        # Corollary discharge neurons (0-3)
        I[0] = motor_command * 12.0             # motor copy
        I[1] = swim_speed * 10.0                # speed signal
        I[2] = turn_rate * 10.0                 # turn signal
        I[3] = self.predicted_flow * 12.0       # predicted flow output

        # Suppression neurons (4-7)
        I[4] = self.suppression_gain * 15.0     # suppression drive
        I[5] = self.corollary_discharge * 12.0  # efference copy
        I[6] = delta_motor * 15.0               # motor onset detection
        I[7] = self.external_flow * 10.0        # residual external signal

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        # Channel 0: flow prediction accuracy (predicted vs actual total flow)
        # Channel 1: suppression adequacy (suppression vs motor magnitude)
        sensory = torch.tensor([lateral_line_input, motor_command],
                               device=self.device, dtype=torch.float32)
        prediction = torch.tensor([self.predicted_flow, self.suppression_gain],
                                  device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'suppression_gain': self.suppression_gain,
            'predicted_flow': self.predicted_flow,
            'external_flow': self.external_flow,
            'corollary_discharge': self.corollary_discharge,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.suppression_gain = 0.0
        self.predicted_flow = 0.0
        self.external_flow = 0.0
        self.corollary_discharge = 0.0
        self._prev_motor = 0.0
        self._flow_prediction_ema = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
