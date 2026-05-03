"""
Spiking inferior olive: cerebellar climbing fiber error signal.

Zebrafish inferior olive (IO):
  - Located in caudal hindbrain (rhombomere 7-8)
  - Sends climbing fibers to cerebellar Purkinje cells
  - Each IO neuron contacts ~1 Purkinje cell (1:1 mapping)
  - Fires complex spikes in Purkinje cells on error/surprise
  - Gap-junction coupled: IO neurons oscillate in synchrony (~5-10 Hz)
  - Teaching signal: unexpected sensory events → IO burst → cerebellar learning

Distinct from SpikingCerebellum: the olive GENERATES the error signal;
cerebellum RECEIVES it via climbing fibers and updates Purkinje cell weights.

Free Energy Principle:
  IO computes sensory prediction error (expected vs actual outcome).
  Large PE → climbing fiber burst → cerebellar adaptation.
  Small PE → no climbing fiber → motor program confirmed.

Architecture:
  8 RS neurons: 4 sensory-error, 4 motor-error
  + 2-channel TwoCompColumn (sensory PE, motor PE)
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingInferiorOlive(nn.Module):
    def __init__(self, n_neurons=8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.5)  # quiescent by default; fires on error
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 2 channels (sensory error, motor error)
        self.pc = TwoCompColumn(n_channels=2, n_per_ch=4, substeps=8, device=device)

        # State
        self.sensory_error = 0.0     # unexpected sensory event magnitude
        self.motor_error = 0.0       # motor outcome mismatch
        self.climbing_fiber = 0.0    # output to cerebellum [0,1]
        self.complex_spike = False   # True when climbing fiber fires burst
        self._prev_sensory = 0.0     # previous sensory prediction
        self._prev_motor = 0.0       # previous motor prediction

        # Gap junction oscillation (IO neurons are electrically coupled)
        self._osc_phase = 0.0
        self._OSC_FREQ = 0.04        # ~8 Hz at 200 steps/sec

        # FEP
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, sensory_surprise: float = 0.0,
                motor_mismatch: float = 0.0,
                cerebellar_pe: float = 0.0) -> dict:
        """
        sensory_surprise: unexpected sensory event (e.g. collision, novel stimulus)
        motor_mismatch: difference between intended and actual motor output
        cerebellar_pe: prediction error from cerebellum (bidirectional coupling)
        """
        import math

        # --- Sensory error computation ---
        # Sudden changes in sensory surprise → IO activation
        delta_sensory = abs(sensory_surprise - self._prev_sensory)
        self.sensory_error = min(1.0, delta_sensory + sensory_surprise * 0.3)
        self._prev_sensory = sensory_surprise

        # --- Motor error computation ---
        delta_motor = abs(motor_mismatch - self._prev_motor)
        self.motor_error = min(1.0, delta_motor + motor_mismatch * 0.3)
        self._prev_motor = motor_mismatch

        # --- Gap junction oscillation (subthreshold) ---
        self._osc_phase += self._OSC_FREQ
        osc = 0.15 * math.sin(2 * math.pi * self._osc_phase)

        # --- Climbing fiber output ---
        # Combined error signal: sensory + motor + cerebellar feedback
        raw_cf = (self.sensory_error * 0.4 +
                  self.motor_error * 0.4 +
                  cerebellar_pe * 0.2)
        # Threshold: only fire climbing fiber on significant error
        if raw_cf > 0.2:
            self.climbing_fiber = min(1.0, raw_cf * 1.5)
            self.complex_spike = True
        else:
            self.climbing_fiber = max(0, raw_cf + osc)
            self.complex_spike = False

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        I[0] = self.sensory_error * 15.0    # sensory error +
        I[1] = delta_sensory * 12.0         # sensory change
        I[2] = self.sensory_error * 8.0     # sensory sustained
        I[3] = osc * 5.0 + 2.0             # gap junction oscillation
        I[4] = self.motor_error * 15.0      # motor error +
        I[5] = delta_motor * 12.0           # motor change
        I[6] = self.motor_error * 8.0       # motor sustained
        I[7] = self.climbing_fiber * 10.0   # climbing fiber output

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        sensory = torch.tensor([self.sensory_error, self.motor_error],
                               device=self.device, dtype=torch.float32)
        prediction = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float32)  # expect no error
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'sensory_error': self.sensory_error,
            'motor_error': self.motor_error,
            'climbing_fiber': self.climbing_fiber,
            'complex_spike': self.complex_spike,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
            'rate_vec': self.rate,  # (n_io,) tensor for 1:1 PC projection
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.sensory_error = 0.0
        self.motor_error = 0.0
        self.climbing_fiber = 0.0
        self.complex_spike = False
        self._prev_sensory = 0.0
        self._prev_motor = 0.0
        self._osc_phase = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
