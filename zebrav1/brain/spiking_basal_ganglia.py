"""
Spiking Basal Ganglia — direct/indirect pathway action selection.

Replaces the numpy gating function with a spiking BG circuit:
  - Striatum D1 (10 neurons): direct pathway → GO signal
  - Striatum D2 (10 neurons): indirect pathway → NO-GO signal
  - GPi (5 neurons): output gate (tonically active, inhibited by D1)
  - STN (5 neurons): subthalamic nucleus (excites GPi, brakes action)

Dopamine modulates D1 (excitatory) and D2 (inhibitory), shifting
the balance between GO and NO-GO.

Neuroscience: zebrafish dorsal telencephalon contains striatal-like
neurons with D1/D2 receptor expression (Mueller & Wullimann 2016).

Torch-based — spiking LIF neurons.
"""
import torch
import torch.nn as nn
import numpy as np


class SpikingBasalGanglia(nn.Module):
    """Spiking BG circuit for action selection.

    Args:
        n_d1: int — D1 striatal neurons (direct/GO)
        n_d2: int — D2 striatal neurons (indirect/NO-GO)
        n_gpi: int — GPi output neurons
        n_stn: int — subthalamic nucleus neurons
        tau: float — membrane time constant
        device: str
    """

    def __init__(self, n_d1=10, n_d2=10, n_gpi=5, n_stn=5,
                 tau=0.8, v_thresh=0.5, device="cpu"):
        super().__init__()
        self.n_total = n_d1 + n_d2 + n_gpi + n_stn
        self.tau = tau
        self.v_thresh = v_thresh
        self.device = device

        # Input weights
        self.W_input_d1 = nn.Linear(2, n_d1, bias=False)  # valL, valR
        self.W_input_d2 = nn.Linear(2, n_d2, bias=False)

        # D1 → GPi (inhibitory: D1 active → GPi suppressed → action GO)
        self.w_d1_gpi = -0.8

        # D2 → STN → GPi (excitatory: D2 active → STN → GPi active → action NO-GO)
        self.w_d2_stn = 0.5
        self.w_stn_gpi = 0.6

        # LIF state
        self.v_d1 = torch.zeros(n_d1, device=device)
        self.v_d2 = torch.zeros(n_d2, device=device)
        self.v_gpi = torch.zeros(n_gpi, device=device)
        self.v_stn = torch.zeros(n_stn, device=device)

        # Tonic GPi activity
        self.gpi_tonic = 0.4

        # Exploration noise
        self.noise = 0.1

        self.to(device)

    @torch.no_grad()
    def step(self, valL, valR, dopa, rpe):
        """Update BG spiking circuit.

        Args:
            valL, valR: float — lateral value signals
            dopa: float [0, 1] — dopamine level
            rpe: float — reward prediction error

        Returns:
            gate: float [-1, 1] — action gate (positive = go right)
        """
        # Input
        x = torch.tensor([valL, valR], dtype=torch.float32,
                          device=self.device)

        # Dopamine modulation: D1 excitation, D2 inhibition
        d1_mod = 1.0 + dopa * 0.5  # DA excites D1
        d2_mod = 1.0 - dopa * 0.3  # DA inhibits D2

        # D1 striatum (direct pathway)
        I_d1 = self.W_input_d1(x) * d1_mod
        noise_d1 = torch.randn_like(self.v_d1) * self.noise
        self.v_d1 = self.tau * self.v_d1 + (1 - self.tau) * (I_d1 + noise_d1)
        d1_spikes = (self.v_d1 >= self.v_thresh).float()
        self.v_d1 *= (1 - d1_spikes)

        # D2 striatum (indirect pathway)
        I_d2 = self.W_input_d2(x) * d2_mod
        noise_d2 = torch.randn_like(self.v_d2) * self.noise
        self.v_d2 = self.tau * self.v_d2 + (1 - self.tau) * (I_d2 + noise_d2)
        d2_spikes = (self.v_d2 >= self.v_thresh).float()
        self.v_d2 *= (1 - d2_spikes)

        # STN (excited by D2 indirect pathway)
        I_stn = d2_spikes.mean() * self.w_d2_stn
        self.v_stn = self.tau * self.v_stn + (1 - self.tau) * I_stn
        stn_spikes = (self.v_stn >= self.v_thresh).float()
        self.v_stn *= (1 - stn_spikes)

        # GPi (tonic active, inhibited by D1, excited by STN)
        I_gpi = (self.gpi_tonic
                 + d1_spikes.mean() * self.w_d1_gpi
                 + stn_spikes.mean() * self.w_stn_gpi)
        self.v_gpi = self.tau * self.v_gpi + (1 - self.tau) * I_gpi
        gpi_spikes = (self.v_gpi >= self.v_thresh).float()
        self.v_gpi *= (1 - gpi_spikes)

        # Gate output: D1 firing rate - GPi firing rate
        # High D1 + low GPi = GO, Low D1 + high GPi = NO-GO
        d1_rate = d1_spikes.mean()
        gpi_rate = gpi_spikes.mean()
        gate = float(d1_rate - gpi_rate)

        return gate

    def reset(self):
        self.v_d1.zero_()
        self.v_d2.zero_()
        self.v_gpi.zero_()
        self.v_stn.zero_()

    def get_diagnostics(self):
        return {
            "d1_v": float(self.v_d1.mean()),
            "d2_v": float(self.v_d2.mean()),
            "gpi_v": float(self.v_gpi.mean()),
            "n_total": self.n_total,
        }
