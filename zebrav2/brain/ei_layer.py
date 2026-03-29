"""
Excitatory-Inhibitory layer: 75% E (RS or IB) + 25% I (FS).
Includes intra-layer E-E, E-I, I-E, I-I connectivity.
Inter-layer input via external AMPA synapse.
"""
import torch
import torch.nn as nn
from zebrav2.spec import (DEVICE, DT, P_EE, P_EI, P_IE, P_II,
                           G_EE, G_EI, G_IE, G_II)
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.synapses import Synapse

class EILayer(nn.Module):
    """
    One E/I layer. Runs SUBSTEPS Izhikevich ms steps per behavioral step.
    E neurons: RS (default) or IB for deep layers.
    I neurons: FS.
    """
    def __init__(self, n_total: int, e_type: str = 'RS', device=DEVICE, name: str = ''):
        super().__init__()
        self.name = name
        self.device = device
        self.n_e = int(0.75 * n_total)
        self.n_i = n_total - self.n_e
        self.n_total = n_total

        # Neuron populations
        self.E = IzhikevichLayer(self.n_e, e_type, device)
        self.I = IzhikevichLayer(self.n_i, 'FS', device)

        # Intra-layer synapses
        self.syn_ee = Synapse(self.n_e, self.n_e, 'AMPA', G_EE, device)
        self.syn_ei = Synapse(self.n_e, self.n_i, 'AMPA', G_EI, device)
        self.syn_ie = Synapse(self.n_i, self.n_e, 'GABA_A', G_IE, device)
        self.syn_ii = Synapse(self.n_i, self.n_i, 'GABA_A', G_II, device)

        # Initialize connectivity
        self.syn_ee.init_sparse(P_EE, 1.0)
        self.syn_ei.init_sparse(P_EI, 1.0)
        self.syn_ie.init_sparse(P_IE, 1.0)
        self.syn_ii.init_sparse(P_II, 1.0)

        # State: accumulated spikes over behavioral step
        self.register_buffer('spike_E', torch.zeros(self.n_e, device=device))
        self.register_buffer('spike_I', torch.zeros(self.n_i, device=device))

    def forward(self, I_ext_e: torch.Tensor, I_ext_i: torch.Tensor = None,
                substeps: int = 50) -> tuple:
        """
        Run `substeps` ms of Izhikevich dynamics.
        I_ext_e: (n_e,) external current to E neurons (pA) — held constant
        I_ext_i: (n_i,) external current to I neurons (optional)
        Returns: (rate_E, rate_I, spike_E_total, spike_I_total)
        """
        if I_ext_i is None:
            I_ext_i = torch.zeros(self.n_i, device=self.device)

        spike_E_acc = torch.zeros(self.n_e, device=self.device)
        spike_I_acc = torch.zeros(self.n_i, device=self.device)

        for _ in range(substeps):
            # Intra-layer currents
            I_ee = self.syn_ee(self.E.rate > 0.1, self.E.v)
            I_ie = self.syn_ie(self.I.rate > 0.1, self.E.v)
            I_ei = self.syn_ei(self.E.rate > 0.1, self.I.v)
            I_ii = self.syn_ii(self.I.rate > 0.1, self.I.v)

            # Step neurons
            sp_e = self.E(I_ext_e + I_ee + I_ie)
            sp_i = self.I(I_ext_i + I_ei + I_ii)

            spike_E_acc += sp_e
            spike_I_acc += sp_i

        self.spike_E.copy_(spike_E_acc)
        self.spike_I.copy_(spike_I_acc)
        return self.E.rate, self.I.rate, spike_E_acc, spike_I_acc

    def get_rate_e(self) -> torch.Tensor:
        """Current E population rate (per ms)."""
        return self.E.rate

    def reset(self):
        self.E.reset()
        self.I.reset()
        for syn in [self.syn_ee, self.syn_ei, self.syn_ie, self.syn_ii]:
            syn.reset()
        self.spike_E.zero_()
        self.spike_I.zero_()
