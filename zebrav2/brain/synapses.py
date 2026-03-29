"""
Conductance-based synaptic dynamics.
Single exponential decay for AMPA, NMDA (with Mg block), GABA_A.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DT, DEVICE, TAU_AMPA, TAU_NMDA, TAU_GABA_A, E_AMPA, E_NMDA, E_GABA_A

class Synapse(nn.Module):
    """
    Single synapse type (AMPA, NMDA, or GABA_A) from pre_n → post_n neurons.
    W: weight matrix (post_n, pre_n), initialized by caller.
    """
    def __init__(self, pre_n: int, post_n: int, syn_type: str = 'AMPA',
                 g_bar: float = 1.0, device=DEVICE):
        super().__init__()
        self.pre_n = pre_n
        self.post_n = post_n
        self.syn_type = syn_type
        self.g_bar = g_bar
        self.device = device

        if syn_type == 'AMPA':
            self.tau = TAU_AMPA
            self.E_rev = E_AMPA
        elif syn_type == 'NMDA':
            self.tau = TAU_NMDA
            self.E_rev = E_NMDA
        elif syn_type == 'GABA_A':
            self.tau = TAU_GABA_A
            self.E_rev = E_GABA_A
        else:
            raise ValueError(f"Unknown synapse type: {syn_type}")

        self.decay = float(torch.exp(torch.tensor(-DT / self.tau)))
        # Sparse random weight matrix
        W = torch.zeros(post_n, pre_n, device=device)
        self.register_buffer('W', W)
        # Synaptic gating variable s per presynaptic neuron
        self.register_buffer('s', torch.zeros(pre_n, device=device))

    def init_sparse(self, p_connect: float, g_scale: float = 1.0):
        """Initialize sparse random connectivity."""
        mask = torch.rand(self.post_n, self.pre_n, device=self.device) < p_connect
        W = torch.rand(self.post_n, self.pre_n, device=self.device) * g_scale
        self.W.copy_(W * mask.float())

    def forward(self, pre_spikes: torch.Tensor, post_v: torch.Tensor) -> torch.Tensor:
        """
        pre_spikes: (pre_n,) binary
        post_v: (post_n,) membrane potential
        Returns: I_syn (post_n,) synaptic current in pA
        """
        # Update gating variable
        self.s.copy_(self.s * self.decay + pre_spikes)
        # Conductance
        g = self.g_bar * (self.W @ self.s)  # (post_n,)
        # Mg block for NMDA
        if self.syn_type == 'NMDA':
            mg_block = 1.0 / (1.0 + (1.0/3.57) * torch.exp(-0.062 * post_v))
            g = g * mg_block
        # Current
        I = g * (self.E_rev - post_v)
        return I

    def reset(self):
        self.s.zero_()
