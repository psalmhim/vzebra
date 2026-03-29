"""
Izhikevich spiking neurons — vectorized over population.
Supports RS, IB, FS, CH, LTS, TC cell types.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DT, DEVICE, E_REST, I_TONIC_MU, I_TONIC_SIG

# Izhikevich parameters per type: (a, b, c, d)
NEURON_PARAMS = {
    'RS':  (0.02, 0.20, -65.0,  8.0),   # Regular spiking (pyramidal excitatory)
    'IB':  (0.02, 0.20, -55.0,  4.0),   # Intrinsic bursting (deep layer)
    'FS':  (0.10, 0.20, -65.0,  2.0),   # Fast spiking (PV+ inhibitory)
    'CH':  (0.02, 0.20, -50.0,  2.0),   # Chattering (superficial tectal)
    'LTS': (0.02, 0.25, -65.0,  2.0),   # Low-threshold spiking
    'TC':  (0.02, 0.25, -65.0,  0.05),  # Thalamocortical (burst/tonic)
    'MSN': (0.02, 0.25, -65.0,  6.0),   # Medium spiny neuron (striatum)
}
V_THRESH = 30.0
V_REST   = -65.0

class IzhikevichLayer(nn.Module):
    """
    Vectorized Izhikevich population.
    State: v (membrane potential), u (recovery variable).
    Input: synaptic current I (pA).
    Output: spikes (binary), v, u, rate (exponential moving average of spikes).
    """
    def __init__(self, n: int, cell_type: str = 'RS', device=DEVICE, tau_rate: float = 0.020):
        super().__init__()
        self.n = n
        self.cell_type = cell_type
        self.device = device
        a, b, c, d = NEURON_PARAMS[cell_type]
        # Store as buffers (not parameters — not trained)
        self.register_buffer('a', torch.full((n,), a, device=device))
        self.register_buffer('b', torch.full((n,), b, device=device))
        self.register_buffer('c', torch.full((n,), c, device=device))
        self.register_buffer('d', torch.full((n,), d, device=device))
        # State
        self.register_buffer('v', torch.full((n,), V_REST, device=device))
        self.register_buffer('u', torch.full((n,), b * V_REST, device=device))
        self.register_buffer('rate', torch.zeros(n, device=device))
        self.tau_rate = tau_rate  # exponential moving average decay
        # Tonic drive
        self.register_buffer('i_tonic',
            torch.normal(I_TONIC_MU, I_TONIC_SIG, (n,)).clamp(0.5, 8.0).to(device))

    def forward(self, I_syn: torch.Tensor) -> torch.Tensor:
        """
        One 1ms Izhikevich step.
        I_syn: (n,) synaptic current in pA
        Returns: spikes (n,) binary float tensor
        """
        I_total = I_syn + self.i_tonic
        # Half-step (midpoint method for stability)
        v_h = self.v + 0.5 * DT * 1000 * (0.04*self.v**2 + 5*self.v + 140 - self.u + I_total)
        u_h = self.u + 0.5 * DT * 1000 * self.a * (self.b * self.v - self.u)
        # Full step
        v_new = v_h + 0.5 * DT * 1000 * (0.04*v_h**2 + 5*v_h + 140 - u_h + I_total)
        u_new = u_h + 0.5 * DT * 1000 * self.a * (self.b * v_h - u_h)
        # Clamp for numerical stability
        v_new = v_new.clamp(-100.0, 35.0)
        # Spike detection and reset
        spikes = (v_new >= V_THRESH).float()
        v_new = torch.where(spikes.bool(), self.c, v_new)
        u_new = torch.where(spikes.bool(), u_new + self.d, u_new)
        # Update state
        self.v.copy_(v_new)
        self.u.copy_(u_new)
        # Update rate (exponential moving average)
        alpha = self.tau_rate
        self.rate.copy_((1 - alpha) * self.rate + alpha * spikes)
        return spikes

    def reset(self):
        """Reset to resting state."""
        self.v.fill_(V_REST)
        a, b = NEURON_PARAMS[self.cell_type][0], NEURON_PARAMS[self.cell_type][1]
        self.u.fill_(b * V_REST)
        self.rate.zero_()
