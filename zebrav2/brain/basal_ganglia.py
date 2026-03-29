"""
D1/D2 Basal Ganglia with GPi disinhibition.
D1 MSNs: activated by DA (direct pathway → GoGPi suppressed)
D2 MSNs: inhibited by DA (indirect pathway → GPi activated)
Action selection via GPi disinhibition.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, N_D1, N_D2, N_GPI, N_PAL_D

class BasalGanglia(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        pal_d_n_e = int(0.75 * N_PAL_D)  # 150 E neurons in pallium-D
        # Input from pallium-D (goal representations)
        self.W_pald_d1 = nn.Linear(pal_d_n_e // 2, N_D1, bias=False)
        self.W_pald_d2 = nn.Linear(pal_d_n_e // 2, N_D2, bias=False)
        # GPi: inhibitory output, receives D1 (inhibition) and D2 (excitation)
        self.W_d1_gpi  = nn.Linear(N_D1, N_GPI, bias=False)  # D1 inhibits GPi
        self.W_d2_gpi  = nn.Linear(N_D2, N_GPI, bias=False)  # D2 excites GPi
        for W in [self.W_pald_d1, self.W_pald_d2, self.W_d1_gpi, self.W_d2_gpi]:
            nn.init.xavier_uniform_(W.weight, gain=0.5)
            W.to(device)
        # State
        self.register_buffer('d1_rate', torch.zeros(N_D1, device=device))
        self.register_buffer('d2_rate', torch.zeros(N_D2, device=device))
        self.register_buffer('gpi_rate', torch.zeros(N_GPI, device=device))
        self.register_buffer('gate', torch.tensor(0.5, device=device))
        self.pal_d_n_e = pal_d_n_e

    def forward(self, pal_d_rate: torch.Tensor, DA: float) -> dict:
        """
        pal_d_rate: (N_PAL_D,) pallium-D E neuron rates
        DA: dopamine level (0-1)
        Returns: gate (0-1, 1=fully open/Go, 0=closed/NoGo)
        """
        pal_e = pal_d_rate[:self.pal_d_n_e // 2]  # E neurons subset
        # D1: excited by DA (direct pathway) — scale up pallium input
        d1 = torch.relu(self.W_pald_d1(pal_e.unsqueeze(0)).squeeze(0).detach() * 50.0)
        d1 = d1 * (1.0 + 0.5 * DA)
        # D2: inhibited by DA (indirect pathway)
        d2 = torch.relu(self.W_pald_d2(pal_e.unsqueeze(0)).squeeze(0).detach() * 50.0)
        d2 = d2 * (1.0 - 0.3 * DA)
        # GPi: lower baseline → easier for D1 to open gate
        gpi_baseline = 0.3
        d1_suppression = self.W_d1_gpi(d1.unsqueeze(0)).squeeze(0).detach().mean()
        d2_excitation = self.W_d2_gpi(d2.unsqueeze(0)).squeeze(0).detach().mean()
        gpi_drive = gpi_baseline - 4.0 * d1_suppression + 2.0 * d2_excitation
        gpi_rate = torch.sigmoid(gpi_drive.detach().clone() if isinstance(gpi_drive, torch.Tensor) else torch.tensor(float(gpi_drive), device=self.device))
        # Gate: high when GPi suppressed (D1 dominates)
        gate = 1.0 - gpi_rate
        self.d1_rate.copy_(d1.clamp(0, 1))
        self.d2_rate.copy_(d2.clamp(0, 1))
        self.gpi_rate.fill_(gpi_rate.item())
        self.gate.copy_(gate)
        return {
            'D1': self.d1_rate, 'D2': self.d2_rate,
            'GPi': self.gpi_rate, 'gate': gate.item()
        }

    def reset(self):
        self.d1_rate.zero_()
        self.d2_rate.zero_()
        self.gpi_rate.zero_()
        self.gate.fill_(0.5)
