"""
Spiking Nucleus Isthmi: winner-take-all visual stimulus selection.

Zebrafish NI anatomy (Fernandes et al. 2012, J Neurosci):
  - Receives glutamatergic input from ipsilateral tectal SGC neurons.
  - Projects GABAergic feedback to contralateral SFGS-b.
  - Cross-hemisphere inhibition: stronger visual stimulus suppresses
    weaker stimulus in the opposite hemifield — prey/threat segregation.

Bilateral organization (30 neurons/hemisphere = 60 total):
  NI_L receives SGC_L → projects GABAergic inhibition to R SFGS-b
  NI_R receives SGC_R → projects GABAergic inhibition to L SFGS-b

Neuron type: IB (intrinsic bursting) — consistent with cholinergic
NI neurons that produce burst responses to retinal motion (Bhatt 2020).

References:
  - Fernandes et al. (2012) "Tectal Neurons in Zebrafish Are Required for Prey
    Capture Behavior" J Neurosci
  - Bhatt et al. (2020) "The optic tectum of teleosts" Nat Neurosci
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer


class NucleusIsthmi(nn.Module):
    def __init__(self, n_per_side: int = 30, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_per_side = n_per_side
        self.n_total = n_per_side * 2

        # Bilateral NI populations (IB: burst-firing, cholinergic)
        self.NI_L = IzhikevichLayer(n_per_side, 'IB', device)
        self.NI_R = IzhikevichLayer(n_per_side, 'IB', device)
        # Strongly inhibited at rest; fires only on significant SGC input
        self.NI_L.i_tonic.fill_(-3.0)
        self.NI_R.i_tonic.fill_(-3.0)

        # State buffers
        self.register_buffer('rate_L', torch.zeros(n_per_side, device=device))
        self.register_buffer('rate_R', torch.zeros(n_per_side, device=device))

    @torch.no_grad()
    def forward(self, sgc_L_mean: float, sgc_R_mean: float) -> dict:
        """
        Parameters
        ----------
        sgc_L_mean : float
            Mean SGC_L firing rate (ipsilateral input to NI_L).
            NI_L then projects GABAergic inhibition to R SFGS-b.
        sgc_R_mean : float
            Mean SGC_R firing rate (ipsilateral input to NI_R).
            NI_R then projects GABAergic inhibition to L SFGS-b.

        Returns
        -------
        dict with:
            'ni_L_rate'     : float  — NI_L mean firing rate
            'ni_R_rate'     : float  — NI_R mean firing rate
            'fb_to_L_sfgsb' : float  — inhibitory pA to deliver to L SFGS-b (from NI_R)
            'fb_to_R_sfgsb' : float  — inhibitory pA to deliver to R SFGS-b (from NI_L)
        """
        # SGC → NI drive: gain=60 so sgc≈0.05 pushes above threshold
        # (IB tonic = -3.0; threshold ~5 pA → need +8 pA from SGC)
        I_L = torch.full((self.n_per_side,), sgc_L_mean * 60.0, device=self.device)
        I_R = torch.full((self.n_per_side,), sgc_R_mean * 60.0, device=self.device)

        for _ in range(15):
            noise = torch.randn(self.n_per_side, device=self.device) * 0.5
            self.NI_L(I_L + noise)
            self.NI_R(I_R + noise)

        self.rate_L.copy_(self.NI_L.rate)
        self.rate_R.copy_(self.NI_R.rate)
        rate_L_mean = float(self.rate_L.mean())
        rate_R_mean = float(self.rate_R.mean())

        # Cross-hemisphere GABAergic feedback (negative = hyperpolarising)
        # NI_L (active when L tectum strong) → suppresses R SFGS-b
        # NI_R (active when R tectum strong) → suppresses L SFGS-b
        fb_to_R = -rate_L_mean * 3.0
        fb_to_L = -rate_R_mean * 3.0

        return {
            'ni_L_rate': rate_L_mean,
            'ni_R_rate': rate_R_mean,
            'fb_to_L_sfgsb': fb_to_L,
            'fb_to_R_sfgsb': fb_to_R,
        }

    def reset(self):
        self.NI_L.reset()
        self.NI_R.reset()
        self.rate_L.zero_()
        self.rate_R.zero_()
