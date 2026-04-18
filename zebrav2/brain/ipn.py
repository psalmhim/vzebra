"""
Spiking interpeduncular nucleus (IPN): habenula relay for behavioral inhibition.

Zebrafish IPN anatomy:
  Ventral IPN (vIPN):
    - Primary target of medial habenula (MHb) via fasciculus retroflexus
    - MHb→IPN is cholinergic (nicotinic) + peptidergic (substance P)
    - Drives behavioral inhibition: slows locomotion, suppresses approach
    - Critical for fear conditioning and aversion memory

  Dorsal IPN (dIPN):
    - Receives weaker lateral habenula (LHb) input
    - Projects to raphe nuclei and VTA (DA modulation)
    - Involved in reward/aversion balance

  Outputs:
    - Raphe (5-HT): modulates mood, patience, social behavior
    - VTA/SNc (DA): suppresses DA during sustained aversion
    - Tegmentum: direct behavioral inhibition (speed × 0.7)
    - Hypothalamus: stress axis activation (HPA/HPI)

  Biology:
    - 24 RS neurons (12 ventral, 12 dorsal)
    - Aversion memory with slow decay (tau=0.95): remembers bad outcomes
    - MHb input is strongest projection in zebrafish brain (Hong et al. 2013)
    - IPN lesion → loss of fear conditioning (Agetsuma et al. 2010)

References:
  - Hong et al. (2013) "Stereotyped axonal arbors of neuronal subtypes in
    the interpeduncular nucleus" Current Biology
  - Agetsuma et al. (2010) "The habenula is crucial for experience-dependent
    modification of fear responses in zebrafish" Nature Neuroscience
  - Okamoto et al. (2012) "Cell-type-specific inhibitory inputs to dendritic
    and somatic compartments of parvalbumin-expressing neurons" Neuron
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingIPN(nn.Module):
    def __init__(self, n_vipn=12, n_dipn=12,
                 aversion_tau=0.95,
                 inhibition_strength=0.3,
                 device=DEVICE):
        super().__init__()
        self.device = device
        self.n_vipn = n_vipn
        self.n_dipn = n_dipn
        self.n_total = n_vipn + n_dipn
        self.aversion_tau = aversion_tau
        self.inhibition_strength = inhibition_strength

        # Ventral IPN: receives strong MHb input → behavioral inhibition
        self.vIPN = IzhikevichLayer(n_vipn, 'RS', device)
        self.vIPN.i_tonic.fill_(-1.0)

        # Dorsal IPN: receives weaker LHb input → DA/5-HT modulation
        self.dIPN = IzhikevichLayer(n_dipn, 'RS', device)
        self.dIPN.i_tonic.fill_(-1.5)

        # State buffers
        self.register_buffer('vipn_rate', torch.zeros(n_vipn, device=device))
        self.register_buffer('dipn_rate', torch.zeros(n_dipn, device=device))
        self.register_buffer('aversion_memory', torch.tensor(0.0, device=device))

        # Output signals
        self.behavioral_inhibition = 0.0  # 0-1: how much to suppress speed
        self.da_feedback = 0.0            # negative = suppress DA
        self.raphe_drive = 0.0            # drive to raphe (5-HT modulation)

    @torch.no_grad()
    def forward(self, mhb_rate: float, lhb_rate: float,
                aversion: float = 0.0) -> dict:
        """
        Process habenula inputs through IPN relay.

        Parameters
        ----------
        mhb_rate : float
            Medial habenula firing rate (aversion / fear signal).
            MHb→vIPN is the strongest projection — drives inhibition.
        lhb_rate : float
            Lateral habenula firing rate (disappointment / negative RPE).
            LHb→dIPN modulates DA/5-HT output.
        aversion : float
            Current aversion level (from amygdala/habenula).

        Returns
        -------
        dict with:
            'behavioral_inhibition' : float 0-1 — speed multiplier reduction
            'da_feedback'           : float — DA suppression signal
            'raphe_drive'           : float — 5-HT modulation drive
            'aversion_memory'       : float — accumulated aversion trace
            'vipn_rate'             : float — ventral IPN mean rate
            'dipn_rate'             : float — dorsal IPN mean rate
            'speed_multiplier'      : float — direct speed scaling factor
        """
        # Drive vIPN with MHb (strong cholinergic) + aversion context
        I_v = torch.full((self.n_vipn,),
                         mhb_rate * 20.0 + aversion * 10.0,
                         device=self.device)

        # Drive dIPN with LHb (weaker, modulatory)
        I_d = torch.full((self.n_dipn,),
                         lhb_rate * 12.0 + aversion * 5.0,
                         device=self.device)

        # Run spiking dynamics (20 substeps)
        for _ in range(20):
            noise_v = torch.randn(self.n_vipn, device=self.device) * 0.5
            noise_d = torch.randn(self.n_dipn, device=self.device) * 0.5
            self.vIPN(I_v + noise_v)
            self.dIPN(I_d + noise_d)

        self.vipn_rate.copy_(self.vIPN.rate)
        self.dipn_rate.copy_(self.dIPN.rate)
        vipn_mean = float(self.vipn_rate.mean())
        dipn_mean = float(self.dipn_rate.mean())

        # Behavioral inhibition: vIPN drives speed reduction
        # Sigmoid-like saturation: strong MHb → max 30% speed reduction
        self.behavioral_inhibition = min(self.inhibition_strength,
                                         vipn_mean * 3.0)

        # Speed multiplier: 1.0 = no effect, 0.7 = max inhibition
        speed_mult = 1.0 - self.behavioral_inhibition

        # DA feedback: dIPN suppresses DA during sustained aversion
        self.da_feedback = -min(0.3, dipn_mean * 2.0)

        # Raphe drive: combined IPN output → 5-HT modulation
        self.raphe_drive = min(0.5, (vipn_mean + dipn_mean) * 2.0)

        # Aversion memory: slow-decaying trace of bad outcomes
        # Remembers contexts that produced aversion → future avoidance
        aversion_input = max(mhb_rate, aversion) * 0.1
        self.aversion_memory.mul_(self.aversion_tau)
        self.aversion_memory.add_(aversion_input)
        self.aversion_memory.clamp_(0.0, 1.0)

        return {
            'behavioral_inhibition': self.behavioral_inhibition,
            'da_feedback': self.da_feedback,
            'raphe_drive': self.raphe_drive,
            'aversion_memory': float(self.aversion_memory),
            'vipn_rate': vipn_mean,
            'dipn_rate': dipn_mean,
            'speed_multiplier': speed_mult,
        }

    def reset(self):
        self.vIPN.reset()
        self.dIPN.reset()
        self.vipn_rate.zero_()
        self.dipn_rate.zero_()
        self.aversion_memory.zero_()
        self.behavioral_inhibition = 0.0
        self.da_feedback = 0.0
        self.raphe_drive = 0.0
