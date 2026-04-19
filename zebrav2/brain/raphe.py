"""
Spiking raphe nucleus: 5-HT (serotonin) source.

Zebrafish dorsal raphe anatomy:
  Dorsal raphe (DR): ~30 serotonergic neurons — main 5-HT output.
    Projects widely: tectum (sensory gating), pallium (mood/patience),
    basal ganglia (impulse control), hypothalamus (satiety).
  Median raphe (MR): ~10 neurons — modulates hippocampal theta rhythm.
    Projects to pallium (memory encoding), place cells.

  Input sources:
    - Habenula LHb: INHIBITORY (disappointment suppresses 5-HT → impatience)
    - IPN: excitatory relay from habenula (ipn.raphe_drive)
    - Amygdala: stress acutely suppresses 5-HT, then rebounds
    - Circadian: 5-HT follows day/night cycle (higher during active phase)

  Replaces scalar neuromod.HT5 EMA with population-coded spiking output.
  Output: float [0, 1] — compatible with all downstream HT5 reads.

References:
  - Lillesaar (2011) "The serotonergic system in fish" J Chem Neuroanat
  - Yokogawa et al. (2012) "The dorsal raphe modulates sensory
    responsiveness during arousal in zebrafish" J Neurosci
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingRaphe(nn.Module):
    def __init__(self, n_dr=30, n_mr=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_dr = n_dr
        self.n_mr = n_mr
        self.n_total = n_dr + n_mr

        # Dorsal raphe: main 5-HT output
        self.DR = IzhikevichLayer(n_dr, 'RS', device)
        self.DR.i_tonic.fill_(-0.5)  # moderate spontaneous firing

        # Median raphe: theta modulation
        self.MR = IzhikevichLayer(n_mr, 'RS', device)
        self.MR.i_tonic.fill_(-1.0)

        # State buffers
        self.register_buffer('dr_rate', torch.zeros(n_dr, device=device))
        self.register_buffer('mr_rate', torch.zeros(n_mr, device=device))
        self.register_buffer('ht5_level', torch.tensor(0.5, device=device))

        # EMA smoothing for stable output (prevents oscillation)
        self._ht5_ema = 0.5

    @torch.no_grad()
    def forward(self, lhb_rate: float = 0.0, ipn_raphe_drive: float = 0.0,
                amygdala_stress: float = 0.0, circadian: float = 0.7,
                flee_active: bool = False) -> dict:
        """
        Parameters
        ----------
        lhb_rate : float — lateral habenula rate (INHIBITS raphe)
        ipn_raphe_drive : float — IPN excitatory relay
        amygdala_stress : float — acute stress level
        circadian : float — activity drive (1.0=day, 0.3=night)
        flee_active : bool — currently fleeing (suppresses 5-HT)

        Returns dict with ht5_level, dr_rate, mr_rate, sensory_gain, patience
        """
        # Base drive: circadian + IPN excitation - LHb inhibition - stress
        base = (3.0 + circadian * 5.0 + ipn_raphe_drive * 12.0
                - lhb_rate * 15.0 - amygdala_stress * 8.0)
        if flee_active:
            base -= 3.0  # suppress 5-HT during flight

        I_dr = torch.full((self.n_dr,), max(0.0, base), device=self.device)
        I_mr = torch.full((self.n_mr,), max(0.0, base * 0.7), device=self.device)

        for _ in range(20):
            self.DR(I_dr + torch.randn(self.n_dr, device=self.device) * 0.5)
            self.MR(I_mr + torch.randn(self.n_mr, device=self.device) * 0.5)

        self.dr_rate.copy_(self.DR.rate)
        self.mr_rate.copy_(self.MR.rate)
        dr_mean = float(self.dr_rate.mean())
        mr_mean = float(self.mr_rate.mean())

        # Convert rate to 5-HT level [0, 1] via sigmoid-like scaling
        raw_ht5 = min(1.0, max(0.0, dr_mean * 8.0))

        # EMA smoothing (matches neuromod.HT5 tau ≈ 0.92)
        self._ht5_ema = 0.9 * self._ht5_ema + 0.1 * raw_ht5
        ht5 = max(0.05, min(0.95, self._ht5_ema))
        self.ht5_level.fill_(ht5)

        return {
            'ht5_level': ht5,
            'dr_rate': dr_mean,
            'mr_rate': mr_mean,
            'sensory_gain': 1.0 - 0.3 * ht5,  # high 5-HT → reduced sensory gain
            'patience': ht5,  # high 5-HT → more patient
        }

    def reset(self):
        self.DR.reset()
        self.MR.reset()
        self.dr_rate.zero_()
        self.mr_rate.zero_()
        self.ht5_level.fill_(0.5)
        self._ht5_ema = 0.5
