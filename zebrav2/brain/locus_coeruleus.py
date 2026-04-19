"""
Spiking locus coeruleus (LC): noradrenaline (NA) source.

Zebrafish LC anatomy:
  Small bilateral nucleus (~10 neurons/side) in posterior hindbrain.
  Two operating modes (Aston-Jones & Cohen 2005):
    Tonic mode: low baseline firing → sustained arousal/vigilance
    Phasic mode: brief high-frequency burst → threat response, memory consolidation

  Input sources:
    - Amygdala: strong drive (threat → phasic burst)
    - Insula: interoceptive arousal (tonic modulation)
    - Circadian: wake/sleep gating

  Output projections:
    - Thalamus: wake/sleep gating (NA → TC burst/tonic transition)
    - Pallium: attention signal (NA enhances signal-to-noise)
    - Amygdala: memory consolidation (NA → LTP strengthening)

  Replaces scalar neuromod.NA EMA with population-coded spiking output.
  Output: float [0, 1] — compatible with all downstream NA reads.

References:
  - Aston-Jones & Cohen (2005) "An integrative theory of locus
    coeruleus-norepinephrine function" Annu Rev Neurosci
  - Singh et al. (2015) "Noradrenergic modulation of brain-wide
    network dynamics" Curr Biol (zebrafish)
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingLocusCoeruleus(nn.Module):
    def __init__(self, n_lc=20, phasic_threshold=0.3, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_lc = n_lc
        self.phasic_threshold = phasic_threshold

        # LC neurons: moderate tonic firing, can burst phasically
        self.LC = IzhikevichLayer(n_lc, 'RS', device)
        self.LC.i_tonic.fill_(0.5)  # moderate spontaneous firing

        # State buffers
        self.register_buffer('lc_rate', torch.zeros(n_lc, device=device))
        self.register_buffer('na_level', torch.tensor(0.3, device=device))

        # Phasic mode state
        self._phasic = False
        self._phasic_cooldown = 0
        self._na_ema = 0.3

    @torch.no_grad()
    def forward(self, amygdala_alpha: float = 0.0,
                insula_arousal: float = 0.5,
                circadian: float = 0.7,
                cms: float = 0.1) -> dict:
        """
        Parameters
        ----------
        amygdala_alpha : float — threat level (>0.3 triggers phasic burst)
        insula_arousal : float — interoceptive arousal (tonic drive)
        circadian : float — activity drive (1.0=day, 0.3=night)
        cms : float — contextual modulation signal

        Returns dict with na_level, phasic, lc_rate, wake_gate, attention
        """
        # Tonic drive: circadian + arousal + novelty
        tonic = 2.0 + circadian * 3.0 + insula_arousal * 5.0 + cms * 3.0

        # Phasic burst: amygdala threat exceeds threshold
        phasic_current = 0.0
        if self._phasic_cooldown > 0:
            self._phasic_cooldown -= 1
        if amygdala_alpha > self.phasic_threshold and self._phasic_cooldown == 0:
            phasic_current = 15.0  # strong burst
            self._phasic = True
            self._phasic_cooldown = 5  # 5-step refractory
        else:
            self._phasic = False

        I_lc = torch.full((self.n_lc,), tonic + phasic_current,
                          device=self.device)

        for _ in range(20):
            self.LC(I_lc + torch.randn(self.n_lc, device=self.device) * 0.5)

        self.lc_rate.copy_(self.LC.rate)
        lc_mean = float(self.lc_rate.mean())

        # Convert rate to NA level [0, 1]
        raw_na = min(1.0, max(0.0, lc_mean * 6.0))
        # Cap phasic NA to prevent runaway flee cascades
        if self._phasic:
            raw_na = min(0.8, raw_na)

        # EMA smoothing
        self._na_ema = 0.85 * self._na_ema + 0.15 * raw_na
        na = max(0.05, min(0.95, self._na_ema))
        self.na_level.fill_(na)

        return {
            'na_level': na,
            'phasic': self._phasic,
            'lc_rate': lc_mean,
            'wake_gate': min(1.0, na * 2.0),  # thalamus gating
            'attention': min(1.0, na * 1.5),   # pallium signal-to-noise
        }

    def reset(self):
        self.LC.reset()
        self.lc_rate.zero_()
        self.na_level.fill_(0.3)
        self._phasic = False
        self._phasic_cooldown = 0
        self._na_ema = 0.3
