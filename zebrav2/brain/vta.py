"""
Spiking ventral tegmental area (VTA) / substantia nigra pars compacta (SNpc):
dopamine (DA) source.

Zebrafish dopaminergic anatomy (Rink & Wullimann 2001; Kastenhuber et al. 2010):
  Posterior tuberculum (PT): teleost homologue of VTA/SNpc.
    ~40 dopaminergic neurons (TH+) per hemisphere.
    Projects to: pallium (mesolimbic reward), striatum (nigrostriatal motor),
    tectum (sensory gain), habenula (aversion modulation).

  Input sources:
    - RPE (reward prediction error): positive RPE → phasic DA burst
      (Schultz et al. 1997: unexpected reward = burst; omission = pause)
    - Habenula LHb: INHIBITORY (disappointment → LHb → DA pause via GABA)
    - Amygdala / IPN: chronic aversion → tonic DA suppression
    - Direct food reward: positive reward event drives phasic burst

  Firing modes (Ungless & Grace 2012):
    - Tonic: ~3–5 Hz baseline (i_tonic = 2.0 pA keeps cells spontaneously active)
    - Phasic burst: 15–25 Hz for 100–200 ms on unexpected reward
    - Pause: suppression to 0 Hz on reward omission / aversion

  Replaces scalar neuromod.DA EMA with population-coded spiking output.
  Output: float [0, 1] — compatible with all downstream DA reads.

References:
  - Schultz et al. (1997) Science 275:1593 — RPE coding
  - Rink & Wullimann (2001) Brain Res 889:316 — zebrafish DA anatomy
  - Kastenhuber et al. (2010) J Comp Neurol 518:116 — PT dopamine neurons
  - Ungless & Grace (2012) Trends Neurosci 35:422 — VTA firing modes
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingVTA(nn.Module):
    def __init__(self, n_da=40, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_da = n_da

        # DA neurons: RS-type, tonically active at baseline
        self.DA_pop = IzhikevichLayer(n_da, 'RS', device)
        # Tonic drive: keeps cells firing at ~3-5 Hz baseline (spontaneous DA release)
        # Calibrated: i_tonic=6.0 → ~0.01 EMA rate/cell → DA 0.5 at baseline
        self.DA_pop.i_tonic.fill_(6.0)

        # State
        self.register_buffer('da_level',     torch.tensor(0.5, device=device))
        self.register_buffer('da_rate',      torch.zeros(n_da, device=device))
        self.register_buffer('rpe_ema',      torch.tensor(0.0, device=device))

        # Pre-allocated noise buffer (avoids per-step MPS allocation)
        self.register_buffer('_noise_da', torch.zeros(n_da, device=device))

    @torch.no_grad()
    def forward(self,
                rpe: float = 0.0,
                lhb_rate: float = 0.0,
                amygdala_stress: float = 0.0,
                hpa_suppression: float = 1.0,
                ipn_da_feedback: float = 0.0) -> dict:
        """
        rpe            : reward prediction error (positive=reward, negative=omission)
        lhb_rate       : lateral habenula mean rate (disappointment → DA pause)
        amygdala_stress: amygdala activation (chronic stress → tonic DA reduction)
        hpa_suppression: HPA cortisol suppression multiplier (0-1)
        ipn_da_feedback: IPN → DA feedback (usually negative)

        Returns:
          da_level : float [0, 1] — current dopamine level
          phasic   : bool  — phasic burst active this step
          da_rate  : Tensor (n_da,) — population firing rate
        """
        # Smooth RPE for tonic component
        self.rpe_ema.mul_(0.95).add_(0.05 * rpe)

        # --- Synaptic drive (pA) ---
        # Phasic: unexpected reward burst → +40 pA (brief, strong drive)
        I_rpe_phasic  = max(0.0, rpe)  * 40.0
        # Omission/punishment: negative RPE → inhibition via GABA interneurons
        I_rpe_pause   = min(0.0, rpe)  * 30.0   # negative → inhibitory
        # LHb: habenula disappointment → strong GABA-mediated pause (Matsumoto & Hikosaka 2007)
        I_lhb_inh     = -lhb_rate      * 35.0
        # Amygdala chronic stress → moderate tonic suppression
        I_stress_inh  = -amygdala_stress * 10.0
        # IPN sustained aversion feedback
        I_ipn         = ipn_da_feedback * 15.0   # usually negative

        I_total = I_rpe_phasic + I_rpe_pause + I_lhb_inh + I_stress_inh + I_ipn

        # 20-substep spiking loop
        spikes_acc = torch.zeros(self.n_da, device=self.device)
        for _ in range(20):
            I_step = torch.full((self.n_da,), I_total, device=self.device) + self._noise_da.normal_(std=0.8)
            sp = self.DA_pop(I_step)
            spikes_acc += sp

        self.da_rate.copy_(self.DA_pop.rate)
        mean_rate = float(self.da_rate.mean())

        # Map firing rate to DA level [0, 1].
        # Calibrated to i_tonic=6.0 baseline: mean_rate≈0.01 → DA=0.5
        # Burst (RPE+): mean_rate≈0.05 → DA≈0.9; Pause (RPE-): mean_rate≈0 → DA=0.1
        da_raw = 0.4 + 10.0 * mean_rate
        da_raw = max(0.0, min(1.0, da_raw))
        # HPA cortisol suppression (anhedonia under chronic stress)
        da_raw *= hpa_suppression

        self.da_level.fill_(da_raw)

        phasic = (rpe > 0.15) and (I_rpe_phasic > 5.0)

        return {
            'da_level': float(self.da_level),
            'phasic':   phasic,
            'da_rate':  self.da_rate,
            'rpe_ema':  float(self.rpe_ema),
        }

    def reset(self):
        self.DA_pop.reset()
        self.da_level.fill_(0.5)
        self.da_rate.zero_()
        self.rpe_ema.zero_()
