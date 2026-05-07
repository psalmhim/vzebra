"""
Spiking Mauthner cell circuit: M-cells + CoLo interneurons.

Zebrafish Mauthner anatomy (Korn & Faber 2005; Bhatt et al. 2007):
  Two M-cells (one per hemisphere), each the largest neuron in the hindbrain.
  Each M-cell receives:
    - Visual looming input from contralateral tectum SGC via crossed
      tectobulbar tract (latency ~5 ms; Del Bene et al. 2010)
    - Acoustic / lateral-line pressure wave via ipsilateral VIII nerve
      club endings: mixed chemical + gap-junction synapses (Pereda 2011)
  M-cell axon crosses the midline immediately and descends to activate
  contralateral axial muscles → fish bends AWAY from the threat (C-start).

  CoLo (commissural local) interneurons — also called PHP cells:
    - Glycinergic, fast spiking (FS type).
    - Ipsilateral M-cell fires → CoLo fires → inhibits contralateral M-cell.
    - Guarantees a unidirectional C-start (Korn & Faber 1975).

Gap junction facilitation (club endings; Pereda & Faber 2011):
  Repeated sub-threshold stimuli accumulate electrical coupling current at
  club endings, lowering the effective C-start threshold.  Modelled as a
  decaying state variable that adds to synaptic drive.

Laterality convention (matching ReticulospinalSystem):
  Right tectum (sgc_R) → LEFT M-cell (threat on left → fish bends right)
  Left  tectum (sgc_L) → RIGHT M-cell (threat on right → fish bends left)
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingMauthner(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device

        # M-cells: 1 per side, RS-type, driven well below threshold at rest.
        # Require ≥25 pA net input to fire reliably in 20 substep window.
        self.M_L = IzhikevichLayer(1, 'RS', device)
        self.M_R = IzhikevichLayer(1, 'RS', device)
        self.M_L.i_tonic.fill_(-3.0)   # high threshold — rarely fires spontaneously
        self.M_R.i_tonic.fill_(-3.0)

        # CoLo interneurons: 2 per side, FS type (fast glycinergic inhibition)
        self.CoLo_L = IzhikevichLayer(2, 'FS', device)
        self.CoLo_R = IzhikevichLayer(2, 'FS', device)
        self.CoLo_L.i_tonic.fill_(-1.0)
        self.CoLo_R.i_tonic.fill_(-1.0)

        # Gap junction state (club endings; Pereda & Faber 2011)
        self.register_buffer('gap_L', torch.tensor(0.0, device=device))
        self.register_buffer('gap_R', torch.tensor(0.0, device=device))
        self.gap_decay = 0.90
        self.gap_gain  = 0.80

        # Rate buffers for visualizer
        self.register_buffer('rate_L', torch.tensor(0.0, device=device))
        self.register_buffer('rate_R', torch.tensor(0.0, device=device))
        self.register_buffer('_noise', torch.zeros(1, device=device))

    @torch.no_grad()
    def forward(self,
                sgc_L: torch.Tensor,
                sgc_R: torch.Tensor,
                ll_pressure: float = 0.0) -> dict:
        """
        sgc_L, sgc_R : (N_SGC_E,) tectal SGC population rates per side
        ll_pressure  : float 0–1  lateral-line high-freq threat proximity

        Returns:
          m_L_spike     : bool   left M-cell fired this step
          m_R_spike     : bool   right M-cell fired this step
          m_L_rate      : float  EMA rate
          m_R_rate      : float  EMA rate
          colo_L_active : bool   CoLo_L fired (suppressing M_R)
          colo_R_active : bool   CoLo_R fired (suppressing M_L)
        """
        sgc_L_mean = float(sgc_L.mean())
        sgc_R_mean = float(sgc_R.mean())

        # Gap junction accumulation: contralateral visual drives gap state
        self.gap_L.mul_(self.gap_decay).add_(self.gap_gain * sgc_R_mean)
        self.gap_R.mul_(self.gap_decay).add_(self.gap_gain * sgc_L_mean)
        gap_L = float(self.gap_L)
        gap_R = float(self.gap_R)

        # Synaptic drive (pA)
        # Visual crossed tectobulbar × 120 (strong fast looming reflex)
        # Acoustic/LL bilateral × 80 (underwater pressure wave)
        # Gap junction × 15 (electrical club-ending facilitation)
        I_vis_L = sgc_R_mean * 120.0
        I_vis_R = sgc_L_mean * 120.0
        I_ll    = ll_pressure * 80.0
        I_gap_L = gap_L * 15.0
        I_gap_R = gap_R * 15.0

        m_L_fired = False
        m_R_fired = False
        colo_L_active = False
        colo_R_active = False
        colo_inh_R = 0.0  # inhibitory pA from CoLo_L → M_R
        colo_inh_L = 0.0  # inhibitory pA from CoLo_R → M_L

        for _ in range(20):
            noise = self._noise.normal_(std=0.5)

            I_ml = (I_vis_L + I_ll + I_gap_L - colo_inh_L) * torch.ones(1, device=self.device) + noise
            I_mr = (I_vis_R + I_ll + I_gap_R - colo_inh_R) * torch.ones(1, device=self.device) + noise

            sp_ml = self.M_L(I_ml)
            sp_mr = self.M_R(I_mr)

            if sp_ml.any():
                m_L_fired = True
                # M_L spike → CoLo_L → strong glycinergic block of M_R
                sp_colo = self.CoLo_L(torch.full((2,), 40.0, device=self.device))
                if sp_colo.any():
                    colo_L_active = True
                    colo_inh_R = float(sp_colo.mean()) * 60.0

            if sp_mr.any():
                m_R_fired = True
                sp_colo = self.CoLo_R(torch.full((2,), 40.0, device=self.device))
                if sp_colo.any():
                    colo_R_active = True
                    colo_inh_L = float(sp_colo.mean()) * 60.0

        self.rate_L.fill_(float(self.M_L.rate.mean()))
        self.rate_R.fill_(float(self.M_R.rate.mean()))

        return {
            'm_L_spike':     m_L_fired,
            'm_R_spike':     m_R_fired,
            'm_L_rate':      float(self.rate_L),
            'm_R_rate':      float(self.rate_R),
            'colo_L_active': colo_L_active,
            'colo_R_active': colo_R_active,
        }

    def reset(self):
        self.M_L.reset()
        self.M_R.reset()
        self.CoLo_L.reset()
        self.CoLo_R.reset()
        self.gap_L.zero_()
        self.gap_R.zero_()
        self.rate_L.zero_()
        self.rate_R.zero_()
