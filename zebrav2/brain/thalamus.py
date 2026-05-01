"""
Thalamo-pallial loop with reticular nucleus (TRN).
TC neurons (LTS type): relay tectum → pallium.
TRN neurons (FS type): gate TC relay based on attention.

This IS a predictive coding microcircuit in spiking form:
  TC = error neurons (superficial pyramidal analog):
    Receive excitatory tectum (sensory) and inhibitory TRN (predictions).
    TC firing rate = E/I balance = prediction error (not analytically computed).
  TRN = precision gate (inhibitory interneuron analog):
    Receive pallium feedback (top-down predictions).
    TRN activity controls gain on TC inhibition = sensory precision.
    High TRN → strong inhibition of TC → low precision → inattention.
    Low TRN → weak inhibition → high precision → attention.

Free Energy (Feldman & Friston 2010):
  PE emerges from TC spiking dynamics (E/I balance), not from arithmetic.
  Precision is encoded as TRN→TC synaptic gain (modulated by NA).
  Free energy = TC error activity × precision = population activity.

Can be instantiated as a full thalamus (default) or as a half-thalamus for
hemispheric organisation by passing sfgs_b_n_e and n_tc at half-size.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, N_TC, N_TRN, N_OT_SFGS_B, N_PAL_S, SUBSTEPS, EI_FRAC_OT
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn

class Thalamus(nn.Module):
    def __init__(self, device=DEVICE,
                 sfgs_b_n_e: int = None,
                 n_tc: int = None,
                 n_trn: int = None):
        """
        sfgs_b_n_e: size of the SFGS-b E input from tectum.
                    Defaults to int(0.75 * N_OT_SFGS_B) = 900.
                    Pass 450 for a half-thalamus receiving one hemisphere.
        n_tc:       number of TC relay neurons.
                    Defaults to N_TC = 300. Pass 150 for a half-thalamus.
        n_trn:      number of TRN inhibitory neurons.
                    Defaults to N_TRN = 80. Kept the same for each half.
        """
        super().__init__()
        self.device = device

        # Resolve sizes (use spec defaults when not specified)
        _sfgs_b_n_e = sfgs_b_n_e if sfgs_b_n_e is not None else int(EI_FRAC_OT[0] * N_OT_SFGS_B)
        _n_tc       = n_tc       if n_tc       is not None else N_TC
        _n_trn      = n_trn      if n_trn      is not None else N_TRN
        # Full pallium feedback always uses complete pallium-S n_e
        _pal_s_n_e  = int(0.75 * N_PAL_S)   # 1200

        # TC neurons: LTS type (burst in low-drive, tonic in high-drive)
        self.TC = IzhikevichLayer(_n_tc, 'LTS', device)
        # TRN neurons: FS type (fast inhibitory gate)
        self.TRN = IzhikevichLayer(_n_trn, 'FS', device)

        # Input projections
        self.W_tect_tc  = nn.Linear(_sfgs_b_n_e, _n_tc,  bias=False)
        self.W_pal_trn  = nn.Linear(_pal_s_n_e,  _n_trn, bias=False)
        self.W_tc_trn   = nn.Linear(_n_tc,        _n_trn, bias=False)
        self.W_trn_tc   = nn.Linear(_n_trn,       _n_tc,  bias=False)  # inhibitory

        for W in [self.W_tect_tc, self.W_pal_trn, self.W_tc_trn, self.W_trn_tc]:
            nn.init.xavier_uniform_(W.weight, gain=0.3)
            W.to(device)

        self.register_buffer('tc_rate',  torch.zeros(_n_tc,  device=device))
        self.register_buffer('trn_rate', torch.zeros(_n_trn, device=device))

        # Prediction error and precision (FEP)
        self.register_buffer('prediction_error', torch.zeros(_n_tc, device=device))
        self.register_buffer('precision', torch.ones(1, device=device))  # scalar precision
        self.free_energy = 0.0

        # FEP: two-compartment thalamic prediction (Lee et al. 2026)
        # 2 channels: sensory relay (tectum→TC), prediction gate (pallium→TRN)
        self.pc = TwoCompColumn(n_channels=2, n_per_ch=4, substeps=8, device=device)

        # TC: stimulus-driven (no tonic), TRN: light tonic for gating
        self.TC.i_tonic.zero_()
        self.TRN.i_tonic.fill_(0.5)

    def forward(self, tect_sfgsb_rate: torch.Tensor,
                pal_s_rate: torch.Tensor,
                NA_level: float = 0.3) -> dict:
        """
        tect_sfgsb_rate: (sfgs_b_n_e,) rate from SFGS-b E neurons
        pal_s_rate: (pal_s_n_e,) feedback from pallium-S
        NA_level: noradrenaline (wakefulness, 0-1)
        """
        # NA raises tonic drive in TC (wake vs sleep)
        na_bias = NA_level * 5.0  # extra pA from LC
        for _ in range(SUBSTEPS):
            # Normalized drives
            def _nd(W, x, tgt=5.0):
                r = W(x.unsqueeze(0)).squeeze(0).detach()
                m = r.abs().mean() + 1e-8
                return r * (tgt / m) if m > 0.001 else r * 0.0

            I_trn = _nd(self.W_tc_trn, self.TC.rate, 3.0) + _nd(self.W_pal_trn, pal_s_rate, 2.0)
            I_tc  = (_nd(self.W_tect_tc, tect_sfgsb_rate, 6.0)
                   - _nd(self.W_trn_tc, self.TRN.rate, 2.0)
                   + na_bias)
            self.TRN(I_trn)
            self.TC(I_tc)
        self.tc_rate.copy_(self.TC.rate)
        self.trn_rate.copy_(self.TRN.rate)

        # --- PE from spiking dynamics (NOT analytical) ---
        # TC firing rate IS the prediction error:
        #   I_tc = tectum(sensory) - TRN(prediction) + NA
        #   TC_rate encodes the E/I residual = PE
        self.prediction_error.copy_(self.tc_rate)

        # --- FEP: two-compartment thalamic prediction (Lee et al. 2026) ---
        # Soma: tectum drive (bottom-up sensory)
        # Apical: pallium feedback (top-down prediction)
        tc_mean = float(self.tc_rate.mean())
        trn_mean = float(self.trn_rate.mean())
        sensory = torch.tensor([tc_mean, trn_mean], device=self.device)
        prediction = torch.tensor([trn_mean * 0.5, tc_mean * 0.3],
                                   device=self.device)
        att = torch.tensor([NA_level * 3.0, NA_level * 2.0], device=self.device)
        self.pc.set_attention(att)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pi = float(pc_out['precision'].mean())
        self.precision.fill_(pi)
        self.free_energy = pc_out['free_energy']

        return {
            'TC': self.tc_rate, 'TRN': self.trn_rate,
            'prediction_error': self.prediction_error,
            'precision': float(self.precision),
            'free_energy': self.free_energy,
        }

    def reset(self):
        self.TC.reset()
        self.TRN.reset()
        self.tc_rate.zero_()
        self.trn_rate.zero_()
        self.prediction_error.zero_()
        self.precision.fill_(1.0)
        self.pc.reset()
        self.free_energy = 0.0
