"""
Optic tectum: bilaterally organised with optic chiasm crossing.

Each hemisphere receives CONTRALATERAL eye input (full decussation, zebrafish):
  L_tectum (sfgs_b_L, sfgs_d_L, sgc_L, so_L) <- R_eye (right visual field)
  R_tectum (sfgs_b_R, sfgs_d_R, sgc_R, so_R) <- L_eye (left visual field)

Layers per hemisphere (each half the original size, total neurons unchanged):
  SFGS-b (CH): 600 neurons -> n_e=450, n_i=150
  SFGS-d (CH): 600 neurons -> n_e=450, n_i=150
  SGC    (IB): 200 neurons -> n_e=150, n_i=50
  SO     (RS): 200 neurons -> n_e=150, n_i=50

Total E per hemisphere: 450+450+150+150 = 1200
Total E across both hemispheres: 2400 (same as before)

Accepts optional I_topdown_L / I_topdown_R for pallium top-down attention.
"""
import torch
import torch.nn as nn
from zebrav2.spec import (DEVICE, N_OT_SFGS_B, N_OT_SFGS_D, N_OT_SGC, N_OT_SO,
                          N_RET_PER_TYPE, N_RET_LOOM, N_RET_DS, SUBSTEPS)
from zebrav2.brain.ei_layer import EILayer


class Tectum(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device

        # Half-sizes per hemisphere
        n_sfgsb_h = N_OT_SFGS_B // 2   # 600
        n_sfgsd_h = N_OT_SFGS_D // 2   # 600
        n_sgc_h   = N_OT_SGC    // 2   # 200
        n_so_h    = N_OT_SO     // 2   # 200

        # LEFT hemisphere (receives R_eye input)
        self.sfgs_b_L = EILayer(n_sfgsb_h, 'CH', device, 'SFGS-b_L')
        self.sfgs_d_L = EILayer(n_sfgsd_h, 'CH', device, 'SFGS-d_L')
        self.sgc_L    = EILayer(n_sgc_h,   'IB', device, 'SGC_L')
        self.so_L     = EILayer(n_so_h,    'RS', device, 'SO_L')

        # RIGHT hemisphere (receives L_eye input)
        self.sfgs_b_R = EILayer(n_sfgsb_h, 'CH', device, 'SFGS-b_R')
        self.sfgs_d_R = EILayer(n_sfgsd_h, 'CH', device, 'SFGS-d_R')
        self.sgc_R    = EILayer(n_sgc_h,   'IB', device, 'SGC_R')
        self.so_R     = EILayer(n_so_h,    'RS', device, 'SO_R')

        # Input projection weights
        # L_tectum <- R_eye (contralateral crossing)
        self.W_on_L   = nn.Linear(N_RET_PER_TYPE, self.sfgs_b_L.n_e, bias=False)
        self.W_off_L  = nn.Linear(N_RET_PER_TYPE, self.sfgs_d_L.n_e, bias=False)
        self.W_loom_L = nn.Linear(N_RET_LOOM,     self.sgc_L.n_e,    bias=False)
        self.W_ds_L   = nn.Linear(N_RET_DS,        self.so_L.n_e,     bias=False)

        # R_tectum <- L_eye (contralateral crossing)
        self.W_on_R   = nn.Linear(N_RET_PER_TYPE, self.sfgs_b_R.n_e, bias=False)
        self.W_off_R  = nn.Linear(N_RET_PER_TYPE, self.sfgs_d_R.n_e, bias=False)
        self.W_loom_R = nn.Linear(N_RET_LOOM,     self.sgc_R.n_e,    bias=False)
        self.W_ds_R   = nn.Linear(N_RET_DS,        self.so_R.n_e,     bias=False)

        # Xavier init for all projection weights
        for W in [self.W_on_L, self.W_off_L, self.W_loom_L, self.W_ds_L,
                  self.W_on_R, self.W_off_R, self.W_loom_R, self.W_ds_R]:
            nn.init.xavier_uniform_(W.weight, gain=1.0)
            W.to(device)

        # Connectome-constrained init for ON/OFF projections
        from zebrav2.brain.connectome import init_connectome_weights
        init_connectome_weights(self.W_on_L.weight,  'retina', 'tectum', 1.0)
        init_connectome_weights(self.W_off_L.weight, 'retina', 'tectum', 1.0)
        init_connectome_weights(self.W_on_R.weight,  'retina', 'tectum', 1.0)
        init_connectome_weights(self.W_off_R.weight, 'retina', 'tectum', 1.0)

        # Tonic bias: sparse, input-driven firing
        for layer in [self.sfgs_b_L, self.sfgs_d_L, self.sgc_L, self.so_L,
                      self.sfgs_b_R, self.sfgs_d_R, self.sgc_R, self.so_R]:
            layer.E.i_tonic.fill_(-2.0)
            layer.I.i_tonic.fill_(0.5)

        # Fused output buffer: all 8 half-layer E rates concatenated
        total_e = (self.sfgs_b_L.n_e + self.sfgs_d_L.n_e +
                   self.sgc_L.n_e    + self.so_L.n_e    +
                   self.sfgs_b_R.n_e + self.sfgs_d_R.n_e +
                   self.sgc_R.n_e    + self.so_R.n_e)
        self.register_buffer('rate_e_all', torch.zeros(total_e, device=device))

        # Looming detection flag
        self.looming = False

    def _norm_drive(self, W, x, target_pA=2.0, max_pA=3.5):
        """Normalize and clamp input current so a sparse subset fires."""
        raw = W(x.unsqueeze(0)).squeeze(0).detach()
        raw_mean = raw.abs().mean() + 1e-8
        scaled = raw * (target_pA / raw_mean)
        return scaled.clamp(-max_pA, max_pA)

    def forward(self, rgc_out: dict,
                I_topdown_L: torch.Tensor = None,
                I_topdown_R: torch.Tensor = None) -> dict:
        """
        rgc_out: dict from RetinaV2 with keys:
            'R_on', 'R_off', 'R_loom', 'R_ds'  -> drive L_tectum (contralateral)
            'L_on', 'L_off', 'L_loom', 'L_ds'  -> drive R_tectum (contralateral)
        I_topdown_L: optional top-down current from pallium to L_tectum SFGS-b (n_e,)
        I_topdown_R: optional top-down current from pallium to R_tectum SFGS-b (n_e,)
        """
        # L_tectum driven by R_eye (optic chiasm crossing)
        I_sfgsb_L = self._norm_drive(self.W_on_L,   rgc_out['R_on'],   3.0, 6.0)
        I_sfgsd_L = self._norm_drive(self.W_off_L,  rgc_out['R_off'],  3.0, 6.0)
        I_sgc_L   = self._norm_drive(self.W_loom_L, rgc_out['R_loom'], 4.0, 8.0)
        I_so_L    = self._norm_drive(self.W_ds_L,   rgc_out['R_ds'],   3.0, 6.0)

        # R_tectum driven by L_eye (optic chiasm crossing)
        I_sfgsb_R = self._norm_drive(self.W_on_R,   rgc_out['L_on'],   3.0, 6.0)
        I_sfgsd_R = self._norm_drive(self.W_off_R,  rgc_out['L_off'],  3.0, 6.0)
        I_sgc_R   = self._norm_drive(self.W_loom_R, rgc_out['L_loom'], 4.0, 8.0)
        I_so_R    = self._norm_drive(self.W_ds_R,   rgc_out['L_ds'],   3.0, 6.0)

        # Apply optional top-down attention
        if I_topdown_L is not None:
            I_sfgsb_L = I_sfgsb_L + I_topdown_L[:self.sfgs_b_L.n_e]
        if I_topdown_R is not None:
            I_sfgsb_R = I_sfgsb_R + I_topdown_R[:self.sfgs_b_R.n_e]

        # Run E/I dynamics for each hemisphere
        sfgs_b_L_rate, _, _, _ = self.sfgs_b_L(I_sfgsb_L, substeps=SUBSTEPS)
        sfgs_d_L_rate, _, _, _ = self.sfgs_d_L(I_sfgsd_L, substeps=SUBSTEPS)
        sgc_L_rate,    _, _, _ = self.sgc_L(I_sgc_L,      substeps=SUBSTEPS)
        so_L_rate,     _, _, _ = self.so_L(I_so_L,        substeps=SUBSTEPS)

        sfgs_b_R_rate, _, _, _ = self.sfgs_b_R(I_sfgsb_R, substeps=SUBSTEPS)
        sfgs_d_R_rate, _, _, _ = self.sfgs_d_R(I_sfgsd_R, substeps=SUBSTEPS)
        sgc_R_rate,    _, _, _ = self.sgc_R(I_sgc_R,      substeps=SUBSTEPS)
        so_R_rate,     _, _, _ = self.so_R(I_so_R,        substeps=SUBSTEPS)

        # Looming: either hemisphere detects threat
        self.looming = (sgc_L_rate.mean() > 0.05) or (sgc_R_rate.mean() > 0.05)

        # Fuse all 8 half-layer E rates
        self.rate_e_all.copy_(torch.cat([
            sfgs_b_L_rate, sfgs_d_L_rate, sgc_L_rate, so_L_rate,
            sfgs_b_R_rate, sfgs_d_R_rate, sgc_R_rate, so_R_rate,
        ]))

        return {
            # Combined bilateral outputs (backward compatibility)
            'sfgs_b': torch.cat([sfgs_b_L_rate, sfgs_b_R_rate]),
            'sfgs_d': torch.cat([sfgs_d_L_rate, sfgs_d_R_rate]),
            'sgc':    torch.cat([sgc_L_rate,    sgc_R_rate]),
            'so':     torch.cat([so_L_rate,     so_R_rate]),
            'all_e':  self.rate_e_all,
            'looming': self.looming,
            # Hemispheric outputs (new)
            'sfgs_b_L':   sfgs_b_L_rate,
            'sfgs_b_R':   sfgs_b_R_rate,
            'sgc_L':      sgc_L_rate,
            'sgc_R':      sgc_R_rate,
            'sgc_L_mean': float(sgc_L_rate.mean()),
            'sgc_R_mean': float(sgc_R_rate.mean()),
        }

    def reset(self):
        for layer in [self.sfgs_b_L, self.sfgs_d_L, self.sgc_L, self.so_L,
                      self.sfgs_b_R, self.sfgs_d_R, self.sgc_R, self.so_R]:
            layer.reset()
        self.rate_e_all.zero_()
        self.looming = False
