"""
Optic tectum: 4 E/I layers (SFGS-b, SFGS-d, SGC, SO).
Receives 4-type RGC input. Projects to pretectum and reticulospinal.
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
        # 4 tectal layers
        self.sfgs_b = EILayer(N_OT_SFGS_B, 'CH', device, 'SFGS-b')
        self.sfgs_d = EILayer(N_OT_SFGS_D, 'CH', device, 'SFGS-d')
        self.sgc    = EILayer(N_OT_SGC,    'IB', device, 'SGC')
        self.so     = EILayer(N_OT_SO,     'RS', device, 'SO')
        # Input projection weights — sizes from spec
        n_on   = N_RET_PER_TYPE  # fused bilateral → pooled
        n_off  = N_RET_PER_TYPE
        n_loom = N_RET_LOOM
        n_ds   = N_RET_DS
        self.W_on_sfgsb = nn.Linear(n_on, self.sfgs_b.n_e, bias=False)
        self.W_off_sfgsd = nn.Linear(n_off, self.sfgs_d.n_e, bias=False)
        self.W_loom_sgc  = nn.Linear(n_loom, self.sgc.n_e,  bias=False)
        self.W_ds_so     = nn.Linear(n_ds,   self.so.n_e,   bias=False)
        # Initialize with connectome-constrained connectivity
        from zebrav2.brain.connectome import init_connectome_weights
        for W in [self.W_on_sfgsb, self.W_off_sfgsd, self.W_loom_sgc, self.W_ds_so]:
            nn.init.xavier_uniform_(W.weight, gain=1.0)
            W.to(device)
        init_connectome_weights(self.W_on_sfgsb.weight, 'retina', 'tectum', 1.0)
        init_connectome_weights(self.W_off_sfgsd.weight, 'retina', 'tectum', 1.0)
        # Tectum: E neurons have strong negative tonic (well below rheobase ~3.8 pA)
        # Only strong stimulus overcomes this → sparse, input-dependent firing
        # Tonic: E=-2 so avg neuron at net 1 pA (subthreshold), top quartile >3.8 (fires)
        for layer in [self.sfgs_b, self.sfgs_d, self.sgc, self.so]:
            layer.E.i_tonic.fill_(-2.0)
            layer.I.i_tonic.fill_(0.5)
        # Fused output (rate of all E neurons concatenated)
        total_e = self.sfgs_b.n_e + self.sfgs_d.n_e + self.sgc.n_e + self.so.n_e
        self.register_buffer('rate_e_all', torch.zeros(total_e, device=device))
        # Looming flag
        self.looming = False

    def forward(self, rgc_out: dict, I_topdown: torch.Tensor = None) -> dict:
        """
        rgc_out: dict from RetinaV2
        I_topdown: optional top-down current from pallium (sfgs_b.n_e,)
        """
        # Drive each layer: normalize and clamp so sparse subset fires
        # target_pA = mean current; max clamped to target + rheobase offset
        def _norm_drive(W, x, target_pA=2.0, max_pA=3.5):
            raw = W(x.unsqueeze(0)).squeeze(0).detach()
            raw_mean = raw.abs().mean() + 1e-8
            scaled = raw * (target_pA / raw_mean)
            return scaled.clamp(-max_pA, max_pA)

        I_sfgsb = _norm_drive(self.W_on_sfgsb, rgc_out['on_fused'], 3.0, 6.0)
        I_sfgsd = _norm_drive(self.W_off_sfgsd, rgc_out['off_fused'], 3.0, 6.0)
        I_sgc   = _norm_drive(self.W_loom_sgc, rgc_out['loom_fused'], 4.0, 8.0)
        I_so    = _norm_drive(self.W_ds_so, rgc_out['ds_fused'], 3.0, 6.0)
        if I_topdown is not None:
            I_sfgsb = I_sfgsb + I_topdown[:self.sfgs_b.n_e]
        # Run E/I dynamics
        rate_b, _, _, _ = self.sfgs_b(I_sfgsb, substeps=SUBSTEPS)
        rate_d, _, _, _ = self.sfgs_d(I_sfgsd, substeps=SUBSTEPS)
        rate_g, _, _, _ = self.sgc(I_sgc, substeps=SUBSTEPS)
        rate_o, _, _, _ = self.so(I_so, substeps=SUBSTEPS)
        # Looming detection from SGC
        self.looming = rate_g.mean().item() > 0.05
        # Concatenate all E rates
        self.rate_e_all.copy_(torch.cat([rate_b, rate_d, rate_g, rate_o]))
        return {
            'sfgs_b': rate_b, 'sfgs_d': rate_d,
            'sgc': rate_g, 'so': rate_o,
            'all_e': self.rate_e_all,
            'looming': self.looming,
        }

    def reset(self):
        for layer in [self.sfgs_b, self.sfgs_d, self.sgc, self.so]:
            layer.reset()
        self.rate_e_all.zero_()
        self.looming = False
