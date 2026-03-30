"""
Pallium: two E/I layers (superficial + deep).
Pallium-S (RS): receives thalamic relay, projects to Pallium-D.
Pallium-D (IB): goal/intent representation, projects to BG and RS.
Two-compartment predictive coding: apical feedback from D→S.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, N_PAL_S, N_PAL_D, N_TC, SUBSTEPS
from zebrav2.brain.ei_layer import EILayer

class Pallium(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.pal_s = EILayer(N_PAL_S, 'RS', device, 'Pallium-S')
        self.pal_d = EILayer(N_PAL_D, 'IB', device, 'Pallium-D')
        # Slower rate EMA for PAL-D so bursts appear intermittent in visualization
        self.pal_d.E.tau_rate = 0.005  # slower than default 0.020
        # Pallium E/I balance: E subthreshold, I light tonic, moderate I→E
        self.pal_s.syn_ie.g_bar = 1.5   # reduced from G_IE=4.0 (cortex < tectum)
        self.pal_d.syn_ie.g_bar = 1.5
        self.pal_s.E.i_tonic.fill_(-2.0)  # subthreshold
        self.pal_d.E.i_tonic.fill_(-5.0)  # well below rheobase → intermittent bursting
        self.pal_s.I.i_tonic.fill_(2.0)   # stronger baseline for real E/I dynamics
        self.pal_d.I.i_tonic.fill_(2.0)
        # Feedforward: thalamus TC → pallium-S
        self.W_tc_pals  = nn.Linear(N_TC, self.pal_s.n_e, bias=False)
        # Feedforward: pallium-S → pallium-D
        self.W_pals_pald = nn.Linear(self.pal_s.n_e, self.pal_d.n_e, bias=False)
        # Feedback: pallium-D → pallium-S (predictive coding)
        self.W_FB = nn.Linear(self.pal_d.n_e, self.pal_s.n_e, bias=False)
        # Attention: goal → pallium (additive somatic bias)
        self.W_goal_att = nn.Linear(4, self.pal_s.n_e, bias=False)
        from zebrav2.brain.connectome import init_connectome_weights
        for W in [self.W_tc_pals, self.W_pals_pald, self.W_FB, self.W_goal_att]:
            nn.init.xavier_uniform_(W.weight, gain=1.0)
            W.to(device)
        init_connectome_weights(self.W_tc_pals.weight, 'thalamus', 'pallium', 1.0)
        init_connectome_weights(self.W_FB.weight, 'pallium', 'pallium', 0.5)
        # Prediction error (apical - somatic)
        self.register_buffer('pred_error', torch.zeros(self.pal_s.n_e, device=device))
        self.register_buffer('rate_s', torch.zeros(self.pal_s.n_e, device=device))
        self.register_buffer('rate_d', torch.zeros(self.pal_d.n_e, device=device))
        # Apical compartment (top-down prediction)
        self.register_buffer('apical_s', torch.zeros(self.pal_s.n_e, device=device))

    def forward(self, tc_rate: torch.Tensor, goal_probs: torch.Tensor,
                ACh_level: float = 0.5) -> dict:
        """
        tc_rate: (N_TC,) thalamic relay rate
        goal_probs: (4,) [forage, flee, explore, social]
        """
        # Normalized feedforward drive
        def _nd(W, x, tgt=5.0):
            r = W(x.unsqueeze(0)).squeeze(0).detach()
            m = r.abs().mean() + 1e-8
            return r * (tgt / m) if m > 0.001 else r * 0.0

        I_ff_s = _nd(self.W_tc_pals, tc_rate, 12.0)  # boosted for E/I balance
        fb_drive = _nd(self.W_FB, self.pal_d.get_rate_e(), 2.0)
        att = _nd(self.W_goal_att, goal_probs, 2.0) * ACh_level
        I_s_total = I_ff_s + fb_drive + att
        rate_s, _, _, _ = self.pal_s(I_s_total, substeps=SUBSTEPS)
        # Prediction error: apical (top-down) - somatic (bottom-up)
        self.apical_s.copy_(fb_drive.clamp(0, 1))
        self.pred_error.copy_((self.apical_s - rate_s).clamp(-1, 1))
        I_ff_d = _nd(self.W_pals_pald, rate_s, 4.0)
        I_ff_d = I_ff_d  # PAL-D receives full drive; intermittency from IB burst dynamics
        rate_d, _, _, _ = self.pal_d(I_ff_d, substeps=SUBSTEPS)
        self.rate_s.copy_(rate_s)
        self.rate_d.copy_(rate_d)
        return {
            'rate_S': rate_s, 'rate_D': rate_d,
            'pred_error': self.pred_error,
            'free_energy': (self.pred_error**2).mean().item(),
        }

    def reset(self):
        self.pal_s.reset()
        self.pal_d.reset()
        self.pred_error.zero_()
        self.rate_s.zero_()
        self.rate_d.zero_()
        self.apical_s.zero_()
