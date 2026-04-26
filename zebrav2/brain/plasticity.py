"""
Three-factor STDP with eligibility traces.
Also: anti-Hebbian feedback weight learning (PE-driven).
Homeostatic synaptic scaling.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, DT, TAU_ELIG

class EligibilitySTDP(nn.Module):
    """
    Three-factor STDP for a feedforward weight matrix W (post x pre).
    e_trace accumulates Hebbian co-activation, DA consolidates into weight.
    """
    def __init__(self, W: nn.Parameter, device=DEVICE,
                 A_plus=0.005, A_minus=0.005,
                 tau_plus=0.020, tau_minus=0.020,
                 w_max=2.0, w_min=-1.0):
        super().__init__()
        self.W = W
        self.device = device
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.decay_elig = float(torch.exp(torch.tensor(-DT / TAU_ELIG)))
        self.decay_pre = float(torch.exp(torch.tensor(-DT / tau_plus)))
        self.decay_post = float(torch.exp(torch.tensor(-DT / tau_minus)))
        self.w_max = w_max
        self.w_min = w_min
        post_n, pre_n = W.shape
        self.register_buffer('e_trace', torch.zeros(post_n, pre_n, device=device))
        self.register_buffer('pre_trace', torch.zeros(pre_n, device=device))
        self.register_buffer('post_trace', torch.zeros(post_n, device=device))
        # Homeostatic rate tracker
        self.register_buffer('post_rate_ema', torch.zeros(post_n, device=device))
        self.R_TARGET = 0.05

    def update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Update eligibility trace (call once per ms substep)."""
        self.pre_trace.copy_(self.pre_trace * self.decay_pre + pre_spikes)
        self.post_trace.copy_(self.post_trace * self.decay_post + post_spikes)
        # STDP kernel: LTP when post fires (pre_trace nonzero), LTD when pre fires
        dW = (self.A_plus * torch.outer(post_spikes, self.pre_trace)
             - self.A_minus * torch.outer(self.post_trace, pre_spikes))
        # Update eligibility trace
        self.e_trace.copy_(self.e_trace * self.decay_elig + dW)

    def consolidate(self, DA: float, ACh: float = 1.0, eta: float = 0.001,
                    dropout_p: float = 0.0):
        """Apply eligibility trace with DA as third factor.

        dropout_p: fraction of synapses randomly silenced per consolidation
        step.  Prevents co-adaptation of synapses and forces distributed
        representations (analogous to biological synaptic unreliability).
        """
        dW = eta * DA * ACh * self.e_trace
        if dropout_p > 0.0:
            mask = torch.rand_like(dW) >= dropout_p
            dW = dW * mask
        with torch.no_grad():
            self.W.data.add_(dW)
            self.W.data.clamp_(self.w_min, self.w_max)

    def homeostatic_scale(self, post_spikes: torch.Tensor, alpha: float = 1e-5):
        """Slow homeostatic normalization of postsynaptic weights."""
        self.post_rate_ema.copy_((1 - alpha) * self.post_rate_ema + alpha * post_spikes)
        scale = (self.R_TARGET / (self.post_rate_ema + 1e-6)).clamp(0.5, 2.0)
        with torch.no_grad():
            self.W.data.mul_(scale.unsqueeze(1))

    def reset(self):
        self.e_trace.zero_()
        self.pre_trace.zero_()
        self.post_trace.zero_()


class FeedbackPELearning(nn.Module):
    """
    Anti-Hebbian feedback weight learning: minimize prediction error.
    ΔW_FB = -η * h_upper.T @ ε (reward-independent)
    """
    def __init__(self, W_FB: nn.Parameter, eta: float = 2e-4,
                 w_max: float = 0.10, device=DEVICE):
        super().__init__()
        self.W_FB = W_FB
        self.eta = eta
        self.w_max = w_max
        self.device = device

    def update(self, h_upper: torch.Tensor, pred_error: torch.Tensor,
               pi: float = 1.0):
        """
        h_upper: (n_upper,) higher layer activity
        pred_error: (n_lower,) apical - somatic mismatch
        """
        dW = -self.eta * pi * torch.outer(pred_error, h_upper)
        with torch.no_grad():
            self.W_FB.data.add_(dW)
            self.W_FB.data.clamp_(-self.w_max, self.w_max)
