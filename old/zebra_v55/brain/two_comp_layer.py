import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================================================
# Surrogate gradient spike function
# ===============================================================
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        out = (v > 0).float()
        ctx.save_for_backward(v)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        grad = 0.3 * torch.clamp(1 - v.abs(), min=0)
        return grad_output * grad


spike_fn = SpikeFn.apply


# ===============================================================
# Two-Compartment Predictive Coding Neuron
# ===============================================================
class TwoCompLayer(nn.Module):
    """
    This is the fundamental predictive-coding spiking neuron used in v55.1.

    Compartments:
        v_b : basal dendrite prediction (feedforward)
        v_a : apical dendrite prediction (feedback)
        v_s : somatic prediction error
        spikes : output spikes (binary)

    Plasticity:
        Hebbian rule modulated by prediction errors:
            dW = pre_trace ⊗ post_error_trace
    """

    def __init__(self,
                 n_in,
                 n_out,
                 fb_dim=None,
                 tau=0.8,
                 use_feedback=False,
                 plastic=False,
                 hebb_lr=1e-5,
                 alpha=0.05,
                 mu_anchor=1e-4,
                 device="cpu"):

        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out

        # ---------------------------------------------------------
        # Predictive coding parameters
        # ---------------------------------------------------------
        self.tau = tau
        self.use_feedback = use_feedback
        self.plastic = plastic

        # feedforward weights
        self.W_ff = nn.Parameter(0.05 * torch.randn(n_in, n_out, device=device))

        # feedback weights
        if use_feedback:
            fb_dim = fb_dim if fb_dim is not None else n_out
            self.W_fb = nn.Parameter(0.05 * torch.randn(fb_dim, n_out, device=device))
        else:
            self.W_fb = None

        # plasticity parameters
        self.hebb_lr = hebb_lr
        self.alpha = alpha
        self.mu_anchor = mu_anchor
        self.W_ff_anchor = None  # stored after global phase

        # internal states
        self.reset_state(1)

    # --------------------------------------------------------------
    # Reset dynamic state
    # --------------------------------------------------------------
    def reset_state(self, B):
        dev = self.device
        out = self.n_out

        self.v_b = torch.zeros(B, out, device=dev)
        self.v_a = torch.zeros(B, out, device=dev)
        self.v_s = torch.zeros(B, out, device=dev)

        self.spikes = torch.zeros(B, out, device=dev)
        self.spike_sum = torch.zeros(B, out, device=dev)

        # traces for hebbian
        self.rbar_pre = torch.zeros(B, self.n_in, device=dev)
        self.ebar_post = torch.zeros(B, out, device=dev)

    # --------------------------------------------------------------
    def set_plastic(self, flag: bool):
        self.plastic = bool(flag)

    def cache_anchor(self):
        self.W_ff_anchor = self.W_ff.detach().clone()

    # --------------------------------------------------------------
    # One time-step update
    # --------------------------------------------------------------
    def step(self, x, fb=None):
        """
        x  : feedforward input [B, n_in]
        fb : feedback input [B, fb_dim]
        """

        # ----------------------------------------------------------
        # Basal prediction (feedforward)
        # ----------------------------------------------------------
        pred_b = x @ self.W_ff
        self.v_b = self.tau * self.v_b + (1 - self.tau) * pred_b

        # ----------------------------------------------------------
        # Apical prediction (feedback)
        # ----------------------------------------------------------
        if self.use_feedback and fb is not None and self.W_fb is not None:
            pred_a = fb @ self.W_fb
        else:
            pred_a = torch.zeros_like(self.v_a)

        self.v_a = self.tau * self.v_a + (1 - self.tau) * pred_a

        # ----------------------------------------------------------
        # Somatic prediction error
        # ----------------------------------------------------------
        self.v_s = self.tau * self.v_s + (1 - self.tau) * (self.v_b - self.v_a)

        # ----------------------------------------------------------
        # Spike generation
        # ----------------------------------------------------------
        self.spikes = spike_fn(self.v_s)
        self.spike_sum += self.spikes

        # ----------------------------------------------------------
        # Hebbian plasticity (local)
        # ----------------------------------------------------------
        if self.plastic:
            # prediction mismatch
            mismatch = self.v_s - self.v_b

            # exponential moving traces
            self.rbar_pre = (1 - self.alpha) * self.rbar_pre + self.alpha * x
            self.ebar_post = (1 - self.alpha) * self.ebar_post + self.alpha * mismatch

            # hebbian update
            dW = self.rbar_pre.transpose(1, 0) @ self.ebar_post / x.size(0)

            with torch.no_grad():
                self.W_ff += self.hebb_lr * dW

                # anchor penalty
                if self.W_ff_anchor is not None:
                    self.W_ff -= self.mu_anchor * (self.W_ff - self.W_ff_anchor)

        return self.spikes

