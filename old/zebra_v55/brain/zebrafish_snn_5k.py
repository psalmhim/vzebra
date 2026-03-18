import torch
import torch.nn as nn
import numpy as np
import math

from .retina_sampling import sample_retina_binocular
from .biological_wiring import (
    apply_topographic_wiring,
    apply_biological_directionality,
    decode_brain_output
)


# ======================================================================
# Spike
# ======================================================================
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


# ======================================================================
# TwoCompLayer
# ======================================================================
class TwoCompLayer(nn.Module):
    def __init__(self, n_in, n_out, tau=0.8, device="cpu",
                 use_feedback=True, fb_dim=None):
        super().__init__()
        self.device = device
        self.tau = tau
        self.use_feedback = use_feedback

        self.W_ff = nn.Parameter(0.05 * torch.randn(n_in, n_out, device=device))
        self.W_fb = nn.Parameter(0.05 * torch.randn(fb_dim, n_out, device=device)) \
                    if use_feedback and fb_dim is not None else None

        self.reset_state(1)

    def reset_state(self, B):
        out = self.W_ff.shape[1]
        dev = self.device
        self.v_b = torch.zeros(B, out, device=dev)
        self.v_a = torch.zeros(B, out, device=dev)
        self.v_s = torch.zeros(B, out, device=dev)
        self.spikes = torch.zeros(B, out, device=dev)

    def step(self, x, fb=None):
        self.v_b = self.tau*self.v_b + (1-self.tau)*(x @ self.W_ff)
        pred_a = fb @ self.W_fb if (self.use_feedback and fb is not None and self.W_fb is not None) else 0
        self.v_a = self.tau*self.v_a + (1-self.tau)*pred_a
        self.v_s = self.tau*self.v_s + (1-self.tau)*(self.v_b - self.v_a)
        self.spikes = spike_fn(self.v_s)
        return self.spikes


# ======================================================================
# Full PC-SNN Brain
# ======================================================================
class ZebrafishSNN_5k(nn.Module):
    def __init__(self, device="cpu", use_sparse_wiring=True):
        super().__init__()
        self.device = device
        self.use_sparse_wiring = use_sparse_wiring

        # Retina → OT sizes
        self.RET = 800
        self.OTL = 600
        self.OTR = 600
        self.OTF = 800
        self.PT  = 400
        self.PC_PER = 120
        self.PC_INT = 30

        # Layers
        self.OT_L = TwoCompLayer(400, self.OTL, device=device, use_feedback=False)
        self.OT_R = TwoCompLayer(400, self.OTR, device=device, use_feedback=False)
        self.OT_fused = TwoCompLayer(self.OTL + self.OTR, self.OTF,
                                     device=device, use_feedback=False)
        self.PT_layer = TwoCompLayer(self.OTF, self.PT, device=device, use_feedback=False)

        self.PC_perc = TwoCompLayer(self.PT, self.PC_PER, device=device,
                                    use_feedback=True, fb_dim=self.PC_INT)
        self.PC_intent = TwoCompLayer(self.PC_PER, self.PC_INT, device=device,
                                      use_feedback=False)

        # Motor heads
        self.head_motor = nn.Linear(self.PC_INT, 200)
        self.head_CPG   = nn.Linear(self.PC_INT, 200)
        self.head_eye   = nn.Linear(self.PC_INT, 100)
        self.head_DA    = nn.Linear(self.PC_INT, 50)

        self.reset_state(1)
        self.to(device)

    def reset_state(self, B):
        for L in [self.OT_L, self.OT_R, self.OT_fused,
                  self.PT_layer, self.PC_perc, self.PC_intent]:
            L.reset_state(B)

    @torch.no_grad()
    def forward_brain(self, retina):
        L = retina[:, :400]
        R = retina[:, 400:]

        otL = self.OT_L.step(L)
        otR = self.OT_R.step(R)

        fused = torch.cat([otL, otR], dim=1)
        otF = self.OT_fused.step(fused)

        pt = self.PT_layer.step(otF)

        fb_int = self.PC_intent.spikes.detach()
        per = self.PC_perc.step(pt, fb=fb_int)
        intent = self.PC_intent.step(per)

        return {
            "motor": self.head_motor(intent),
            "CPG":   self.head_CPG(intent),
            "eye":   self.head_eye(intent),
            "DA":    self.head_DA(intent),
            "per": per,
            "intent": intent
        }

    @torch.no_grad()
    def step_with_retina(self, position, heading, world, T=1):
        retL, retR = sample_retina_binocular(position, heading, world, device=self.device)
        retina = torch.cat([retL, retR], dim=1)
        for _ in range(T):
            out = self.forward_brain(retina)
        return out
