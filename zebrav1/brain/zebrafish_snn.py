"""
Zebrafish SNN v1 — Two-Compartment Predictive Coding with Attention.

Based on Lee, Lee & Park (2026) "Modulation of bias in predictive-coding
spiking neural networks for selective attention."

Each trainable neuron has two compartments:
  - Soma (V_s): integrates bottom-up feedforward input + bounded apical
    contribution + attention bias
  - Apical dendrite (V_a): integrates top-down feedback from the next
    higher layer (1-step delayed)

Prediction error = V_a - V_s (apical-somatic mismatch).
Attention = additive somatic bias from goal-driven attention neurons.

OT_L / OT_R remain as simple TwoComp (frozen topographic maps).
"""

import torch
import torch.nn as nn
from .retina_sampling import sample_retina_binocular
from .tectum_topography import apply_tectal_topography
from .pc_precision import PrecisionUnit
from .lateral_inhibition import lateral_inhibition_pair
from .saccade_module import SaccadeStabilizer


# ---------------------------------------------------------------------------
# Neuron models
# ---------------------------------------------------------------------------

class TwoComp(nn.Module):
    """Original single-compartment leaky integrator (used for frozen OT_L/OT_R)."""

    def __init__(self, n_in, n_out, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(n_in, n_out, device=device))
        self.v = torch.zeros(1, n_out, device=device)
        self.device = device

    def step(self, x):
        self.v = 0.8 * self.v + 0.2 * (x @ self.W)
        return self.v


class PredictiveTwoComp(nn.Module):
    """Two-compartment predictive coding neuron with attention modulation.

    Dynamics (per timestep):
        V_a = tau_a * V_a + (1 - tau_a) * (fb @ W_FB)        # apical
        V_s = tau_s * V_s + (1 - tau_s) * (x @ W_FF + f(V_a) + alpha * m)  # soma
        PE  = V_a - V_s                                       # prediction error

    f_apical(x) = 0.5 * (sigmoid(x) - 0.5)  bounded to [-0.25, +0.25]

    The apical compartment is updated AFTER the somatic step using the
    current higher-layer output, so feedback is effectively 1-step delayed.
    """

    TARGET_RMS = 5.0  # homeostatic gain control target

    def __init__(self, n_in, n_out, n_fb=0, device="cpu",
                 tau_s=0.8, tau_a=0.65, alpha_att=0.1):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        self.tau_s = tau_s
        self.tau_a = tau_a
        self.alpha_att = alpha_att

        # Feedforward weight (same role as old TwoComp.W)
        self.W_FF = nn.Parameter(
            0.01 * torch.randn(n_in, n_out, device=device))

        # Feedback weight (top-down from higher layer)
        # 2x W_FF scale: feedback source has fewer neurons than feedforward,
        # so W_FB must be larger to produce comparable apical drive
        if n_fb > 0:
            self.W_FB = nn.Parameter(
                0.02 * torch.randn(n_fb, n_out, device=device))
        else:
            self.W_FB = None

        # State tensors (not parameters — reset each episode)
        self.v_s = torch.zeros(1, n_out, device=device)       # somatic
        self.v_a = torch.zeros(1, n_out, device=device)       # apical
        self.m_att = torch.zeros(1, n_out, device=device)     # attention mod
        self.pred_error = torch.zeros(1, n_out, device=device)

        # Backward-compatible alias
        self.v = self.v_s

    def reset(self):
        """Reset all membrane states to zero."""
        self.v_s = torch.zeros(1, self.n_out, device=self.device)
        self.v_a = torch.zeros(1, self.n_out, device=self.device)
        self.m_att = torch.zeros(1, self.n_out, device=self.device)
        self.pred_error = torch.zeros(1, self.n_out, device=self.device)
        self.v = self.v_s

    @staticmethod
    def f_apical(x):
        """Bounded apical-to-soma coupling: [-0.25, +0.25]."""
        return 0.5 * (torch.sigmoid(x) - 0.5)

    def step(self, x, att_proj=None):
        """Somatic forward step (bottom-up + apical + attention).

        Args:
            x: bottom-up input [1, n_in]
            att_proj: per-neuron attention projection [1, n_out] or None
        Returns:
            v_s: somatic membrane potential [1, n_out]
        """
        # Attention modulator update (slow dynamics)
        if att_proj is not None:
            self.m_att = 0.9 * self.m_att + 0.1 * att_proj

        # Bounded apical contribution (from previous step's feedback)
        apical_drive = self.f_apical(self.v_a)

        # Somatic integration: feedforward + apical + attention
        ff_drive = x @ self.W_FF
        self.v_s = (self.tau_s * self.v_s
                    + (1.0 - self.tau_s) * ff_drive
                    + apical_drive
                    + self.alpha_att * self.m_att)
        # Homeostatic gain control: suppress overactive layers.
        # Deep layers (PC_int, motor) have low activity by design —
        # their signal reaches motor output via the reticulospinal
        # shortcut (OT_L/R → motor) rather than the deep pathway.
        v_rms = (self.v_s ** 2).mean().sqrt()
        if v_rms > self.TARGET_RMS:
            self.v_s = self.v_s * (self.TARGET_RMS / v_rms)

        # Compute prediction error (apical prediction - somatic evidence)
        self.pred_error = self.v_a - self.v_s

        # Keep backward-compatible .v alias
        self.v = self.v_s
        return self.v_s

    def update_apical(self, fb):
        """Update apical dendrite with top-down feedback.

        Called AFTER the feedforward pass, so this feedback takes effect
        on the NEXT timestep's somatic computation (1-step delay).

        Args:
            fb: feedback from the next higher layer [1, n_fb]
        """
        if self.W_FB is not None:
            self._last_fb = fb.detach()  # store for PE minimization
            self.v_a = (self.tau_a * self.v_a
                        + (1.0 - self.tau_a) * (fb @ self.W_FB))
            # Clamp apical to prevent runaway
            self.v_a = torch.clamp(self.v_a, -5.0, 5.0)


class IzhikevichTwoComp(nn.Module):
    """Two-compartment predictive coding with Izhikevich spike generation.

    Combines:
      - Apical dendrite (V_a): continuous top-down feedback, unchanged from
        PredictiveTwoComp — dendrites have graded potentials in biology.
      - Soma: Izhikevich (v, u) dynamics for realistic spike generation.
        v' = 0.04v² + 5v + 140 - u + I    (Izhikevich 2003)
        u' = a(bv - u)
        if v ≥ 30: v ← c, u ← u + d       (spike + reset)

    Prediction error PE = V_a - norm(v_s), as in PredictiveTwoComp but
    normalised to comparable scale (soma in mV, apical in small units).

    Firing-type presets (a, b, c, d):
      rs  (0.02, 0.20, -65, 8)  — regular spiking       (PT_L, PC_per)
      fs  (0.10, 0.20, -65, 2)  — fast spiking           (OT_F)
      ch  (0.02, 0.20, -50, 2)  — chattering             (PC_int)
      ib  (0.02, 0.20, -55, 4)  — intrinsically bursting (DA, mot, eye)
    """

    V_THRESH = 30.0    # spike threshold (mV)
    V_REST   = -65.0   # resting potential (mV)
    V_SCALE  = 8.0     # scale W_FF output → mV input current

    PRESETS = {
        "rs": (0.02, 0.20, -65.0,  8.0),
        "fs": (0.10, 0.20, -65.0,  2.0),
        "ch": (0.02, 0.20, -50.0,  2.0),
        "ib": (0.02, 0.20, -55.0,  4.0),
    }

    # Keep TARGET_RMS for compatibility with any code that reads it,
    # but homeostatic gain control is disabled (Izhikevich self-regulates).
    TARGET_RMS = 5.0

    def __init__(self, n_in, n_out, n_fb=0, device="cpu",
                 tau_a=0.65, alpha_att=0.1, izhi_type="rs"):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        self.tau_a = tau_a
        self.alpha_att = alpha_att

        a, b, c, d = self.PRESETS[izhi_type]
        self.izhi_a = a
        self.izhi_b = b
        self.izhi_c = c
        self.izhi_d = d

        # Feedforward weight (same as PredictiveTwoComp)
        self.W_FF = nn.Parameter(
            0.01 * torch.randn(n_in, n_out, device=device))

        # Tonic bias: subthreshold drive keeps neurons near threshold.
        # Izhikevich I_crit ≈ 4.0 mV for all presets (b=0.2).
        # Bias = 3.5 mV keeps neurons ~0.5 mV below critical point.
        # Neurons only fire when driven by upstream input (no spontaneous noise),
        # while remaining near threshold for fast response to visual drive.
        self.bias = nn.Parameter(
            torch.full((n_out,), 3.5, device=device))

        # Feedback weight
        if n_fb > 0:
            self.W_FB = nn.Parameter(
                0.02 * torch.randn(n_fb, n_out, device=device))
        else:
            self.W_FB = None

        # State tensors (not parameters — reset each episode)
        self._init_state()

    def _init_state(self):
        """Initialise / reset all membrane state tensors."""
        dev = self.device
        self.v_s  = torch.full((1, self.n_out), self.V_REST,  device=dev)
        self.u    = torch.full((1, self.n_out), self.izhi_b * self.V_REST, device=dev)
        self.spikes = torch.zeros(1, self.n_out, device=dev)
        self.v_a  = torch.zeros(1, self.n_out,   device=dev)
        self.m_att = torch.zeros(1, self.n_out,  device=dev)
        self.pred_error = torch.zeros(1, self.n_out, device=dev)
        self.v = self.v_s   # backward-compatible alias

    def reset(self):
        self._init_state()

    @staticmethod
    def f_apical(x):
        """Bounded apical-to-soma coupling: [-0.25, +0.25]."""
        return 0.5 * (torch.sigmoid(x) - 0.5)

    def step(self, x, att_proj=None):
        """One timestep: Izhikevich soma + continuous apical.

        Args:
            x: bottom-up input [1, n_in]
            att_proj: attention projection [1, n_out] or None
        Returns:
            v_s: somatic membrane potential [1, n_out] (mV)
        """
        # Attention modulator (slow dynamics, same as PredictiveTwoComp)
        if att_proj is not None:
            self.m_att = 0.9 * self.m_att + 0.1 * att_proj

        # Input current: feedforward scaled to mV + tonic bias + apical + attention
        apical_drive = self.f_apical(self.v_a) * 20.0   # scale to mV
        I = ((x @ self.W_FF) * self.V_SCALE
             + self.bias
             + apical_drive
             + self.alpha_att * self.m_att * 10.0)

        # Izhikevich dynamics — 2 Euler half-steps for numerical stability
        v, u = self.v_s, self.u
        for _ in range(2):
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
            du = self.izhi_a * (self.izhi_b * v - u)
            v = v + 0.5 * dv
            u = u + 0.5 * du

        # Spike detection and reset
        fired = (v >= self.V_THRESH)
        self.spikes = fired.float()
        v = torch.where(fired, torch.full_like(v, self.izhi_c), v)
        u = torch.where(fired, u + self.izhi_d, u)

        # Clamp for numerical stability
        v = torch.clamp(v, -100.0, 35.0)
        u = torch.clamp(u, -100.0, 100.0)

        self.v_s = v
        self.u   = u

        # Normalise soma to [0, 1]: 0 at rest (-65 mV), 1 at spike threshold (30 mV).
        # This keeps inter-layer inputs bounded regardless of mV scale,
        # preventing noise amplification through random W_FF chains.
        v_norm = torch.clamp(
            (self.v_s - self.V_REST) / (self.V_THRESH - self.V_REST),
            0.0, 1.0)

        # Prediction error: apical (small units) - normalised soma (0..1)
        self.pred_error = self.v_a - v_norm

        self.v = v_norm   # backward-compatible alias → always bounded
        return v_norm     # return normalised for inter-layer communication

    def update_apical(self, fb):
        """Update apical dendrite with 1-step delayed top-down feedback."""
        if self.W_FB is not None:
            self._last_fb = fb.detach()
            self.v_a = (self.tau_a * self.v_a
                        + (1.0 - self.tau_a) * (fb @ self.W_FB))
            self.v_a = torch.clamp(self.v_a, -5.0, 5.0)


class AttentionModulator(nn.Module):
    """Goal-driven attention modulation for the SNN hierarchy.

    Maps 4 goal probabilities → 8 attention neurons → per-layer projections.
    Implements slow neuromodulatory dynamics (tau_att >> tau_s).

    Based on Lee et al.: attention as additive somatic excitability bias.
    """

    N_GOALS = 4
    N_ATT = 8  # 2 attention neurons per goal

    def __init__(self, n_goals=4, device="cpu"):
        super().__init__()
        self.device = device

        # Goal → attention neuron weights
        # Structured init: each goal gets 2 dedicated neurons
        W_att_init = torch.zeros(n_goals, self.N_ATT, device=device)
        for g in range(n_goals):
            W_att_init[g, g * 2] = 0.1
            W_att_init[g, g * 2 + 1] = 0.1
        self.W_att = nn.Parameter(W_att_init)

        # Attention neuron state (slow dynamics)
        self.m = torch.zeros(1, self.N_ATT, device=device)
        self.tau_att = 0.93   # slow: ~14 step half-life

        # Per-layer projection weights
        self.W_proj = nn.ModuleDict()
        self._last_goal = None

        self.to(device)

    def register_layer(self, name, n_out):
        """Register a target layer for attention projection."""
        self.W_proj[name] = nn.Linear(self.N_ATT, n_out, bias=False)
        nn.init.normal_(self.W_proj[name].weight, std=0.01)
        self.W_proj[name].to(self.device)

    def step(self, goal_probs):
        """Update attention neuron state.

        Args:
            goal_probs: [1, 4] goal probability tensor
        Returns:
            m: [1, N_ATT] attention neuron activations
        """
        self._last_goal = goal_probs.detach()
        drive = goal_probs @ self.W_att   # [1, N_ATT]
        self.m = self.tau_att * self.m + (1.0 - self.tau_att) * drive
        return self.m

    def project(self, layer_name):
        """Project attention state to a specific layer.

        Returns:
            [1, n_out] attention signal for that layer, or None
        """
        if layer_name not in self.W_proj:
            return None
        return self.W_proj[layer_name](self.m)  # [1, n_out]

    def reset(self):
        """Reset attention neuron state."""
        self.m = torch.zeros(1, self.N_ATT, device=self.device)
        self._last_goal = None


# ---------------------------------------------------------------------------
# Main SNN model
# ---------------------------------------------------------------------------

class ZebrafishSNN(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        # Retina: 800 per eye (400 intensity + 400 type channel)
        self.RET = 800
        self.OTL = 600
        self.OTR = 600
        self.OTF = 800
        self.PT = 400
        self.PC_PER = 120
        self.PC_INT = 30

        # --- Frozen layers (simple TwoComp, topographic map) ---
        self.OT_L = TwoComp(self.RET, self.OTL, device)
        self.OT_R = TwoComp(self.RET, self.OTR, device)

        # --- Trainable layers (IzhikevichTwoComp: real spikes + PC) ---
        # Layer-specific firing types match biological zebrafish circuits:
        #   OT_F: fast-spiking interneurons (tectal GABAergic)
        #   PT_L: regular spiking pretectal projection neurons
        #   PC_per: chattering cortical-like PC neurons
        #   PC_int: intrinsically bursting intent layer
        #   mot/eye/DA: intrinsically bursting output heads
        self.OT_F = IzhikevichTwoComp(
            self.OTL + self.OTR, self.OTF, n_fb=self.PT,
            device=device, izhi_type="fs")
        self.PT_L = IzhikevichTwoComp(
            self.OTF, self.PT, n_fb=self.PC_PER,
            device=device, izhi_type="rs")
        self.PC_per = IzhikevichTwoComp(
            self.PT, self.PC_PER, n_fb=self.PC_INT,
            device=device, izhi_type="ch")
        self.PC_int = IzhikevichTwoComp(
            self.PC_PER, self.PC_INT, n_fb=50,
            device=device, izhi_type="ch")

        # precision units
        self.prec_OT = PrecisionUnit(self.OTF, device)
        self.prec_PC = PrecisionUnit(self.PC_PER, device)

        # output heads (no higher layer → n_fb=0)
        # Bias=0: output heads should only fire when driven by PC_int,
        # not spontaneously — spontaneous motor noise creates turn bias artifacts.
        self.mot = IzhikevichTwoComp(
            self.PC_INT, 200, n_fb=0, device=device, izhi_type="ib")
        self.eye = IzhikevichTwoComp(
            self.PC_INT, 100, n_fb=0, device=device, izhi_type="ib")
        self.DA = IzhikevichTwoComp(
            self.PC_INT, 50,  n_fb=0, device=device, izhi_type="ib")
        # Output heads: same subthreshold bias (3.5) — only fire when PC_int drives them
        # (no spontaneous activity → no random motor turn bias)

        # Reticulospinal shortcut: crossed OT_L/R → motor (trainable)
        # Left tectum → right motor, right tectum → left motor
        self.reticulo_L = nn.Linear(self.OTL, 100, bias=False)
        self.reticulo_R = nn.Linear(self.OTR, 100, bias=False)
        nn.init.zeros_(self.reticulo_L.weight)  # start at zero, train online
        nn.init.zeros_(self.reticulo_R.weight)

        # Classification head (800 type pixels + 4 aggregate pixel counts)
        self.cls_hidden = nn.Linear(804, 128)
        self.cls_out = nn.Linear(128, 5)

        self.saccade = SaccadeStabilizer()

        # Attention modulator
        self.attention = AttentionModulator(n_goals=4, device=device)
        self.attention.register_layer("OT_F", self.OTF)
        self.attention.register_layer("PT_L", self.PT)
        self.attention.register_layer("PC_per", self.PC_PER)
        self.attention.register_layer("PC_int", self.PC_INT)

        # Move all parameters to the target device
        self.to(device)

        apply_tectal_topography(self)

    def reset(self):
        """Reset all state tensors, detaching from any computation graph."""
        self.OT_L.v = torch.zeros(1, self.OTL, device=self.device)
        self.OT_R.v = torch.zeros(1, self.OTR, device=self.device)
        for layer in [self.OT_F, self.PT_L, self.PC_per, self.PC_int,
                      self.mot, self.eye, self.DA]:
            layer.reset()
        self.attention.reset()

    def forward(self, pos, heading, world, depth_shading=False,
                depth_scale=80.0, max_dist=200, goal_probs=None):
        # 1. Retinal sampling
        L, R = sample_retina_binocular(
            pos, heading, world, device=self.device,
            depth_shading=depth_shading, depth_scale=depth_scale,
            max_dist=max_dist)

        # 2. OT left/right (frozen TwoComp)
        oL = self.OT_L.step(L)
        oR = self.OT_R.step(R)
        oL, oR = lateral_inhibition_pair(oL, oR, lam=0.25)
        fused = torch.cat([oL, oR], dim=1)

        # 3. Attention modulator (goal-driven)
        if goal_probs is None:
            goal_probs = torch.zeros(1, 4, device=self.device)
        att = self.attention.step(goal_probs)

        # 4. Feedforward pass (soma uses previous step's apical state)
        att_OTF = self.attention.project("OT_F")
        oF = self.OT_F.step(fused, att_proj=att_OTF)

        pi_OT = self.prec_OT.compute_pi()
        oF_weighted = pi_OT * oF

        att_PT = self.attention.project("PT_L")
        pt = self.PT_L.step(oF_weighted, att_proj=att_PT)

        att_PC_per = self.attention.project("PC_per")
        per = self.PC_per.step(pt, att_proj=att_PC_per)
        pi_PC = self.prec_PC.compute_pi()
        per = pi_PC * per

        att_PC_int = self.attention.project("PC_int")
        intent = self.PC_int.step(per, att_proj=att_PC_int)

        m = self.mot.step(intent)
        e = self.eye.step(intent)
        d = self.DA.step(intent)

        # Reticulospinal: crossed OT_L/R → motor L/R (trained shortcut)
        retic_from_L = self.reticulo_L(oL)  # left tectum → right motor
        retic_from_R = self.reticulo_R(oR)  # right tectum → left motor
        retic_motor = torch.cat([retic_from_R, retic_from_L], dim=1)
        m = m + retic_motor  # blend deep + shortcut

        # 5. Feedback pass (top-down, update apical for NEXT timestep)
        self.PC_int.update_apical(d)         # DA → PC_int
        self.PC_per.update_apical(intent)    # PC_int → PC_per
        self.PT_L.update_apical(per)         # PC_per → PT_L
        self.OT_F.update_apical(pt)          # PT_L → OT_F

        # 6. Classification
        type_L = L[:, 400:]
        type_R = R[:, 400:]
        type_features = torch.cat([type_L, type_R], dim=1)  # [1, 800]
        # Add aggregate pixel counts (obstacle vs enemy disambiguation)
        obs_count = ((torch.abs(type_L - 0.75) < 0.1).float().sum(dim=1, keepdim=True)
                     + (torch.abs(type_R - 0.75) < 0.1).float().sum(dim=1, keepdim=True))
        ene_count = ((torch.abs(type_L - 0.5) < 0.1).float().sum(dim=1, keepdim=True)
                     + (torch.abs(type_R - 0.5) < 0.1).float().sum(dim=1, keepdim=True))
        food_count = ((torch.abs(type_L - 1.0) < 0.1).float().sum(dim=1, keepdim=True)
                      + (torch.abs(type_R - 1.0) < 0.1).float().sum(dim=1, keepdim=True))
        boundary_count = ((torch.abs(type_L - 0.12) < 0.05).float().sum(dim=1, keepdim=True)
                          + (torch.abs(type_R - 0.12) < 0.05).float().sum(dim=1, keepdim=True))
        # Normalize counts to [0, 1] range (max ~400 pixels per eye)
        scale = 1.0 / 50.0
        agg = torch.cat([obs_count * scale, ene_count * scale,
                         food_count * scale, boundary_count * scale], dim=1)
        cls_input = torch.cat([type_features, agg], dim=1)  # [1, 804]
        cls_h = torch.relu(self.cls_hidden(cls_input))
        cls = self.cls_out(cls_h)

        # 7. Per-layer prediction errors
        pe_OTF = self.OT_F.pred_error.abs().mean().item()
        pe_PT = self.PT_L.pred_error.abs().mean().item()
        pe_PC_per = self.PC_per.pred_error.abs().mean().item()
        pe_PC_int = self.PC_int.pred_error.abs().mean().item()

        # 8. Store intermediates for Hebbian / diagnostics
        self._last_oL = oL
        self._last_oR = oR
        self._last_oF = oF
        self._last_fused = fused
        self._last_pi_OT = pi_OT
        self._last_pi_PC = pi_PC
        self._last_att = att

        retL_intensity = L[:, :400]
        retR_intensity = R[:, :400]

        return {
            "motor": m,
            "eye": e,
            "DA": d,
            "cls": cls,
            "retL": retL_intensity,
            "retR": retR_intensity,
            "retL_full": L,
            "retR_full": R,
            "oL": oL,
            "oR": oR,
            "oF": oF,
            "fused": fused,
            "pt": pt,
            "per": per,
            "intent": intent,
            "pi_OT": pi_OT.mean().item(),
            "pi_PC": pi_PC.mean().item(),
            # Prediction error outputs
            "pe_OTF": pe_OTF,
            "pe_PT": pe_PT,
            "pe_PC_per": pe_PC_per,
            "pe_PC_int": pe_PC_int,
            "att_signals": att.detach(),
            # Izhikevich spike events (binary, for raster)
            "spikes_OTF":    self.OT_F.spikes,
            "spikes_PT":     self.PT_L.spikes,
            "spikes_PC_per": self.PC_per.spikes,
            "spikes_PC_int": self.PC_int.spikes,
            "spikes_mot":    self.mot.spikes,
            "spikes_DA":     self.DA.spikes,
        }

    def get_saveable_state(self):
        """Return all learned weights for checkpoint persistence."""
        return self.state_dict()

    def load_saveable_state(self, state):
        """Restore learned weights from a checkpoint.

        Handles old checkpoints that use TwoComp.W or nn.Linear (weight/bias)
        instead of PredictiveTwoComp.W_FF by remapping parameter names.
        """
        has_old_keys = any(
            (k.endswith('.W') and not k.endswith('.W_FF')
             and not k.endswith('.W_FB'))
            or (k.split('.')[0] in ('mot', 'eye', 'DA')
                and k.endswith('.weight'))
            for k in state.keys()
            if k.split('.')[0] in ('OT_F', 'PT_L', 'PC_per', 'PC_int',
                                   'mot', 'eye', 'DA')
        )
        if has_old_keys:
            remapped = {}
            remap_layers = {'OT_F', 'PT_L', 'PC_per', 'PC_int',
                            'mot', 'eye', 'DA'}
            # Layers that were nn.Linear (weight is transposed vs W_FF)
            linear_layers = {'mot', 'eye', 'DA'}
            for k, v in state.items():
                parts = k.split('.')
                layer = parts[0]
                if len(parts) == 2 and layer in remap_layers:
                    if parts[1] == 'W':
                        remapped[f"{layer}.W_FF"] = v
                    elif parts[1] == 'weight' and layer in linear_layers:
                        remapped[f"{layer}.W_FF"] = v.t()
                    elif parts[1] == 'bias' and layer in linear_layers:
                        pass  # drop bias — PredictiveTwoComp has none
                    else:
                        remapped[k] = v
                else:
                    remapped[k] = v
            state = remapped
            print("[SNN] Migrated old checkpoint → PredictiveTwoComp")

        # Shape-safe loading: skip keys where saved shape differs from current
        # (handles classifier head size changes across architecture versions)
        current = self.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in current and v.shape == current[k].shape}
        skipped = [k for k in state if k in current
                   and state[k].shape != current[k].shape]
        if skipped:
            print(f"[SNN] Skipped {len(skipped)} shape-mismatched keys: "
                  f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}")
        self.load_state_dict(filtered, strict=False)

    def minimize_pe(self, lr=0.0005):
        """Layer-wise PE minimization: nudge W_FB to reduce |V_a - V_s|².

        Gradient: ΔW_FB = -lr * (1-tau_a) * fb^T @ (V_a - V_s)
        This brings V_a closer to V_s on the next step.
        Applied every timestep with a very small lr to avoid disrupting
        existing behavior while gradually reviving deep layers.

        Phase 2 SNN: extends reticulospinal STDP to all feedback weights.
        """
        with torch.no_grad():
            for layer in [self.OT_F, self.PT_L, self.PC_per, self.PC_int]:
                if layer.W_FB is None:
                    continue
                if not hasattr(layer, '_last_fb'):
                    continue
                pe = layer.pred_error          # [1, n_out] = V_a - V_s
                fb = layer._last_fb            # [1, n_fb]
                if pe.abs().mean() < 1e-6:
                    continue
                # dW = -lr * (1-tau_a) * fb^T @ pe  → shape [n_fb, n_out]
                dW = lr * (1.0 - layer.tau_a) * fb.T @ pe
                layer.W_FB.data -= dW
                layer.W_FB.data.clamp_(-2.0, 2.0)

    def compute_free_energy(self):
        """Compute variational free energy F as precision-weighted hierarchical PE.

        F = Σ_l (π_l/2) * ||ε_l||²  +  D_KL(q(θ)||p(θ))

        The PE term measures prediction accuracy (how well the generative
        model explains observations).  The KL term measures complexity
        (how far the posterior deviates from the prior).  F upper-bounds
        Bayesian surprise: F ≥ -ln p(o).

        Also computes Bayesian surprise as the absolute change in free
        energy between timesteps: large ΔF signals a regime change.
        """
        accuracy = 0.0
        # Precision-weighted layers
        for layer, prec in [(self.OT_F, self.prec_OT),
                            (self.PC_per, self.prec_PC)]:
            pe = layer.pred_error
            if pe is not None:
                pi = prec.compute_pi().mean().item()
                accuracy += 0.5 * pi * float((pe ** 2).mean())
        # Non-precision-weighted layers (implicit π=1)
        for layer in [self.PT_L, self.PC_int]:
            pe = layer.pred_error
            if pe is not None:
                accuracy += 0.5 * float((pe ** 2).mean())

        # Complexity: KL divergence between current and prior precision
        # D_KL(π||π_prior) ≈ Σ (γ_i - γ_prior)² / 2 for Gaussian approx
        complexity = 0.0
        for prec in [self.prec_OT, self.prec_PC]:
            complexity += 0.5 * float((prec.gamma.detach() ** 2).mean())

        F = accuracy + 0.01 * complexity  # small weight on complexity

        # Bayesian surprise: |F(t) - F(t-1)|
        prev_F = getattr(self, '_prev_free_energy', F)
        self._bayesian_surprise = abs(F - prev_F)
        self._prev_free_energy = F
        self._accuracy_term = accuracy
        self._complexity_term = complexity

        return F


@torch.no_grad()
def enforce_motor_directionality(model):
    # PredictiveTwoComp uses W_FF instead of W
    W = model.mot.W_FF  # shape [30, 200]
    W[:] = 0.0

    left_size = model.PC_INT // 2

    for pi in range(left_size):
        for mi in range(100):
            W[pi, mi] = 1.0

    for pi in range(left_size, model.PC_INT):
        for mi in range(100, 200):
            W[pi, mi] = 1.0

    print("[v1] Motor directional mapping applied.")
