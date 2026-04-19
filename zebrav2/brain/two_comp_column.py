"""
Two-compartment predictive coding column for zebrav2.

Unifies the v1 IzhikevichTwoComp + PrecisionUnit with the FEP framework.
Each neuron computes PE internally via apical-somatic mismatch, making
population-level E/I subtraction unnecessary.

Biological mapping (Larkum 2013, Sacramento et al. 2018):
  Apical dendrite (V_a) = top-down prediction (from higher cortical area)
  Soma/basal (V_s)      = bottom-up sensory evidence
  PE = V_a - V_s        = prediction error (within-neuron, not between populations)
  bias                   = precision (synaptic gain / excitability modulation)
  W_FB                   = generative model (learned predictions)

FEP mapping (Friston 2005, Bogacz 2017):
  PE:        ε = V_a - normalize(V_s) → apical-somatic mismatch
  Precision: π = σ(γ) where γ = learnable precision parameter
             Implemented as: bias = bias_base + π * attention_gain
             Higher π → more excitable soma → PE has stronger effect
  Free energy: F = Σ π_i * ε_i² (sum over neurons)
  Belief update: V_a += η * ε (apical tracks sensory → reduces PE)
  Precision update: dγ/dt = α * (|ε| - β) (Bogacz 2017 eq. 7)

Key insight from Lee, Lee & Park (2026):
  Attention bias IS precision. The additive somatic excitability bias
  controls how strongly each neuron reports prediction errors. This is
  exactly Feldman & Friston (2010): precision = gain on PE neurons.

Architecture per channel:
  n_neurons IzhikevichTwoComp cells (Izhikevich soma + continuous apical)
  1 PrecisionUnit (learnable γ → π)
  Connections: W_FF (feedforward), W_FB (feedback/prediction)

References:
  Larkum (2013) A cellular mechanism for cortical associations. Nature Neuroscience
  Sacramento, Costa, Bengio & Senn (2018) Dendritic cortical microcircuits
    approximate the backpropagation algorithm. NeurIPS
  Bogacz (2017) A tutorial on the free-energy framework. J Math Psych
  Lee, Lee & Park (2026) Modulation of bias in predictive-coding SNNs
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE


# ---------------------------------------------------------------------------
# Izhikevich presets
# ---------------------------------------------------------------------------

IZHI_PRESETS = {
    'RS':  (0.02, 0.20, -65.0,  8.0),   # regular spiking (pyramidal)
    'FS':  (0.10, 0.20, -65.0,  2.0),   # fast spiking (inhibitory)
    'IB':  (0.02, 0.20, -55.0,  4.0),   # intrinsic bursting (deep layer)
    'CH':  (0.02, 0.20, -50.0,  2.0),   # chattering
    'LTS': (0.02, 0.25, -65.0,  2.0),   # low-threshold spiking
}

V_THRESH = 30.0
V_REST = -65.0


# ---------------------------------------------------------------------------
# Two-Compartment Predictive Coding Column
# ---------------------------------------------------------------------------

class TwoCompColumn(nn.Module):
    """
    Two-compartment predictive coding column.

    Each neuron has:
      - Soma: Izhikevich spiking dynamics, receives bottom-up + apical + bias
      - Apical dendrite: continuous graded potential, receives top-down feedback
      - PE = V_a - norm(V_s): computed within each neuron
      - Bias = precision: modulates somatic excitability (= PE gain)

    Args:
        n_channels:  number of independent channels
        n_per_ch:    neurons per channel (default 6)
        cell_type:   Izhikevich preset ('RS', 'IB', 'FS', etc.)
        substeps:    Izhikevich simulation steps per call
    """

    def __init__(self, n_channels: int, n_per_ch: int = 6,
                 cell_type: str = 'RS', substeps: int = 10,
                 device=DEVICE):
        super().__init__()
        self.device = device
        self.n_ch = n_channels
        self.n_per = n_per_ch
        self.n_total = n_channels * n_per_ch
        self.substeps = substeps

        # Izhikevich parameters
        a, b, c, d = IZHI_PRESETS[cell_type]
        self.register_buffer('izhi_a', torch.full((self.n_total,), a, device=device))
        self.register_buffer('izhi_b', torch.full((self.n_total,), b, device=device))
        self.register_buffer('izhi_c', torch.full((self.n_total,), c, device=device))
        self.register_buffer('izhi_d', torch.full((self.n_total,), d, device=device))

        # --- Somatic state ---
        self.register_buffer('v_s', torch.full((self.n_total,), V_REST, device=device))
        self.register_buffer('u', torch.full((self.n_total,), b * V_REST, device=device))
        self.register_buffer('spikes', torch.zeros(self.n_total, device=device))
        self.register_buffer('rate', torch.zeros(self.n_total, device=device))
        self.tau_rate = 0.02  # EMA decay for firing rate

        # --- Apical dendrite state ---
        self.register_buffer('v_a', torch.zeros(self.n_total, device=device))
        self.tau_a = 0.65  # apical time constant

        # --- Bias = precision (per channel) ---
        # Base tonic drive: keeps neurons near threshold
        self.register_buffer('bias_base',
            torch.full((n_channels,), 3.5, device=device))
        # Learnable precision parameter γ → π = σ(γ)
        self.register_buffer('gamma',
            torch.zeros(n_channels, device=device))
        # Structural bounds on γ (for disorder models)
        # gamma_floor/ceiling constrain the operating range permanently.
        # Schizophrenia: ceiling < 0 → precision can never recover to WT levels.
        self.gamma_floor = -3.0
        self.gamma_ceiling = 3.0
        # Precision learning parameters (PrecisionUnit)
        self.precision_lr = 0.05
        self.precision_beta = 0.1  # desired error level
        self.register_buffer('prev_pe_mag',
            torch.zeros(n_channels, device=device))

        # --- Attention modulation (slow dynamics) ---
        self.register_buffer('m_att',
            torch.zeros(n_channels, device=device))
        self.alpha_att = 0.1  # attention coupling strength
        self.tau_att = 0.9    # attention time constant

        # --- Apical belief update rate ---
        self.apical_lr = 0.05  # how fast apical tracks sensory (belief update)

        # --- Output buffers ---
        self.register_buffer('pe',
            torch.zeros(n_channels, device=device))
        self.register_buffer('rate_ch',
            torch.zeros(n_channels, device=device))
        self.free_energy = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def precision(self) -> torch.Tensor:
        """Current precision per channel: π = σ(γ)."""
        return torch.sigmoid(self.gamma)

    @property
    def effective_bias(self) -> torch.Tensor:
        """Somatic bias = base + precision_gain + attention.
        This IS the precision modulation from Lee et al. (2026)."""
        pi = self.precision
        att_gain = self.alpha_att * self.m_att
        # Higher precision → higher bias → more excitable → PE has stronger effect
        return self.bias_base + pi * 2.0 + att_gain

    # ------------------------------------------------------------------
    # Channel ↔ population indexing
    # ------------------------------------------------------------------

    def _expand(self, ch_values: torch.Tensor) -> torch.Tensor:
        """(n_ch,) → (n_total,) by repeating each channel value n_per times."""
        return ch_values.repeat_interleave(self.n_per)

    def _pool(self, pop_values: torch.Tensor) -> torch.Tensor:
        """(n_total,) → (n_ch,) by averaging within each channel."""
        return pop_values.view(self.n_ch, self.n_per).mean(dim=1)

    # ------------------------------------------------------------------
    # Precision interface (neuromodulatory control)
    # ------------------------------------------------------------------

    def set_attention(self, att: torch.Tensor):
        """Set attention modulation per channel (from neuromodulators)."""
        self.m_att = self.tau_att * self.m_att + (1 - self.tau_att) * att.to(self.device)

    def set_precision_channel(self, channel: int, gamma_val: float):
        """Directly set precision parameter for a channel."""
        self.gamma[channel] = max(-3.0, min(3.0, gamma_val))

    # ------------------------------------------------------------------
    # Forward: one predictive coding cycle
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, sensory_drive: torch.Tensor,
                prediction_drive: torch.Tensor) -> dict:
        """
        Run one two-compartment predictive coding cycle.

        Args:
            sensory_drive:    (n_channels,) bottom-up input [0, ~1]
            prediction_drive: (n_channels,) top-down belief [0, ~1]

        The key difference from PredictiveCodingColumn:
          PE is computed WITHIN each neuron (V_a - V_s),
          not between separate populations via E/I balance.

        Returns dict with:
            pe:           (n_channels,) prediction error (apical - somatic)
            precision:    (n_channels,) current precision π = σ(γ)
            free_energy:  scalar, precision-weighted PE²
            rate:         (n_channels,) somatic firing rate
            v_a:          (n_channels,) apical potential (beliefs)
            spikes_mean:  scalar, mean spike rate
        """
        sensory_drive = sensory_drive.to(self.device)
        prediction_drive = prediction_drive.to(self.device)

        # --- 1. Update apical dendrite (top-down prediction) ---
        # Apical integrates prediction with slow dynamics (1-step delayed)
        # Keep on same scale as normalized soma [0, 1] for balanced PE
        pred_expanded = self._expand(prediction_drive)
        self.v_a = (self.tau_a * self.v_a
                    + (1.0 - self.tau_a) * pred_expanded)
        self.v_a.clamp_(-2.0, 2.0)

        # --- 2. Bounded apical coupling: f(V_a) → [-0.25, 0.25] ---
        # Sigmoid bounding from Lee et al. (2026), then scale to mV for soma
        apical_coupling = 0.5 * (torch.sigmoid(self.v_a) - 0.5)
        apical_current = apical_coupling * 15.0  # scale to mV range

        # --- 3. Somatic current: sensory + apical + bias (= precision) ---
        sens_expanded = self._expand(sensory_drive)
        bias_expanded = self._expand(self.effective_bias)

        I_total = (sens_expanded * 12.0  # sensory feedforward
                   + apical_current       # top-down via apical dendrite
                   + bias_expanded)       # precision + attention bias

        # --- 4. Izhikevich spiking dynamics ---
        for _ in range(self.substeps):
            noise = torch.randn(self.n_total, device=self.device) * 0.3
            I_step = I_total + noise

            # Half-step Euler (numerical stability)
            v_h = self.v_s + 0.5 * (0.04 * self.v_s ** 2 + 5 * self.v_s
                                     + 140 - self.u + I_step)
            u_h = self.u + 0.5 * self.izhi_a * (self.izhi_b * self.v_s - self.u)

            v_new = v_h + 0.5 * (0.04 * v_h ** 2 + 5 * v_h + 140 - u_h + I_step)
            u_new = u_h + 0.5 * self.izhi_a * (self.izhi_b * v_h - u_h)

            v_new = v_new.clamp(-100.0, 35.0)

            # Spike detection and reset
            fired = (v_new >= V_THRESH).float()
            v_new = torch.where(fired.bool(), self.izhi_c, v_new)
            u_new = torch.where(fired.bool(), u_new + self.izhi_d, u_new)

            self.v_s.copy_(v_new)
            self.u.copy_(u_new)
            self.spikes.copy_(fired)

            # Update firing rate EMA
            self.rate.mul_(1 - self.tau_rate).add_(self.tau_rate * fired)

        # --- 5. Normalize soma for PE computation ---
        # Map mV to [0, 1]: 0 at rest (-65), 1 at threshold (30)
        v_norm = ((self.v_s - V_REST) / (V_THRESH - V_REST)).clamp(0.0, 1.0)

        # --- 6. Prediction error: apical - somatic (WITHIN each neuron) ---
        # This is the core of two-compartment predictive coding:
        # PE > 0: prediction exceeds evidence (over-prediction)
        # PE < 0: evidence exceeds prediction (under-prediction)
        pe_per_neuron = self.v_a - v_norm
        pe_ch = self._pool(pe_per_neuron)
        self.pe.copy_(pe_ch)

        # Firing rate per channel
        rate_ch = self._pool(self.rate)
        self.rate_ch.copy_(rate_ch)

        # --- 7. Free energy: precision-weighted PE² ---
        # F = Σ πᵢ * εᵢ²  (directly from within-neuron PE and learned precision)
        pi = self.precision
        self.free_energy = float(0.5 * (pi * pe_ch ** 2).sum())

        # --- 8. Precision update (PrecisionUnit logic) ---
        # dγ/dt = α * (|ε| - β) — increase precision on high PE, decrease on low
        pe_mag = pe_ch.abs()
        # Novelty: sudden change in PE magnitude → precision dip
        novelty = (pe_mag - self.prev_pe_mag).abs()
        self.prev_pe_mag.mul_(0.8).add_(0.2 * pe_mag)

        d_gamma = self.precision_lr * (pe_mag - self.precision_beta)
        # Novelty-driven precision dip (surprise → temporary drop)
        d_gamma -= 0.05 * novelty * (novelty > 0.03).float()
        self.gamma.add_(d_gamma)
        self.gamma.clamp_(self.gamma_floor, self.gamma_ceiling)

        # --- 9. Apical belief update (minimize PE) ---
        # V_a moves toward sensory evidence to reduce future PE
        # This IS free energy minimization through belief updating
        self.v_a.add_(self._expand(-self.apical_lr * pe_ch))
        self.v_a.clamp_(-2.0, 2.0)

        return {
            'pe': pe_ch,
            'precision': pi,
            'free_energy': self.free_energy,
            'rate': rate_ch,
            'v_a': self._pool(self.v_a),
            'spikes_mean': float(self.rate.mean()),
            'effective_bias': self._pool(bias_expanded),
        }

    # ------------------------------------------------------------------

    def reset(self):
        """Reset spiking state but keep learned precision and apical beliefs."""
        self.v_s.fill_(V_REST)
        b = float(self.izhi_b[0])
        self.u.fill_(b * V_REST)
        self.spikes.zero_()
        self.rate.zero_()
        self.pe.zero_()
        self.rate_ch.zero_()
        self.m_att.zero_()
        self.prev_pe_mag.zero_()
        # Keep v_a (learned beliefs), gamma (learned precision)
        self.free_energy = 0.0

    def hard_reset(self):
        """Full reset including learned state (episode boundary)."""
        self.reset()
        self.v_a.zero_()
        self.gamma.zero_()
