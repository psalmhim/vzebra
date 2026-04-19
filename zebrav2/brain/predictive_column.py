"""
Canonical predictive coding microcircuit (Bastos et al. 2012).

Implements the cortical column where prediction error (PE), precision,
and free energy emerge from SPIKING DYNAMICS rather than scalar arithmetic.

Architecture (per channel):
  SP+ (superficial pyramidal, up-error):
      Fires when sensory > prediction (under-prediction).
      Receives: excitatory from sensory, inhibitory from IN.
  SP- (superficial pyramidal, down-error):
      Fires when prediction > sensory (over-prediction).
      Receives: excitatory from prediction, inhibitory from sensory-IN.
  DP  (deep pyramidal, prediction neurons):
      Encodes generative model belief. Driven by top-down input.
      Updated by ascending PE from SP (teaches predictions via STDP).
  IN  (fast-spiking interneurons):
      Relay predictions as inhibition to SP+ neurons.
      Synaptic gain = precision (neuromodulatory control).

Synaptic wiring:
  Sensory input ──exc──→ SP+ (bottom-up)
  DP ──exc──→ IN (prediction relay)
  IN ──inh──→ SP+ (creates subtraction: SP+ ≈ sensory − prediction)
  DP ──exc──→ SP- (over-prediction signal)
  Sensory ──exc──→ IN_s → SP- (inh) (sensory cancels over-prediction)
  SP+ ──exc──→ DP (ascending PE teaches predictions)
  SP- ──exc──→ DP (ascending PE, sign-flipped)

Precision:
  Encoded as gain g_inh on IN→SP+ pathway (and IN_s→SP- pathway).
  Modulated by neuromodulators (DA, NA, ACh, 5-HT).
  High precision = strong inhibition = PE neurons only fire on real mismatch.
  Low precision = weak inhibition = noisy PE (as in schizophrenia).

Free energy:
  F = mean(SP+.rate² + SP-.rate²) — directly from spiking activity.
  No explicit formula needed; it emerges from the circuit.

STDP (predictive learning):
  SP→DP connections use rate-based STDP approximation:
    Δw = η * (err_rate × sensory_drive − w × err_rate²)
  This teaches DP to predict sensory input, reducing future PE.

References:
  Bastos, Usrey, Adams, Mangun, Fries & Friston (2012) Canonical microcircuits
    for predictive coding. Neuron 76:695-711
  Shipp (2016) Neural elements for predictive coding. Frontiers in Psychology 7:1792
  Feldman & Friston (2010) Attention, uncertainty, and free-energy.
    Frontiers in Human Neuroscience 4:215
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer


class PredictiveCodingColumn(nn.Module):
    """
    Reusable predictive coding microcircuit.

    n_channels: number of independent prediction channels
    n_per_ch:   neurons per channel per population (default 4)

    Total neurons: n_channels × n_per_ch × 4 populations
      (SP+, SP-, DP, IN — 4 × n_per_ch per channel)
    """

    def __init__(self, n_channels: int, n_per_ch: int = 4,
                 substeps: int = 8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_ch = n_channels
        self.n_per = n_per_ch
        self.n_pop = n_channels * n_per_ch  # neurons per population
        self.substeps = substeps

        # ---- Spiking populations ----
        # SP+: under-prediction error (L2/3 superficial pyramidal, RS)
        self.sp_up = IzhikevichLayer(self.n_pop, 'RS', device)
        self.sp_up.i_tonic.fill_(-1.0)  # light inhibition (fires on strong PE)

        # SP-: over-prediction error (L2/3, RS)
        self.sp_down = IzhikevichLayer(self.n_pop, 'RS', device)
        self.sp_down.i_tonic.fill_(-1.0)

        # DP: deep pyramidal prediction neurons (L5/6, IB — intrinsic bursting)
        self.dp = IzhikevichLayer(self.n_pop, 'IB', device)
        self.dp.i_tonic.fill_(0.0)  # neutral — driven by top-down input

        # IN: fast-spiking interneurons (precision gate)
        n_inh = max(n_channels, n_channels * (n_per_ch // 2))
        self.n_inh = n_inh
        self.inh = IzhikevichLayer(n_inh, 'FS', device)
        self.inh.i_tonic.fill_(-0.5)

        # ---- Synaptic conductances (per-channel) ----
        # Drive scale: inputs are [0,1], RS neurons need ~4+ pA to fire
        # g_sens: sensory → SP+ gain (feedforward excitation)
        self.register_buffer('g_sens',
            torch.full((n_channels,), 15.0, device=device))

        # g_pred: DP → IN gain (prediction relay)
        self.register_buffer('g_pred_inh',
            torch.full((n_channels,), 12.0, device=device))

        # g_inh: IN → SP+ gain = PRECISION (neuromodulatory)
        # This IS the precision parameter — encoded as synaptic gain
        self.register_buffer('g_precision',
            torch.ones(n_channels, device=device) * 5.0)

        # g_err_dp: SP → DP gain (ascending PE, teaches predictions)
        self.register_buffer('g_err_dp',
            torch.full((n_channels,), 8.0, device=device))

        # ---- STDP state for prediction learning ----
        # Eligibility trace: tracks correlated SP↔DP activity
        self.register_buffer('eligibility',
            torch.zeros(n_channels, device=device))
        self.stdp_lr = 0.005  # STDP learning rate
        self.tau_elig = 0.9   # eligibility trace decay

        # ---- Prediction bias (learned internal model) ----
        # This is what STDP adjusts — the DP neurons' baseline prediction
        self.register_buffer('pred_bias',
            torch.zeros(n_channels, device=device))

        # ---- Output buffers ----
        self.register_buffer('pe_signed',
            torch.zeros(n_channels, device=device))
        self.register_buffer('pe_magnitude',
            torch.zeros(n_channels, device=device))
        self.register_buffer('dp_rate_ch',
            torch.zeros(n_channels, device=device))
        self.free_energy = 0.0

    # ------------------------------------------------------------------
    # Channel ↔ population indexing
    # ------------------------------------------------------------------

    def _expand(self, ch_values: torch.Tensor, n_per: int) -> torch.Tensor:
        """(n_ch,) → (n_ch * n_per,) by repeating each channel value."""
        return ch_values.repeat_interleave(n_per)

    def _expand_inh(self, ch_values: torch.Tensor) -> torch.Tensor:
        """(n_ch,) → (n_inh,) by repeating."""
        n_per_inh = self.n_inh // self.n_ch
        return ch_values.repeat_interleave(n_per_inh)

    def _pool(self, pop_values: torch.Tensor, n_per: int) -> torch.Tensor:
        """(n_ch * n_per,) → (n_ch,) by averaging within each channel."""
        return pop_values.view(self.n_ch, n_per).mean(dim=1)

    def _pool_inh(self, pop_values: torch.Tensor) -> torch.Tensor:
        """(n_inh,) → (n_ch,) by averaging."""
        n_per_inh = self.n_inh // self.n_ch
        return pop_values.view(self.n_ch, n_per_inh).mean(dim=1)

    # ------------------------------------------------------------------
    # Precision interface (neuromodulatory control)
    # ------------------------------------------------------------------

    def set_precision(self, channel: int, value: float):
        """Set precision for a specific channel (used by neuromodulators)."""
        self.g_precision[channel] = max(0.1, min(20.0, value))

    def set_precision_all(self, values: torch.Tensor):
        """Set precision for all channels at once."""
        self.g_precision.copy_(values.to(self.device).clamp(0.1, 20.0))

    def get_precision(self) -> torch.Tensor:
        """Current precision per channel (readable for diagnostics)."""
        return self.g_precision.clone()

    # ------------------------------------------------------------------
    # Forward: one predictive coding cycle
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, sensory_drive: torch.Tensor,
                prediction_drive: torch.Tensor) -> dict:
        """
        Run one predictive coding cycle.

        Args:
            sensory_drive:    (n_channels,) bottom-up sensory input [0, ~1]
            prediction_drive: (n_channels,) top-down belief/goal [0, ~1]

        Returns dict with:
            pe_signed:     (n_channels,) signed PE (positive = under-prediction)
            pe_magnitude:  (n_channels,) absolute PE magnitude
            free_energy:   scalar, total variational free energy
            precision:     (n_channels,) current precision per channel
            dp_rate:       (n_channels,) prediction neuron firing rates
            sp_up_rate:    (n_channels,) under-prediction error rates
            sp_down_rate:  (n_channels,) over-prediction error rates
        """
        # Ensure inputs are on correct device
        sensory_drive = sensory_drive.to(self.device)
        prediction_drive = prediction_drive.to(self.device)

        # Add learned prediction bias to top-down drive
        pred_total = prediction_drive + self.pred_bias

        # Expand channel values to population-level currents
        I_sens_base = self._expand(sensory_drive * self.g_sens, self.n_per)
        I_pred_base = self._expand(pred_total * self.g_pred_inh, self.n_per)

        noise_scale = 0.4

        for _ in range(self.substeps):
            noise_pop = torch.randn(self.n_pop, device=self.device) * noise_scale
            noise_inh = torch.randn(self.n_inh, device=self.device) * noise_scale

            # ---- 1. DP: prediction neurons driven by top-down ----
            # Also receives ascending PE from previous substep
            sp_up_rate_pop = self.sp_up.rate    # (n_pop,)
            sp_down_rate_pop = self.sp_down.rate

            # Ascending PE: SP+ increases DP drive, SP- decreases
            I_asc = self._expand(
                self._pool(sp_up_rate_pop, self.n_per) * self.g_err_dp
                - self._pool(sp_down_rate_pop, self.n_per) * self.g_err_dp * 0.5,
                self.n_per
            )
            I_dp = I_pred_base + I_asc + noise_pop
            self.dp(I_dp)

            # ---- 2. IN: interneurons relay DP predictions ----
            # DP rate → IN excitation (prediction becomes inhibition downstream)
            dp_rate_pooled = self._pool(self.dp.rate, self.n_per)
            I_in = self._expand_inh(dp_rate_pooled * self.g_pred_inh)
            self.inh(I_in + noise_inh)

            # ---- 3. SP+: under-prediction error ----
            # Excitation: sensory (bottom-up)
            # Inhibition: IN (prediction-via-interneuron) × precision
            inh_rate_ch = self._pool_inh(self.inh.rate)
            I_sp_up_inh = self._expand(
                inh_rate_ch * self.g_precision, self.n_per)
            I_sp_up = I_sens_base - I_sp_up_inh + noise_pop
            self.sp_up(I_sp_up)

            # ---- 4. SP-: over-prediction error ----
            # Excitation: prediction (top-down, direct)
            # Inhibition: sensory (via separate inhibitory pathway)
            I_sp_down_exc = self._expand(
                pred_total * self.g_pred_inh, self.n_per)
            I_sp_down_inh = self._expand(
                sensory_drive * self.g_precision, self.n_per)
            I_sp_down = I_sp_down_exc - I_sp_down_inh + noise_pop
            self.sp_down(I_sp_down)

        # ---- 5. Read out PE from spiking rates ----
        sp_up_ch = self._pool(self.sp_up.rate, self.n_per)     # (n_ch,)
        sp_down_ch = self._pool(self.sp_down.rate, self.n_per)  # (n_ch,)
        dp_ch = self._pool(self.dp.rate, self.n_per)            # (n_ch,)

        # Signed PE: positive = sensory > prediction (under-predicted)
        pe_signed = sp_up_ch - sp_down_ch
        # Magnitude: total error activity
        pe_magnitude = sp_up_ch + sp_down_ch

        self.pe_signed.copy_(pe_signed)
        self.pe_magnitude.copy_(pe_magnitude)
        self.dp_rate_ch.copy_(dp_ch)

        # ---- 6. Free energy: directly from error population activity ----
        # F ∝ Σ (precision × error_rate²)  — emerges from spiking, no formula
        # The precision-weighting is already baked into the synaptic gain,
        # so high-precision channels contribute more to total FE.
        self.free_energy = float(
            0.5 * (self.g_precision * pe_magnitude ** 2).sum()
        )

        # ---- 7. STDP: teach predictions to match sensory ----
        # Rate-based approximation of spike-timing dependent plasticity
        # Eligibility: correlated error↔sensory activity
        self.eligibility.mul_(self.tau_elig).add_(
            sp_up_ch * sensory_drive - sp_down_ch * dp_ch
        )
        # Update prediction bias: reduce PE by adjusting internal model
        self.pred_bias.add_(self.stdp_lr * self.eligibility)
        self.pred_bias.clamp_(-2.0, 2.0)  # prevent runaway

        # ---- 8. Online precision update (optional) ----
        # Precision adapts to prediction accuracy:
        # If PE is consistently low → increase precision (more confident)
        # If PE is high → decrease precision (less reliable channel)
        # This is variational Bayes on the precision parameter
        # dπ/dt = -∂F/∂π ∝ 1/(2π) - ε²/2
        precision_grad = 0.5 / (self.g_precision + 1e-4) - 0.5 * pe_signed ** 2
        self.g_precision.add_(0.001 * precision_grad)
        self.g_precision.clamp_(0.5, 15.0)

        return {
            'pe_signed': pe_signed,
            'pe_magnitude': pe_magnitude,
            'free_energy': self.free_energy,
            'precision': self.g_precision.clone(),
            'dp_rate': dp_ch,
            'sp_up_rate': sp_up_ch,
            'sp_down_rate': sp_down_ch,
        }

    # ------------------------------------------------------------------

    def reset(self):
        self.sp_up.reset()
        self.sp_down.reset()
        self.dp.reset()
        self.inh.reset()
        self.pe_signed.zero_()
        self.pe_magnitude.zero_()
        self.dp_rate_ch.zero_()
        self.eligibility.zero_()
        # Keep pred_bias (learned) and g_precision (adapted)
        self.free_energy = 0.0
