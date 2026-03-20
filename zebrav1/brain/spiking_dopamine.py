"""
Spiking Dopamine System — VTA tonic/phasic DA neurons.

Replaces the numpy TD-error formula with a spiking VTA circuit:
  - 5 tonic DA neurons: maintain baseline dopamine level
  - 5 phasic DA neurons: fire bursts on positive RPE, pause on negative
  - RPE computed from prediction error in reward-predicting neurons

Neuroscience: VTA dopamine neurons encode reward prediction error
(Schultz et al. 1997). Tonic firing (~4 Hz) maintains baseline
motivation; phasic bursts signal unexpected reward (Dabney et al. 2020).

Torch-based — spiking LIF neurons.
"""
import torch
import torch.nn as nn
import numpy as np


class SpikingDopamine(nn.Module):
    """Spiking VTA dopamine circuit.

    Args:
        n_tonic: int — tonic DA neurons (baseline)
        n_phasic: int — phasic DA neurons (RPE burst/pause)
        tau: float — membrane time constant
        v_thresh: float — spike threshold
        device: str
    """

    def __init__(self, n_tonic=5, n_phasic=5, tau=0.85,
                 v_thresh=0.4, device="cpu"):
        super().__init__()
        self.n_tonic = n_tonic
        self.n_phasic = n_phasic
        self.tau = tau
        self.v_thresh = v_thresh
        self.device = device

        # Tonic neurons receive constant drive + noise
        self.tonic_drive = nn.Parameter(torch.full((n_tonic,), 0.3))

        # Phasic neurons receive reward prediction error
        self.W_rpe = nn.Linear(3, n_phasic, bias=True)  # input: [reward, predicted, F_visual]

        # Reward predictor (learns expected reward from state)
        self.W_pred = nn.Linear(5, 1, bias=True)  # input: classifier probs

        # LIF state
        self.v_tonic = torch.zeros(n_tonic, device=device)
        self.v_phasic = torch.zeros(n_phasic, device=device)

        # Output state (dopa_level + alias 'dopa' for compatibility)
        self.dopa_level = 0.5  # [0, 1]
        self.dopa = 0.5        # alias
        self.rpe = 0.0
        self.beta = 1.0  # gain parameter

        # Learning
        self._predicted_reward = 0.0
        self._prev_cls = torch.zeros(5, device=device)

        self.to(device)

    @torch.no_grad()
    def step(self, F_visual, oL_mean, oR_mean, eaten=0, cls_probs=None):
        """Update dopamine via spiking VTA dynamics.

        Args:
            F_visual: float — visual free energy
            oL_mean: float — left OT activity
            oR_mean: float — right OT activity
            eaten: int — food eaten this step
            cls_probs: tensor [5] or None — classifier output

        Returns:
            dopa: float [0, 1] — dopamine level
            rpe: float — reward prediction error
            valL: float — left value
            valR: float — right value
        """
        # Reward signal
        reward = float(eaten) * 1.0 - F_visual * 0.1

        # Predicted reward from classifier state
        if cls_probs is not None:
            cls_t = torch.tensor(cls_probs, dtype=torch.float32,
                                 device=self.device)
            self._predicted_reward = float(self.W_pred(cls_t).item())
            self._prev_cls = cls_t

        # RPE = actual - predicted
        self.rpe = reward - self._predicted_reward

        # Tonic neurons: constant drive + noise → baseline firing
        noise = torch.randn(self.n_tonic, device=self.device) * 0.1
        self.v_tonic = (self.tau * self.v_tonic
                        + (1 - self.tau) * (self.tonic_drive + noise))
        tonic_spikes = (self.v_tonic >= self.v_thresh).float()
        self.v_tonic = self.v_tonic * (1 - tonic_spikes)
        tonic_rate = tonic_spikes.mean()

        # Phasic neurons: RPE-driven
        rpe_input = torch.tensor([reward, self._predicted_reward, F_visual],
                                 dtype=torch.float32, device=self.device)
        I_phasic = self.W_rpe(rpe_input)
        self.v_phasic = (self.tau * self.v_phasic
                         + (1 - self.tau) * I_phasic)
        phasic_spikes = (self.v_phasic >= self.v_thresh).float()
        self.v_phasic = self.v_phasic * (1 - phasic_spikes)
        phasic_rate = phasic_spikes.mean()

        # Dopamine level: tonic baseline + phasic modulation
        self.dopa_level = float(torch.clamp(
            0.3 * tonic_rate + 0.7 * (0.5 + self.rpe * 0.5), 0.0, 1.0))
        self.dopa = self.dopa_level  # alias for compatibility

        # Lateral value (for BG gating)
        valL = oL_mean * self.dopa_level
        valR = oR_mean * self.dopa_level

        return self.dopa_level, self.rpe, valL, valR

    def reset(self):
        self.v_tonic.zero_()
        self.v_phasic.zero_()
        self.dopa_level = 0.5
        self.rpe = 0.0
        self._predicted_reward = 0.0

    def get_diagnostics(self):
        return {
            "dopa": self.dopa_level,
            "rpe": self.rpe,
            "predicted_reward": self._predicted_reward,
        }
