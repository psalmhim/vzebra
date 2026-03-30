"""
Spiking habenula: lateral (disappointment) + medial (aversion).

Zebrafish dorsal habenula (Hb) anatomy:
  Lateral Hb (LHb): fires when reward < expected (negative RPE).
    Projects to raphe → suppresses 5-HT → reduces patience.
    Projects to VTA → suppresses DA → reduces reward seeking.
  Medial Hb (MHb): aversive state / fear memory.
    Projects to interpeduncular nucleus (IPN).
    Drives behavioral inhibition and learned helplessness.

v1-equivalent features:
  - Per-goal frustration accumulation (4 goals)
  - Strategy switching: when frustration > threshold → force goal change
  - Goal avoidance bias: frustrated goals get EFE penalty
  - DA modulation: low DA amplifies helplessness
"""
import numpy as np
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingHabenula(nn.Module):
    def __init__(self, n_lhb=30, n_mhb=20,
                 helplessness_threshold=0.4,
                 helplessness_decay=0.998,
                 helplessness_gain=0.05,
                 device=DEVICE):
        super().__init__()
        self.device = device
        self.n_lhb = n_lhb
        self.n_mhb = n_mhb

        # Neuron populations
        self.LHb = IzhikevichLayer(n_lhb, 'RS', device)
        self.MHb = IzhikevichLayer(n_mhb, 'RS', device)
        self.LHb.i_tonic.fill_(-1.0)
        self.MHb.i_tonic.fill_(-1.0)

        # State
        self.register_buffer('lhb_rate', torch.zeros(n_lhb, device=device))
        self.register_buffer('mhb_rate', torch.zeros(n_mhb, device=device))
        self.register_buffer('disappointment', torch.tensor(0.0, device=device))
        self.register_buffer('aversion_level', torch.tensor(0.0, device=device))

        # Expected reward tracker
        self.expected_reward = 0.0
        self.reward_ema_alpha = 0.05

        # Output signals
        self.da_suppression = 0.0
        self.ht5_suppression = 0.0

        # --- v1-equivalent: per-goal frustration & strategy switching ---
        self.threshold = helplessness_threshold
        self.decay = helplessness_decay
        self.gain = helplessness_gain
        self.frustration = np.zeros(4, dtype=np.float32)
        self.helplessness = 0.0
        self._switch_cooldown = 0

    @torch.no_grad()
    def forward(self, reward: float, expected_reward: float = None,
                aversion: float = 0.0,
                current_goal: int = 0, DA: float = 0.5) -> dict:
        """
        reward: current reward signal
        expected_reward: from critic (if None, uses internal EMA)
        aversion: amygdala fear / threat level
        current_goal: active goal index (0-3) for per-goal frustration
        DA: dopamine level for modulation
        Returns: disappointment, DA/5-HT suppression, switch signal, goal bias
        """
        # Compute RPE
        if expected_reward is None:
            expected_reward = self.expected_reward
        rpe = reward - expected_reward
        self.expected_reward += self.reward_ema_alpha * (reward - self.expected_reward)
        disappoint = max(0.0, -rpe)

        # --- Spiking dynamics ---
        I_lhb = torch.full((self.n_lhb,), disappoint * 20.0, device=self.device)
        I_mhb = torch.full((self.n_mhb,), aversion * 15.0, device=self.device)
        lhb_spikes = torch.zeros(self.n_lhb, device=self.device)
        mhb_spikes = torch.zeros(self.n_mhb, device=self.device)
        for _ in range(20):  # reduced substeps
            sp_l = self.LHb(I_lhb + torch.randn(self.n_lhb, device=self.device) * 0.5)
            sp_m = self.MHb(I_mhb + torch.randn(self.n_mhb, device=self.device) * 0.5)
            lhb_spikes += sp_l
            mhb_spikes += sp_m
        self.lhb_rate.copy_(self.LHb.rate)
        self.mhb_rate.copy_(self.MHb.rate)
        lhb_mean = float(self.lhb_rate.mean())
        mhb_mean = float(self.mhb_rate.mean())

        # Neuromod suppression
        self.da_suppression = min(0.5, lhb_mean * 5.0)
        self.ht5_suppression = min(0.3, lhb_mean * 3.0)
        self.disappointment.fill_(disappoint)
        self.aversion_level.fill_(aversion)

        # --- Per-goal frustration (v1 logic) ---
        self.frustration *= self.decay
        if rpe < -0.05:
            self.frustration[current_goal] += self.gain * abs(rpe)
        if rpe > 0.1:
            self.frustration[current_goal] *= 0.8
        self.frustration = np.clip(self.frustration, 0.0, 1.0)
        self.helplessness = float(self.frustration.max())

        # Strategy switch
        switch = False
        if self._switch_cooldown > 0:
            self._switch_cooldown -= 1
        elif self.frustration[current_goal] > self.threshold:
            switch = True
            self._switch_cooldown = 15
            self.frustration[current_goal] *= 0.3

        # Goal avoidance bias (positive = avoid)
        dopa_mod = max(0.5, 2.0 * (1.0 - DA))
        goal_bias = self.frustration * 0.5 * dopa_mod

        return {
            'disappointment': disappoint,
            'da_suppression': self.da_suppression,
            'ht5_suppression': self.ht5_suppression,
            'lhb_rate': lhb_mean,
            'mhb_rate': mhb_mean,
            'explore_drive': min(1.0, disappoint * 2.0 + lhb_mean * 3.0),
            'switch_signal': switch,
            'goal_bias': goal_bias.copy(),
            'helplessness': self.helplessness,
            'frustration': self.frustration.copy(),
        }

    def reset(self):
        self.LHb.reset()
        self.MHb.reset()
        self.lhb_rate.zero_()
        self.mhb_rate.zero_()
        self.disappointment.zero_()
        self.aversion_level.zero_()
        self.expected_reward = 0.0
        self.da_suppression = 0.0
        self.ht5_suppression = 0.0
        self.frustration[:] = 0.0
        self.helplessness = 0.0
        self._switch_cooldown = 0
