"""
Spinal Central Pattern Generator — spiking half-centre oscillator.

Architecture (per side):
  8 V2a excitatory → drive ipsilateral motor neurons
  4 V0d inhibitory → cross-inhibit contralateral V2a
  4 motor neurons  → output to muscles

Total: 32 spiking LIF neurons (16 per side).
Brain provides tonic drive (speed) + turn bias → CPG adds phasic L/R oscillation.

Zebrafish spinal cord: V0-V2 interneurons form half-centre oscillator
(Kinkhabwala et al. 2011). Swim frequency scales with drive amplitude
(McLean et al. 2007).
"""
import torch
import torch.nn as nn


class SpinalCPG(nn.Module):
    def __init__(self, n_exc=8, n_inh=4, n_mot=4,
                 tau_m=0.5, tau_syn=0.6,
                 v_thresh=0.5, v_reset=0.0,
                 w_exc_mot=0.6, w_inh_cross=0.8,
                 w_exc_inh=0.5, noise=0.15, device='cpu'):
        super().__init__()
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_mot = n_mot
        self.n_per_side = n_exc + n_inh + n_mot  # 16
        self.tau_m = tau_m
        self.tau_syn = tau_syn
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.noise = noise
        self.device = device

        self.n_total = 2 * self.n_per_side  # 32
        self.v = torch.zeros(self.n_total, device=device)
        self.spike = torch.zeros(self.n_total, device=device)
        self.syn = torch.zeros(self.n_total, device=device)

        self.w_exc_mot = w_exc_mot
        self.w_inh_cross = w_inh_cross
        self.w_exc_inh = w_exc_inh

        self._phase = 0.0
        self._cycle_count = 0
        self._prev_L_active = False
        self.motor_L = 0.0
        self.motor_R = 0.0
        self.bout_active = False

    @torch.no_grad()
    def step(self, descending_drive: float, turn_bias: float = 0.0):
        """
        descending_drive: float [0, 1] — brain speed command
        turn_bias: float [-1, 1] — brain turn command
        Returns: motor_L, motor_R, speed, turn, diag dict
        """
        ne, ni, nm = self.n_exc, self.n_inh, self.n_mot
        nps = self.n_per_side

        L_exc = slice(0, ne)
        L_inh = slice(ne, ne + ni)
        L_mot = slice(ne + ni, nps)
        R_exc = slice(nps, nps + ne)
        R_inh = slice(nps + ne, nps + ne + ni)
        R_mot = slice(nps + ne + ni, 2 * nps)

        drive_L = descending_drive + max(0, turn_bias) * 0.3
        drive_R = descending_drive + max(0, -turn_bias) * 0.3

        I_ext = torch.zeros_like(self.v)
        I_ext[L_exc] = drive_L
        I_ext[R_exc] = drive_R

        I_syn = torch.zeros_like(self.v)
        L_exc_rate = self.syn[L_exc].mean()
        R_exc_rate = self.syn[R_exc].mean()
        I_syn[L_mot] += self.w_exc_mot * L_exc_rate
        I_syn[R_mot] += self.w_exc_mot * R_exc_rate
        I_syn[L_inh] += self.w_exc_inh * L_exc_rate
        I_syn[R_inh] += self.w_exc_inh * R_exc_rate
        L_inh_rate = self.syn[L_inh].mean()
        R_inh_rate = self.syn[R_inh].mean()
        I_syn[R_exc] -= self.w_inh_cross * L_inh_rate
        I_syn[L_exc] -= self.w_inh_cross * R_inh_rate

        noise_t = torch.randn(self.n_total, device=self.device) * self.noise
        v_new = self.tau_m * self.v + (1 - self.tau_m) * (I_ext + I_syn + noise_t)
        spike_new = (v_new >= self.v_thresh).float()
        v_new = torch.where(spike_new.bool(), torch.tensor(self.v_reset, device=self.device), v_new)
        self.v = v_new
        self.spike = spike_new
        self.syn = self.tau_syn * self.syn + spike_new

        self.motor_L = float(self.syn[L_mot].mean())
        self.motor_R = float(self.syn[R_mot].mean())
        self.bout_active = max(self.motor_L, self.motor_R) > 0.2

        L_active = self.motor_L > self.motor_R
        if L_active and not self._prev_L_active:
            self._cycle_count += 1
        self._prev_L_active = L_active
        if self.motor_L + self.motor_R > 0.01:
            self._phase = self.motor_R / (self.motor_L + self.motor_R + 1e-8)
        else:
            self._phase = 0.5

        speed = (self.motor_L + self.motor_R) / 2.0
        turn = (self.motor_R - self.motor_L) * 2.0

        return self.motor_L, self.motor_R, speed, turn, {
            'phase': self._phase,
            'v_L': self.motor_L, 'v_R': self.motor_R,
            'bout_active': self.bout_active,
            'cycle_count': self._cycle_count,
            'spikes_L': float(self.spike[:nps].sum()),
            'spikes_R': float(self.spike[nps:].sum()),
        }

    def reset(self):
        self.v = torch.zeros(self.n_total, device=self.device)
        self.spike = torch.zeros(self.n_total, device=self.device)
        self.syn = torch.zeros(self.n_total, device=self.device)
        self._phase = 0.0
        self._cycle_count = 0
        self._prev_L_active = False
        self.motor_L = 0.0
        self.motor_R = 0.0
        self.bout_active = False
