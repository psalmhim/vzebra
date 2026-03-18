"""
Spinal Central Pattern Generator — spiking motor circuits (Step 38).

A spiking half-centre oscillator using leaky integrate-and-fire (LIF)
neurons with mutual inhibition.  Each side has an excitatory pool
(V2a interneurons) and an inhibitory commissural pool (V0d) that
crosses to inhibit the opposite side.

The CPG receives descending reticulospinal drive from the brain
and produces alternating left/right motor neuron bursts.  Swim
frequency scales with drive amplitude (McLean et al. 2007).

Neuroscience: zebrafish spinal cord contains V0-V2 interneurons
forming a half-centre oscillator (Kinkhabwala et al. 2011).
Slow/fast motor neuron recruitment follows a size principle
driven by descending drive amplitude (McLean et al. 2008).

Torch-based — spiking LIF neurons with synaptic dynamics.
"""
import math
import numpy as np
import torch
import torch.nn as nn


class SpinalCPG(nn.Module):
    """Spiking half-centre oscillator for rhythmic swimming.

    Architecture (per side):
      - 8 excitatory V2a neurons (drive ipsilateral motor neurons)
      - 4 inhibitory V0d neurons (cross-inhibit contralateral V2a)
      - 4 motor neurons (output to muscles)

    Total: 32 spiking neurons (16 per side).

    Args:
        n_exc: int — excitatory neurons per side
        n_inh: int — inhibitory commissural neurons per side
        n_mot: int — motor neurons per side
        tau_m: float — membrane time constant
        tau_syn: float — synaptic time constant
        v_thresh: float — spike threshold
        v_reset: float — post-spike reset
        w_exc_mot: float — excitatory→motor weight
        w_inh_cross: float — inhibitory cross-connection weight
        w_exc_inh: float — excitatory→inhibitory weight
        noise: float — membrane noise sigma
    """

    def __init__(self, n_exc=8, n_inh=4, n_mot=4,
                 tau_m=0.7, tau_syn=0.8,
                 v_thresh=0.5, v_reset=0.0,
                 w_exc_mot=0.8, w_inh_cross=1.0,
                 w_exc_inh=0.6, noise=0.08,
                 device="cpu"):
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

        # Membrane potentials: [L_exc, L_inh, L_mot, R_exc, R_inh, R_mot]
        n_total = 2 * self.n_per_side  # 32
        self.v = torch.zeros(n_total, device=device)
        self.spike = torch.zeros(n_total, device=device)
        self.syn = torch.zeros(n_total, device=device)  # synaptic trace

        # Connectivity (fixed, not learned)
        # L_exc (0:8) → L_mot (8:12): excitatory
        # L_exc (0:8) → L_inh (4:8 of left): excitatory (but cross-wired)
        # Actually: L_exc → L_inh → R_exc (inhibitory)
        self.w_exc_mot = w_exc_mot
        self.w_inh_cross = w_inh_cross
        self.w_exc_inh = w_exc_inh

        # Phase tracking
        self._phase = 0.0
        self._cycle_count = 0
        self._prev_L_active = False

        # Output state
        self.motor_L = 0.0
        self.motor_R = 0.0
        self.bout_active = False

    @torch.no_grad()
    def step(self, descending_drive, turn_bias=0.0):
        """Update spiking CPG one timestep.

        Args:
            descending_drive: float [0, 1] — brain speed command
            turn_bias: float [-1, 1] — brain turn command

        Returns:
            motor_L, motor_R: float — motor neuron firing rates
            speed: float — effective swim speed
            turn: float — effective turn rate
            diag: dict
        """
        ne, ni, nm = self.n_exc, self.n_inh, self.n_mot
        nps = self.n_per_side

        # Index ranges
        L_exc = slice(0, ne)
        L_inh = slice(ne, ne + ni)
        L_mot = slice(ne + ni, nps)
        R_exc = slice(nps, nps + ne)
        R_inh = slice(nps + ne, nps + ne + ni)
        R_mot = slice(nps + ne + ni, 2 * nps)

        # Descending drive (asymmetric for turning)
        drive_L = descending_drive + max(0, turn_bias) * 0.3
        drive_R = descending_drive + max(0, -turn_bias) * 0.3

        # External current
        I_ext = torch.zeros_like(self.v)
        I_ext[L_exc] = drive_L
        I_ext[R_exc] = drive_R

        # Synaptic current from spikes
        I_syn = torch.zeros_like(self.v)

        # L_exc → L_mot (excitatory)
        L_exc_rate = self.syn[L_exc].mean()
        I_syn[L_mot] += self.w_exc_mot * L_exc_rate

        # R_exc → R_mot (excitatory)
        R_exc_rate = self.syn[R_exc].mean()
        I_syn[R_mot] += self.w_exc_mot * R_exc_rate

        # L_exc → L_inh (excitatory, same side)
        I_syn[L_inh] += self.w_exc_inh * L_exc_rate

        # R_exc → R_inh
        I_syn[R_inh] += self.w_exc_inh * R_exc_rate

        # L_inh → R_exc (cross-inhibitory)
        L_inh_rate = self.syn[L_inh].mean()
        I_syn[R_exc] -= self.w_inh_cross * L_inh_rate

        # R_inh → L_exc (cross-inhibitory)
        R_inh_rate = self.syn[R_inh].mean()
        I_syn[L_exc] -= self.w_inh_cross * R_inh_rate

        # Membrane dynamics (LIF)
        noise_t = torch.randn_like(self.v) * self.noise
        self.v = (self.tau_m * self.v
                  + (1 - self.tau_m) * (I_ext + I_syn + noise_t))

        # Spike generation
        self.spike = (self.v >= self.v_thresh).float()
        self.v[self.v >= self.v_thresh] = self.v_reset

        # Synaptic trace (exponential decay + spike)
        self.syn = self.tau_syn * self.syn + self.spike

        # Motor output: firing rate of motor neurons
        self.motor_L = float(self.syn[L_mot].mean())
        self.motor_R = float(self.syn[R_mot].mean())

        # Bout detection
        self.bout_active = max(self.motor_L, self.motor_R) > 0.2

        # Phase tracking (detect L/R alternation)
        L_active = self.motor_L > self.motor_R
        if L_active and not self._prev_L_active:
            self._cycle_count += 1
        self._prev_L_active = L_active
        # Smooth phase estimate
        if self.motor_L + self.motor_R > 0.01:
            self._phase = self.motor_R / (self.motor_L + self.motor_R + 1e-8)
        else:
            self._phase = 0.5

        # Convert to speed and turn
        speed = (self.motor_L + self.motor_R) / 2.0
        turn = (self.motor_R - self.motor_L) * 2.0

        return self.motor_L, self.motor_R, speed, turn, {
            "phase": self._phase,
            "v_L": self.motor_L,
            "v_R": self.motor_R,
            "bout_active": self.bout_active,
            "cycle_count": self._cycle_count,
            "spikes_L": float(self.spike[:nps].sum()),
            "spikes_R": float(self.spike[nps:].sum()),
        }

    def reset(self):
        self.v.zero_()
        self.spike.zero_()
        self.syn.zero_()
        self._phase = 0.0
        self._cycle_count = 0
        self._prev_L_active = False
        self.motor_L = 0.0
        self.motor_R = 0.0
        self.bout_active = False

    def get_diagnostics(self):
        return {
            "phase": self._phase,
            "v_L": self.motor_L,
            "v_R": self.motor_R,
            "motor_L": self.motor_L,
            "motor_R": self.motor_R,
            "bout_active": self.bout_active,
        }
