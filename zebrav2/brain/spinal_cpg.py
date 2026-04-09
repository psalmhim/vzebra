"""
Spinal Central Pattern Generator — V2 biologically accurate bout-glide swimming.

Circuit per side (48 neurons, total 96):
  V2a   (excitatory RS):    16 — rhythm generation, drive ipsilateral MN
  V0d   (inh, commissural):  8 — cross-inhibit contralateral V2a (half-centre)
  dI6   (inh, local):        4 — local burst-shaping inhibition
  MN    (motor, RS):        12 — project to muscle
  Renshaw (inh, RS):         8 — recurrent inhibition from MN collaterals → terminates bout

Total per side: 16+8+4+12+8 = 48  (N_CPG_PER_SIDE = 48)

Biological features:
  - Renshaw / V1-like recurrent inhibition creates bout termination
  - Bout-glide state machine: discrete swim bouts + passive glide phases
  - Half-centre oscillation via V0d commissural inhibition
  - dI6 shapes burst waveform (local, not cross-midline)
  - Frequency scales with descending drive (low→~5 Hz, high→continuous)

References:
  Kinkhabwala et al. 2011 (V0/V2 half-centre)
  McLean et al. 2007 (frequency scaling with drive)
  Bhatt et al. 2007 (bout-glide kinematics)
  Higashijima et al. 2004 (Renshaw-like recurrent inhibition)
"""
import torch
import torch.nn as nn


# Neuron counts per side — must sum to N_CPG_PER_SIDE = 48
N_V2A   = 16
N_V0D   =  8
N_DI6   =  4
N_MN    = 12
N_RSH   =  8   # Renshaw (V1-like)
N_PER_SIDE = N_V2A + N_V0D + N_DI6 + N_MN + N_RSH  # 48


class SpinalCPG(nn.Module):
    """
    Spiking LIF central pattern generator with bout-glide dynamics.

    Constructor: SpinalCPG(device=device)
    After init:  cpg.noise = p['cpg_noise']   (settable attribute)

    Step:  mL, mR, cpg_speed, cpg_turn, cpg_diag = cpg.step(cpg_drive, turn)
    """

    def __init__(self,
                 tau_m:   float = 0.5,
                 tau_syn: float = 0.6,
                 v_thresh: float = 0.5,
                 v_reset:  float = 0.0,
                 # synaptic weight parameters
                 w_v2a_mn:     float = 0.7,   # V2a → MN (excitatory)
                 w_v2a_v0d:    float = 0.6,   # V2a → V0d (excitatory)
                 w_v0d_cross:  float = 0.9,   # V0d → contralateral V2a (inhibitory)
                 w_v2a_di6:    float = 0.4,   # V2a → dI6
                 w_di6_v2a:    float = 0.3,   # dI6 → V2a (shaping)
                 w_di6_mn:     float = 0.2,   # dI6 → MN (shaping)
                 w_mn_rsh:     float = 0.5,   # MN → Renshaw (MN collateral)
                 w_rsh_v2a:    float = 0.6,   # Renshaw → V2a (bout termination)
                 noise:        float = 0.15,
                 device: str = 'cpu'):
        super().__init__()

        # LIF parameters
        self.tau_m    = tau_m
        self.tau_syn  = tau_syn
        self.v_thresh = v_thresh
        self.v_reset  = v_reset
        self.noise    = noise
        self.device   = device

        # Synaptic weights
        self.w_v2a_mn    = w_v2a_mn
        self.w_v2a_v0d   = w_v2a_v0d
        self.w_v0d_cross = w_v0d_cross
        self.w_v2a_di6   = w_v2a_di6
        self.w_di6_v2a   = w_di6_v2a
        self.w_di6_mn    = w_di6_mn
        self.w_mn_rsh    = w_mn_rsh
        self.w_rsh_v2a   = w_rsh_v2a

        self.n_total = 2 * N_PER_SIDE  # 96

        # --- index slices (left side) ---
        self._L_v2a = slice(0,                     N_V2A)
        self._L_v0d = slice(N_V2A,                 N_V2A + N_V0D)
        self._L_di6 = slice(N_V2A + N_V0D,         N_V2A + N_V0D + N_DI6)
        self._L_mn  = slice(N_V2A + N_V0D + N_DI6, N_V2A + N_V0D + N_DI6 + N_MN)
        self._L_rsh = slice(N_V2A + N_V0D + N_DI6 + N_MN, N_PER_SIDE)

        # --- index slices (right side, offset by N_PER_SIDE) ---
        _o = N_PER_SIDE
        self._R_v2a = slice(_o,                          _o + N_V2A)
        self._R_v0d = slice(_o + N_V2A,                  _o + N_V2A + N_V0D)
        self._R_di6 = slice(_o + N_V2A + N_V0D,          _o + N_V2A + N_V0D + N_DI6)
        self._R_mn  = slice(_o + N_V2A + N_V0D + N_DI6,  _o + N_V2A + N_V0D + N_DI6 + N_MN)
        self._R_rsh = slice(_o + N_V2A + N_V0D + N_DI6 + N_MN, 2 * N_PER_SIDE)

        # Neuron state
        self.v   = torch.zeros(self.n_total, device=device)
        self.syn = torch.zeros(self.n_total, device=device)

        # Bout-glide state
        self.in_bout     = False
        self.bout_timer  = 0
        self.glide_timer = 0

        # Diagnostics
        self._phase          = 0.5
        self._cycle_count    = 0
        self._prev_L_active  = False
        self.motor_L         = 0.0
        self.motor_R         = 0.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, cpg_drive: float, turn: float = 0.0):
        """
        cpg_drive : float [0,1] — descending speed command
        turn      : float [-1,1] — turn bias (positive = turn right)
        Returns   : (motor_L, motor_R, speed, turn_out, diag)
        """
        # ---- 1. During glide phase: passively coast, no CPG output ----
        if self.glide_timer > 0:
            self.glide_timer -= 1
            diag = self._make_diag(0.0, 0.0)
            return 0.0, 0.0, 0.0, 0.0, diag

        # ---- 2. Descending drive with turn asymmetry ----
        # turn > 0 → turn right → more drive to right side (motor_R > motor_L)
        # turn_out = (motor_R - motor_L)*2 so right motor must get more drive
        drive_L = cpg_drive + max(0.0, -turn) * 0.3
        drive_R = cpg_drive + max(0.0,  turn) * 0.3
        # Scale to bring V2a near threshold: resting I = 0.5, max I = 2.5
        I_v2a_L = drive_L * 2.0 + 0.5
        I_v2a_R = drive_R * 2.0 + 0.5

        # ---- 3. Build external input ----
        I_ext = torch.zeros(self.n_total, device=self.device)
        I_ext[self._L_v2a] = I_v2a_L
        I_ext[self._R_v2a] = I_v2a_R

        # ---- 4. Compute synaptic currents from previous syn state ----
        Lv2a = self.syn[self._L_v2a].mean()
        Rv2a = self.syn[self._R_v2a].mean()
        Lv0d = self.syn[self._L_v0d].mean()
        Rv0d = self.syn[self._R_v0d].mean()
        Ldi6 = self.syn[self._L_di6].mean()
        Rdi6 = self.syn[self._R_di6].mean()
        Lmn  = self.syn[self._L_mn].mean()
        Rmn  = self.syn[self._R_mn].mean()
        Lrsh = self.syn[self._L_rsh].mean()
        Rrsh = self.syn[self._R_rsh].mean()

        I_syn = torch.zeros(self.n_total, device=self.device)

        # V2a → V0d  (ipsilateral excitation)
        I_syn[self._L_v0d] += self.w_v2a_v0d * Lv2a
        I_syn[self._R_v0d] += self.w_v2a_v0d * Rv2a

        # V0d → contralateral V2a  (half-centre inhibition)
        I_syn[self._R_v2a] -= self.w_v0d_cross * Lv0d
        I_syn[self._L_v2a] -= self.w_v0d_cross * Rv0d

        # V2a → MN  (excitatory drive to motor neurons)
        I_syn[self._L_mn] += self.w_v2a_mn * Lv2a
        I_syn[self._R_mn] += self.w_v2a_mn * Rv2a

        # MN → Renshaw  (MN collateral drives Renshaw)
        I_syn[self._L_rsh] += self.w_mn_rsh * Lmn
        I_syn[self._R_rsh] += self.w_mn_rsh * Rmn

        # Renshaw → V2a  (recurrent inhibition — terminates bout)
        I_syn[self._L_v2a] -= self.w_rsh_v2a * Lrsh
        I_syn[self._R_v2a] -= self.w_rsh_v2a * Rrsh

        # V2a → dI6  (local burst shaping)
        I_syn[self._L_di6] += self.w_v2a_di6 * Lv2a
        I_syn[self._R_di6] += self.w_v2a_di6 * Rv2a

        # dI6 → V2a  (local shaping, not cross-midline)
        I_syn[self._L_v2a] -= self.w_di6_v2a * Ldi6
        I_syn[self._R_v2a] -= self.w_di6_v2a * Rdi6

        # dI6 → MN  (local shaping)
        I_syn[self._L_mn] -= self.w_di6_mn * Ldi6
        I_syn[self._R_mn] -= self.w_di6_mn * Rdi6

        # ---- 5. LIF integration ----
        noise_t = torch.randn(self.n_total, device=self.device) * self.noise
        I_net = I_ext + I_syn + noise_t
        v_new = self.tau_m * self.v + (1.0 - self.tau_m) * I_net
        spike  = (v_new >= self.v_thresh).float()
        v_new  = torch.where(spike.bool(),
                             torch.tensor(self.v_reset, device=self.device),
                             v_new)
        self.v   = v_new
        self.syn = self.tau_syn * self.syn + spike

        # ---- 6. Motor output ----
        motor_L = float(self.syn[self._L_mn].mean())
        motor_R = float(self.syn[self._R_mn].mean())
        self.motor_L = motor_L
        self.motor_R = motor_R

        # ---- 7. Bout-glide state machine ----
        motor_peak = max(motor_L, motor_R)
        if not self.in_bout and motor_peak > 0.15:
            # Start a new bout; duration scales with drive (1-3 behavioral steps)
            self.in_bout    = True
            self.bout_timer = max(1, int(cpg_drive * 3))

        if self.in_bout:
            self.bout_timer -= 1
            if self.bout_timer <= 0:
                self.in_bout    = False
                # Glide duration: longer at low drive (3 steps), near-zero at high drive
                self.glide_timer = max(1, int((1.0 - cpg_drive) * 3))

        # ---- 8. Phase / cycle tracking ----
        L_active = motor_L > motor_R
        if L_active and not self._prev_L_active:
            self._cycle_count += 1
        self._prev_L_active = L_active

        total = motor_L + motor_R
        self._phase = (motor_R / (total + 1e-8)) if total > 0.01 else 0.5

        # ---- 9. Outputs ----
        speed_out = (motor_L + motor_R) / 2.0
        turn_out  = (motor_R - motor_L) * 2.0

        diag = self._make_diag(motor_L, motor_R)
        return motor_L, motor_R, speed_out, turn_out, diag

    # ------------------------------------------------------------------
    def _make_diag(self, motor_L: float, motor_R: float) -> dict:
        return {
            'phase':        self._phase,
            'v_L':          motor_L,
            'v_R':          motor_R,
            'bout_active':  self.in_bout,
            'cycle_count':  self._cycle_count,
            'spikes_L':     float(self.syn[:N_PER_SIDE].sum()),
            'spikes_R':     float(self.syn[N_PER_SIDE:].sum()),
            'renshaw_L':    float(self.syn[self._L_rsh].mean()),
            'renshaw_R':    float(self.syn[self._R_rsh].mean()),
            'glide_active': self.glide_timer > 0,
        }

    # ------------------------------------------------------------------
    def reset(self):
        """Reset all neuron state and bout-glide counters."""
        self.v   = torch.zeros(self.n_total, device=self.device)
        self.syn = torch.zeros(self.n_total, device=self.device)
        self.in_bout      = False
        self.bout_timer   = 0
        self.glide_timer  = 0
        self._phase       = 0.5
        self._cycle_count = 0
        self._prev_L_active = False
        self.motor_L = 0.0
        self.motor_R = 0.0
