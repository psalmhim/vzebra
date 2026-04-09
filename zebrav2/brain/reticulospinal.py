"""
Named reticulospinal neurons (21 per side).

Anatomy notes:
  Mauthner cell (M-cell): large, myelinated hindbrain neuron. ONE per side.
    - Receives direct input from VIII nerve (acoustic/vestibular) and tectum (visual looming).
    - Axon crosses the midline immediately → activates CONTRALATERAL axial muscles.
    - Left M-cell fires → right body wall contracts → fish bends RIGHT (away from left threat).
    - Crossed inhibition via CoLo cells (commissural local interneurons, also called PHP cells):
      when M-cell fires it simultaneously drives CoLo → suppresses contralateral M-cell,
      guaranteeing a unidirectional C-start.

  MiD2 / MiD3: T-start (non-Mauthner escape) interneurons.
    - Activated by sub-threshold or sustained looming; longer latency than M-cell.
    - Produce a smaller body bend (~80°) vs M-cell (~150°), but are not refractory-blocked
      while M-cell is recovering.

  RoM2a-d: voluntary turn (descend from pallium-D / striatum via BG gate).
  MeM1-8:  sustained forward locomotion.
  CaD1-6:  speed modulation (scale overall motor drive).

C-start stage anatomy (mapped to 4 simulation substeps):
  Stage 1 (~10 ms real): ipsilateral fast-twitch bend — sharp high-amplitude turn, low speed.
  Stage 2 (~15 ms real): contralateral counterbend + rostro-caudal propulsion wave.

T-start stage anatomy (3 substeps):
  Smaller initial bend, faster transition to propulsion.
"""
import math
import random
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE


class ReticulospinalSystem(nn.Module):
    NEURON_NAMES = [
        'Mauthner', 'MiD2', 'MiD3',          # fast escape
        'RoM2a', 'RoM2b', 'RoM2c', 'RoM2d',  # voluntary turn
        'MeM1', 'MeM2', 'MeM3', 'MeM4',      # sustained locomotion
        'MeM5', 'MeM6', 'MeM7', 'MeM8',
        'CaD1', 'CaD2', 'CaD3', 'CaD4',      # speed modulation
        'CaD5', 'CaD6',
    ]  # 21 neurons per side

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_rs = len(self.NEURON_NAMES)  # 21

        # Per-side firing rate buffers (used by visualizer)
        self.register_buffer('rate_L', torch.zeros(self.n_rs, device=device))
        self.register_buffer('rate_R', torch.zeros(self.n_rs, device=device))

        # Motor output buffers
        self.register_buffer('motor_turn',  torch.tensor(0.0, device=device))
        self.register_buffer('motor_speed', torch.tensor(1.0, device=device))

        # --- Mauthner (C-start) state machines — one per side ---
        # refrac: countdown to zero; while > 0 the cell cannot fire.
        # active: countdown driving motor sequence; while > 0 we are in a C-start.
        self.mauthner_refrac_L = 0  # refractory counter for left  M-cell
        self.mauthner_refrac_R = 0  # refractory counter for right M-cell
        self.mauthner_active_L = 0  # motor-sequence step for left  M-cell
        self.mauthner_active_R = 0  # motor-sequence step for right M-cell

        # --- T-start (MiD2/MiD3) state ---
        self.tstart_refrac  = 0
        self.tstart_active  = 0
        self.tstart_dir     = 1.0   # direction when T-start was triggered

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cstart_motor(step: int, direction: float):
        """
        C-start motor sequence.  step counts DOWN from 4 to 1.
          step 4: Stage-1 sharp ipsilateral bend (large turn, low speed)
          step 3: late Stage 1 / early Stage 2 extension
          step 2: Stage-2 counterbend + propulsion onset
          step 1: full propulsion burst
        """
        if step == 4:
            return 1.5 * direction, 0.3
        elif step == 3:
            return 1.0 * direction, 0.8
        elif step == 2:
            return 0.2 * direction, 1.5
        else:  # step == 1
            return 0.0,             1.8

    @staticmethod
    def _tstart_motor(step: int, direction: float):
        """
        T-start motor sequence.  step counts DOWN from 3 to 1.
          step 3: moderate initial bend
          step 2: partial extension + propulsion
          step 1: full propulsion
        """
        if step == 3:
            return 0.8 * direction, 0.4
        elif step == 2:
            return 0.3 * direction, 1.2
        else:  # step == 1
            return 0.0,             1.5

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self,
                sgc_rate:   torch.Tensor,
                bg_gate:    float,
                pal_d_rate: torch.Tensor,
                flee_dir:   float,
                goal_speed: float,
                looming:    bool) -> dict:
        """
        sgc_rate   : (N_SGC_E,)  looming input from tectal SGC cells
        bg_gate    : float 0-1   basal ganglia go-signal for voluntary movement
        pal_d_rate : (N_PAL_D,)  pallium-D firing rates (L half vs R half → turn bias)
        flee_dir   : float -1..1 escape direction from downstream threat assessment
                     convention: negative = threat on left  → flee right (positive turn)
                                 positive = threat on right → flee left  (negative turn)
        goal_speed : float 0-2   desired cruising speed
        looming    : bool        tectum looming flag

        Returns dict with keys:
          'turn'   : float  (-1 .. 1)
          'speed'  : float  (0  .. 2)
          'cstart' : bool
          'tstart' : bool
          'rate_L' : Tensor (n_rs,)
          'rate_R' : Tensor (n_rs,)
        """
        sgc_mean = sgc_rate.mean().item()

        # ----------------------------------------------------------
        # 1. Decrement all refractory / sequence counters
        # ----------------------------------------------------------
        if self.mauthner_refrac_L > 0: self.mauthner_refrac_L -= 1
        if self.mauthner_refrac_R > 0: self.mauthner_refrac_R -= 1
        if self.tstart_refrac      > 0: self.tstart_refrac     -= 1

        # ----------------------------------------------------------
        # 2. C-start: serve ongoing motor sequence (highest priority)
        # ----------------------------------------------------------
        if self.mauthner_active_L > 0:
            direction = -1.0   # left M-cell → right body bend → leftward turn
            turn, speed = self._cstart_motor(self.mauthner_active_L, direction)
            self.mauthner_active_L -= 1
            self._fill_motor(turn, speed)
            return self._make_out(turn, speed, cstart=True, tstart=False)

        if self.mauthner_active_R > 0:
            direction = +1.0   # right M-cell → left body bend → rightward turn
            turn, speed = self._cstart_motor(self.mauthner_active_R, direction)
            self.mauthner_active_R -= 1
            self._fill_motor(turn, speed)
            return self._make_out(turn, speed, cstart=True, tstart=False)

        # ----------------------------------------------------------
        # 3. T-start: serve ongoing motor sequence (second priority)
        # ----------------------------------------------------------
        if self.tstart_active > 0:
            turn, speed = self._tstart_motor(self.tstart_active, self.tstart_dir)
            self.tstart_active -= 1
            self._fill_motor(turn, speed)
            return self._make_out(turn, speed, cstart=False, tstart=True)

        # ----------------------------------------------------------
        # 4. Check for new C-start trigger (looming + SGC above threshold)
        # ----------------------------------------------------------
        if looming and sgc_mean > 0.05:
            # Determine which M-cell to fire based on threat laterality.
            # M-cells are contralateral: threat on RIGHT → fire LEFT M-cell
            #                            threat on LEFT  → fire RIGHT M-cell
            enemy_on_right = flee_dir > 0   # positive flee_dir = threat on right
            enemy_on_left  = flee_dir < 0

            trigger_L = False
            trigger_R = False

            if enemy_on_right and self.mauthner_refrac_L == 0 and self.mauthner_refrac_R == 0:
                # Threat on right → left M-cell fires → fish bends left (away)
                trigger_L = True
            elif enemy_on_left and self.mauthner_refrac_R == 0 and self.mauthner_refrac_L == 0:
                # Threat on left → right M-cell fires → fish bends right (away)
                trigger_R = True
            elif self.mauthner_refrac_L == 0 and self.mauthner_refrac_R == 0:
                # No laterality info — random tie-break (50/50)
                if random.random() > 0.5:
                    trigger_L = True
                else:
                    trigger_R = True

            if trigger_L and self.mauthner_refrac_L == 0 and self.mauthner_refrac_R == 0:
                # Left M-cell fires; CoLo cells lock out right M-cell (longer block)
                self.mauthner_active_L = 4
                self.mauthner_refrac_L = 12   # ~600 ms own refractory
                self.mauthner_refrac_R = 20   # crossed inhibition: contralateral blocked longer
                direction = -1.0
                turn, speed = self._cstart_motor(4, direction)
                self.mauthner_active_L -= 1   # consumed step 4 on this frame
                self._fill_motor(turn, speed)
                return self._make_out(turn, speed, cstart=True, tstart=False)

            elif trigger_R and self.mauthner_refrac_R == 0 and self.mauthner_refrac_L == 0:
                self.mauthner_active_R = 4
                self.mauthner_refrac_R = 12
                self.mauthner_refrac_L = 20
                direction = +1.0
                turn, speed = self._cstart_motor(4, direction)
                self.mauthner_active_R -= 1
                self._fill_motor(turn, speed)
                return self._make_out(turn, speed, cstart=True, tstart=False)

        # ----------------------------------------------------------
        # 5. Check for new T-start trigger (MiD2/MiD3 pathway)
        # ----------------------------------------------------------
        if looming and self.tstart_refrac == 0:
            m_refrac_active = (self.mauthner_refrac_L > 0 or self.mauthner_refrac_R > 0)

            # Sub-threshold SGC: primary T-start (M-cell not triggered)
            primary_tstart = (sgc_mean > 0.02 and sgc_mean < 0.05
                              and self.mauthner_refrac_L == 0
                              and self.mauthner_refrac_R == 0)

            # Backup T-start: M-cell already in refractory but threat persists
            backup_tstart = (sgc_mean > 0.02 and m_refrac_active)

            if primary_tstart or backup_tstart:
                # Direction mirrors C-start logic
                if flee_dir > 0:
                    self.tstart_dir = -1.0   # threat right → turn left
                elif flee_dir < 0:
                    self.tstart_dir = +1.0   # threat left  → turn right
                else:
                    self.tstart_dir = -1.0 if random.random() > 0.5 else 1.0
                self.tstart_active = 3
                self.tstart_refrac = 15
                turn, speed = self._tstart_motor(3, self.tstart_dir)
                self.tstart_active -= 1
                self._fill_motor(turn, speed)
                return self._make_out(turn, speed, cstart=False, tstart=True)

        # ----------------------------------------------------------
        # 6. Voluntary movement (pallium-D asymmetry × BG gate)
        # ----------------------------------------------------------
        half = len(pal_d_rate) // 2
        pal_L = pal_d_rate[:half].mean().item()
        pal_R = pal_d_rate[half:].mean().item()
        voluntary_turn = (pal_R - pal_L) * bg_gate * 2.0

        # Blend with flee direction
        turn = float(math.copysign(1, flee_dir) * abs(flee_dir) * 0.7
                     + voluntary_turn * 0.3)
        turn = max(-1.0, min(1.0, turn))

        # Speed: flee bypasses BG gate; voluntary is gated
        if abs(flee_dir) > 0.1:
            speed = goal_speed
        else:
            speed = goal_speed * (0.5 + 0.5 * bg_gate)

        self._fill_motor(turn, speed)
        return self._make_out(turn, speed, cstart=False, tstart=False)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _fill_motor(self, turn: float, speed: float):
        self.motor_turn.fill_(turn)
        self.motor_speed.fill_(speed)

    def _make_out(self, turn: float, speed: float,
                  cstart: bool, tstart: bool) -> dict:
        return {
            'turn':   turn,
            'speed':  speed,
            'cstart': cstart,
            'tstart': tstart,
            'rate_L': self.rate_L,
            'rate_R': self.rate_R,
        }

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self):
        self.mauthner_refrac_L = 0
        self.mauthner_refrac_R = 0
        self.mauthner_active_L = 0
        self.mauthner_active_R = 0
        self.tstart_refrac     = 0
        self.tstart_active     = 0
        self.tstart_dir        = 1.0
        self.rate_L.zero_()
        self.rate_R.zero_()
        self.motor_turn.zero_()
        self.motor_speed.fill_(1.0)
