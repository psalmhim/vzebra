"""
Spiking lateral line: mechanoreceptive multi-source detection.

Zebrafish lateral line has two neuromast types:
  Superficial neuromasts (SN) — velocity-sensitive, 0-50 Hz
    Detect slow-moving objects: prey, schooling fish, gentle currents.
  Canal neuromasts (CN) — acceleration-sensitive, 50-300 Hz
    Detect fast-moving objects: predator rushes, escape turns.

Architecture (24 neurons total):
  superficial_ant:  8 RS neurons  — anterior, velocity-sensitive
  superficial_post: 8 RS neurons  — posterior, velocity-sensitive
  canal_ant:        4 CH neurons  — anterior, acceleration-sensitive
  canal_post:       4 CH neurons  — posterior, acceleration-sensitive

Sources processed independently (priority: predator > conspecific > prey):
  Predator  — high-speed approach → canal drive → high_freq_threat
  Prey/food — slow nearby objects → superficial drive → low_freq_prey
  Conspecifics — swimming neighbours → superficial drive → conspecific_dist
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer

_SUBSTEPS = 10  # fixed — too expensive to use full SUBSTEPS here


class SpikingLateralLine(nn.Module):
    # Detection ranges (pixels)
    SN_RANGE = 200.0   # superficial: velocity, slow objects
    CN_RANGE = 150.0   # canal: acceleration, fast objects
    PREY_RANGE = 100.0

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device

        # --- Superficial neuromasts (RS, velocity-sensitive) ---
        self.superficial_ant  = IzhikevichLayer(8, 'RS', device)
        self.superficial_post = IzhikevichLayer(8, 'RS', device)
        self.superficial_ant.i_tonic.fill_(-2.0)
        self.superficial_post.i_tonic.fill_(-2.0)

        # --- Canal neuromasts (CH, acceleration-sensitive) ---
        self.canal_ant  = IzhikevichLayer(4, 'CH', device)
        self.canal_post = IzhikevichLayer(4, 'CH', device)
        self.canal_ant.i_tonic.fill_(-2.0)
        self.canal_post.i_tonic.fill_(-2.0)

        # Rate buffers
        self.register_buffer('ant_rate',  torch.zeros(8, device=device))
        self.register_buffer('post_rate', torch.zeros(8, device=device))

        # Scalar state (not torch buffers — scalars updated each step)
        self.proximity      = 0.0
        self.flow_direction = 0.0
        self.flow_magnitude = 0.0

        # Velocity history for canal (acceleration) computation
        self._prev_pred_vx = 0.0
        self._prev_pred_vy = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rel_angle(dx: float, dy: float, fish_heading: float) -> float:
        """Bearing to (dx,dy) relative to fish heading, in (-π, π)."""
        angle_to = math.atan2(dy, dx)
        return math.atan2(
            math.sin(angle_to - fish_heading),
            math.cos(angle_to - fish_heading),
        )

    @staticmethod
    def _sn_decay(dist: float) -> float:
        """Superficial neuromast proximity decay — 1/(1+d/50)²."""
        return 1.0 / (1.0 + dist / 50.0) ** 2

    @staticmethod
    def _cn_decay(dist: float) -> float:
        """Canal neuromast proximity decay — 1/(1+d/30)²."""
        return 1.0 / (1.0 + dist / 30.0) ** 2

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self,
                fish_x: float, fish_y: float, fish_heading: float,
                pred_x: float, pred_y: float,
                pred_vx: float = 0.0, pred_vy: float = 0.0,
                foods: list = None,
                conspecifics: list = None) -> dict:

        if foods is None:
            foods = []
        if conspecifics is None:
            conspecifics = []

        # === 1. PREDATOR — canal (acceleration) + superficial (velocity) ===

        pred_dx = pred_x - fish_x
        pred_dy = pred_y - fish_y
        pred_dist = math.sqrt(pred_dx * pred_dx + pred_dy * pred_dy) + 1e-6

        pred_accel = math.sqrt(
            (pred_vx - self._prev_pred_vx) ** 2 +
            (pred_vy - self._prev_pred_vy) ** 2
        )
        self._prev_pred_vx = pred_vx
        self._prev_pred_vy = pred_vy

        pred_speed = math.sqrt(pred_vx ** 2 + pred_vy ** 2)

        if pred_dist <= self.CN_RANGE:
            pred_rel = self._rel_angle(pred_dx, pred_dy, fish_heading)
            cn_prox  = self._cn_decay(pred_dist)
            # Frontal/caudal split via cosine projection
            frontal = max(0.0,  math.cos(pred_rel))
            caudal  = max(0.0, -math.cos(pred_rel))

            # Canal drive: acceleration-based
            cn_drive_ant  = cn_prox * pred_accel * frontal * 20.0
            cn_drive_post = cn_prox * pred_accel * caudal  * 16.0

            # Superficial contribution from predator (velocity)
            if pred_dist <= self.SN_RANGE:
                sn_prox = self._sn_decay(pred_dist)
                sn_pred_ant  = sn_prox * pred_speed * frontal * 12.0
                sn_pred_post = sn_prox * pred_speed * caudal  * 10.0
            else:
                sn_pred_ant = sn_pred_post = 0.0

            self.proximity      = max(0.0, 1.0 - pred_dist / self.CN_RANGE)
            self.flow_direction = math.sin(pred_rel)
            self.flow_magnitude = min(1.0, pred_speed * self.proximity * 0.1)
            pred_reported_dist  = pred_dist
        else:
            cn_drive_ant  = cn_drive_post = 0.0
            sn_pred_ant   = sn_pred_post  = 0.0
            self.proximity      = 0.0
            self.flow_direction = 0.0
            self.flow_magnitude = 0.0
            pred_reported_dist  = 999.0

        # === 2. PREY / FOOD — superficial (velocity via relative flow) ===

        best_prey_dist  = 999.0
        best_prey_angle = 0.0
        sn_prey_ant     = 0.0
        sn_prey_post    = 0.0

        for food in foods:
            try:
                fx, fy = float(food[0]), float(food[1])
            except (IndexError, TypeError):
                continue
            fdx  = fx - fish_x
            fdy  = fy - fish_y
            fdist = math.sqrt(fdx * fdx + fdy * fdy) + 1e-6

            if fdist > self.PREY_RANGE:
                continue

            # Static food generates relative-flow signal proportional to 1/dist²
            signal = min(1.0, (30.0 / fdist) ** 2)
            frel   = self._rel_angle(fdx, fdy, fish_heading)
            frontal_f = max(0.0,  math.cos(frel))
            caudal_f  = max(0.0, -math.cos(frel))

            sn_prey_ant  += signal * frontal_f * 8.0
            sn_prey_post += signal * caudal_f  * 6.0

            if fdist < best_prey_dist:
                best_prey_dist  = fdist
                best_prey_angle = frel

        # Clamp accumulated prey drives
        sn_prey_ant  = min(sn_prey_ant,  15.0)
        sn_prey_post = min(sn_prey_post, 12.0)

        # === 3. CONSPECIFICS — superficial (medium-frequency swimming) ===

        best_conspc_dist   = 999.0
        sn_conspc_ant      = 0.0
        sn_conspc_post     = 0.0

        for fish in conspecifics:
            try:
                cx = float(fish['x'])
                cy = float(fish['y'])
                spd = float(fish.get('speed', 1.0))
            except (KeyError, TypeError):
                continue
            cdx   = cx - fish_x
            cdy   = cy - fish_y
            cdist = math.sqrt(cdx * cdx + cdy * cdy) + 1e-6

            if cdist > 120.0:
                continue

            cn_prox_c = self._sn_decay(cdist)
            signal_c  = cn_prox_c * spd
            crel      = self._rel_angle(cdx, cdy, fish_heading)
            frontal_c = max(0.0,  math.cos(crel))
            caudal_c  = max(0.0, -math.cos(crel))

            sn_conspc_ant  += signal_c * frontal_c * 6.0
            sn_conspc_post += signal_c * caudal_c  * 5.0

            if cdist < best_conspc_dist:
                best_conspc_dist = cdist

        sn_conspc_ant  = min(sn_conspc_ant,  12.0)
        sn_conspc_post = min(sn_conspc_post, 10.0)

        # === 4. Assemble current injections ===

        # Superficial: prey + conspecific + predator-slow
        I_sup_ant  = torch.full((8,),
                                sn_prey_ant + sn_conspc_ant + sn_pred_ant,
                                device=self.device)
        I_sup_post = torch.full((8,),
                                sn_prey_post + sn_conspc_post + sn_pred_post,
                                device=self.device)

        # Canal: predator acceleration only
        I_can_ant  = torch.full((4,), cn_drive_ant,  device=self.device)
        I_can_post = torch.full((4,), cn_drive_post, device=self.device)

        # === 5. Simulate ===

        for _ in range(_SUBSTEPS):
            noise4 = torch.randn(4, device=self.device) * 0.5
            noise8 = torch.randn(8, device=self.device) * 0.5
            self.superficial_ant(I_sup_ant   + noise8)
            self.superficial_post(I_sup_post + noise8)
            self.canal_ant(I_can_ant         + noise4)
            self.canal_post(I_can_post       + noise4)

        # === 6. Read rates ===

        self.ant_rate.copy_(self.superficial_ant.rate)
        self.post_rate.copy_(self.superficial_post.rate)

        can_ant_mean  = float(self.canal_ant.rate.mean())
        can_post_mean = float(self.canal_post.rate.mean())
        sup_ant_mean  = float(self.superficial_ant.rate.mean())
        sup_post_mean = float(self.superficial_post.rate.mean())

        # === 7. Derived output signals ===

        high_freq_threat = (can_ant_mean + can_post_mean) / 2.0

        # low_freq_prey: superficial activity gated by whether prey is present
        prey_fraction = 1.0 if best_prey_dist < self.PREY_RANGE else 0.0
        low_freq_prey = (sup_ant_mean + sup_post_mean) / 2.0 * prey_fraction

        # Threshold calibrated to IzhikevichLayer tau_rate=0.02 with 10 substeps:
        # food within ~35px gives low_freq_prey ≈ 0.009; baseline = 0.000
        prey_detected = bool(low_freq_prey > 0.005 and best_prey_dist < self.PREY_RANGE)

        return {
            # --- Legacy keys (preserved) ---
            'proximity':       self.proximity,
            'flow_direction':  self.flow_direction,
            'flow_magnitude':  self.flow_magnitude,
            'ant_rate':        float(self.ant_rate.mean()),
            'post_rate':       float(self.post_rate.mean()),
            'dist':            pred_reported_dist,
            # --- New keys ---
            'prey_detected':   prey_detected,
            'prey_direction':  best_prey_angle,
            'prey_dist':       best_prey_dist,
            'high_freq_threat': min(1.0, high_freq_threat),
            'low_freq_prey':   min(1.0, low_freq_prey),
            'conspecific_dist': best_conspc_dist,
        }

    # ------------------------------------------------------------------

    def reset(self):
        self.superficial_ant.reset()
        self.superficial_post.reset()
        self.canal_ant.reset()
        self.canal_post.reset()
        self.ant_rate.zero_()
        self.post_rate.zero_()
        self.proximity      = 0.0
        self.flow_direction = 0.0
        self.flow_magnitude = 0.0
        self._prev_pred_vx  = 0.0
        self._prev_pred_vy  = 0.0
