"""
Spiking olfactory system: alarm substance + food odor.

Zebrafish olfaction:
  Olfactory epithelium → Olfactory bulb → Telencephalon (Dp, Dl)

Alarm substance (Schreckstoff): released by injured conspecifics,
triggers antipredator behavior (Speedie & Gerlai 2008).

Food odor: amino acid gradients guide foraging (Friedrich & Korsching 1997).

Architecture:
  10 alarm neurons (RS)     — detect alarm substance concentration
  10 food-odor neurons (RS) — detect amino acid gradient
   8 bilateral neurons (RS)  — 4 L + 4 R, driven by L/R nostril difference

Diffusion model (Fick steady-state):
  C(r) = S * exp(-r / λ) / (r/50 + 1)
  λ_food  = 70 px  (amino acids diffuse slowly)
  λ_alarm = 100 px (Schreckstoff spreads farther)

Bilateral sampling:
  Two nares offset ±5 px perpendicular to heading.
  Difference C_R − C_L divided by sum gives turn signal (−1 to +1).

Temporal gradient:
  Tracks Δ concentration step-to-step; decrease suggests wrong direction.

Receptor adaptation:
  Sustained strong odor desensitises receptors (0.7 max suppression).
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer

# ---------------------------------------------------------------------------
# Physics helper
# ---------------------------------------------------------------------------

def _concentration(source_strength: float, dist: float,
                   lambda_diff: float = 70.0) -> float:
    """
    Steady-state Fick diffusion concentration.

    C = S * exp(-r / λ) / (r/50 + 1)

    Args:
        source_strength: dimensionless source intensity [0, 1]
        dist:            distance in pixels
        lambda_diff:     diffusion length constant (pixels)
    """
    return source_strength * math.exp(-dist / lambda_diff) / (dist / 50.0 + 1.0)


class SpikingOlfaction(nn.Module):
    def __init__(self, n_alarm: int = 10, n_food: int = 10,
                 n_bilateral: int = 8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_alarm = n_alarm
        self.n_food = n_food
        self.n_bilateral = n_bilateral          # must be even
        self.n_bilateral_side = n_bilateral // 2

        # Spiking populations
        self.alarm_pop    = IzhikevichLayer(n_alarm,    'RS', device)
        self.food_pop     = IzhikevichLayer(n_food,     'RS', device)
        self.bilateral_pop = IzhikevichLayer(n_bilateral, 'RS', device)

        # Alarm neurons: light suppression to prevent noise-driven spikes,
        # but low enough that alarm signals (I > ~4 pA) reliably fire.
        # Food/bilateral: stronger suppression — only fire on salient odor.
        self.alarm_pop.i_tonic.fill_(-1.0)
        self.food_pop.i_tonic.fill_(-2.0)
        self.bilateral_pop.i_tonic.fill_(-2.0)

        self.register_buffer('alarm_rate',    torch.zeros(n_alarm,    device=device))
        self.register_buffer('food_rate',     torch.zeros(n_food,     device=device))
        self.register_buffer('bilateral_rate', torch.zeros(n_bilateral, device=device))

        # Public state (read by brain_v2)
        self.alarm_level       = 0.0
        self.food_gradient_dir = 0.0   # relative angle to strongest food odor (rad)
        self.food_odor_strength = 0.0  # effective concentration [0, 1]
        self.bilateral_diff    = 0.0   # C_R − C_L (normalised, −1 to +1)
        self.temporal_gradient = 0.0   # Δ concentration per step
        self.receptor_adapt    = 0.0   # adaptation state [0, 1]

        # Geometry
        self._nare_offset = 5.0        # nostril lateral offset (px)

        # Internal state
        self._prev_concentration = 0.0
        self._receptor_adapt_state = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nostril_positions(self, fish_x: float, fish_y: float,
                           fish_heading: float):
        """Return (x_L, y_L), (x_R, y_R) for the two nares."""
        d = self._nare_offset
        # Perpendicular to heading: left = heading + π/2
        sin_h = math.sin(fish_heading)
        cos_h = math.cos(fish_heading)
        # Left nostril: +90° from heading direction
        nL_x = fish_x - d * sin_h
        nL_y = fish_y + d * cos_h
        # Right nostril: −90° from heading direction
        nR_x = fish_x + d * sin_h
        nR_y = fish_y - d * cos_h
        return (nL_x, nL_y), (nR_x, nR_y)

    def _sample_food_concentration(self, px: float, py: float,
                                   foods: list) -> float:
        """Sum Fick concentration from all food sources at position (px, py)."""
        total = 0.0
        for food in foods:
            fx, fy = food[0], food[1]
            dist = math.sqrt((fx - px) ** 2 + (fy - py) ** 2) + 1e-6
            total += _concentration(1.0, dist, lambda_diff=70.0)
        return total

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, fish_x: float, fish_y: float, fish_heading: float,
                foods: list, conspecific_injured: bool = False,
                pred_dist: float = 999.0,
                conspc_dist: float = 30.0) -> dict:

        # ── 1. Bilateral nostril sampling ──────────────────────────────
        (nL_x, nL_y), (nR_x, nR_y) = self._nostril_positions(
            fish_x, fish_y, fish_heading)

        C_L = self._sample_food_concentration(nL_x, nL_y, foods)
        C_R = self._sample_food_concentration(nR_x, nR_y, foods)

        # Also sample from fish centre for temporal tracking
        C_centre = self._sample_food_concentration(fish_x, fish_y, foods)
        total_C_now = C_centre  # use centre for scalar concentration

        # Bilateral difference: range −1 to +1 (positive → turn right)
        bilateral_diff = (C_R - C_L) / (C_R + C_L + 1e-8)

        # ── 2. Temporal gradient ───────────────────────────────────────
        temporal_gradient = total_C_now - self._prev_concentration
        self._prev_concentration = total_C_now

        # Temporal turn bias: if concentration is decreasing, suggest reversal
        if abs(temporal_gradient) > 0.01:
            temporal_turn_bias = -math.copysign(0.3, temporal_gradient)
        else:
            temporal_turn_bias = 0.0

        # ── 3. Receptor adaptation ─────────────────────────────────────
        self._receptor_adapt_state = (
            0.9 * self._receptor_adapt_state
            + 0.1 * min(1.0, total_C_now * 2.0)
        )
        effective_C = total_C_now * (1.0 - 0.7 * self._receptor_adapt_state)

        # ── 4. Gradient direction estimation ──────────────────────────
        if abs(bilateral_diff) > 0.1:
            # Bilateral difference gives instantaneous turn direction
            gradient_dir = bilateral_diff * math.pi * 0.5
        else:
            # Fallback: direct bearing to the nearest/strongest food source
            best_dist  = 1e9
            best_angle = 0.0
            for food in foods:
                fx, fy = food[0], food[1]
                dx = fx - fish_x
                dy = fy - fish_y
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                c = _concentration(1.0, dist, lambda_diff=70.0)
                if c > _concentration(1.0, best_dist, lambda_diff=70.0):
                    best_dist  = dist
                    best_angle = math.atan2(dy, dx)

            rel = best_angle - fish_heading
            gradient_dir = math.atan2(math.sin(rel), math.cos(rel))

        # Blend in temporal hint: if we're moving in the wrong direction
        if temporal_turn_bias != 0.0 and abs(bilateral_diff) < 0.05:
            # flip estimated direction by π when temporal says we're receding
            gradient_dir = math.atan2(
                math.sin(gradient_dir + math.pi * math.copysign(0.15, temporal_turn_bias)),
                math.cos(gradient_dir + math.pi * math.copysign(0.15, temporal_turn_bias))
            )

        self.food_gradient_dir  = gradient_dir
        self.food_odor_strength = min(1.0, effective_C)
        self.bilateral_diff     = bilateral_diff
        self.temporal_gradient  = temporal_gradient
        self.receptor_adapt     = self._receptor_adapt_state

        # ── 5. Alarm substance (Fick, λ=100px) ────────────────────────
        # Sources are independent: injured conspecific diffuses from conspc_dist,
        # predator-proximity alarm diffuses from pred_dist.
        alarm_C = 0.0
        if conspecific_injured:
            # Schreckstoff released by injured conspecific at conspc_dist
            alarm_C = 0.8 * math.exp(-conspc_dist / 100.0)
        if pred_dist < 50:
            # Very close predator (fish cornered) also triggers alarm response
            pred_alarm = 0.8 * max(0.0, 1.0 - pred_dist / 50.0) * math.exp(-pred_dist / 100.0)
            alarm_C = max(alarm_C, pred_alarm)
        alarm_C = min(1.0, alarm_C)
        alarm_drive = alarm_C
        self.alarm_level = alarm_drive  # raw level before spiking scale

        I_alarm = torch.full((self.n_alarm,), alarm_drive * 20.0,
                             device=self.device)

        # ── 6. Spiking populations ────────────────────────────────────
        # Food population (directionally biased)
        I_food = torch.full((self.n_food,), self.food_odor_strength * 12.0,
                            device=self.device)
        if self.food_gradient_dir > 0:   # food on left (CCW)
            I_food[:self.n_food // 2] *= 1.5
        else:
            I_food[self.n_food // 2:] *= 1.5

        # Bilateral population: left neurons driven by C_L dominance,
        #                        right neurons driven by C_R dominance
        I_bilateral = torch.zeros(self.n_bilateral, device=self.device)
        I_bilateral_L = max(0.0, -bilateral_diff) * 10.0  # left nostril stronger
        I_bilateral_R = max(0.0,  bilateral_diff) * 10.0  # right nostril stronger
        I_bilateral[:self.n_bilateral_side]  = I_bilateral_L
        I_bilateral[self.n_bilateral_side:]  = I_bilateral_R

        noise = 0.3
        for _ in range(10):   # reduced substeps
            self.alarm_pop(
                I_alarm + torch.randn(self.n_alarm, device=self.device) * noise)
            self.food_pop(
                I_food  + torch.randn(self.n_food,  device=self.device) * noise)
            self.bilateral_pop(
                I_bilateral + torch.randn(self.n_bilateral, device=self.device) * noise)

        self.alarm_rate.copy_(self.alarm_pop.rate)
        self.food_rate.copy_(self.food_pop.rate)
        self.bilateral_rate.copy_(self.bilateral_pop.rate)

        self.alarm_level = float(self.alarm_rate.mean()) * 5.0

        return {
            'alarm_level':      self.alarm_level,
            'food_gradient_dir': self.food_gradient_dir,
            'food_odor_strength': self.food_odor_strength,
            'alarm_rate':       float(self.alarm_rate.mean()),
            'food_odor_rate':   float(self.food_rate.mean()),
            # new keys
            'bilateral_diff':   bilateral_diff,
            'temporal_gradient': temporal_gradient,
            'receptor_adapt':   self._receptor_adapt_state,
            'C_L':              C_L,
            'C_R':              C_R,
        }

    # ------------------------------------------------------------------
    # Bias outputs (unchanged interface)
    # ------------------------------------------------------------------

    def get_forage_bias(self) -> float:
        """Olfactory EFE bias toward food (negative = attract)."""
        return -self.food_odor_strength * 0.2

    def get_flee_bias(self) -> float:
        """Alarm substance drives flee."""
        return -self.alarm_level * 0.3

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        self.alarm_pop.reset()
        self.food_pop.reset()
        self.bilateral_pop.reset()
        self.alarm_rate.zero_()
        self.food_rate.zero_()
        self.bilateral_rate.zero_()
        self.alarm_level          = 0.0
        self.food_gradient_dir    = 0.0
        self.food_odor_strength   = 0.0
        self.bilateral_diff       = 0.0
        self.temporal_gradient    = 0.0
        self.receptor_adapt       = 0.0
        self._prev_concentration  = 0.0
        self._receptor_adapt_state = 0.0
