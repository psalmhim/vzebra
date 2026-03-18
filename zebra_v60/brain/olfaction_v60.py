"""
Olfactory System — chemical gradient navigation (Step 35).

Implements two olfactory channels:
  1. Food odour: diffusion-based concentration gradient from food items
  2. Alarm substance (Schreckstoff): released by injured conspecifics,
     triggers group flight response

The olfactory epithelium samples concentration at bilateral nostrils
(left/right of the head), providing directional chemotaxis via
concentration difference.

Neuroscience: zebrafish olfactory rosette has ~140k sensory neurons
(Yoshihara 2009). Food odour activates lateral olfactory tract (LOT)
→ posterior telencephalon (foraging). Alarm substance activates medial
tract (MOT) → habenula → interpeduncular nucleus (fear/avoidance)
(Jesuthasan & Bhatt 2012).

Pure numpy — no torch.
"""
import math
import numpy as np


class OlfactorySystem:
    """Bilateral olfactory system with food and alarm channels.

    Args:
        nostril_spacing: float — distance between L/R nostrils (px)
        food_diffusion_radius: float — how far food odour spreads (px)
        alarm_diffusion_radius: float — alarm substance spread (px)
        alarm_decay: float — alarm intensity decay per step
    """

    def __init__(self, nostril_spacing=8.0, food_diffusion_radius=200.0,
                 alarm_diffusion_radius=250.0, alarm_decay=0.95):
        self.nostril_spacing = nostril_spacing
        self.food_diff_r = food_diffusion_radius
        self.alarm_diff_r = alarm_diffusion_radius
        self.alarm_decay = alarm_decay

        # Alarm substance sources (from injured/fleeing conspecifics)
        self._alarm_sources = []  # [(x, y, intensity)]

    def step(self, fish_pos, fish_heading, foods, alarm_events=None):
        """Sample olfactory concentration at bilateral nostrils.

        Args:
            fish_pos: [x, y] — gym pixel position
            fish_heading: float — radians
            foods: list of [x, y, ...] — food item positions
            alarm_events: list of {"x", "y", "intensity"} or None
                          — new alarm substance releases this step

        Returns:
            food_L, food_R: float — food concentration at L/R nostril
            alarm_L, alarm_R: float — alarm concentration at L/R nostril
            diag: dict — aggregate features
        """
        fx, fy = float(fish_pos[0]), float(fish_pos[1])
        cos_h = math.cos(fish_heading)
        sin_h = math.sin(fish_heading)
        half_s = self.nostril_spacing / 2

        # Nostril positions (left = +90° from heading, right = -90°)
        lx = fx + cos_h * 5 - sin_h * half_s
        ly = fy + sin_h * 5 + cos_h * half_s
        rx = fx + cos_h * 5 + sin_h * half_s
        ry = fy + sin_h * 5 - cos_h * half_s

        # Food odour: sum of Gaussian plumes from all food items
        food_L = 0.0
        food_R = 0.0
        for food in foods:
            food_x, food_y = food[0], food[1]
            dL = math.sqrt((lx - food_x) ** 2 + (ly - food_y) ** 2)
            dR = math.sqrt((rx - food_x) ** 2 + (ry - food_y) ** 2)
            food_L += math.exp(-dL / self.food_diff_r)
            food_R += math.exp(-dR / self.food_diff_r)

        # Alarm substance: update sources, then sample
        if alarm_events:
            for evt in alarm_events:
                self._alarm_sources.append(
                    [evt["x"], evt["y"], evt.get("intensity", 1.0)])

        # Decay existing alarm sources
        new_sources = []
        for src in self._alarm_sources:
            src[2] *= self.alarm_decay
            if src[2] > 0.01:
                new_sources.append(src)
        self._alarm_sources = new_sources

        alarm_L = 0.0
        alarm_R = 0.0
        for src in self._alarm_sources:
            sx, sy, sint = src
            dL = math.sqrt((lx - sx) ** 2 + (ly - sy) ** 2)
            dR = math.sqrt((rx - sx) ** 2 + (ry - sy) ** 2)
            alarm_L += sint * math.exp(-dL / self.alarm_diff_r)
            alarm_R += sint * math.exp(-dR / self.alarm_diff_r)

        # Chemotaxis direction: concentration gradient
        food_gradient = food_R - food_L  # positive = food to the right
        alarm_gradient = alarm_R - alarm_L
        total_food = food_L + food_R
        total_alarm = alarm_L + alarm_R

        return food_L, food_R, alarm_L, alarm_R, {
            "food_gradient": food_gradient,
            "total_food_odour": total_food,
            "alarm_gradient": alarm_gradient,
            "total_alarm": total_alarm,
            "n_alarm_sources": len(self._alarm_sources),
        }

    def add_alarm(self, x, y, intensity=1.0):
        """Add alarm substance at position (from injured conspecific)."""
        self._alarm_sources.append([x, y, intensity])

    def get_food_heading_bias(self, food_gradient, total_food):
        """Convert food gradient to turn bias for chemotaxis.

        Returns:
            turn_bias: float [-0.3, 0.3] — turn toward food odour
        """
        if total_food < 0.01:
            return 0.0
        # Normalised gradient → turn bias
        norm_grad = food_gradient / (total_food + 0.01)
        return float(np.clip(norm_grad * 0.3, -0.3, 0.3))

    def get_alarm_response(self, total_alarm):
        """Convert alarm concentration to flee urgency.

        Returns:
            flee_boost: float [0, 0.3] — additive to p_enemy
        """
        return float(min(0.3, total_alarm * 0.15))

    def reset(self):
        self._alarm_sources = []

    def get_diagnostics(self):
        return {
            "n_alarm_sources": len(self._alarm_sources),
            "total_alarm_intensity": sum(s[2] for s in self._alarm_sources),
        }
