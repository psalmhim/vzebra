"""
Shoaling module: Boids-style social behavior + social learning.

Zebrafish larvae show attraction to conspecifics and modulation of
activity by social context (Dreosti et al. 2015). Schooling emerges
from three local rules (separation, cohesion, alignment).

Social learning: observe neighbors' speed/direction to infer:
  - Danger: fast-moving neighbors fleeing = predator nearby
  - Food: slow neighbors clustered = food patch found
  - Safety: calm group = no threat

Ported from v1 with v2 integration hooks.
"""
import math
import numpy as np


class ShoalingModule:
    def __init__(self,
                 separation_radius=40.0,
                 cohesion_radius=150.0,
                 alignment_radius=100.0,
                 separation_weight=1.2,
                 cohesion_weight=0.6,
                 alignment_weight=0.4,
                 max_turn_bias=0.8):
        self.separation_radius = separation_radius
        self.cohesion_radius = cohesion_radius
        self.alignment_radius = alignment_radius
        self.separation_weight = separation_weight
        self.cohesion_weight = cohesion_weight
        self.alignment_weight = alignment_weight
        self.max_turn_bias = max_turn_bias
        self.turn_bias = 0.0
        self.speed_mod = 1.0
        self.social_alarm = 0.0
        self.social_safety = 0.5
        self.n_neighbours = 0

    def step(self, fish_x, fish_y, fish_heading, colleagues):
        """
        Compute shoaling turn bias and speed modulation.
        colleagues: list of dicts with keys "x", "y", "heading", "speed"
        Returns: turn_bias, speed_mod, diagnostics
        """
        if not colleagues:
            self.turn_bias = 0.0
            self.speed_mod = 1.0
            self.n_neighbours = 0
            return 0.0, 1.0, {'n_neighbours': 0}

        sep_x, sep_y = 0.0, 0.0
        coh_x, coh_y = 0.0, 0.0
        align_sin, align_cos = 0.0, 0.0
        n_sep, n_coh, n_align = 0, 0, 0

        for c in colleagues:
            dx = c['x'] - fish_x
            dy = c['y'] - fish_y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-8

            if dist < self.separation_radius:
                sep_x -= dx / (dist * dist)
                sep_y -= dy / (dist * dist)
                n_sep += 1
            if dist < self.cohesion_radius:
                coh_x += dx
                coh_y += dy
                n_coh += 1
            if dist < self.alignment_radius:
                align_sin += math.sin(c['heading'])
                align_cos += math.cos(c['heading'])
                n_align += 1

        desired_dx, desired_dy = 0.0, 0.0
        if n_sep > 0:
            desired_dx += self.separation_weight * sep_x
            desired_dy += self.separation_weight * sep_y
        if n_coh > 0:
            desired_dx += self.cohesion_weight * (coh_x / n_coh)
            desired_dy += self.cohesion_weight * (coh_y / n_coh)
        if n_align > 0:
            avg_heading = math.atan2(align_sin / n_align, align_cos / n_align)
            align_diff = math.atan2(
                math.sin(avg_heading - fish_heading),
                math.cos(avg_heading - fish_heading))
            desired_dx += self.alignment_weight * math.cos(fish_heading + align_diff)
            desired_dy += self.alignment_weight * math.sin(fish_heading + align_diff)

        if abs(desired_dx) + abs(desired_dy) > 1e-6:
            desired_angle = math.atan2(desired_dy, desired_dx)
            angle_diff = math.atan2(
                math.sin(desired_angle - fish_heading),
                math.cos(desired_angle - fish_heading))
            turn_bias = float(np.clip(angle_diff, -self.max_turn_bias, self.max_turn_bias))
        else:
            turn_bias = 0.0

        speed_mod = max(0.7, 1.0 - 0.1 * n_sep) if n_sep > 0 else 1.0

        self.turn_bias = turn_bias
        self.speed_mod = speed_mod
        self.n_neighbours = n_coh
        return turn_bias, speed_mod, {'n_neighbours': n_coh, 'n_sep': n_sep}

    def observe_social_cues(self, fish_x, fish_y, colleagues):
        """Infer environmental state from conspecific behavior."""
        if not colleagues:
            self.social_alarm = 0.0
            self.social_safety = 0.5
            return {'social_alarm': 0.0, 'social_food_bearing': None, 'social_safety': 0.5}

        alarm = 0.0
        slow_x, slow_y, n_slow = 0.0, 0.0, 0
        speeds = []

        for c in colleagues:
            dx = c['x'] - fish_x
            dy = c['y'] - fish_y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-8
            if dist > 200:
                continue
            speed = c.get('speed', 0.5)
            speeds.append(speed)
            if speed > 1.5:
                alarm += (1.0 - dist / 200.0) * min(1.0, speed / 2.0)
            if speed < 0.4:
                slow_x += c['x']
                slow_y += c['y']
                n_slow += 1

        self.social_alarm = min(1.0, alarm)
        social_food_bearing = None
        if n_slow >= 2:
            social_food_bearing = math.atan2(
                slow_y / n_slow - fish_y, slow_x / n_slow - fish_x)
        self.social_safety = max(0.0, 1.0 - sum(speeds) / (len(speeds) * 1.5)) if speeds else 0.5

        return {
            'social_alarm': self.social_alarm,
            'social_food_bearing': social_food_bearing,
            'social_safety': self.social_safety,
        }

    def get_efe_bias(self):
        """EFE bias from social cues."""
        return {
            'social_forage': -0.1 if self.social_safety > 0.6 else 0.0,
            'social_flee': -self.social_alarm * 0.2,
            'social_explore': -0.05 if self.n_neighbours < 2 else 0.05,
        }

    def reset(self):
        self.turn_bias = 0.0
        self.speed_mod = 1.0
        self.social_alarm = 0.0
        self.social_safety = 0.5
        self.n_neighbours = 0
