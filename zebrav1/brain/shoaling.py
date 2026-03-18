"""
Shoaling Module (Step 20) — Boids-style social behavior.

Computes three classic flocking forces when GOAL_SOCIAL is active:
  - Separation: avoid collision with nearby colleagues
  - Cohesion:   steer toward group centroid
  - Alignment:  match heading of neighbours

Returns turn bias and speed modulation that the brain agent
blends into its motor output.

Pure numpy — no torch dependency.
"""
import math
import numpy as np


class ShoalingModule:
    """Boids-style shoaling for zebrafish social behavior."""

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

    def step(self, fish_x, fish_y, fish_heading, colleagues):
        """Compute shoaling turn bias and speed modulation.

        Args:
            fish_x, fish_y: float — agent fish position (gym pixel coords)
            fish_heading: float — agent heading in radians
            colleagues: list of dicts with keys "x", "y", "heading", "speed"

        Returns:
            turn_bias: float in [-max_turn_bias, max_turn_bias]
            speed_mod: float — multiplier (0.7–1.0)
            diagnostics: dict
        """
        if not colleagues:
            return 0.0, 1.0, {"n_neighbours": 0}

        sep_x, sep_y = 0.0, 0.0
        coh_x, coh_y = 0.0, 0.0
        align_sin, align_cos = 0.0, 0.0
        n_sep, n_coh, n_align = 0, 0, 0

        for c in colleagues:
            dx = c["x"] - fish_x
            dy = c["y"] - fish_y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-8

            # Separation
            if dist < self.separation_radius:
                # Push away (inverse distance weighting)
                sep_x -= dx / (dist * dist)
                sep_y -= dy / (dist * dist)
                n_sep += 1

            # Cohesion
            if dist < self.cohesion_radius:
                coh_x += dx
                coh_y += dy
                n_coh += 1

            # Alignment
            if dist < self.alignment_radius:
                align_sin += math.sin(c["heading"])
                align_cos += math.cos(c["heading"])
                n_align += 1

        # Combine forces into a desired heading offset
        desired_dx, desired_dy = 0.0, 0.0

        if n_sep > 0:
            desired_dx += self.separation_weight * sep_x
            desired_dy += self.separation_weight * sep_y

        if n_coh > 0:
            desired_dx += self.cohesion_weight * (coh_x / n_coh)
            desired_dy += self.cohesion_weight * (coh_y / n_coh)

        if n_align > 0:
            avg_heading = math.atan2(align_sin / n_align, align_cos / n_align)
            align_diff = avg_heading - fish_heading
            align_diff = math.atan2(math.sin(align_diff), math.cos(align_diff))
            desired_dx += self.alignment_weight * math.cos(
                fish_heading + align_diff)
            desired_dy += self.alignment_weight * math.sin(
                fish_heading + align_diff)

        # Convert desired direction to turn bias
        if abs(desired_dx) + abs(desired_dy) > 1e-6:
            desired_angle = math.atan2(desired_dy, desired_dx)
            angle_diff = desired_angle - fish_heading
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            turn_bias = float(np.clip(
                angle_diff, -self.max_turn_bias, self.max_turn_bias))
        else:
            turn_bias = 0.0

        # Speed modulation: slow down when very close to neighbours
        speed_mod = 1.0
        if n_sep > 0:
            speed_mod = max(0.7, 1.0 - 0.1 * n_sep)

        diagnostics = {
            "n_neighbours": n_coh,
            "n_separation": n_sep,
            "n_alignment": n_align,
            "turn_bias": turn_bias,
            "speed_mod": speed_mod,
        }

        return turn_bias, speed_mod, diagnostics

    def observe_social_cues(self, fish_x, fish_y, colleagues):
        """Social learning: infer environmental state from conspecific behavior.

        Zebrafish observe neighbours' speed and direction to infer:
        - Danger: fast-moving neighbours fleeing = predator nearby
        - Food:   slow neighbours in one area = food patch found
        - Safety: calm group = no immediate threat

        Biological basis: social information transfer in zebrafish shoals
        (Arganda et al. 2012, Sosna et al. 2019).

        Returns:
            dict with social cues:
                social_alarm: float [0, 1] — conspecific flee signal
                social_food_bearing: float or None — direction toward food
                social_safety: float [0, 1] — group calmness
        """
        if not colleagues:
            return {"social_alarm": 0.0, "social_food_bearing": None,
                    "social_safety": 0.5}

        alarm = 0.0
        slow_x, slow_y = 0.0, 0.0
        n_slow = 0
        speeds = []

        for c in colleagues:
            dx = c["x"] - fish_x
            dy = c["y"] - fish_y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-8
            if dist > 200:
                continue  # too far to observe

            speed = c.get("speed", 0.5)
            speeds.append(speed)

            # Fast-moving neighbour = alarm signal (fleeing)
            if speed > 1.5:
                # Weight by proximity (closer neighbour = stronger signal)
                alarm += (1.0 - dist / 200.0) * min(1.0, speed / 2.0)

            # Slow-moving neighbour = foraging indicator
            if speed < 0.4:
                slow_x += c["x"]
                slow_y += c["y"]
                n_slow += 1

        # Social alarm: any neighbour fleeing → danger
        social_alarm = min(1.0, alarm)

        # Social food bearing: direction toward slow group (foraging patch)
        social_food_bearing = None
        if n_slow >= 2:
            cx = slow_x / n_slow
            cy = slow_y / n_slow
            social_food_bearing = math.atan2(cy - fish_y, cx - fish_x)

        # Social safety: low mean speed + many neighbours = safe
        if speeds:
            mean_speed = sum(speeds) / len(speeds)
            social_safety = max(0.0, 1.0 - mean_speed / 1.5)
        else:
            social_safety = 0.5

        return {
            "social_alarm": social_alarm,
            "social_food_bearing": social_food_bearing,
            "social_safety": social_safety,
            "n_observed": len(speeds),
        }

    def reset(self):
        """No persistent state to reset."""
        pass
