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

    def reset(self):
        """No persistent state to reset."""
        pass
