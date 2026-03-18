"""
Lateral Line Organ — mechanosensory neuromast array (Step 32).

16 virtual neuromasts (8 per side, anterior-to-posterior) detect water
flow from nearby moving entities, boundary proximity, and self-motion.
Efference copy from the motor command cancels reafferent flow,
isolating exafferent signals from external sources.

Key outputs:
  - Bilateral flow activations: flow_L[8], flow_R[8]
  - Rear wake intensity (predator approach from behind)
  - Boundary proximity (wall/obstacle reflected flow)
  - Wake alarm signal (fast-moving entity nearby)

Neuroscience: posterior lateral line ganglion → medial octavolateral
nucleus (MON) → optic tectum + cerebellum (Coombs & Montgomery 1999).
Larval zebrafish have ~30 neuromasts; we model 16 for computational
efficiency (Liao & Haehnel-Taguchi 2015).

Pure numpy — no torch.
"""
import math
import numpy as np


class LateralLineOrgan:
    """Neuromast array for hydrodynamic flow detection.

    Args:
        n_per_side: int — neuromasts per side (anterior→posterior)
        detection_radius: float — max detection distance (px)
        reafference_gain: float — fraction of self-motion to cancel
        body_length: float — fish body length for neuromast spacing
    """

    def __init__(self, n_per_side=8, detection_radius=80.0,
                 reafference_gain=0.9, body_length=30.0):
        self.n_per_side = n_per_side
        self.detection_radius = detection_radius
        self.reafference_gain = reafference_gain
        self.det_r2 = detection_radius ** 2

        # Neuromast positions: body-relative offsets (front→back)
        # Left neuromasts are at +90° from heading, right at -90°
        spacing = body_length / (n_per_side + 1)
        self._offsets = np.zeros((n_per_side, 2), dtype=np.float32)
        for i in range(n_per_side):
            # Along body axis, front to back
            self._offsets[i, 0] = (n_per_side / 2 - i) * spacing  # forward
            self._offsets[i, 1] = 5.0  # lateral offset (half body width)

        # Previous entity positions keyed by type for velocity estimation
        self._prev_by_type = {}

    def step(self, fish_pos, fish_heading, fish_speed,
             entities, efference_speed=0.0):
        """Compute flow at all neuromasts.

        Args:
            fish_pos: array-like [2] — gym pixel position
            fish_heading: float — radians
            fish_speed: float — current speed [0, ~1.6]
            entities: list of dicts with "x", "y", "type" keys
                      type: "predator", "fish", "food"
            efference_speed: float — motor command speed (for reafference)

        Returns:
            flow_L: np.array[n_per_side] — left neuomast activations
            flow_R: np.array[n_per_side] — right neuromast activations
            diag: dict — aggregate features
        """
        cos_h = math.cos(fish_heading)
        sin_h = math.sin(fish_heading)
        fx, fy = float(fish_pos[0]), float(fish_pos[1])

        flow_L = np.zeros(self.n_per_side, dtype=np.float32)
        flow_R = np.zeros(self.n_per_side, dtype=np.float32)

        # Compute entity velocities from position delta (keyed by type+index)
        entity_vels = self._estimate_velocities(entities)

        for i in range(self.n_per_side):
            # World-space neuromast positions (left and right)
            ox, oy = self._offsets[i]
            # Left: +90° from heading
            lx = fx + cos_h * ox - sin_h * oy
            ly = fy + sin_h * ox + cos_h * oy
            # Right: -90° from heading
            rx = fx + cos_h * ox + sin_h * oy
            ry = fy + sin_h * ox - cos_h * oy

            for j, ent in enumerate(entities):
                ex, ey = ent["x"], ent["y"]
                key = (ent.get("type", ""), j)
                vx, vy = entity_vels.get(key, (0.0, 0.0))
                ent_speed = math.sqrt(vx * vx + vy * vy)
                if ent_speed < 0.01:
                    continue

                # Left neuromast
                flow_L[i] += self._dipole_flow(
                    lx, ly, ex, ey, vx, vy, ent_speed)
                # Right neuromast
                flow_R[i] += self._dipole_flow(
                    rx, ry, ex, ey, vx, vy, ent_speed)

        # Reafference cancellation: subtract predicted self-motion flow
        # Self-motion creates forward flow that rear neuromasts feel more
        reafference = np.zeros(self.n_per_side, dtype=np.float32)
        if efference_speed > 0.01:
            for i in range(self.n_per_side):
                # Posterior neuromasts feel more self-motion wake
                rear_factor = (self.n_per_side - i) / self.n_per_side
                reafference[i] = efference_speed * 0.3 * rear_factor

        exafferent_L = np.maximum(0, flow_L - self.reafference_gain * reafference)
        exafferent_R = np.maximum(0, flow_R - self.reafference_gain * reafference)

        # Aggregate diagnostics
        rear_wake = float(np.mean(exafferent_L[-3:] + exafferent_R[-3:]))
        front_flow = float(np.mean(exafferent_L[:3] + exafferent_R[:3]))
        lateral_diff = float(np.mean(exafferent_L) - np.mean(exafferent_R))
        total_flow = float(np.sum(exafferent_L) + np.sum(exafferent_R))

        # Update entity tracking (keyed by type + index)
        self._prev_by_type = {
            (e.get("type", ""), i): (e["x"], e["y"])
            for i, e in enumerate(entities)}

        return exafferent_L, exafferent_R, {
            "rear_wake_intensity": rear_wake,
            "front_flow": front_flow,
            "lateral_diff": lateral_diff,
            "total_flow": total_flow,
            "flow_L": exafferent_L.copy(),
            "flow_R": exafferent_R.copy(),
        }

    def _dipole_flow(self, nx, ny, ex, ey, vx, vy, speed):
        """Hydrodynamic dipole flow at neuromast from moving entity.

        Flow decays as 1/r² and depends on angle between velocity
        vector and neuromast-to-entity direction.
        """
        dx = ex - nx
        dy = ey - ny
        r2 = dx * dx + dy * dy
        if r2 > self.det_r2 or r2 < 1.0:
            return 0.0

        r = math.sqrt(r2)
        # Angle between entity velocity and entity-to-neuromast direction
        cos_theta = (-vx * dx - vy * dy) / (speed * r + 1e-8)
        # Dipole: flow ~ speed * cos(theta) / r²
        flow = speed * max(0.0, cos_theta) / (r2 / 100.0 + 1.0)
        return float(flow)

    def _estimate_velocities(self, entities):
        """Estimate entity velocities from position deltas."""
        vels = {}
        for i, ent in enumerate(entities):
            key = (ent.get("type", ""), i)
            if key in self._prev_by_type:
                px, py = self._prev_by_type[key]
                vels[key] = (ent["x"] - px, ent["y"] - py)
            else:
                vels[key] = (0.0, 0.0)
        return vels

    def reset(self):
        self._prev_by_type = {}

    def get_diagnostics(self):
        return {}  # filled by step() return
