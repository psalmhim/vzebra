# ============================================================
# ENVIRONMENT: PreyPredatorEnv
# AUTHOR: H.J. Park & GPT-5
# VERSION: v1.0 (2025-12-04)
#
# PURPOSE:
#     A simple 2D world where a zebrafish tracks a moving object
#     (prey, predator, or neutral). Generates visual input compatible
#     with RetinaPC and VisualCortexPC processing pipeline.
# ============================================================

import numpy as np
import torch


class PreyPredatorEnv:
    """
    2D environment:
      - agent fixed at (0,0)
      - object moves in world space
      - object has label: prey (+1), predator (−1), neutral (0)

    Observation returned:
      ret = {
        "dx": delta x,
        "dy": delta y,
        "dist": distance,
        "size": object size,
        "type": label
      }
    """

    def __init__(self, world_size=100.0, T=400, device="cpu"):
        self.world_size = world_size
        self.device = device
        self.T = T  # max steps per episode
        self.object_pos = np.zeros(2)
        self.object_vel = np.zeros(2)
        self.object_size = 5.0
        self.object_type = 0  # +1 prey, -1 predator, 0 neutral

    # ------------------------------------------------------------
    def reset(self, obj_type=None):
        # random object class
        if obj_type is None:
            self.object_type = np.random.choice([+1, -1, 0])
        else:
            self.object_type = obj_type

        # random initial position
        self.object_pos = np.random.uniform(-40, 40, size=2)

        # velocity depends on type
        if self.object_type == +1:      # prey moves erratically
            speed = np.random.uniform(1.0, 3.0)
        elif self.object_type == -1:    # predator approaches fast
            speed = np.random.uniform(3.0, 6.0)
        else:                           # neutral object drift
            speed = np.random.uniform(0.5, 1.5)

        angle = np.random.uniform(0, 2 * np.pi)
        self.object_vel = speed * np.array([np.cos(angle), np.sin(angle)])

        # object size (predator > prey)
        if self.object_type == -1:
            self.object_size = np.random.uniform(6, 10)
        elif self.object_type == +1:
            self.object_size = np.random.uniform(3, 6)
        else:
            self.object_size = np.random.uniform(2, 4)

        return self._get_obs()

    # ------------------------------------------------------------
    def step(self):
        """Move object and return observation."""
        self.object_pos += self.object_vel

        # bounce at boundaries
        for i in range(2):
            if abs(self.object_pos[i]) > self.world_size:
                self.object_vel[i] *= -1

        return self._get_obs()

    # ------------------------------------------------------------
    def _get_obs(self):
        """Return sensory features encoding object relative to fish."""
        dx = float(self.object_pos[0])
        dy = float(self.object_pos[1])
        dist = np.sqrt(dx*dx + dy*dy)

        obs = {
            "dx": dx,
            "dy": dy,
            "dist": dist,
            "size": float(self.object_size),
            "type": float(self.object_type)
        }

        return obs

    # ------------------------------------------------------------
    def render(self):
        """Optional: print simple text visualization."""
        print(f"Object: pos=({self.object_pos[0]:.1f},{self.object_pos[1]:.1f}) "
              f"vel=({self.object_vel[0]:.2f},{self.object_vel[1]:.2f}) "
              f"type={self.object_type}")

