# ============================================================
# ENVIRONMENT: PreyPredatorEnv
# ============================================================

import numpy as np

class PreyPredatorEnv:
    def __init__(self, world_size=100.0, T=400, device="cpu"):
        self.world_size = world_size
        self.device = device
        self.T = T
        self.object_pos = np.zeros(2)
        self.object_vel = np.zeros(2)
        self.object_size = 5.0
        self.object_type = 0

    def reset(self, obj_type=None):
        if obj_type is None:
            self.object_type = np.random.choice([+1, -1, 0])
        else:
            self.object_type = obj_type

        self.object_pos = np.random.uniform(-40, 40, size=2)

        if self.object_type == +1:
            speed = np.random.uniform(1.0, 3.0)
        elif self.object_type == -1:
            speed = np.random.uniform(3.0, 6.0)
        else:
            speed = np.random.uniform(0.5, 1.5)

        angle = np.random.uniform(0, 2 * np.pi)
        self.object_vel = speed * np.array([np.cos(angle), np.sin(angle)])

        if self.object_type == -1:
            self.object_size = np.random.uniform(6, 10)
        elif self.object_type == +1:
            self.object_size = np.random.uniform(3, 6)
        else:
            self.object_size = np.random.uniform(2, 4)

        return self._get_obs()

    def step(self):
        self.object_pos += self.object_vel

        for i in range(2):
            if abs(self.object_pos[i]) > self.world_size:
                self.object_vel[i] *= -1

        return self._get_obs()

    def _get_obs(self):
        dx = float(self.object_pos[0])
        dy = float(self.object_pos[1])
        dist = np.sqrt(dx**2 + dy**2)

        return {
            "dx": dx,
            "dy": dy,
            "dist": dist,
            "size": float(self.object_size),
            "type": float(self.object_type),
        }

