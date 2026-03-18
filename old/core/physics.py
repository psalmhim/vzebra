import numpy as np
from .gpu_array import xp, GPU_ENABLED

class HydroPhysics:
    def __init__(self, mass=1.0, drag=0.04):
        self.mass = mass
        self.drag = drag
        self.vx = 0.0
        self.vy = 0.0

    def compute_thrust(self, spine_x, spine_y, t, amp):
        """
        Tail-generated thrust from traveling wave:
        F = sum(amp_i * sin(k*y + w*t)) across joints
        """
        X = spine_x
        Y = spine_y
        k = 0.25
        w = 0.20

        sin_term = xp(amp * xp.sin(k*Y + w*t))
        F = xp.sum(sin_term) * 0.08
        if GPU_ENABLED:
            F = float(cp.asnumpy(F))
        return F

    def update(self, heading, thrust, dt=1.0):
        # acceleration from thrust
        ax = (thrust/self.mass) * np.cos(heading)
        ay = (thrust/self.mass) * np.sin(heading)

        # update velocity
        self.vx += ax * dt
        self.vy += ay * dt

        # drag
        self.vx *= (1 - self.drag)
        self.vy *= (1 - self.drag)

        return self.vx, self.vy
