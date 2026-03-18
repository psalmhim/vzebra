import numpy as np


# ======================================================================
# Fish physics integrator
# ======================================================================
class FishPhysics:
    """
    Handles physical movement of the zebrafish.
    Position, heading, velocity, friction.
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # radians

        self.vx = 0.0
        self.vy = 0.0

        # water friction
        self.drag = 0.92

        # inertia factor
        self.mass = 1.0

    # ==================================================================
    # Apply decoded motor commands
    # ==================================================================
    def apply_motor(self, turn_force, forward_drive, dt=0.02):
        """
        turn_force: -1..+1
        forward_drive: 0..1
        """
        # 1) heading turn
        self.heading += float(turn_force) * 0.12
        self.heading = (self.heading + np.pi * 2) % (np.pi * 2)

        # 2) forward thrust
        thrust = forward_drive * 50.0

        # push in heading direction
        fx = np.cos(self.heading) * thrust
        fy = np.sin(self.heading) * thrust

        # integrate velocity
        self.vx += fx * dt / self.mass
        self.vy += fy * dt / self.mass

    # ==================================================================
    # Integrate movement
    # ==================================================================
    def step(self, dt=0.02):
        # water drag
        self.vx *= self.drag
        self.vy *= self.drag

        # position update
        self.x += self.vx * dt
        self.y += self.vy * dt

        return self.x, self.y, self.heading
