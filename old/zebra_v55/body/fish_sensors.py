import numpy as np


# =======================================================================
# FishSensors: environment→fish (non-visual low-level sensors)
# =======================================================================

class FishSensors:
    def __init__(self, fish_physics):
        self.phys = fish_physics

        # short-range tactile radius
        self.touch_radius = 12.0

        # wall-avoidance reflex gain
        self.wall_gain = 0.5

        # food reflex gain
        self.food_gain = 0.3

    # ===================================================================
    # Sense nearest food distance and bearing
    # ===================================================================
    def sense_food(self, world):
        """
        world.foods: list of (x,y)
        Returns:
            distance
            bearing relative to heading
        """
        if len(world.foods) == 0:
            return 999.0, 0.0

        fx, fy = None, None
        min_d = 1e9

        for (ax, ay) in world.foods:
            dx = ax - self.phys.x
            dy = ay - self.phys.y
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_d:
                min_d = d
                fx, fy = ax, ay

        if fx is None:
            return 999.0, 0.0

        # bearing
        ang = np.arctan2(fy - self.phys.y, fx - self.phys.x)
        dtheta = ang - self.phys.heading
        dtheta = (dtheta + np.pi*3) % (np.pi*2) - np.pi

        return min_d, dtheta

    # ===================================================================
    # Basic wall proximity (PyQt6 world uses boundaries)
    # ===================================================================
    def sense_wall(self, world):
        """
        Simple distance to nearest wall for reflex.
        """
        x = self.phys.x
        y = self.phys.y

        d_left = x - world.xmin
        d_right = world.xmax - x
        d_up = world.ymax - y
        d_down = y - world.ymin

        return min(d_left, d_right, d_up, d_down)

    # ===================================================================
    # Reflex motor contribution
    # ===================================================================
    def compute_reflex(self, world):
        """
        Returns:
            reflex_turn
            reflex_forward
        """
        # wall reflex
        d_wall = self.sense_wall(world)
        reflex_turn = 0.0

        if d_wall < self.touch_radius:
            if self.phys.x < world.xmin + self.touch_radius:
                reflex_turn += self.wall_gain
            if self.phys.x > world.xmax - self.touch_radius:
                reflex_turn -= self.wall_gain
            if self.phys.y < world.ymin + self.touch_radius:
                reflex_turn -= self.wall_gain
            if self.phys.y > world.ymax - self.touch_radius:
                reflex_turn += self.wall_gain

        # food reflex: closer food → subtle attraction
        d_food, theta_food = self.sense_food(world)
        reflex_forward = 0.0

        if d_food < 30.0:
            reflex_forward += self.food_gain
            reflex_turn += self.food_gain * np.tanh(theta_food)

        return reflex_turn, reflex_forward
