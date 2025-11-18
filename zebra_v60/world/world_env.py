import numpy as np


# ======================================================================
# WORLD ENVIRONMENT FOR ZEBRAFISH V55.1
# ======================================================================

class WorldEnv:
    """
    A very simple 2D environment:
        - rect boundary
        - food items placed at (x,y)
        - ray sampling for retina
    """

    def __init__(self,
                 xmin=-200, xmax=200,
                 ymin=-150, ymax=150,
                 n_food=10):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # food list
        self.foods = []
        self.generate_food(n_food)

        # optional obstacles
        self.obstacles = []   # list of dicts: {"x":..., "y":..., "r":...}

    # ==================================================================
    # Random food generation
    # ==================================================================
    def generate_food(self, n):
        for _ in range(n):
            fx = np.random.uniform(self.xmin + 20, self.xmax - 20)
            fy = np.random.uniform(self.ymin + 20, self.ymax - 20)
            self.foods.append((fx, fy))

    # ==================================================================
    # Add obstacle
    # ==================================================================
    def add_obstacle(self, x, y, r):
        self.obstacles.append({"x": x, "y": y, "r": r})

    # ==================================================================
    # Ray sampling for retina
    # ==================================================================
    def sample_direction(self, origin, dx, dy, max_dist=200, step=4.0):
        """
        Steps a ray forward until:
            - hit food → strong signal
            - hit obstacle → medium signal
            - hit boundary → strong signal
            - nothing → weak background
        """
        ox, oy = origin
        t = 0.0

        while t < max_dist:
            x = ox + dx * t
            y = oy + dy * t

            # boundary detection
            if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
                return 1.0  # strong bright edge

            # food detection
            for (fx, fy) in self.foods:
                if (x - fx)**2 + (y - fy)**2 < 8**2:
                    return 1.0  # bright pixel

            # obstacle detection
            for obs in self.obstacles:
                if (x - obs["x"])**2 + (y - obs["y"])**2 < obs["r"]**2:
                    return 0.6

            t += step

        return 0.05  # background darkness


    # ==================================================================
    # Eat food if fish is close enough
    # ==================================================================
    def try_eat(self, fish_x, fish_y):
        new_foods = []
        eaten = 0
        for (fx, fy) in self.foods:
            if (fish_x - fx)**2 + (fish_y - fy)**2 < 20**2:
                eaten += 1
                continue
            new_foods.append((fx, fy))
        self.foods = new_foods
        return eaten

