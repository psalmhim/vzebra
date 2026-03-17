import numpy as np


# ======================================================================
# WORLD ENVIRONMENT FOR ZEBRAFISH V60
# Multi-entity: food, enemies, colleagues, obstacles, boundaries
# Each entity type has distinct visual signature (intensity + radius)
# ======================================================================

# Entity type constants
ENTITY_NONE = 0
ENTITY_FOOD = 1
ENTITY_ENEMY = 2
ENTITY_COLLEAGUE = 3
ENTITY_BOUNDARY = 4
ENTITY_OBSTACLE = 5
ENTITY_PREY = 6

# Visual signatures: (intensity, detection_radius)
ENTITY_VISUAL = {
    ENTITY_FOOD:      (1.0,  15),   # bright, small prey (default radius)
    ENTITY_ENEMY:     (0.75, 45),   # dimmer, 1.5x larger body — visible from far
    ENTITY_COLLEAGUE: (0.50, 18),   # moderate, same-size fish
    ENTITY_BOUNDARY:  (0.30, None), # dull edge signal (detected by ray clipping)
    ENTITY_OBSTACLE:  (0.60, 25),   # rock visible within 25-unit margin around AABB
    ENTITY_PREY:      (0.85, 28),   # bright, fish-sized prey (predator's target)
}

# Multi-size food: detection_radius varies by size
FOOD_SIZE_RADIUS = {
    "small": 10,   # plankton: small, many spawned
    "large": 22,   # artemia/worm: large, few spawned
}

BACKGROUND_INTENSITY = 0.05


def _food_xy(food):
    """Extract (x, y) from food item (tuple or dict)."""
    if isinstance(food, dict):
        return food["x"], food["y"]
    return food[0], food[1]


def _food_size(food):
    """Extract size from food item, defaulting to 'small'."""
    if isinstance(food, dict):
        return food.get("size", "small")
    if isinstance(food, (list, tuple)) and len(food) > 2:
        return food[2]
    return "small"


class WorldEnv:
    """
    2D environment with multiple entity types:
        - food: prey items the fish can eat
        - enemies: predators to avoid
        - colleagues: conspecifics (other fish)
        - obstacles: static objects
        - boundaries: rect boundary walls
    Each entity type has a distinct visual intensity and detection radius.
    """

    def __init__(self,
                 xmin=-200, xmax=200,
                 ymin=-150, ymax=150,
                 n_food=10, n_enemies=0, n_colleagues=0):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.foods = []
        self.enemies = []
        self.colleagues = []
        self.obstacles = []   # list of dicts: {"x":..., "y":..., "hw":..., "hh":...}
        self.prey = []        # prey targets (fish seen by predator)

        self.generate_food(n_food)
        self.generate_enemies(n_enemies)
        self.generate_colleagues(n_colleagues)

    # ==================================================================
    # Entity generation
    # ==================================================================
    def generate_food(self, n, size="small"):
        for _ in range(n):
            fx = np.random.uniform(self.xmin + 20, self.xmax - 20)
            fy = np.random.uniform(self.ymin + 20, self.ymax - 20)
            self.foods.append({"x": fx, "y": fy, "size": size})

    def generate_enemies(self, n):
        for _ in range(n):
            ex = np.random.uniform(self.xmin + 30, self.xmax - 30)
            ey = np.random.uniform(self.ymin + 30, self.ymax - 30)
            self.enemies.append((ex, ey))

    def generate_colleagues(self, n):
        for _ in range(n):
            cx = np.random.uniform(self.xmin + 20, self.xmax - 20)
            cy = np.random.uniform(self.ymin + 20, self.ymax - 20)
            self.colleagues.append((cx, cy))

    def add_obstacle(self, x, y, hw, hh):
        self.obstacles.append({"x": x, "y": y, "hw": hw, "hh": hh})

    # ==================================================================
    # Ray sampling — returns (intensity, entity_type)
    # ==================================================================
    def sample_direction(self, origin, dx, dy, max_dist=200, step=3.0):
        """
        Cast ray, return (intensity, entity_type).
        Checks all entity types with type-specific detection radii.
        Returns the FIRST hit (closest entity).
        """
        ox, oy = origin
        t = 0.0

        # Pre-compute per-food detection radii squared
        food_r2_list = []
        for food in self.foods:
            fx, fy = _food_xy(food)
            sz = _food_size(food)
            r = FOOD_SIZE_RADIUS.get(sz, FOOD_SIZE_RADIUS["small"])
            food_r2_list.append((fx, fy, r * r))

        enemy_r2 = ENTITY_VISUAL[ENTITY_ENEMY][1] ** 2
        colleague_r2 = ENTITY_VISUAL[ENTITY_COLLEAGUE][1] ** 2
        prey_r2 = ENTITY_VISUAL[ENTITY_PREY][1] ** 2
        _obs_margin = ENTITY_VISUAL[ENTITY_OBSTACLE][1]

        while t < max_dist:
            x = ox + dx * t
            y = oy + dy * t

            # Boundary detection
            if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
                return ENTITY_VISUAL[ENTITY_BOUNDARY][0], ENTITY_BOUNDARY

            # Food detection (per-food radius for multi-size support)
            for fx, fy, fr2 in food_r2_list:
                if (x - fx) ** 2 + (y - fy) ** 2 < fr2:
                    return ENTITY_VISUAL[ENTITY_FOOD][0], ENTITY_FOOD

            # Prey detection (fish seen by predator)
            for (px, py) in self.prey:
                if (x - px) ** 2 + (y - py) ** 2 < prey_r2:
                    return ENTITY_VISUAL[ENTITY_PREY][0], ENTITY_PREY

            # Enemy detection
            for (ex, ey) in self.enemies:
                if (x - ex) ** 2 + (y - ey) ** 2 < enemy_r2:
                    return ENTITY_VISUAL[ENTITY_ENEMY][0], ENTITY_ENEMY

            # Colleague detection
            for (cx, cy) in self.colleagues:
                if (x - cx) ** 2 + (y - cy) ** 2 < colleague_r2:
                    return ENTITY_VISUAL[ENTITY_COLLEAGUE][0], ENTITY_COLLEAGUE

            # Obstacle detection (axis-aligned rectangle + margin)
            for obs in self.obstacles:
                if (abs(x - obs["x"]) < obs["hw"] + _obs_margin
                        and abs(y - obs["y"]) < obs["hh"] + _obs_margin):
                    return ENTITY_VISUAL[ENTITY_OBSTACLE][0], ENTITY_OBSTACLE

            t += step

        return BACKGROUND_INTENSITY, ENTITY_NONE

    # ==================================================================
    # Ray sampling with depth — returns (intensity, entity_type, distance)
    # ==================================================================
    def sample_direction_with_depth(self, origin, dx, dy, max_dist=200, step=3.0):
        """
        Cast ray, return (intensity, entity_type, hit_distance).
        Same as sample_direction() but also returns the distance t at first hit.
        No-hit returns (BACKGROUND_INTENSITY, ENTITY_NONE, max_dist).
        """
        ox, oy = origin
        t = 0.0

        # Pre-compute per-food detection radii squared
        food_r2_list = []
        for food in self.foods:
            fx, fy = _food_xy(food)
            sz = _food_size(food)
            r = FOOD_SIZE_RADIUS.get(sz, FOOD_SIZE_RADIUS["small"])
            food_r2_list.append((fx, fy, r * r))

        enemy_r2 = ENTITY_VISUAL[ENTITY_ENEMY][1] ** 2
        colleague_r2 = ENTITY_VISUAL[ENTITY_COLLEAGUE][1] ** 2
        prey_r2 = ENTITY_VISUAL[ENTITY_PREY][1] ** 2
        _obs_margin = ENTITY_VISUAL[ENTITY_OBSTACLE][1]

        while t < max_dist:
            x = ox + dx * t
            y = oy + dy * t

            # Boundary detection
            if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
                return ENTITY_VISUAL[ENTITY_BOUNDARY][0], ENTITY_BOUNDARY, t

            # Food detection (per-food radius)
            for fx, fy, fr2 in food_r2_list:
                if (x - fx) ** 2 + (y - fy) ** 2 < fr2:
                    return ENTITY_VISUAL[ENTITY_FOOD][0], ENTITY_FOOD, t

            # Prey detection (fish seen by predator)
            for (px, py) in self.prey:
                if (x - px) ** 2 + (y - py) ** 2 < prey_r2:
                    return ENTITY_VISUAL[ENTITY_PREY][0], ENTITY_PREY, t

            # Enemy detection
            for (ex, ey) in self.enemies:
                if (x - ex) ** 2 + (y - ey) ** 2 < enemy_r2:
                    return ENTITY_VISUAL[ENTITY_ENEMY][0], ENTITY_ENEMY, t

            # Colleague detection
            for (cx, cy) in self.colleagues:
                if (x - cx) ** 2 + (y - cy) ** 2 < colleague_r2:
                    return ENTITY_VISUAL[ENTITY_COLLEAGUE][0], ENTITY_COLLEAGUE, t

            # Obstacle detection (axis-aligned rectangle + margin)
            for obs in self.obstacles:
                if (abs(x - obs["x"]) < obs["hw"] + _obs_margin
                        and abs(y - obs["y"]) < obs["hh"] + _obs_margin):
                    return ENTITY_VISUAL[ENTITY_OBSTACLE][0], ENTITY_OBSTACLE, t

            t += step

        return BACKGROUND_INTENSITY, ENTITY_NONE, max_dist

    # ==================================================================
    # Backward-compatible: return intensity only
    # ==================================================================
    def sample_direction_intensity(self, origin, dx, dy, max_dist=200, step=3.0):
        """Return only intensity (for backward compatibility)."""
        intensity, _ = self.sample_direction(origin, dx, dy, max_dist, step)
        return intensity

    # ==================================================================
    # Eat food if fish is close enough (radius aligned with visibility)
    # ==================================================================
    def try_eat(self, fish_x, fish_y, eat_radius=18):
        new_foods = []
        eaten = 0
        eaten_sizes = []
        r2 = eat_radius ** 2
        for food in self.foods:
            fx, fy = _food_xy(food)
            if (fish_x - fx) ** 2 + (fish_y - fy) ** 2 < r2:
                eaten += 1
                eaten_sizes.append(_food_size(food))
                continue
            new_foods.append(food)
        self.foods = new_foods
        return eaten, eaten_sizes

    # ==================================================================
    # Flee from enemy (check proximity)
    # ==================================================================
    def check_enemy_proximity(self, fish_x, fish_y, danger_radius=30):
        """Return distance to nearest enemy, or None."""
        min_dist = float('inf')
        for ex, ey in self.enemies:
            d2 = (fish_x - ex) ** 2 + (fish_y - ey) ** 2
            if d2 < min_dist:
                min_dist = d2
        if min_dist < danger_radius ** 2:
            return min_dist ** 0.5
        return None
