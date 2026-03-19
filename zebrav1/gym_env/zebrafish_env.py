"""
Gymnasium environment for zebrafish predator-prey simulation.

The zebrafish (blue triangle, prey) must forage for food (green dots)
while avoiding a predator (red triangle, 1.5x bigger) that chases it.

Features (Step 14):
  A. Hunger mechanic (energy budget)
  B. Obstacle collision physics
  C. Vision strip rendering (1D retinal overlay)
  D. Energy HUD

Observation: [fish_x, fish_y, fish_heading, fish_speed,
              pred_x, pred_y, pred_heading, pred_dist, pred_angle,
              nearest_food_dist, nearest_food_angle,
              wall_N, wall_S, wall_E, wall_W,
              n_food_eaten,
              energy_normalized]   ← NEW dim 17

Action (continuous): [turn_rate, speed_mod]
  turn_rate ∈ [-1, 1] → heading change
  speed_mod ∈ [0, 1]  → speed multiplier
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _make_rock_formation(cx, cy, base_r, n_lobes=5, seed=None):
    """Build a composite rock: central AABB + radial lobe AABBs + visual polygon.

    Returns dict with keys:
        cx, cy      – center in gym pixel coords
        base_r      – nominal radius (for compat with old obs["r"])
        r           – alias of base_r
        aabbs       – list of {"x","y","hw","hh"} rectangles
        polygon     – list of (px, py) for organic outline
    """
    rng = np.random.RandomState(seed)

    # --- Central AABB ---
    central_hw = base_r * 0.65
    central_hh = base_r * 0.55
    aabbs = [{"x": cx, "y": cy, "hw": central_hw, "hh": central_hh}]

    # --- Radial lobes ---
    angle_step = 2 * math.pi / n_lobes
    for i in range(n_lobes):
        angle = angle_step * i + rng.uniform(-0.3, 0.3)
        dist = base_r * rng.uniform(0.4, 0.7)
        lx = cx + dist * math.cos(angle)
        ly = cy + dist * math.sin(angle)
        lhw = base_r * rng.uniform(0.25, 0.45)
        lhh = base_r * rng.uniform(0.20, 0.40)
        aabbs.append({"x": lx, "y": ly, "hw": lhw, "hh": lhh})

    # --- Visual polygon (noisy circle for organic look) ---
    n_pts = 14
    polygon = []
    for i in range(n_pts):
        angle = 2 * math.pi * i / n_pts
        r_noise = base_r * rng.uniform(0.7, 1.1)
        px = cx + r_noise * math.cos(angle)
        py = cy + r_noise * math.sin(angle)
        polygon.append((px, py))

    return {
        "cx": cx, "cy": cy, "base_r": base_r, "r": base_r,
        "aabbs": aabbs, "polygon": polygon,
    }


class ZebrafishPreyPredatorEnv(gym.Env):
    """Zebrafish predator-prey environment with Gymnasium API."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    # Food energy values by size
    FOOD_ENERGY = {"small": 2.0, "large": 5.0}

    def __init__(self, render_mode=None, arena_size=800, n_food=15,
                 max_steps=2000, n_colleagues=3, side_panels=False,
                 use_predator_brain=False,
                 n_food_small=None, n_food_large=None):
        super().__init__()

        self.use_predator_brain = use_predator_brain
        self.arena_w = arena_size
        self.arena_h = int(arena_size * 0.75)  # 4:3 aspect ratio
        # Multi-size food: default 20 small + 5 large if not specified
        if n_food_small is not None or n_food_large is not None:
            self.n_food_small_init = n_food_small or 20
            self.n_food_large_init = n_food_large or 5
            self.n_food_init = self.n_food_small_init + self.n_food_large_init
        else:
            self.n_food_init = n_food
            self.n_food_small_init = max(1, n_food - n_food // 5)
            self.n_food_large_init = n_food // 5
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.n_colleagues = n_colleagues

        # Side panels: move brain/predator monitors outside the arena
        self._side_panels = side_panels
        self._panel_l = 180 if side_panels else 0   # left margin
        self._panel_r = 200 if side_panels else 0   # right margin

        # Fish parameters
        self.fish_size = 12        # triangle half-length
        self.fish_speed_base = 2.0
        self.fish_turn_max = 0.15  # radians per step
        self.eat_radius = 28.0

        # Predator parameters (1.5x bigger)
        self.pred_size = 18        # 1.5x fish
        self.pred_speed = 1.4      # threatening but escapable with C-start
        self.pred_chase_radius = 280.0
        self.pred_catch_radius = 16.0
        self.pred_wander_turn = 0.03

        # Smart predator state machine
        self.pred_state = "PATROL"  # PATROL, STALK, HUNT, AMBUSH
        self.pred_hunger = 0.3     # 0=full, 1=starving
        self.pred_stamina = 1.0    # 1=fresh, 0=exhausted
        self.pred_state_timer = 0
        self.pred_ambush_target = None  # (x,y) food cluster center
        self.pred_diag = {}        # diagnostics for monitoring

        # Predator state gating (Step 22 curriculum support)
        # When all 4 states are allowed, behavior is unchanged.
        # Empty list → predator drifts harmlessly in PATROL with speed=0.
        self._pred_allowed_states = ["PATROL", "STALK", "HUNT", "AMBUSH"]

        # Motor primitive bout dynamics (Marques et al. 2018)
        self._bout_phase = "IDLE"
        self._bout_timer = 0
        self._bout_speed_peak = 0.0
        self._bout_turn_rate = 0.0
        self._bout_ibi_timer = 0
        self._bout_ibi = 3
        self._bout_count = 0
        self._bout_type_name = "IDLE"
        self._bout_goal_mod = 2   # default EXPLORE

        # Food parameters
        self.food_radius = 4.0

        # Energy parameters (Feature A)
        self.energy_max = 100.0
        self.energy_drain_base = 0.05
        self.energy_drain_speed = 0.03
        self.energy_food_gain = 25.0

        # Obstacle parameters (Feature B)
        self.n_obstacles_min = 3
        self.n_obstacles_max = 5
        self.obstacle_r_min = 15.0
        self.obstacle_r_max = 30.0

        # Observation: 17-dim continuous (was 16, +1 for energy)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(17,), dtype=np.float32)

        # Action: [turn_rate, speed_mod] continuous
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32)

        # Pygame surface (lazy init)
        self._screen = None
        self._clock = None
        self._font = None
        self._font_big = None
        self._sim_speed = 1       # 1x, 2x, 4x, 8x speed multiplier
        self._death_timer = 0     # frames to show death splash
        self._death_pos = None    # (x, y) of death

        # Vision strip buffers (Feature C) — set by brain agent
        self.vision_strip_L = None
        self.vision_strip_R = None
        self.vision_type_L = None
        self.vision_type_R = None
        self._brain_diag = {}
        self._food_prospects = []
        self._flee_active = False
        self._panic_intensity = 0.0
        self.pred_speed_current = 0.0

        # Predator brain agent (lazy init after reset)
        self._predator_agent = None

        self.reset()

        if self.use_predator_brain:
            from zebrav1.gym_env.predator_brain_agent import PredatorBrainAgent
            self._predator_agent = PredatorBrainAgent(device="auto")
            self._predator_agent.reset()

    @property
    def render_width(self):
        """Full render surface width (arena + side panel margins)."""
        return self._panel_l + self.arena_w + self._panel_r

    @property
    def render_height(self):
        """Full render surface height."""
        return self.arena_h

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Fish state
        self.fish_x = self.arena_w * 0.5
        self.fish_y = self.arena_h * 0.5
        self.fish_heading = self.np_random.uniform(-math.pi, math.pi)
        self.fish_speed = 0.0

        # Energy (Feature A) — start at 91% so fish forages before socializing
        self.fish_energy = self.energy_max * 0.91

        # Predator state (start visible but gives foraging room: 200–300px)
        for _ in range(20):
            angle = self.np_random.uniform(-math.pi, math.pi)
            dist = self.np_random.uniform(200, 300)
            self.pred_x = self.fish_x + dist * math.cos(angle)
            self.pred_y = self.fish_y + dist * math.sin(angle)
            # Clamp to arena
            self.pred_x = float(np.clip(self.pred_x, 50, self.arena_w - 50))
            self.pred_y = float(np.clip(self.pred_y, 50, self.arena_h - 50))
            # Verify distance after clamping
            ddx = self.pred_x - self.fish_x
            ddy = self.pred_y - self.fish_y
            if math.sqrt(ddx * ddx + ddy * ddy) >= 100:
                break
        self.pred_heading = self.np_random.uniform(-math.pi, math.pi)
        self.pred_state = "PATROL"
        self.pred_hunger = 0.3
        self.pred_stamina = 1.0
        self.pred_state_timer = 0
        self.pred_ambush_target = None
        self.pred_diag = {}
        # Preserve curriculum-set gating; default = all states allowed
        if not hasattr(self, '_pred_allowed_states'):
            self._pred_allowed_states = ["PATROL", "STALK", "HUNT", "AMBUSH"]
        # Reset predator brain if active
        if self._predator_agent is not None:
            self._predator_agent.reset()

        # Rock formations — large mountain-like obstacles
        self.rock_formations = []
        self._spawn_rock_formations()
        self.plankton_patches = self._make_plankton_patches()

        # Legacy obstacles list (for predator circle-based deflection)
        self.obstacles = [{"x": r["cx"], "y": r["cy"], "r": r["base_r"]}
                          for r in self.rock_formations]

        # Food positions (after rocks, so spawning avoids them)
        # Multi-size food: large food preferentially in shelter (behind rocks),
        # small food distributed evenly. 40% hidden, 60% open.
        self.foods = []

        # Large food: mostly shelter patches (higher value, harder to reach)
        for _ in range(self.n_food_large_init):
            placed = False
            for _attempt in range(100):
                fx, fy = self._spawn_food_pos()
                if self._has_rock_occlusion(fx, fy):
                    self.foods.append([fx, fy, "large"])
                    placed = True
                    break
            if not placed:
                fx, fy = self._spawn_food_pos()
                self.foods.append([fx, fy, "large"])

        # Small food: mostly open water (easy to find, low energy)
        for _ in range(self.n_food_small_init):
            fx, fy = self._spawn_food_open()
            self.foods.append([fx, fy, "small"])

        self.step_count = 0
        self.total_eaten = 0
        self.food_eaten_this_step = 0
        self.alive = True

        # Colleagues (Step 20: social shoaling)
        self.colleagues = []
        for _ in range(self.n_colleagues):
            self.colleagues.append(self._spawn_colleague())

        # Reset vision strip and diagnostics
        self.vision_strip_L = None
        self.vision_strip_R = None
        self.vision_type_L = None
        self.vision_type_R = None
        self._brain_diag = {}
        self._food_prospects = []
        self._flee_active = False
        self._panic_intensity = 0.0
        self.pred_speed_current = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _spawn_obstacle(self):
        """Spawn a circular obstacle, rejection-sampled to avoid fish/others."""
        for _ in range(50):
            ox = self.np_random.uniform(60, self.arena_w - 60)
            oy = self.np_random.uniform(60, self.arena_h - 60)
            r = self.np_random.uniform(self.obstacle_r_min, self.obstacle_r_max)

            # Check distance from fish start (center)
            dx = ox - self.arena_w * 0.5
            dy = oy - self.arena_h * 0.5
            if dx * dx + dy * dy < (r + 50) ** 2:
                continue

            # Check distance from other obstacles
            too_close = False
            for other in self.obstacles:
                ddx = ox - other["x"]
                ddy = oy - other["y"]
                min_dist = r + other["r"] + 20
                if ddx * ddx + ddy * ddy < min_dist * min_dist:
                    too_close = True
                    break
            if too_close:
                continue

            self.obstacles.append({"x": ox, "y": oy, "r": r})
            return

    def _spawn_rock_formations(self):
        """Spawn 2-3 large mountain-like rock formations in different quadrants."""
        n_rocks = int(self.np_random.integers(2, 4))  # 2 or 3

        # First rock always left-upper (1/3 height), rest shuffled
        left_upper = (self.arena_w * 0.22, self.arena_h * 0.22)
        others = [
            (self.arena_w * 0.72, self.arena_h * 0.28),   # right upper
            (self.arena_w * 0.35, self.arena_h * 0.70),   # left lower
            (self.arena_w * 0.75, self.arena_h * 0.72),   # right lower
        ]
        self.np_random.shuffle(others)
        quadrants = [left_upper] + list(others)

        fish_cx, fish_cy = self.arena_w * 0.5, self.arena_h * 0.5

        for i in range(n_rocks):
            qx, qy = quadrants[i]
            # Jitter within quadrant
            cx = qx + self.np_random.uniform(-40, 40)
            cy = qy + self.np_random.uniform(-30, 30)
            # Clamp to safe area
            cx = float(np.clip(cx, 80, self.arena_w - 80))
            cy = float(np.clip(cy, 80, self.arena_h - 80))

            # Enforce min distance from fish spawn center (>90px)
            dx, dy = cx - fish_cx, cy - fish_cy
            if math.sqrt(dx * dx + dy * dy) < 90:
                # Push outward
                d = math.sqrt(dx * dx + dy * dy) + 1e-8
                cx = fish_cx + dx / d * 95
                cy = fish_cy + dy / d * 95

            # Enforce min distance between rocks (>50px center-to-center)
            too_close = False
            for prev in self.rock_formations:
                ddx, ddy = cx - prev["cx"], cy - prev["cy"]
                if math.sqrt(ddx * ddx + ddy * ddy) < 50 + prev["base_r"]:
                    too_close = True
                    break
            if too_close:
                continue

            base_r = float(self.np_random.uniform(55, 85))
            seed = int(self.np_random.integers(0, 100000))
            rock = _make_rock_formation(cx, cy, base_r, n_lobes=5, seed=seed)
            self.rock_formations.append(rock)

    def _make_plankton_patches(self):
        """Create plankton patch descriptors based on rock positions.

        Food is hidden behind rocks (away from fish spawn at arena center)
        to create a foraging challenge requiring exploration.

        Patch types:
            0 – behind-rock shelter patch (high density, weight 0.30)
            1 – second shelter patch behind rock[1] if available (weight 0.25)
            2 – open-water patch (moderate density, weight 0.30)
            3 – background sprinkling (arena-wide, weight 0.15)
        """
        patches = []
        center_x = self.arena_w * 0.5
        center_y = self.arena_h * 0.5

        if self.rock_formations:
            # Patch 0: behind first rock — angle points away from fish spawn
            r0 = self.rock_formations[0]
            angle = math.atan2(r0["cy"] - center_y, r0["cx"] - center_x)
            offset = r0["base_r"] * 1.8
            p0_cx = r0["cx"] + offset * math.cos(angle)
            p0_cy = r0["cy"] + offset * math.sin(angle)
            p0_cx = float(np.clip(p0_cx, 40, self.arena_w - 40))
            p0_cy = float(np.clip(p0_cy, 40, self.arena_h - 40))
            patches.append({
                "cx": p0_cx, "cy": p0_cy,
                "radius": r0["base_r"] * 1.2,
                "weight": 0.30,
                "label": "shelter",
            })

            # Patch 1: behind second rock if available
            if len(self.rock_formations) >= 2:
                r1 = self.rock_formations[1]
                angle1 = math.atan2(r1["cy"] - center_y, r1["cx"] - center_x)
                offset1 = r1["base_r"] * 1.8
                p1_cx = r1["cx"] + offset1 * math.cos(angle1)
                p1_cy = r1["cy"] + offset1 * math.sin(angle1)
                p1_cx = float(np.clip(p1_cx, 40, self.arena_w - 40))
                p1_cy = float(np.clip(p1_cy, 40, self.arena_h - 40))
                patches.append({
                    "cx": p1_cx, "cy": p1_cy,
                    "radius": r1["base_r"] * 1.2,
                    "weight": 0.25,
                    "label": "shelter",
                })
        else:
            # Fallback: random location
            patches.append({
                "cx": float(self.arena_w * 0.3),
                "cy": float(self.arena_h * 0.4),
                "radius": 80.0,
                "weight": 0.50,
                "label": "shelter",
            })

        # Open water patch — far from all rocks
        for _ in range(30):
            ox = float(self.np_random.uniform(60, self.arena_w - 60))
            oy = float(self.np_random.uniform(60, self.arena_h - 60))
            far_enough = True
            for rock in self.rock_formations:
                ddx, ddy = ox - rock["cx"], oy - rock["cy"]
                if math.sqrt(ddx * ddx + ddy * ddy) < rock["base_r"] * 2.5:
                    far_enough = False
                    break
            if far_enough:
                break
        patches.append({
            "cx": ox, "cy": oy,
            "radius": 90.0,
            "weight": 0.30,
            "label": "open_water",
        })

        # Arena-wide background
        patches.append({
            "cx": float(self.arena_w * 0.5),
            "cy": float(self.arena_h * 0.5),
            "radius": float(max(self.arena_w, self.arena_h) * 0.5),
            "weight": 0.15,
            "label": "background",
        })

        return patches

    def _has_rock_occlusion(self, fx, fy):
        """Check if any rock AABB blocks line-of-sight from fish spawn (center) to (fx,fy).

        Uses AABB-line segment intersection test.
        """
        ox, oy = self.arena_w * 0.5, self.arena_h * 0.5
        dx, dy = fx - ox, fy - oy

        for rock in self.rock_formations:
            for aabb in rock["aabbs"]:
                # Translate to AABB-local coords
                lx, ly = ox - aabb["x"], oy - aabb["y"]
                hw, hh = aabb["hw"], aabb["hh"]

                # Ray-AABB slab intersection
                t_min, t_max = 0.0, 1.0
                for axis_d, axis_l, half in [(dx, lx, hw), (dy, ly, hh)]:
                    if abs(axis_d) < 1e-8:
                        # Ray parallel to slab
                        if abs(axis_l) > half:
                            t_min = 2.0  # no intersection
                            break
                    else:
                        inv_d = 1.0 / axis_d
                        t1 = (-half - axis_l) * inv_d
                        t2 = (half - axis_l) * inv_d
                        if t1 > t2:
                            t1, t2 = t2, t1
                        t_min = max(t_min, t1)
                        t_max = min(t_max, t2)
                        if t_min > t_max:
                            break

                if t_min <= t_max:
                    return True
        return False

    def _spawn_food_pos(self):
        """Spawn food using patch-biased Gaussian sampling."""
        # Fallback to uniform if no patches yet
        if not hasattr(self, 'plankton_patches') or not self.plankton_patches:
            return self._spawn_food_pos_uniform()

        weights = np.array([p["weight"] for p in self.plankton_patches])
        weights /= weights.sum()

        for _ in range(80):
            # Pick a patch by weight
            idx = self.np_random.choice(len(self.plankton_patches), p=weights)
            patch = self.plankton_patches[idx]

            # Gaussian offset within patch radius
            fx = patch["cx"] + self.np_random.normal(0, patch["radius"] * 0.4)
            fy = patch["cy"] + self.np_random.normal(0, patch["radius"] * 0.4)

            # Clamp to arena
            fx = float(np.clip(fx, 30, self.arena_w - 30))
            fy = float(np.clip(fy, 30, self.arena_h - 30))

            # Avoid predator
            dx = fx - self.pred_x
            dy = fy - self.pred_y
            if dx * dx + dy * dy < 60 ** 2:
                continue

            # Reject if inside any rock AABB (8px margin)
            inside_rock = False
            for rock in self.rock_formations:
                for aabb in rock["aabbs"]:
                    if (abs(fx - aabb["x"]) < aabb["hw"] + 8 and
                            abs(fy - aabb["y"]) < aabb["hh"] + 8):
                        inside_rock = True
                        break
                if inside_rock:
                    break
            if inside_rock:
                continue

            return fx, fy

        # Fallback
        return self._spawn_food_pos_uniform()

    def _spawn_food_open(self):
        """Spawn food in open water — far from rocks, clear line-of-sight."""
        for _ in range(100):
            fx = float(self.np_random.uniform(60, self.arena_w - 60))
            fy = float(self.np_random.uniform(60, self.arena_h - 60))

            # Must be far from all rock AABBs (>60px margin)
            too_close = False
            for rock in getattr(self, 'rock_formations', []):
                rdx = fx - rock["cx"]
                rdy = fy - rock["cy"]
                if math.sqrt(rdx * rdx + rdy * rdy) < rock["base_r"] * 2.0:
                    too_close = True
                    break
            if too_close:
                continue

            # Must not be occluded from fish spawn (center)
            if self._has_rock_occlusion(fx, fy):
                continue

            return fx, fy

        # Fallback: uniform spawn
        return self._spawn_food_pos_uniform()

    def _spawn_food_pos_uniform(self):
        """Fallback uniform food spawn (avoids rocks)."""
        for _ in range(50):
            fx = float(self.np_random.uniform(30, self.arena_w - 30))
            fy = float(self.np_random.uniform(30, self.arena_h - 30))

            inside = False
            for rock in getattr(self, 'rock_formations', []):
                for aabb in rock["aabbs"]:
                    if (abs(fx - aabb["x"]) < aabb["hw"] + 8 and
                            abs(fy - aabb["y"]) < aabb["hh"] + 8):
                        inside = True
                        break
                if inside:
                    break
            if inside:
                continue
            return fx, fy

        return (float(self.np_random.uniform(30, self.arena_w - 30)),
                float(self.np_random.uniform(30, self.arena_h - 30)))

    # ------------------------------------------------------------------
    # Colleague helpers (Step 20)
    # ------------------------------------------------------------------

    def _spawn_colleague(self):
        """Spawn a colleague fish at a random position away from predator."""
        for _ in range(30):
            cx = float(self.np_random.uniform(60, self.arena_w - 60))
            cy = float(self.np_random.uniform(60, self.arena_h - 60))
            # Avoid predator
            dx = cx - self.pred_x
            dy = cy - self.pred_y
            if dx * dx + dy * dy < 80 ** 2:
                continue
            # Avoid rocks
            inside = False
            for rock in getattr(self, 'rock_formations', []):
                rdx = cx - rock["cx"]
                rdy = cy - rock["cy"]
                if math.sqrt(rdx * rdx + rdy * rdy) < rock["base_r"] * 1.2:
                    inside = True
                    break
            if inside:
                continue
            return {
                "x": cx, "y": cy,
                "heading": float(self.np_random.uniform(-math.pi, math.pi)),
                "speed": float(self.np_random.uniform(0.8, 1.2)),
            }
        # Fallback
        return {
            "x": float(self.arena_w * 0.5 + self.np_random.uniform(-50, 50)),
            "y": float(self.arena_h * 0.5 + self.np_random.uniform(-50, 50)),
            "heading": float(self.np_random.uniform(-math.pi, math.pi)),
            "speed": 1.0,
        }

    def _update_colleagues(self):
        """Move colleagues: Reynolds flocking + gaze-aware predator escape.

        Each colleague applies separation, cohesion, alignment with all
        neighbors (other colleagues + main fish), plus predator evasion
        with per-individual jitter.
        """
        if not self.colleagues:
            return

        # Flocking parameters
        sep_radius = 25.0
        coh_radius = 100.0
        align_radius = 60.0
        sep_w = 1.0
        coh_w = 0.5
        align_w = 0.3

        pred_speed = getattr(self, 'pred_speed_current', 0.0)

        for i, c in enumerate(self.colleagues):
            jitter = c["speed"] - 1.0  # ≈ [-0.2, +0.2]

            # --- Reynolds flocking ---
            sep_x, sep_y = 0.0, 0.0
            coh_x, coh_y = 0.0, 0.0
            align_sin, align_cos = 0.0, 0.0
            n_sep, n_coh, n_align = 0, 0, 0

            # Check neighbors: other colleagues + main fish
            neighbors = [(self.fish_x, self.fish_y, self.fish_heading)]
            for j, other in enumerate(self.colleagues):
                if i != j:
                    neighbors.append((other["x"], other["y"], other["heading"]))

            for nx, ny, nh in neighbors:
                dx = nx - c["x"]
                dy = ny - c["y"]
                dist = math.sqrt(dx * dx + dy * dy) + 1e-8
                if dist < sep_radius:
                    sep_x -= dx / (dist * dist)
                    sep_y -= dy / (dist * dist)
                    n_sep += 1
                if dist < coh_radius:
                    coh_x += dx
                    coh_y += dy
                    n_coh += 1
                if dist < align_radius:
                    align_sin += math.sin(nh)
                    align_cos += math.cos(nh)
                    n_align += 1

            desired_dx, desired_dy = 0.0, 0.0
            if n_sep > 0:
                desired_dx += sep_w * sep_x
                desired_dy += sep_w * sep_y
            if n_coh > 0:
                desired_dx += coh_w * (coh_x / n_coh)
                desired_dy += coh_w * (coh_y / n_coh)
            if n_align > 0:
                avg_h = math.atan2(align_sin / n_align, align_cos / n_align)
                align_diff = avg_h - c["heading"]
                align_diff = math.atan2(
                    math.sin(align_diff), math.cos(align_diff))
                desired_dx += align_w * math.cos(c["heading"] + align_diff)
                desired_dy += align_w * math.sin(c["heading"] + align_diff)

            if abs(desired_dx) + abs(desired_dy) > 1e-6:
                desired_angle = math.atan2(desired_dy, desired_dx)
                angle_diff = desired_angle - c["heading"]
                angle_diff = math.atan2(
                    math.sin(angle_diff), math.cos(angle_diff))
                c["heading"] += max(-0.15, min(0.15, angle_diff))

            # Random walk noise (reduced for cohesive schooling)
            c["heading"] += float(self.np_random.uniform(-0.04, 0.04))

            # --- Gaze-aware predator escape ---
            pdx = c["x"] - self.pred_x
            pdy = c["y"] - self.pred_y
            pred_dist = math.sqrt(pdx * pdx + pdy * pdy) + 1e-8

            angle_pred_to_c = math.atan2(pdy, pdx)
            facing_diff = self.pred_heading - angle_pred_to_c
            facing_diff = math.atan2(
                math.sin(facing_diff), math.cos(facing_diff))
            c_facing_score = max(0.0, 1.0 - abs(facing_diff) / math.pi)

            pred_vx = pred_speed * math.cos(self.pred_heading)
            pred_vy = pred_speed * math.sin(self.pred_heading)
            ux = pdx / pred_dist
            uy = pdy / pred_dist
            closing_speed = -(pred_vx * ux + pred_vy * uy)
            c_ttc = (pred_dist / closing_speed) if closing_speed > 0.1 else 999.0

            detect_range = 100.0 + jitter * 100.0

            c_panic = 0.0
            facing_thresh = 0.5 + jitter * 0.15
            if abs(facing_diff) < facing_thresh and c_ttc < (40 + jitter * 10):
                c_panic = c_facing_score * max(0.0, 1.0 - c_ttc / 40.0)
            elif c_ttc < 20:
                c_panic = 0.5 * max(0.0, 1.0 - c_ttc / 20.0)
            c_panic = min(1.0, c_panic)

            if pred_dist < detect_range:
                flee_angle = math.atan2(pdy, pdx)
                flee_diff = flee_angle - c["heading"]
                flee_diff = math.atan2(
                    math.sin(flee_diff), math.cos(flee_diff))
                flee_gain = 0.15 + 0.15 * c_facing_score
                c["heading"] += flee_gain * flee_diff

            # Rock avoidance
            for rock in getattr(self, 'rock_formations', []):
                rdx = c["x"] - rock["cx"]
                rdy = c["y"] - rock["cy"]
                rdist = math.sqrt(rdx * rdx + rdy * rdy) + 1e-8
                if rdist < rock["base_r"] * 1.3:
                    away = math.atan2(rdy, rdx)
                    diff = away - c["heading"]
                    diff = math.atan2(math.sin(diff), math.cos(diff))
                    c["heading"] += 0.2 * diff

            # Wall avoidance
            margin = 40
            if c["x"] < margin:
                c["heading"] += 0.1
            elif c["x"] > self.arena_w - margin:
                c["heading"] -= 0.1
            if c["y"] < margin:
                c["heading"] += 0.1 * math.copysign(1, math.cos(c["heading"]))
            elif c["y"] > self.arena_h - margin:
                c["heading"] -= 0.1 * math.copysign(1, math.cos(c["heading"]))

            c["heading"] = math.atan2(
                math.sin(c["heading"]), math.cos(c["heading"]))

            # Move — panic sprint boosts speed
            base_spd = c["speed"] * 1.2
            if c_panic > 0.1:
                base_spd = max(base_spd, base_spd * (1.0 + 0.5 * c_panic))
            c["x"] += base_spd * math.cos(c["heading"])
            c["y"] += base_spd * math.sin(c["heading"])

            # Clamp to arena
            c["x"] = float(np.clip(c["x"], 10, self.arena_w - 10))
            c["y"] = float(np.clip(c["y"], 10, self.arena_h - 10))

    def set_vision_strip(self, retL, retR, typeL, typeR):
        """Set the vision strip buffers for rendering (Feature C).

        Args:
            retL, retR: 1D arrays of retinal intensity [N]
            typeL, typeR: 1D arrays of entity type encoding [N]
        """
        self.vision_strip_L = np.asarray(retL).flatten()
        self.vision_strip_R = np.asarray(retR).flatten()
        self.vision_type_L = np.asarray(typeL).flatten()
        self.vision_type_R = np.asarray(typeR).flatten()

    def set_brain_diagnostics(self, diag):
        """Set brain diagnostics dict for monitoring panel rendering."""
        self._brain_diag = diag

    def set_flee_active(self, active, panic_intensity=0.0):
        """Signal whether the fish is currently fleeing (higher energy cost).

        Args:
            active: bool — fleeing state
            panic_intensity: float [0, 1] — panic level scales energy drain
        """
        self._flee_active = bool(active)
        self._panic_intensity = float(np.clip(panic_intensity, 0.0, 1.0))

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        turn_rate = float(action[0])
        speed_mod = float(action[1])

        # === Energy-based speed capping (Feature A) ===
        # Below 30% energy: fish weakens, max speed scales down linearly
        # At 30% → cap at 0.7, at 0% → cap at 0.15 (barely moving)
        energy_ratio_now = self.fish_energy / self.energy_max
        if energy_ratio_now < 0.20:
            starvation_cap = 0.15 + 0.55 * (energy_ratio_now / 0.20)
            speed_mod = min(speed_mod, starvation_cap)

        # === Fish movement (motor primitive bout dynamics) ===
        # Bouts modulate speed timing + add turn noise; brain controls direction
        bout_turn_noise, bout_speed = self._update_bout_state(speed_mod, turn_rate)
        # Brain turn + biological noise from motor primitive
        self.fish_heading += turn_rate * self.fish_turn_max + bout_turn_noise
        self.fish_heading = math.atan2(
            math.sin(self.fish_heading), math.cos(self.fish_heading))
        self.fish_speed = self.fish_speed_base * bout_speed
        self.fish_x += self.fish_speed * math.cos(self.fish_heading)
        self.fish_y += self.fish_speed * math.sin(self.fish_heading)

        # Wall clamping
        margin = 5
        self.fish_x = np.clip(self.fish_x, margin, self.arena_w - margin)
        self.fish_y = np.clip(self.fish_y, margin, self.arena_h - margin)

        # === Obstacle collision (Feature B) ===
        self._resolve_obstacle_collisions()

        # === Predator AI ===
        if self._predator_agent is not None:
            self._update_predator_brain()
        else:
            self._update_predator()

        # === Colleague movement (Step 20) ===
        self._update_colleagues()

        # === Food eating (multi-size, strike-enlarged radius) ===
        effective_eat_r = self.eat_radius
        if getattr(self, '_strike_active', False):
            effective_eat_r = self.eat_radius * 1.4  # Feature 5: strike lunge
        eaten = 0
        energy_gained = 0.0
        remaining = []
        for food in self.foods:
            fx, fy = food[0], food[1]
            sz = food[2] if len(food) > 2 else "small"
            dx = self.fish_x - fx
            dy = self.fish_y - fy
            if dx * dx + dy * dy < effective_eat_r ** 2:
                eaten += 1
                energy_gained += self.FOOD_ENERGY.get(sz, 2.0)
            else:
                remaining.append(food)
        self.foods = remaining
        self.total_eaten += eaten
        self.food_eaten_this_step = eaten

        # Energy gain from eating (variable by food size)
        self.fish_energy = min(self.energy_max,
                               self.fish_energy + energy_gained)

        # Respawn food to maintain type ratios
        n_small = sum(1 for f in self.foods
                      if (f[2] if len(f) > 2 else "small") == "small")
        n_large = sum(1 for f in self.foods
                      if (f[2] if len(f) > 2 else "small") == "large")
        while n_small < self.n_food_small_init:
            fx, fy = self._spawn_food_open()
            self.foods.append([fx, fy, "small"])
            n_small += 1
        while n_large < self.n_food_large_init:
            fx, fy = self._spawn_food_pos()
            self.foods.append([fx, fy, "large"])
            n_large += 1

        # === Energy drain (Feature A) ===
        # Flee multiplier: escape bursts cost 2.5–3.5x (panic-scaled)
        if self._flee_active:
            flee_mult = 2.5 + 1.0 * self._panic_intensity
        else:
            flee_mult = 1.0

        # Starvation pressure: below 50% energy, metabolic cost rises
        # (weakened fish burns reserves faster). Below 30%, severe penalty.
        energy_ratio = self.fish_energy / self.energy_max
        if energy_ratio < 0.30:
            starvation_mult = 1.3   # critically starving
        elif energy_ratio < 0.50:
            starvation_mult = 1.15  # danger zone
        else:
            starvation_mult = 1.0

        actual_speed = self.fish_speed / max(0.01, self.fish_speed_base)
        self.fish_energy -= (self.energy_drain_base
                             + self.energy_drain_speed * actual_speed * flee_mult
                             ) * starvation_mult
        self.fish_energy = max(0.0, self.fish_energy)

        # === Check predator catch ===
        dx = self.fish_x - self.pred_x
        dy = self.fish_y - self.pred_y
        pred_dist = math.sqrt(dx * dx + dy * dy)
        caught = pred_dist < self.pred_catch_radius

        if caught:
            self.alive = False
            self._death_timer = 60  # show splash for 60 frames
            self._death_pos = (self.fish_x, self.fish_y)
            # Feed catch event to predator brain
            if self._predator_agent is not None:
                self._predator_agent.update_post_step(caught=True)

        # === Predator catches colleagues ===
        catch_r2 = (self.pred_catch_radius * 1.5) ** 2  # slightly larger radius
        surviving = []
        colleague_caught = False
        for c in getattr(self, 'colleagues', []):
            cdx = c["x"] - self.pred_x
            cdy = c["y"] - self.pred_y
            if cdx * cdx + cdy * cdy < catch_r2:
                colleague_caught = True
                # Predator eats colleague — satisfy hunger
                self.pred_hunger = max(0.0, self.pred_hunger - 0.3)
            else:
                surviving.append(c)
        if colleague_caught:
            self.colleagues = surviving
            if self._predator_agent is not None:
                self._predator_agent.update_post_step(caught=True)
            # Respawn colleague after a delay (instant for now)
            while len(self.colleagues) < self.n_colleagues:
                self.colleagues.append(self._spawn_colleague())

        # === Check starvation (Feature A) ===
        starved = self.fish_energy <= 0.0
        if starved and self.alive:
            self._death_timer = 60
            self._death_pos = (self.fish_x, self.fish_y)

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        terminated = caught or starved

        # === Reward ===
        reward = 0.0
        reward += eaten * 10.0                  # food reward
        reward -= 0.01                          # time penalty

        if caught:
            reward -= 50.0                      # death penalty
        elif starved:
            reward -= 30.0                      # starvation penalty
        elif pred_dist < 50:
            reward -= 0.1 * (50 - pred_dist) / 50  # proximity danger

        # Energy reward shaping (Feature A)
        energy_ratio = self.fish_energy / self.energy_max
        reward += 0.002 * energy_ratio
        if energy_ratio < 0.2:
            reward -= 0.05 * (1.0 - energy_ratio)

        # Wall avoidance
        wall_dists = [
            self.fish_y,
            self.arena_h - self.fish_y,
            self.fish_x,
            self.arena_w - self.fish_x,
        ]
        if min(wall_dists) < 20:
            reward -= 0.05

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Motor Primitive Repertoire
    # Real zebrafish select from ~7 stereotyped bout types, each with
    # characteristic kinematics. The brain provides "intent" (direction +
    # urgency); the motor system selects the closest matching primitive.
    # Based on Marques et al. (2018), Dunn et al. (2016), Johnson et al. (2020).
    #
    # Each primitive: (name, turn_mean, turn_std, speed, duration, ibi)
    # turn in radians, speed as multiplier, duration in steps
    # ------------------------------------------------------------------
    BOUT_TYPES = {
        "IDLE":         {"turn": 0.0,  "turn_std": 0.03, "speed": 0.0,  "dur": 3, "ibi": 4},
        "ROUTINE_FWD":  {"turn": 0.0,  "turn_std": 0.08, "speed": 0.5,  "dur": 3, "ibi": 3},
        "ROUTINE_L":    {"turn": -0.3, "turn_std": 0.10, "speed": 0.4,  "dur": 3, "ibi": 3},
        "ROUTINE_R":    {"turn": 0.3,  "turn_std": 0.10, "speed": 0.4,  "dur": 3, "ibi": 3},
        "J_TURN_L":     {"turn": -0.5, "turn_std": 0.05, "speed": 0.25, "dur": 4, "ibi": 2},
        "J_TURN_R":     {"turn": 0.5,  "turn_std": 0.05, "speed": 0.25, "dur": 4, "ibi": 2},
        "BURST_FWD":    {"turn": 0.0,  "turn_std": 0.05, "speed": 1.0,  "dur": 4, "ibi": 2},
        "BURST_L":      {"turn": -0.4, "turn_std": 0.08, "speed": 0.9,  "dur": 3, "ibi": 2},
        "BURST_R":      {"turn": 0.4,  "turn_std": 0.08, "speed": 0.9,  "dur": 3, "ibi": 2},
        "C_START_L":    {"turn": -0.9, "turn_std": 0.05, "speed": 0.2,  "dur": 2, "ibi": 1},
        "C_START_R":    {"turn": 0.9,  "turn_std": 0.05, "speed": 0.2,  "dur": 2, "ibi": 1},
        "CAPTURE":      {"turn": 0.0,  "turn_std": 0.02, "speed": 1.3,  "dur": 2, "ibi": 3},
    }

    def _select_bout_type(self, turn_intent, speed_intent):
        """Select the best-matching motor primitive for the brain's intent.

        Maps continuous (turn, speed) intent from the brain to the closest
        discrete bout type. Adds biological noise to the selected bout.

        Args:
            turn_intent: float [-1, 1] — desired turn direction
            speed_intent: float [0, 1.5] — desired speed
        Returns:
            bout_type name, effective (turn_rate, speed) for this step
        """
        import random

        # Scale brain turn to radians (fish_turn_max ≈ 0.15 rad)
        desired_turn = turn_intent * 0.15  # convert [-1,1] to ~[-0.15, 0.15]

        # Score each bout type by similarity to intent
        best_type = "ROUTINE_FWD"
        best_score = float('inf')
        for name, params in self.BOUT_TYPES.items():
            # Skip C-start unless speed is very high (Mauthner override)
            if name.startswith("C_START") and speed_intent < 1.2:
                continue
            # Skip CAPTURE unless speed > 1.2 (prey strike)
            if name == "CAPTURE" and speed_intent < 1.2:
                continue
            # Skip IDLE unless speed is very low
            if name == "IDLE" and speed_intent > 0.15:
                continue

            turn_err = (desired_turn - params["turn"] * 0.15) ** 2
            speed_err = (speed_intent - params["speed"]) ** 2
            score = turn_err * 3.0 + speed_err  # weight turn matching more
            if score < best_score:
                best_score = score
                best_type = name

        params = self.BOUT_TYPES[best_type]
        # Add biological noise to the selected primitive
        noisy_turn = params["turn"] + random.gauss(0, params["turn_std"])
        noisy_speed = params["speed"] * (0.85 + 0.3 * random.random())

        return best_type, noisy_turn, noisy_speed, params["dur"], params["ibi"]

    def _update_bout_state(self, speed_mod, turn_rate=0.0):
        """Motor primitive bout dynamics.

        The brain controls DIRECTION (turn_rate). The motor system controls
        TIMING (burst-glide-idle cycle) and adds BIOLOGICAL NOISE.

        Real zebrafish kinematics: bouts are 50-100ms bursts at 2-5 Hz,
        with glide phases between. Turn variability adds natural trajectory
        curvature (Marques et al. 2018).

        Returns:
            (turn_noise, effective_speed) — noise is ADDED to brain turn
        """
        import random

        # Goal-dependent IBI
        IBI_BY_GOAL = {0: 2, 1: 1, 2: 3, 3: 3}
        goal = getattr(self, '_bout_goal_mod', 2)

        # Escape/strike: bypass bout timing — sustained burst
        if speed_mod > 1.0:
            self._bout_phase = "BURST"
            self._bout_timer = 5  # longer escape burst
            self._bout_speed_peak = speed_mod
            self._bout_count += 1
            self._bout_type_name = "ESCAPE"
            return random.gauss(0, 0.01), speed_mod

        # Flee mode: faster bouts, shorter glide, minimal IBI
        is_fleeing = (goal == 1)  # GOAL_FLEE
        if is_fleeing and speed_mod > 0.5:
            self._bout_phase = "BURST"
            self._bout_timer = 4
            self._bout_speed_peak = speed_mod
            self._bout_count += 1
            self._bout_type_name = "FLEE_BURST"
            return random.gauss(0, 0.02), speed_mod

        # Continue current bout
        if self._bout_phase == "BURST":
            self._bout_timer -= 1
            if self._bout_timer <= 0:
                self._bout_phase = "GLIDE"
                self._bout_timer = 2
            # During burst: full speed + small turn noise
            noise = random.gauss(0, 0.03)
            return noise, self._bout_speed_peak * (0.85 + 0.15 * random.random())

        if self._bout_phase == "GLIDE":
            self._bout_timer -= 1
            glide_frac = max(0.0, self._bout_timer / 2.0)
            if self._bout_timer <= 0:
                self._bout_phase = "IDLE"
                self._bout_ibi_timer = IBI_BY_GOAL.get(goal, 3)
            # During glide: decaying speed, brain turn still applies
            return random.gauss(0, 0.02), self._bout_speed_peak * (0.3 + 0.3 * glide_frac)

        # IDLE: wait for IBI, then initiate new bout
        # Starvation or strong intent interrupts IBI
        energy_ratio = self.fish_energy / self.energy_max
        starving = energy_ratio < 0.30
        urgent = abs(turn_rate) > 0.5 or speed_mod > 0.5 or starving
        if self._bout_ibi_timer > 0 and not urgent:
            self._bout_ibi_timer -= 1
            # Idle fidgeting: micro-movements for natural appearance
            fidget = random.gauss(0, 0.015)
            return fidget, speed_mod * 0.15

        # Initiate new bout
        if speed_mod > 0.05 or abs(turn_rate) > 0.1:
            self._bout_phase = "BURST"
            dur = random.randint(3, 4) if speed_mod < 0.7 else random.randint(3, 5)
            self._bout_timer = dur
            self._bout_speed_peak = speed_mod
            self._bout_count += 1
            # Classify bout type for diagnostics
            if abs(turn_rate) > 0.6:
                self._bout_type_name = "ROUTINE_TURN"
            elif speed_mod > 0.7:
                self._bout_type_name = "BURST_SWIM"
            else:
                self._bout_type_name = "ROUTINE_FWD"
            return random.gauss(0, 0.03), speed_mod

        # Truly idle
        return random.gauss(0, 0.015), 0.05

    # ------------------------------------------------------------------
    # Interactive commands (keyboard during demo)
    # ------------------------------------------------------------------
    def _command_predator_attack(self):
        """[P key] Send predator charging directly at fish."""
        angle = math.atan2(self.fish_y - self.pred_y,
                           self.fish_x - self.pred_x)
        self.pred_heading = angle
        self.pred_state = "HUNT"
        self.pred_stamina = 1.0
        self.pred_state_timer = 0
        print(f"[CMD] Predator ATTACK → fish at ({self.fish_x:.0f}, {self.fish_y:.0f})")

    def _command_predator_retreat(self):
        """[R key] Send predator to far corner."""
        # Move to opposite corner from fish
        cx = self.arena_w - self.fish_x
        cy = self.arena_h - self.fish_y
        self.pred_x = max(30, min(self.arena_w - 30, cx))
        self.pred_y = max(30, min(self.arena_h - 30, cy))
        self.pred_state = "PATROL"
        self.pred_stamina = 1.0
        print(f"[CMD] Predator RETREAT → ({self.pred_x:.0f}, {self.pred_y:.0f})")

    def _command_spawn_food_cluster(self):
        """[F key] Spawn a dense food cluster near the fish."""
        import random
        cx = self.fish_x + random.uniform(-60, 60)
        cy = self.fish_y + random.uniform(-60, 60)
        for _ in range(8):
            fx = cx + random.gauss(0, 25)
            fy = cy + random.gauss(0, 25)
            fx = max(20, min(self.arena_w - 20, fx))
            fy = max(20, min(self.arena_h - 20, fy))
            self.foods.append([fx, fy, "small"])
        print(f"[CMD] Spawned 8 food near ({cx:.0f}, {cy:.0f})")

    def _resolve_obstacle_collisions(self):
        """Push fish out of rock formation AABBs (Feature B).

        Strong push-out + heading deflection to prevent sticking.
        """
        fish_r = self.fish_size * 0.5
        for rock in getattr(self, 'rock_formations', []):
            for aabb in rock["aabbs"]:
                dx = self.fish_x - aabb["x"]
                dy = self.fish_y - aabb["y"]
                overlap_x = aabb["hw"] + fish_r - abs(dx)
                overlap_y = aabb["hh"] + fish_r - abs(dy)

                if overlap_x > 0 and overlap_y > 0:
                    # Push out with generous margin to prevent re-entry
                    push_margin = 4.0
                    if overlap_x < overlap_y:
                        push_dir = 1.0 if dx > 0 else -1.0
                        self.fish_x += push_dir * (overlap_x + push_margin)
                    else:
                        push_dir = 1.0 if dy > 0 else -1.0
                        self.fish_y += push_dir * (overlap_y + push_margin)

                    # Deflect heading away from rock center (Fix D: 0.7 gain)
                    away_angle = math.atan2(dy, dx)
                    angle_diff = away_angle - self.fish_heading
                    angle_diff = math.atan2(
                        math.sin(angle_diff), math.cos(angle_diff))
                    # Strong heading deflection to prevent re-approach
                    self.fish_heading += 0.7 * angle_diff
                    self.fish_heading = math.atan2(
                        math.sin(self.fish_heading),
                        math.cos(self.fish_heading))

    def _find_food_cluster(self):
        """Find the densest food cluster center for ambush strategy."""
        if len(self.foods) < 3:
            return None
        best_center = None
        best_count = 0
        for food in self.foods:
            fx, fy = food[0], food[1]
            count = 0
            for food2 in self.foods:
                fx2, fy2 = food2[0], food2[1]
                if (fx - fx2) ** 2 + (fy - fy2) ** 2 < 80 ** 2:
                    count += 1
            if count > best_count:
                best_count = count
                best_center = (fx, fy)
        return best_center if best_count >= 3 else None

    def _update_predator_brain(self):
        """Update predator using SNN brain agent instead of state machine."""
        agent = self._predator_agent

        # Hunger increases over time
        self.pred_hunger = min(1.0, self.pred_hunger + 0.0003)

        # Get brain action
        turn_rate, speed_mod = agent.act(self)

        # Apply action (slightly faster turning than fish for pursuit advantage)
        pred_turn_max = 0.18
        self.pred_heading += turn_rate * pred_turn_max
        self.pred_heading = math.atan2(
            math.sin(self.pred_heading), math.cos(self.pred_heading))

        speed = self.pred_speed * speed_mod
        self.pred_x += speed * math.cos(self.pred_heading)
        self.pred_y += speed * math.sin(self.pred_heading)
        self.pred_speed_current = speed

        # Stamina dynamics: slow drain during sprint, faster recovery when slow
        if speed_mod > 0.6:
            self.pred_stamina = max(0.0, self.pred_stamina - 0.002)
        else:
            self.pred_stamina = min(1.0, self.pred_stamina + 0.006)

        # Obstacle deflection (same as state machine predator)
        pred_r = self.pred_size * 0.5
        for obs in self.obstacles:
            ddx = self.pred_x - obs["x"]
            ddy = self.pred_y - obs["y"]
            d = math.sqrt(ddx * ddx + ddy * ddy) + 1e-8
            min_d = obs["r"] + pred_r
            if d < min_d:
                nx = ddx / d
                ny = ddy / d
                self.pred_x = obs["x"] + nx * (min_d + 1.0)
                self.pred_y = obs["y"] + ny * (min_d + 1.0)
                self.pred_heading = math.atan2(ny, nx)

        # Wall bounce
        margin = 20
        if self.pred_x < margin:
            self.pred_x = margin
            self.pred_heading = math.pi - self.pred_heading
        if self.pred_x > self.arena_w - margin:
            self.pred_x = self.arena_w - margin
            self.pred_heading = math.pi - self.pred_heading
        if self.pred_y < margin:
            self.pred_y = margin
            self.pred_heading = -self.pred_heading
        if self.pred_y > self.arena_h - margin:
            self.pred_y = self.arena_h - margin
            self.pred_heading = -self.pred_heading

        # Map brain goal to state machine state name (for diagnostics/rendering)
        from zebrav1.gym_env.predator_brain_agent import GOAL_NAMES as PRED_GOAL_NAMES
        diag = agent.last_diagnostics
        goal_name = diag.get("goal_name", "PATROL")
        self.pred_state = goal_name  # reuse pred_state for rendering

        dx = self.fish_x - self.pred_x
        dy = self.fish_y - self.pred_y
        dist = math.sqrt(dx * dx + dy * dy)

        self.pred_diag = {
            "state": goal_name,
            "prev_state": goal_name,
            "hunger": self.pred_hunger,
            "stamina": self.pred_stamina,
            "dist_to_fish": dist,
            "speed": speed,
            "strategy": f"brain:{goal_name.lower()} conf={diag.get('confidence', 0):.2f}",
            "timer": self._step_count if hasattr(self, '_step_count') else 0,
            "ambush_target": None,
        }

    def _update_predator(self):
        """Smart predator AI with state machine: PATROL, STALK, HUNT, AMBUSH."""
        dx = self.fish_x - self.pred_x
        dy = self.fish_y - self.pred_y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8
        angle_to_fish = math.atan2(dy, dx)

        # --- Hunger increases over time, decreases on catch ---
        self.pred_hunger = min(1.0, self.pred_hunger + 0.0003)

        # --- State transitions ---
        self.pred_state_timer += 1
        prev_state = self.pred_state

        # Allowed states for curriculum gating (Step 22)
        allowed = self._pred_allowed_states

        if self.pred_state == "PATROL":
            # Transition to STALK if fish detected and STALK is allowed
            if (dist < self.pred_chase_radius
                    and "STALK" in allowed):
                self.pred_state = "STALK"
                self.pred_state_timer = 0
            # Transition to AMBUSH if hungry and AMBUSH is allowed
            elif (self.pred_hunger > 0.5 and self.pred_state_timer > 60
                    and "AMBUSH" in allowed):
                cluster = self._find_food_cluster()
                if cluster is not None:
                    self.pred_ambush_target = cluster
                    self.pred_state = "AMBUSH"
                    self.pred_state_timer = 0

        elif self.pred_state == "STALK":
            # Transition to HUNT if close enough and HUNT is allowed
            if (dist < 80 and self.pred_stamina > 0.2
                    and "HUNT" in allowed):
                self.pred_state = "HUNT"
                self.pred_state_timer = 0
            # Lose interest if fish escapes far
            elif dist > self.pred_chase_radius * 1.3:
                self.pred_state = "PATROL"
                self.pred_state_timer = 0
            # Fall back to PATROL if STALK no longer allowed
            elif "STALK" not in allowed:
                self.pred_state = "PATROL"
                self.pred_state_timer = 0

        elif self.pred_state == "HUNT":
            # Active sprint — burns stamina proportional to chase effort
            _chase_drain = 0.005 * (1.0 + 0.5 * getattr(self, '_last_chase_boost', 0.0))
            self.pred_stamina = max(0.0, self.pred_stamina - _chase_drain)
            # Give up if exhausted or fish too far
            if self.pred_stamina < 0.03 or dist > self.pred_chase_radius:
                self.pred_state = "PATROL"
                self.pred_state_timer = 0
            # Transition to stalk if fish gets far but still visible
            elif dist > 120 and self.pred_stamina < 0.4:
                if "STALK" in allowed:
                    self.pred_state = "STALK"
                else:
                    self.pred_state = "PATROL"
                self.pred_state_timer = 0
            # Fall back to PATROL if HUNT no longer allowed
            elif "HUNT" not in allowed:
                self.pred_state = "PATROL"
                self.pred_state_timer = 0

        elif self.pred_state == "AMBUSH":
            # Wait near food cluster, switch to hunt if fish approaches
            if dist < 120 and "HUNT" in allowed:
                self.pred_state = "HUNT"
                self.pred_state_timer = 0
            # Give up ambush after timeout
            elif self.pred_state_timer > 200:
                self.pred_state = "PATROL"
                self.pred_state_timer = 0
            # Fall back to PATROL if AMBUSH no longer allowed
            elif "AMBUSH" not in allowed:
                self.pred_state = "PATROL"
                self.pred_state_timer = 0

        # --- Stamina recovery when not hunting ---
        if self.pred_state != "HUNT":
            self.pred_stamina = min(1.0, self.pred_stamina + 0.005)

        # --- Movement based on state ---
        turn_rate_max = 0.08
        speed = self.pred_speed * 0.5
        strategy_detail = ""

        if self.pred_state == "PATROL":
            # Systematic search: gentle turns with occasional direction change
            if self.pred_state_timer % 80 < 40:
                self.pred_heading += 0.02
            else:
                self.pred_heading -= 0.02
            self.pred_heading += self.np_random.uniform(-0.01, 0.01)
            speed = self.pred_speed * 0.55
            strategy_detail = "searching"

        elif self.pred_state == "STALK":
            # Approach slowly, turning toward fish
            angle_diff = angle_to_fish - self.pred_heading
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            self.pred_heading += np.clip(angle_diff, -0.04, 0.04)
            speed = self.pred_speed * 0.5
            strategy_detail = f"closing d={dist:.0f}"

        elif self.pred_state == "HUNT":
            # Predictive intercept: aim where the fish will be
            fish_vx = self.fish_speed * math.cos(self.fish_heading)
            fish_vy = self.fish_speed * math.sin(self.fish_heading)
            # Predict fish position ~10 steps ahead
            intercept_steps = min(10.0, dist / max(0.5, self.pred_speed))
            target_x = self.fish_x + fish_vx * intercept_steps
            target_y = self.fish_y + fish_vy * intercept_steps
            # Clamp to arena
            target_x = np.clip(target_x, 30, self.arena_w - 30)
            target_y = np.clip(target_y, 30, self.arena_h - 30)

            angle_to_target = math.atan2(target_y - self.pred_y,
                                          target_x - self.pred_x)
            angle_diff = angle_to_target - self.pred_heading
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            # Faster turning during hunt
            self.pred_heading += np.clip(angle_diff, -0.15, 0.15)
            # Sprint: faster when close, hunger-boosted, chase-accelerated
            hunger_boost = 1.0 + 0.2 * self.pred_hunger
            dist_factor = min(1.0, dist / 60.0 + 0.6)
            # Chase acceleration: up to 1.4x when locked on target
            aim_quality = max(0.0, 1.0 - abs(angle_diff) / (math.pi * 0.3))
            chase_boost = 1.0 + 0.25 * aim_quality
            speed = (self.pred_speed * dist_factor * hunger_boost
                     * self.pred_stamina * chase_boost)
            self._last_chase_boost = chase_boost
            strategy_detail = (f"intercept d={dist:.0f} "
                               f"stam={self.pred_stamina:.1f} "
                               f"chase={chase_boost:.2f}")

        elif self.pred_state == "AMBUSH":
            # Move toward food cluster and wait
            if self.pred_ambush_target is not None:
                ax, ay = self.pred_ambush_target
                adx = ax - self.pred_x
                ady = ay - self.pred_y
                ambush_dist = math.sqrt(adx * adx + ady * ady) + 1e-8
                if ambush_dist > 20:
                    angle_to_ambush = math.atan2(ady, adx)
                    angle_diff = angle_to_ambush - self.pred_heading
                    angle_diff = math.atan2(math.sin(angle_diff),
                                            math.cos(angle_diff))
                    self.pred_heading += np.clip(angle_diff, -0.06, 0.06)
                    speed = self.pred_speed * 0.4
                else:
                    # At ambush position — slow drift
                    self.pred_heading += self.np_random.uniform(-0.02, 0.02)
                    speed = self.pred_speed * 0.1
                strategy_detail = f"waiting d={ambush_dist:.0f}"
            else:
                self.pred_state = "PATROL"
                self.pred_state_timer = 0

        self.pred_heading = math.atan2(
            math.sin(self.pred_heading), math.cos(self.pred_heading))

        self.pred_x += speed * math.cos(self.pred_heading)
        self.pred_y += speed * math.sin(self.pred_heading)
        self.pred_speed_current = speed

        # --- Obstacle deflection ---
        pred_r = self.pred_size * 0.5
        for obs in self.obstacles:
            ddx = self.pred_x - obs["x"]
            ddy = self.pred_y - obs["y"]
            d = math.sqrt(ddx * ddx + ddy * ddy) + 1e-8
            min_d = obs["r"] + pred_r
            if d < min_d:
                nx = ddx / d
                ny = ddy / d
                self.pred_x = obs["x"] + nx * (min_d + 1.0)
                self.pred_y = obs["y"] + ny * (min_d + 1.0)
                self.pred_heading = math.atan2(ny, nx)

        # --- Wall bounce ---
        margin = 20
        if self.pred_x < margin:
            self.pred_x = margin
            self.pred_heading = math.pi - self.pred_heading
        if self.pred_x > self.arena_w - margin:
            self.pred_x = self.arena_w - margin
            self.pred_heading = math.pi - self.pred_heading
        if self.pred_y < margin:
            self.pred_y = margin
            self.pred_heading = -self.pred_heading
        if self.pred_y > self.arena_h - margin:
            self.pred_y = self.arena_h - margin
            self.pred_heading = -self.pred_heading

        # --- Populate diagnostics ---
        self.pred_diag = {
            "state": self.pred_state,
            "prev_state": prev_state,
            "hunger": self.pred_hunger,
            "stamina": self.pred_stamina,
            "dist_to_fish": dist,
            "speed": speed,
            "strategy": strategy_detail,
            "timer": self.pred_state_timer,
            "ambush_target": self.pred_ambush_target,
        }

    def _get_obs(self):
        """Construct normalized observation vector (17-dim)."""
        fx_n = 2 * self.fish_x / self.arena_w - 1
        fy_n = 2 * self.fish_y / self.arena_h - 1
        fh_n = self.fish_heading / math.pi
        fs_n = self.fish_speed / self.fish_speed_base

        dx = self.pred_x - self.fish_x
        dy = self.pred_y - self.fish_y
        pred_dist = math.sqrt(dx * dx + dy * dy)
        pred_angle = math.atan2(dy, dx) - self.fish_heading
        pred_angle = math.atan2(math.sin(pred_angle), math.cos(pred_angle))

        px_n = 2 * self.pred_x / self.arena_w - 1
        py_n = 2 * self.pred_y / self.arena_h - 1
        ph_n = self.pred_heading / math.pi
        pd_n = np.clip(pred_dist / (self.arena_w * 0.5), 0, 1) * 2 - 1
        pa_n = pred_angle / math.pi

        if len(self.foods) > 0:
            dists = []
            for food in self.foods:
                ffx, ffy = food[0], food[1]
                d = math.sqrt((ffx - self.fish_x) ** 2
                              + (ffy - self.fish_y) ** 2)
                dists.append(d)
            idx = int(np.argmin(dists))
            nf_dist = dists[idx]
            nf_angle = math.atan2(
                self.foods[idx][1] - self.fish_y,
                self.foods[idx][0] - self.fish_x) - self.fish_heading
            nf_angle = math.atan2(math.sin(nf_angle), math.cos(nf_angle))
            nfd_n = np.clip(nf_dist / 200, 0, 1) * 2 - 1
            nfa_n = nf_angle / math.pi
        else:
            nfd_n = 1.0
            nfa_n = 0.0

        wall_n = np.clip(self.fish_y / 100, 0, 1) * 2 - 1
        wall_s = np.clip((self.arena_h - self.fish_y) / 100, 0, 1) * 2 - 1
        wall_e = np.clip((self.arena_w - self.fish_x) / 100, 0, 1) * 2 - 1
        wall_w = np.clip(self.fish_x / 100, 0, 1) * 2 - 1

        eaten_n = np.clip(self.total_eaten / 20, 0, 1) * 2 - 1

        # Energy normalized to [-1, 1] (Feature A)
        energy_n = 2 * (self.fish_energy / self.energy_max) - 1

        obs = np.array([
            fx_n, fy_n, fh_n, fs_n,
            px_n, py_n, ph_n, pd_n, pa_n,
            nfd_n, nfa_n,
            wall_n, wall_s, wall_e, wall_w,
            eaten_n,
            energy_n,
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        return {
            "fish_pos": (self.fish_x, self.fish_y),
            "pred_pos": (self.pred_x, self.pred_y),
            "total_eaten": self.total_eaten,
            "food_eaten_this_step": self.food_eaten_this_step,
            "alive": self.alive,
            "step": self.step_count,
            "n_food": len(self.foods),
            "fish_energy": self.fish_energy,
            "obstacles": self.obstacles,
            "fish_heading": self.fish_heading,
            "plankton_patches": [
                {"cx": p["cx"], "cy": p["cy"], "label": p["label"],
                 "weight": p["weight"]}
                for p in getattr(self, 'plankton_patches', [])
            ],
            "rock_formations": [
                {"cx": r["cx"], "cy": r["cy"], "base_r": r["base_r"]}
                for r in getattr(self, 'rock_formations', [])
            ],
            "colleagues": getattr(self, 'colleagues', []),
        }

    def render(self):
        """Render using pygame.

        Keyboard controls (human mode):
          UP / +   : increase simulation speed (2x, 4x, 8x)
          DOWN / - : decrease simulation speed
          Q        : quit
        """
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. "
                              "Install with: pip install pygame")

        if self._screen is None:
            pygame.init()
            rw, rh = self.render_width, self.render_height
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((rw, rh))
                pygame.display.set_caption("Zebrafish Predator-Prey")
            else:
                self._screen = pygame.Surface((rw, rh))
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont(None, 18)
            self._font_big = pygame.font.SysFont(None, 48)

        main_surf = self._screen

        # Side-panel mode: dark background for margins, arena is a subsurface
        if self._side_panels:
            main_surf.fill((25, 28, 35))
            surface = main_surf.subsurface(
                (self._panel_l, 0, self.arena_w, self.arena_h))
        else:
            surface = main_surf

        # Background — circadian-modulated water color (Step 40)
        # Day: light blue (220, 232, 240), Night: dark blue (40, 60, 90)
        _circ_activity = getattr(self, '_circadian_activity', 1.0)
        _bg_r = int(40 + 180 * _circ_activity)
        _bg_g = int(60 + 172 * _circ_activity)
        _bg_b = int(90 + 150 * _circ_activity)
        surface.fill((_bg_r, _bg_g, _bg_b))

        # Draw rock formations — organic multi-layer rendering
        for rock in getattr(self, 'rock_formations', []):
            poly = rock["polygon"]
            if len(poly) < 3:
                continue
            int_poly = [(int(px), int(py)) for px, py in poly]

            # Shadow (offset, dark)
            shadow_poly = [(px + 4, py + 4) for px, py in int_poly]
            pygame.draw.polygon(surface, (60, 40, 20), shadow_poly)

            # Base fill (brown)
            pygame.draw.polygon(surface, (139, 100, 55), int_poly)

            # Highlight (shrunk lighter polygon)
            cx, cy = int(rock["cx"]), int(rock["cy"])
            highlight_poly = [
                (int(cx + (px - cx) * 0.7), int(cy + (py - cy) * 0.7))
                for px, py in int_poly
            ]
            pygame.draw.polygon(surface, (170, 130, 80), highlight_poly)

            # Craggy outline
            pygame.draw.polygon(surface, (80, 50, 20), int_poly, 2)

        # Draw plankton patch glow (subtle teal circles — cosmetic only)
        for patch in getattr(self, 'plankton_patches', []):
            if patch["label"] == "background":
                continue
            pcx, pcy = int(patch["cx"]), int(patch["cy"])
            pr = int(patch["radius"])
            glow_surf = pygame.Surface((pr * 2, pr * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (0, 180, 160, 25),
                               (pr, pr), pr)
            surface.blit(glow_surf, (pcx - pr, pcy - pr))

        # Draw food (similar size, density indicates value)
        for food in self.foods:
            fx, fy = food[0], food[1]
            sz = food[2] if len(food) > 2 else "small"
            r = 4 if sz == "large" else 3
            color = (0, 180, 20)
            outline = (0, 110, 10)
            pygame.draw.circle(surface, color, (int(fx), int(fy)), r)
            pygame.draw.circle(surface, outline, (int(fx), int(fy)), r, 1)

        # Draw receptive field cones (before fish so they appear behind)
        if self.alive:
            self._draw_receptive_fields(surface)

        # Draw zebrafish (blue, base=head with eyes, point=tail)
        if self.alive:
            # Vestibular tilt effect (Step 37): slight size oscillation
            _vest_balance = getattr(self, '_vest_balance', 1.0)
            _fish_render_size = self.fish_size * (0.9 + 0.1 * _vest_balance)

            self._draw_zebrafish(
                surface, self.fish_x, self.fish_y, self.fish_heading,
                _fish_render_size, (30, 60, 220))

            # Proprioceptive collision flash (Step 41)
            _collision = getattr(self, '_prop_collision', 0.0)
            if _collision > 0.2:
                flash_r = int(self.fish_size * 2 * _collision)
                flash_surf = pygame.Surface((flash_r * 2, flash_r * 2),
                                            pygame.SRCALPHA)
                alpha = int(min(120, _collision * 200))
                pygame.draw.circle(flash_surf, (255, 100, 50, alpha),
                                   (flash_r, flash_r), flash_r)
                surface.blit(flash_surf,
                             (int(self.fish_x - flash_r),
                              int(self.fish_y - flash_r)))

        # Draw colleagues (teal triangles, Step 20)
        for c in getattr(self, 'colleagues', []):
            self._draw_zebrafish(
                surface, c["x"], c["y"], c["heading"],
                self.fish_size * 0.85, (0, 180, 170))

        # Draw predator (red, base=head with mouth, point=tail)
        self._draw_predator(
            surface, self.pred_x, self.pred_y, self.pred_heading,
            self.pred_size, (200, 50, 50))

        # Death splash
        if self._death_timer > 0:
            self._draw_death_splash(surface)
            self._death_timer -= 1

        # Food prospect highlights (rings around evaluated foods)
        self._draw_food_prospects(surface)

        # Vision strip overlay (Feature C) — with L/R labels
        self._draw_vision_strip(surface)

        # Brain monitoring panel + Predator panel
        if self._side_panels:
            # Panels in side margins (on main_surf, outside arena)
            brain_x = self._panel_l + self.arena_w + 3
            self._draw_brain_monitor(main_surf, panel_x=brain_x, panel_y=10)
            self._draw_predator_monitor(
                main_surf, panel_x=3, panel_y=10, arena_surface=surface)
        else:
            # Overlay panels on the arena (legacy)
            self._draw_brain_monitor(surface)
            self._draw_predator_monitor(surface)

        # HUD — bottom panel
        self._draw_hud(surface)

        if self.render_mode == "human":
            pygame.display.flip()
            target_fps = self.metadata["render_fps"] * self._sim_speed
            self._clock.tick(target_fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.close()
                    elif event.key in (pygame.K_UP, pygame.K_PLUS,
                                       pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self._sim_speed = min(16, self._sim_speed * 2)
                    elif event.key in (pygame.K_DOWN, pygame.K_MINUS,
                                       pygame.K_KP_MINUS):
                        self._sim_speed = max(1, self._sim_speed // 2)
                    elif event.key == pygame.K_p:
                        # Send predator charging toward fish
                        self._command_predator_attack()
                    elif event.key == pygame.K_r:
                        # Retreat predator to corner
                        self._command_predator_retreat()
                    elif event.key == pygame.K_f:
                        # Spawn food cluster near fish
                        self._command_spawn_food_cluster()
        else:
            # Return full surface (including side panels if enabled)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(main_surf)),
                axes=(1, 0, 2))

    def _draw_hud(self, surface):
        """Draw HUD: energy bar, stats, speed indicator."""
        import pygame

        y_base = self.arena_h - 40

        # Energy bar (wide, bottom of screen)
        bar_x, bar_y = 10, y_base
        bar_w, bar_h = 200, 12
        ratio = self.fish_energy / self.energy_max

        if ratio > 0.5:
            color = (40, 180, 40)
        elif ratio > 0.2:
            color = (220, 160, 30)
        else:
            color = (200, 40, 40)

        pygame.draw.rect(surface, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * ratio)
        if fill_w > 0:
            pygame.draw.rect(surface, color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(surface, (30, 30, 30), (bar_x, bar_y, bar_w, bar_h), 1)

        # Energy text
        energy_text = self._font.render(
            f"Energy: {self.fish_energy:.0f}/{self.energy_max:.0f}  "
            f"(-{self.energy_drain_base + self.energy_drain_speed:.2f}/step)",
            True, (40, 40, 40))
        surface.blit(energy_text, (bar_x, bar_y - 16))

        # Stats text (right side)
        stats = self._font.render(
            f"Food: {self.total_eaten}  Step: {self.step_count}  "
            f"Speed: {self._sim_speed}x  [UP/DOWN to change]",
            True, (40, 40, 40))
        surface.blit(stats, (bar_x + bar_w + 20, bar_y - 2))

        # Alive status
        if not self.alive:
            dead_text = self._font.render("DEAD", True, (200, 30, 30))
            surface.blit(dead_text, (bar_x + bar_w + 20, bar_y - 18))

    def _draw_vision_strip(self, surface):
        """Draw 1D vision strip at top with L/R eye labels.

        Color = object type, Brightness = distance (closer = brighter).
        """
        import pygame

        if self.vision_strip_L is None or self.vision_strip_R is None:
            return

        strip_h = 30
        label_h = 14
        y_top = 0
        retL = self.vision_strip_L
        retR = self.vision_strip_R
        typeL = self.vision_type_L
        typeR = self.vision_type_R
        n_pixels = len(retL)

        half_w = self.arena_w // 2
        px_w = max(1, half_w // n_pixels)

        # Distance-based brightness: intensity encodes proximity
        # (higher retinal activation = closer object = brighter color)
        def type_to_color(t_val, intensity):
            # Intensity 0→far(dark), 1→close(bright)
            # Minimum brightness floor so distant entities remain visible
            floor = 0.18 if t_val > 0.05 else 0.05
            b = max(floor, min(1.0, intensity))
            if t_val > 0.8:      # food (1.0) — green
                return (int(20 * b), int(220 * b), int(20 * b))
            elif t_val > 0.6:    # obstacle (0.75) — brown/orange
                return (int(180 * b), int(100 * b), int(30 * b))
            elif t_val > 0.4:    # enemy (0.5) — red
                return (int(220 * b), int(30 * b), int(30 * b))
            elif t_val > 0.2:    # colleague — cyan
                return (int(20 * b), int(160 * b), int(220 * b))
            elif t_val > 0.05:   # boundary — gray
                return (int(140 * b), int(140 * b), int(140 * b))
            else:                # nothing — dark
                return (int(25 * b), int(25 * b), int(30 * b))

        # Background strip
        pygame.draw.rect(surface, (10, 10, 15),
                         (0, y_top, self.arena_w, strip_h + label_h))

        # Left eye pixels
        for i in range(n_pixels):
            x = i * px_w
            c = type_to_color(typeL[i] if typeL is not None else 0.0,
                              retL[i])
            pygame.draw.rect(surface, c,
                             (x, y_top + label_h, px_w, strip_h))

        # Right eye pixels
        for i in range(n_pixels):
            x = half_w + i * px_w
            c = type_to_color(typeR[i] if typeR is not None else 0.0,
                              retR[i])
            pygame.draw.rect(surface, c,
                             (x, y_top + label_h, px_w, strip_h))

        # Divider
        pygame.draw.line(surface, (255, 255, 255),
                         (half_w, y_top), (half_w, y_top + strip_h + label_h), 2)

        # Labels
        lbl_l = self._font.render("LEFT EYE", True, (180, 180, 200))
        lbl_r = self._font.render("RIGHT EYE", True, (180, 180, 200))
        surface.blit(lbl_l, (half_w // 2 - lbl_l.get_width() // 2, y_top))
        surface.blit(lbl_r,
                     (half_w + half_w // 2 - lbl_r.get_width() // 2, y_top))

        # Legend (small, bottom-right of strip)
        leg = self._font.render(
            "bright=close dark=far | green=food red=enemy brown=obstacle gray=wall",
            True, (140, 140, 160))
        surface.blit(leg, (4, y_top + label_h + strip_h + 1))

    def _draw_food_prospects(self, surface):
        """Highlight evaluated food targets with colored rings."""
        import pygame

        prospects = getattr(self, '_food_prospects', [])
        if not prospects:
            return

        for rank, p in enumerate(prospects[:3]):
            fx, fy = p["gym_pos"]
            risk = p["risk"]
            score = p["prospect_score"]

            # Color: green=good prospect, red=risky, yellow=moderate
            if rank == 0:
                ring_color = (0, 255, 100)  # best target — bright green
            elif risk > 0.5:
                ring_color = (255, 80, 80)  # dangerous
            else:
                ring_color = (255, 220, 50)  # moderate

            radius = 10 + rank * 3
            pygame.draw.circle(surface, ring_color,
                               (int(fx), int(fy)), radius, 2)

            # Rank number
            rank_txt = self._font.render(str(rank + 1), True, ring_color)
            surface.blit(rank_txt, (int(fx) + radius + 2, int(fy) - 6))

    def _draw_brain_monitor(self, surface, panel_x=None, panel_y=None):
        """Draw comprehensive brain state monitoring panel."""
        import pygame

        diag = getattr(self, '_brain_diag', {})
        if not diag:
            return

        # Panel position and dimensions
        panel_w = 195
        if panel_x is None:
            panel_x = self.arena_w - panel_w - 5
        if panel_y is None:
            panel_y = 52  # below vision strip
        line_h = 14
        y = panel_y

        # Semi-transparent background
        bg_h = 360 if self._side_panels else 310
        bg_surf = pygame.Surface((panel_w, bg_h), pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 160))
        surface.blit(bg_surf, (panel_x, panel_y))

        font = self._font

        # --- Goal state ---
        goal_names = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
        goal_colors = [(40, 200, 40), (220, 40, 40), (80, 140, 255),
                       (0, 180, 170)]
        goal = diag.get("goal", 2)
        goal_name = (goal_names[goal] if goal < len(goal_names)
                     else "?")
        goal_color = (goal_colors[goal] if goal < len(goal_colors)
                      else (180, 180, 180))

        # Large goal label
        goal_txt = font.render(f"GOAL: {goal_name}", True, goal_color)
        surface.blit(goal_txt, (panel_x + 4, y))
        y += line_h

        # Shortcut indicator
        sc = diag.get("shortcut_active", False)
        mode_txt = "HABIT" if sc else "DELIBERATE"
        mode_color = (255, 200, 50) if sc else (160, 160, 180)
        surface.blit(font.render(f"Mode: {mode_txt}", True, mode_color),
                     (panel_x + 4, y))
        y += line_h

        # Confidence bar
        conf = diag.get("confidence", 0.0)
        surface.blit(font.render(f"Conf: {conf:.2f}", True, (180, 180, 200)),
                     (panel_x + 4, y))
        bar_x = panel_x + 80
        bar_w = panel_w - 85
        pygame.draw.rect(surface, (50, 50, 50), (bar_x, y + 2, bar_w, 8))
        fill = int(bar_w * conf)
        if fill > 0:
            pygame.draw.rect(surface, goal_color, (bar_x, y + 2, fill, 8))
        y += line_h + 2

        # --- EFE values ---
        surface.blit(font.render("-- EFE (lower=preferred) --", True,
                                 (140, 140, 160)), (panel_x + 4, y))
        y += line_h
        efe = diag.get("efe_vec", [0, 0, 0])
        for i, (name, color) in enumerate(zip(goal_names, goal_colors)):
            val = efe[i] if i < len(efe) else 0
            marker = ">" if i == goal else " "
            txt = font.render(f"{marker}{name:8s} {val:+.3f}", True, color)
            surface.blit(txt, (panel_x + 4, y))
            y += line_h
        y += 2

        # --- Classification ---
        surface.blit(font.render("-- Classification --", True,
                                 (140, 140, 160)), (panel_x + 4, y))
        y += line_h
        cls_probs = diag.get("cls_probs", [])
        cls_names = ["none", "food", "enemy", "coll", "env"]
        cls_colors = [(100, 100, 100), (40, 200, 40), (220, 40, 40),
                      (40, 160, 220), (140, 140, 140)]
        for i, (name, color) in enumerate(zip(cls_names, cls_colors)):
            if i < len(cls_probs):
                p = cls_probs[i]
                bar_fill = int((panel_w - 65) * min(1.0, p))
                surface.blit(font.render(f"{name:5s}", True, color),
                             (panel_x + 4, y))
                bx = panel_x + 45
                pygame.draw.rect(surface, (40, 40, 40),
                                 (bx, y + 2, panel_w - 65, 8))
                if bar_fill > 0:
                    pygame.draw.rect(surface, color,
                                     (bx, y + 2, bar_fill, 8))
                surface.blit(font.render(f"{p:.2f}", True, (180, 180, 180)),
                             (bx + panel_w - 62, y))
                y += line_h
        y += 2

        # --- Signals ---
        surface.blit(font.render("-- Signals --", True,
                                 (140, 140, 160)), (panel_x + 4, y))
        y += line_h
        dopa = diag.get("dopa", 0)
        rpe = diag.get("rpe", 0)
        fe = diag.get("F_visual", 0)
        surface.blit(font.render(f"DA:{dopa:.2f} RPE:{rpe:+.2f} FE:{fe:.2f}",
                                 True, (180, 180, 200)), (panel_x + 4, y))
        y += line_h

        eye = diag.get("eye_pos", 0)
        bg = diag.get("bg_gate", 0)
        spd = diag.get("speed", 0)
        surface.blit(font.render(f"Eye:{eye:+.2f} BG:{bg:+.2f} Spd:{spd:.2f}",
                                 True, (180, 180, 200)), (panel_x + 4, y))
        y += line_h

        # Saccade/flee indicators
        sac = diag.get("saccade_active", False)
        flee = diag.get("flee_burst", 0)
        if sac or flee > 0:
            alert_txt = []
            if sac:
                alert_txt.append("SACCADE")
            if flee > 0:
                alert_txt.append(f"FLEE-BURST({flee})")
            surface.blit(font.render(" ".join(alert_txt), True,
                                     (255, 100, 50)), (panel_x + 4, y))
        y += line_h + 2

        # --- Food prospects (top 3) ---
        prospects = diag.get("food_prospects", [])
        if prospects:
            surface.blit(font.render("-- Food Prospects --", True,
                                     (140, 140, 160)), (panel_x + 4, y))
            y += line_h
            surface.blit(font.render("# dist  cost  risk  score", True,
                                     (120, 120, 140)), (panel_x + 4, y))
            y += line_h
            for rank, p in enumerate(prospects[:3]):
                dist = p["dist"]
                cost = p["metabolic_cost"]
                risk = p["risk"]
                score = p["prospect_score"]
                risk_color = (220, 40, 40) if risk > 0.5 else (180, 180, 180)
                txt = f"{rank+1} {dist:5.0f} {cost:5.1f} {risk:.2f} {score:+.2f}"
                surface.blit(font.render(txt, True, risk_color),
                             (panel_x + 4, y))
                y += line_h

    def _draw_predator_monitor(self, surface, panel_x=None, panel_y=None,
                               arena_surface=None):
        """Draw predator state monitoring panel.

        Args:
            surface: surface to draw the panel text/bars on.
            panel_x, panel_y: override position (used for side-panel mode).
            arena_surface: if given, arena overlays (ambush circle, hunt line)
                are drawn here instead of on *surface*.
        """
        import pygame

        diag = getattr(self, 'pred_diag', {})
        if not diag:
            return

        panel_w = 175
        if panel_x is None:
            panel_x = 5
        if panel_y is None:
            panel_y = 52
        line_h = 14
        y = panel_y

        # Semi-transparent background
        bg_surf = pygame.Surface((panel_w, 210), pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 160))
        surface.blit(bg_surf, (panel_x, panel_y))

        font = self._font

        # --- Title ---
        surface.blit(font.render("PREDATOR", True, (220, 80, 80)),
                     (panel_x + 4, y))
        y += line_h

        # --- State ---
        state = diag.get("state", "?")
        state_colors = {
            "PATROL": (140, 140, 200),
            "STALK": (220, 180, 50),
            "HUNT": (255, 60, 60),
            "AMBUSH": (180, 100, 220),
            "REST": (100, 200, 100),
        }
        sc = state_colors.get(state, (180, 180, 180))
        surface.blit(font.render(f"State: {state}", True, sc),
                     (panel_x + 4, y))
        y += line_h

        # Strategy detail
        strat = diag.get("strategy", "")
        surface.blit(font.render(strat, True, (160, 160, 180)),
                     (panel_x + 4, y))
        y += line_h + 2

        # --- Hunger bar ---
        hunger = diag.get("hunger", 0)
        surface.blit(font.render(f"Hunger:", True, (180, 180, 200)),
                     (panel_x + 4, y))
        bx = panel_x + 60
        bw = panel_w - 65
        pygame.draw.rect(surface, (40, 40, 40), (bx, y + 2, bw, 8))
        hw = int(bw * hunger)
        if hw > 0:
            hc = (220, 60, 40) if hunger > 0.7 else (220, 160, 40)
            pygame.draw.rect(surface, hc, (bx, y + 2, hw, 8))
        surface.blit(font.render(f"{hunger:.2f}", True, (160, 160, 160)),
                     (bx + bw + 3, y))
        y += line_h

        # --- Stamina bar ---
        stamina = diag.get("stamina", 1.0)
        surface.blit(font.render(f"Stamina:", True, (180, 180, 200)),
                     (panel_x + 4, y))
        pygame.draw.rect(surface, (40, 40, 40), (bx, y + 2, bw, 8))
        sw = int(bw * stamina)
        if sw > 0:
            stc = (40, 180, 40) if stamina > 0.5 else (220, 160, 40)
            pygame.draw.rect(surface, stc, (bx, y + 2, sw, 8))
        surface.blit(font.render(f"{stamina:.2f}", True, (160, 160, 160)),
                     (bx + bw + 3, y))
        y += line_h + 2

        # --- Distance to prey ---
        dist = diag.get("dist_to_fish", 0)
        speed = diag.get("speed", 0)
        surface.blit(font.render(f"Prey dist: {dist:.0f}", True,
                                 (180, 180, 200)), (panel_x + 4, y))
        y += line_h
        surface.blit(font.render(f"Speed: {speed:.2f}", True,
                                 (180, 180, 200)), (panel_x + 4, y))
        y += line_h

        # --- Timer ---
        timer = diag.get("timer", 0)
        surface.blit(font.render(f"State timer: {timer}", True,
                                 (140, 140, 160)), (panel_x + 4, y))
        y += line_h

        # --- Arena overlays (ambush circle, hunt line) ---
        overlay = arena_surface if arena_surface is not None else surface

        ambush = diag.get("ambush_target", None)
        if ambush is not None and state == "AMBUSH":
            surface.blit(font.render(
                f"Ambush: ({ambush[0]:.0f},{ambush[1]:.0f})", True,
                (180, 100, 220)), (panel_x + 4, y))
            y += line_h

            ax, ay = int(ambush[0]), int(ambush[1])
            pygame.draw.circle(overlay, (180, 100, 220), (ax, ay), 25, 2)
            pygame.draw.circle(overlay, (180, 100, 220), (ax, ay), 4)

        if state == "HUNT":
            px, py = int(self.pred_x), int(self.pred_y)
            gaze_len = 40
            gx = px + int(gaze_len * math.cos(self.pred_heading))
            gy = py + int(gaze_len * math.sin(self.pred_heading))
            pygame.draw.line(overlay, (255, 80, 80),
                             (px, py), (gx, gy), 2)

    def _draw_receptive_fields(self, surface):
        """Draw semi-transparent receptive field cones for L and R eyes."""
        import pygame

        cone_len = 60  # length of the cone lines
        fov_half = 0.8  # half field-of-view in radians (~45 degrees per eye)
        x, y, h = self.fish_x, self.fish_y, self.fish_heading

        # In gym coords (y-down): left eye = heading - offset, right = heading + offset
        eye_offset = 0.55
        for eye_dir, label, color in [
            (h - eye_offset, "L", (100, 100, 255, 40)),
            (h + eye_offset, "R", (255, 100, 100, 40)),
        ]:
            p1 = (int(x), int(y))
            p2 = (int(x + cone_len * math.cos(eye_dir + fov_half)),
                  int(y + cone_len * math.sin(eye_dir + fov_half)))
            p3 = (int(x + cone_len * math.cos(eye_dir - fov_half)),
                  int(y + cone_len * math.sin(eye_dir - fov_half)))

            # Semi-transparent cone
            cone_surf = pygame.Surface((self.arena_w, self.arena_h),
                                       pygame.SRCALPHA)
            pygame.draw.polygon(cone_surf, color, [p1, p2, p3])
            surface.blit(cone_surf, (0, 0))

            # Cone outline
            pygame.draw.lines(surface, color[:3], False, [p2, p1, p3], 1)

    def _draw_death_splash(self, surface):
        """Draw death splash effect (red burst + KILLED text)."""
        import pygame

        if self._death_pos is None:
            return

        dx, dy = int(self._death_pos[0]), int(self._death_pos[1])
        t = self._death_timer  # 60→0

        # Expanding red circles
        alpha = min(255, t * 6)
        for r_mult in [1.0, 1.8, 2.6]:
            radius = int((60 - t) * r_mult + 5)
            splash_surf = pygame.Surface((self.arena_w, self.arena_h),
                                         pygame.SRCALPHA)
            pygame.draw.circle(splash_surf,
                               (255, 50, 30, min(alpha, int(180 / r_mult))),
                               (dx, dy), radius)
            surface.blit(splash_surf, (0, 0))

        # "KILLED" text
        if t > 20:
            killed = self._font_big.render("KILLED!", True, (220, 20, 20))
            tx = dx - killed.get_width() // 2
            ty = dy - 50
            surface.blit(killed, (tx, ty))

    def _draw_zebrafish(self, surface, x, y, heading, size, color):
        """Draw zebrafish larva: smooth segmented body with undulating tail.

        10 connected elliptical segments taper from head to tail.
        Muscle-driven oscillation bends the body like a real larva.
        """
        import pygame

        n_seg = 10
        seg_len = size * 0.35
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        perp_x = -sin_h
        perp_y = cos_h

        # Body undulation driven by actual turn rate + swim speed
        # Turn rate determines sustained body curve direction
        # Speed determines tail-beat amplitude
        _turn_rate = getattr(self, '_last_turn_rate', 0.0)
        speed_ratio = self.fish_speed / max(self.fish_speed_base, 0.01)
        phase = self.step_count * 0.6
        amplitude = size * 0.25 * max(speed_ratio, 0.15)
        # Body curve from turning: positive turn → curve right
        muscle_diff = _turn_rate * 0.5

        # Build segment centres along the body (head=0, tail=n_seg-1)
        seg_cx = []
        seg_cy = []
        seg_widths = []
        cumulative_angle = heading

        px, py = x, y
        for i in range(n_seg):
            t = i / (n_seg - 1)  # 0 at head, 1 at tail

            # Body width: widest at head (0.7*size), tapers to tail (0.15*size)
            width = size * (0.7 - 0.55 * t)
            seg_widths.append(width)

            # Undulation: traveling wave from head to tail
            wave = amplitude * t * math.sin(phase + i * 0.9)
            # Muscle bias: asymmetric bend
            bias = muscle_diff * size * 0.3 * t

            # Position: advance along body axis + lateral wave
            if i > 0:
                px -= seg_len * math.cos(cumulative_angle)
                py -= seg_len * math.sin(cumulative_angle)
                # Bend the body axis
                bend = (wave + bias) * 0.08
                cumulative_angle += bend

            seg_cx.append(px + perp_x * (wave + bias) * 0.3)
            seg_cy.append(py + perp_y * (wave + bias) * 0.3)

        # Draw body segments (back to front for proper layering)
        for i in range(n_seg - 1, -1, -1):
            t = i / (n_seg - 1)
            w = int(seg_widths[i])
            cx, cy = int(seg_cx[i]), int(seg_cy[i])

            # Body color: darker stripes (zebrafish pattern)
            if i % 3 == 0:
                seg_color = (max(0, color[0] - 30),
                             max(0, color[1] - 20),
                             min(255, color[2] + 20))
            else:
                seg_color = color

            # Tail fin: last 2 segments are transparent/thin
            if i >= n_seg - 2:
                fin_color = (min(255, color[0] + 60),
                             min(255, color[1] + 80),
                             min(255, color[2] + 30))
                if w > 2:
                    pygame.draw.ellipse(surface, fin_color,
                                        (cx - w, cy - w // 2, w * 2, w))
            else:
                pygame.draw.ellipse(surface, seg_color,
                                    (cx - w // 2, cy - w // 2, w, w))

        # Draw body outline (smooth curve through segment centres)
        if len(seg_cx) > 2:
            # Left edge
            left_pts = []
            right_pts = []
            for i in range(n_seg):
                cx, cy = seg_cx[i], seg_cy[i]
                w = seg_widths[i] * 0.5
                # Perpendicular to body axis at this segment
                if i < n_seg - 1:
                    dx = seg_cx[i + 1] - seg_cx[i]
                    dy = seg_cy[i + 1] - seg_cy[i]
                else:
                    dx = seg_cx[i] - seg_cx[i - 1]
                    dy = seg_cy[i] - seg_cy[i - 1]
                d = math.sqrt(dx * dx + dy * dy) + 1e-8
                nx, ny = -dy / d, dx / d
                left_pts.append((int(cx + nx * w), int(cy + ny * w)))
                right_pts.append((int(cx - nx * w), int(cy - ny * w)))

            outline = left_pts + list(reversed(right_pts))
            if len(outline) > 2:
                pygame.draw.polygon(surface, color, outline)
                pygame.draw.polygon(surface, (20, 40, 120), outline, 1)

        # Eyes — anchored to head segment, lateral to body axis
        head_x = seg_cx[0]
        head_y = seg_cy[0]
        eye_r = max(2, int(size * 0.3))
        # Eye position: slightly forward and to the sides of the head
        eye_fwd = size * 0.3
        eye_lat = size * 0.4
        eye_l_x = head_x + eye_fwd * cos_h - eye_lat * sin_h
        eye_l_y = head_y + eye_fwd * sin_h + eye_lat * cos_h
        eye_r_x = head_x + eye_fwd * cos_h + eye_lat * sin_h
        eye_r_y = head_y + eye_fwd * sin_h - eye_lat * cos_h

        # Zebrafish larva eyes: large dark lens with thin silver iris
        # Dark lens (fills most of the eye)
        pygame.draw.circle(surface, (15, 15, 25),
                           (int(eye_l_x), int(eye_l_y)), eye_r)
        pygame.draw.circle(surface, (15, 15, 25),
                           (int(eye_r_x), int(eye_r_y)), eye_r)
        # Silver-gold iris ring
        pygame.draw.circle(surface, (180, 170, 120),
                           (int(eye_l_x), int(eye_l_y)), eye_r, 1)
        pygame.draw.circle(surface, (180, 170, 120),
                           (int(eye_r_x), int(eye_r_y)), eye_r, 1)
        # Bright reflection spot (forward of centre)
        ref_r = max(1, eye_r // 3)
        ref_off = eye_r * 0.25
        pygame.draw.circle(surface, (220, 230, 255),
                           (int(eye_l_x + ref_off * cos_h),
                            int(eye_l_y + ref_off * sin_h)), ref_r)
        pygame.draw.circle(surface, (220, 230, 255),
                           (int(eye_r_x + ref_off * cos_h),
                            int(eye_r_y + ref_off * sin_h)), ref_r)

    def _draw_predator(self, surface, x, y, heading, size, color):
        """Draw predator: isosceles triangle with base=head, eyes, mouth, vertex=tail."""
        import pygame

        spread = 0.55

        # Base corners (head)
        base_l_x = x + size * math.cos(heading + spread)
        base_l_y = y + size * math.sin(heading + spread)
        base_r_x = x + size * math.cos(heading - spread)
        base_r_y = y + size * math.sin(heading - spread)

        # Tail vertex
        tail_x = x - size * 1.5 * math.cos(heading)
        tail_y = y - size * 1.5 * math.sin(heading)

        points = [(base_l_x, base_l_y), (base_r_x, base_r_y),
                  (tail_x, tail_y)]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (120, 20, 20), points, 2)

        # Eyes — bigger than zebrafish (predator is 1.5x)
        eye_r = max(3, int(size * 0.3))
        eye_off = 0.3
        eye_l_x = base_l_x + size * eye_off * math.cos(heading)
        eye_l_y = base_l_y + size * eye_off * math.sin(heading)
        eye_r_x = base_r_x + size * eye_off * math.cos(heading)
        eye_r_y = base_r_y + size * eye_off * math.sin(heading)

        # Yellow sclera, red iris for menacing look
        pygame.draw.circle(surface, (255, 220, 50),
                           (int(eye_l_x), int(eye_l_y)), eye_r)
        pygame.draw.circle(surface, (0, 0, 0),
                           (int(eye_l_x), int(eye_l_y)), eye_r, 1)
        pygame.draw.circle(surface, (180, 20, 20),
                           (int(eye_l_x), int(eye_l_y)), max(1, eye_r // 2))

        pygame.draw.circle(surface, (255, 220, 50),
                           (int(eye_r_x), int(eye_r_y)), eye_r)
        pygame.draw.circle(surface, (0, 0, 0),
                           (int(eye_r_x), int(eye_r_y)), eye_r, 1)
        pygame.draw.circle(surface, (180, 20, 20),
                           (int(eye_r_x), int(eye_r_y)), max(1, eye_r // 2))

        # Mouth — V-shape on the base center
        base_cx = (base_l_x + base_r_x) / 2
        base_cy = (base_l_y + base_r_y) / 2
        mouth_w = size * 0.35
        mouth_l_x = base_cx + mouth_w * math.cos(heading + 1.57)
        mouth_l_y = base_cy + mouth_w * math.sin(heading + 1.57)
        mouth_r_x = base_cx + mouth_w * math.cos(heading - 1.57)
        mouth_r_y = base_cy + mouth_w * math.sin(heading - 1.57)
        mouth_inner_x = base_cx - size * 0.2 * math.cos(heading)
        mouth_inner_y = base_cy - size * 0.2 * math.sin(heading)
        pygame.draw.lines(surface, (60, 0, 0), False, [
            (int(mouth_l_x), int(mouth_l_y)),
            (int(mouth_inner_x), int(mouth_inner_y)),
            (int(mouth_r_x), int(mouth_r_y)),
        ], 2)

    def close(self):
        if self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._screen = None
