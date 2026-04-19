"""
Behavioral assay arenas for ZebrafishBrainV2.

Each assay is a recorder/context that wraps an existing env.
Use record(fish_x, fish_y, fish_heading, speed, goal) each step,
then call compute_metrics() to get summary statistics.

Assays:
  NovelTankTest       — thigmotaxis / anxiety (portrait arena)
  LightDarkTest       — photophobia preference (landscape arena)
  SocialPreferenceTest — 3-chamber social/conspecific preference
  OpenFieldTest       — exploration vs anxiety (square arena)

Runner:
  run_assay(brain, assay, ...) — headless episode runner
"""
import os
import sys
import math
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav2.brain.sensory_bridge import inject_sensory


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BehavioralAssay:
    """Base recorder for behavioral assay data.

    Usage pattern::

        env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=0)
        assay = NovelTankTest()
        assay.setup_env(env)
        obs, _ = env.reset(seed=42)
        for _ in range(500):
            inject_sensory(env)
            out = brain.step(obs, env)
            obs, *_ = env.step([out['turn'], out['speed']])
            assay.record(env.fish_x, env.fish_y, env.fish_heading,
                         out['speed'], out['goal'])
        metrics = assay.compute_metrics()
    """

    def __init__(self, arena_w: int = 800, arena_h: int = 600):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self._step = 0
        self._records = []

    def setup_env(self, env):
        """Configure env geometry for this assay. Override in subclasses."""
        pass

    def record(self, fish_x: float, fish_y: float, fish_heading: float,
               speed: float, goal: int):
        """Record one step. Override to add per-step derived fields."""
        self._step += 1
        self._records.append({
            'x': fish_x,
            'y': fish_y,
            'heading': fish_heading,
            'speed': speed,
            'goal': goal,
            't': self._step,
        })

    def compute_metrics(self) -> dict:
        """Return summary dict. Must be overridden by subclasses."""
        raise NotImplementedError

    def reset(self):
        self._step = 0
        self._records = []

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _near_wall(self, x: float, y: float, margin: float) -> bool:
        return (x < margin or x > self.arena_w - margin
                or y < margin or y > self.arena_h - margin)

    @staticmethod
    def _count_bouts(speeds, threshold: float, min_len: int) -> int:
        """Count continuous bouts where speed < threshold lasting >= min_len steps."""
        bouts = 0
        in_bout = False
        bout_len = 0
        for s in speeds:
            if s < threshold:
                in_bout = True
                bout_len += 1
            else:
                if in_bout and bout_len >= min_len:
                    bouts += 1
                in_bout = False
                bout_len = 0
        if in_bout and bout_len >= min_len:
            bouts += 1
        return bouts

    @staticmethod
    def _path_entropy(xs, ys, arena_w: int, arena_h: int,
                      grid_cols: int = 6, grid_rows: int = 6) -> float:
        """Shannon entropy of visit distribution across a grid of cells."""
        counts = np.zeros(grid_cols * grid_rows, dtype=np.float32)
        for x, y in zip(xs, ys):
            col = int(np.clip(x / arena_w * grid_cols, 0, grid_cols - 1))
            row = int(np.clip(y / arena_h * grid_rows, 0, grid_rows - 1))
            counts[row * grid_cols + col] += 1
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        # Avoid log(0) — only sum non-zero cells
        nz = probs[probs > 0]
        return float(-np.sum(nz * np.log(nz)))

    @staticmethod
    def _distance_traveled(xs, ys) -> float:
        if len(xs) < 2:
            return 0.0
        dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        return float(dists.sum())


# ---------------------------------------------------------------------------
# Assay 1: Novel Tank Test
# ---------------------------------------------------------------------------

class NovelTankTest(BehavioralAssay):
    """Novel tank diving test — measures anxiety via thigmotaxis and zone preference.

    Arena: 400 wide × 600 tall (portrait orientation, mimicking a tall tank).
    Zones:
      bottom_zone — y > arena_h * 0.6  (high anxiety, fish hug the floor)
      top_zone    — y <= arena_h * 0.6 (low anxiety, fish explore up)

    First 60 steps are acclimation and excluded from zone metrics.
    """

    ACCLIMATION = 60
    WALL_MARGIN = 60       # px — thigmotaxis boundary
    FREEZE_SPEED = 0.2     # below → freezing
    FREEZE_MIN_LEN = 5     # minimum consecutive steps to count as a bout
    ERRATIC_SPEED = 2.5    # above → burst swimming

    def __init__(self):
        super().__init__(arena_w=400, arena_h=600)
        self._bottom_zone_y = self.arena_h * 0.6

    def setup_env(self, env):
        """Set portrait tank dimensions and place fish at bottom center."""
        try:
            env.arena_w = self.arena_w
            env.arena_h = self.arena_h
        except AttributeError:
            pass
        # Place fish at bottom center — high-anxiety start position
        try:
            env.fish_x = self.arena_w / 2
            env.fish_y = self.arena_h * 0.85
            env.fish_heading = -math.pi / 2   # facing up
        except AttributeError:
            pass
        # No food, no predator for this assay
        try:
            env.foods = []
        except AttributeError:
            pass
        try:
            env.pred_state = 'IDLE'
        except AttributeError:
            pass

    def compute_metrics(self) -> dict:
        recs = self._records
        if not recs:
            return {}

        post = [r for r in recs if r['t'] > self.ACCLIMATION]
        all_recs = recs  # use all for speed/wall stats

        xs = np.array([r['x'] for r in post]) if post else np.array([])
        ys = np.array([r['y'] for r in post]) if post else np.array([])
        speeds_post = np.array([r['speed'] for r in post]) if post else np.array([])
        speeds_all = np.array([r['speed'] for r in all_recs])
        xs_all = np.array([r['x'] for r in all_recs])
        ys_all = np.array([r['y'] for r in all_recs])

        n_post = len(post)

        # Zone fractions (post-acclimation)
        if n_post > 0:
            in_top = ys <= self._bottom_zone_y
            top_fraction = float(in_top.sum() / n_post)
        else:
            top_fraction = 0.0
        bottom_fraction = 1.0 - top_fraction

        # Freezing bouts (post-acclimation)
        freezing_bouts = self._count_bouts(
            speeds_post.tolist() if len(speeds_post) else [],
            self.FREEZE_SPEED, self.FREEZE_MIN_LEN)

        # Erratic swimming (post-acclimation)
        erratic_swimming = (float((speeds_post > self.ERRATIC_SPEED).sum() / n_post)
                            if n_post > 0 else 0.0)

        # Latency to top zone (any step)
        latency_to_top = None
        for r in recs:
            if r['y'] <= self._bottom_zone_y:
                latency_to_top = r['t']
                break

        # Mean speed (all steps)
        mean_speed = float(speeds_all.mean()) if len(speeds_all) else 0.0

        # Thigmotaxis score — near any wall (all steps)
        near_wall_count = sum(
            1 for r in all_recs
            if self._near_wall(r['x'], r['y'], self.WALL_MARGIN)
        )
        thigmotaxis_score = near_wall_count / len(all_recs) if all_recs else 0.0

        return {
            'top_fraction': top_fraction,
            'bottom_fraction': bottom_fraction,
            'freezing_bouts': freezing_bouts,
            'erratic_swimming': erratic_swimming,
            'latency_to_top': latency_to_top,
            'mean_speed': mean_speed,
            'thigmotaxis_score': float(thigmotaxis_score),
            'n_steps_total': len(recs),
            'n_steps_post_acclimation': n_post,
        }


# ---------------------------------------------------------------------------
# Assay 2: Light-Dark Test
# ---------------------------------------------------------------------------

class LightDarkTest(BehavioralAssay):
    """Light-dark preference test — zebrafish are naturally photophobic.

    Arena: 800 × 300 (landscape tank).
    Zones:
      dark_zone  — x < arena_w * 0.5  (left half)
      light_zone — x >= arena_w * 0.5 (right half)

    First 30 steps are acclimation and excluded from preference metrics.
    Fish start in center.
    """

    ACCLIMATION = 30

    def __init__(self):
        super().__init__(arena_w=800, arena_h=300)
        self._dark_boundary = self.arena_w * 0.5

    def setup_env(self, env):
        """Set landscape tank dimensions and place fish at center."""
        try:
            env.arena_w = self.arena_w
            env.arena_h = self.arena_h
        except AttributeError:
            pass
        try:
            env.fish_x = self.arena_w / 2
            env.fish_y = self.arena_h / 2
            env.fish_heading = 0.0
        except AttributeError:
            pass
        try:
            env.foods = []
        except AttributeError:
            pass
        try:
            env.pred_state = 'IDLE'
        except AttributeError:
            pass

    def _in_dark(self, x: float) -> bool:
        return x < self._dark_boundary

    def compute_metrics(self) -> dict:
        recs = self._records
        if not recs:
            return {}

        post = [r for r in recs if r['t'] > self.ACCLIMATION]
        n_post = len(post)

        # Zone fractions (post-acclimation)
        if n_post > 0:
            dark_steps = sum(1 for r in post if self._in_dark(r['x']))
            dark_fraction = dark_steps / n_post
        else:
            dark_fraction = 0.0
        light_fraction = 1.0 - dark_fraction

        # Transition rate per 100 steps (post-acclimation)
        transitions = 0
        if n_post > 1:
            prev_dark = self._in_dark(post[0]['x'])
            for r in post[1:]:
                cur_dark = self._in_dark(r['x'])
                if cur_dark != prev_dark:
                    transitions += 1
                prev_dark = cur_dark
        transition_rate = (transitions / n_post * 100) if n_post > 0 else 0.0

        # Latency to first dark entry (any step including acclimation)
        latency_to_dark = None
        for r in recs:
            if self._in_dark(r['x']):
                latency_to_dark = r['t']
                break

        # Dark preference index
        dark_pref_index = (dark_fraction - light_fraction)  # range [-1, 1]

        # Mean speeds per zone (post-acclimation)
        dark_speeds = [r['speed'] for r in post if self._in_dark(r['x'])]
        light_speeds = [r['speed'] for r in post if not self._in_dark(r['x'])]
        mean_speed_dark = float(np.mean(dark_speeds)) if dark_speeds else 0.0
        mean_speed_light = float(np.mean(light_speeds)) if light_speeds else 0.0

        return {
            'dark_fraction': float(dark_fraction),
            'light_fraction': float(light_fraction),
            'transition_rate': float(transition_rate),
            'latency_to_dark': latency_to_dark,
            'dark_preference_index': float(dark_pref_index),
            'mean_speed_dark': mean_speed_dark,
            'mean_speed_light': mean_speed_light,
            'n_steps_total': len(recs),
            'n_steps_post_acclimation': n_post,
        }


# ---------------------------------------------------------------------------
# Assay 3: Social Preference Test
# ---------------------------------------------------------------------------

class SocialPreferenceTest(BehavioralAssay):
    """3-chamber social preference test.

    Arena: 900 × 300 (three 300 × 300 chambers side by side).
    Zones:
      left_chamber   — x < 300           (empty)
      center_chamber — 300 <= x < 600    (acclimation zone)
      right_chamber  — x >= 600          (simulated social / conspecific)

    Acclimation: first 60 steps (fish should stay in center).
    Test phase: remaining steps.

    Conspecific attractiveness is simulated by placing a food cluster
    in the right chamber (no food in left).
    """

    ACCLIMATION = 60
    LEFT_BOUND = 300.0
    RIGHT_BOUND = 600.0

    def __init__(self):
        super().__init__(arena_w=900, arena_h=300)

    def setup_env(self, env):
        """Set 3-chamber dimensions and populate right chamber with food."""
        try:
            env.arena_w = self.arena_w
            env.arena_h = self.arena_h
        except AttributeError:
            pass
        # Start fish in center chamber
        try:
            env.fish_x = 450.0
            env.fish_y = 150.0
            env.fish_heading = 0.0
        except AttributeError:
            pass
        try:
            env.pred_state = 'IDLE'
        except AttributeError:
            pass
        # Simulate conspecific zone with food cluster in right chamber
        try:
            food_cluster = []
            rng = np.random.RandomState(0)
            for _ in range(8):
                fx = rng.uniform(620, 880)
                fy = rng.uniform(60, 240)
                food_cluster.append([float(fx), float(fy), 'small'])
            env.foods = food_cluster
        except AttributeError:
            pass

    def _zone(self, x: float) -> str:
        if x < self.LEFT_BOUND:
            return 'left'
        if x < self.RIGHT_BOUND:
            return 'center'
        return 'right'

    def compute_metrics(self) -> dict:
        recs = self._records
        if not recs:
            return {}

        test = [r for r in recs if r['t'] > self.ACCLIMATION]
        n_test = len(test)

        # Zone counts during test phase
        zone_counts = {'left': 0, 'center': 0, 'right': 0}
        for r in test:
            zone_counts[self._zone(r['x'])] += 1

        social_steps = zone_counts['right']
        empty_steps = zone_counts['left']
        center_steps = zone_counts['center']

        social_zone_fraction = social_steps / n_test if n_test else 0.0
        empty_zone_fraction = empty_steps / n_test if n_test else 0.0
        center_time_fraction = center_steps / n_test if n_test else 0.0

        denom = social_steps + empty_steps
        social_preference_index = (
            (social_steps - empty_steps) / denom if denom > 0 else 0.0
        )

        # Approach latency — first test-phase entry into right chamber
        approach_latency = None
        for r in test:
            if self._zone(r['x']) == 'right':
                approach_latency = r['t'] - self.ACCLIMATION
                break

        return {
            'social_zone_fraction': float(social_zone_fraction),
            'empty_zone_fraction': float(empty_zone_fraction),
            'social_preference_index': float(social_preference_index),
            'center_time_fraction': float(center_time_fraction),
            'approach_latency': approach_latency,
            'n_steps_total': len(recs),
            'n_steps_test_phase': n_test,
        }


# ---------------------------------------------------------------------------
# Assay 4: Open Field Test
# ---------------------------------------------------------------------------

class OpenFieldTest(BehavioralAssay):
    """Open field test — exploration vs thigmotaxis.

    Arena: 600 × 600.
    Zones:
      center_zone   — 150 <= x <= 450 AND 150 <= y <= 450 (inner 50%)
      periphery_zone — everything else

    Food is scattered; no predator.
    """

    WALL_MARGIN = 50  # px — thigmotaxis boundary
    CENTER_X_MIN, CENTER_X_MAX = 150.0, 450.0
    CENTER_Y_MIN, CENTER_Y_MAX = 150.0, 450.0

    def __init__(self):
        super().__init__(arena_w=600, arena_h=600)

    def setup_env(self, env):
        """Set square arena and scatter food."""
        try:
            env.arena_w = self.arena_w
            env.arena_h = self.arena_h
        except AttributeError:
            pass
        try:
            env.fish_x = self.arena_w / 2
            env.fish_y = self.arena_h / 2
            env.fish_heading = 0.0
        except AttributeError:
            pass
        try:
            env.pred_state = 'IDLE'
        except AttributeError:
            pass
        # Scatter food evenly across arena
        try:
            rng = np.random.RandomState(0)
            env.foods = [
                [float(rng.uniform(40, self.arena_w - 40)),
                 float(rng.uniform(40, self.arena_h - 40)),
                 'small']
                for _ in range(15)
            ]
        except AttributeError:
            pass

    def _in_center(self, x: float, y: float) -> bool:
        return (self.CENTER_X_MIN <= x <= self.CENTER_X_MAX
                and self.CENTER_Y_MIN <= y <= self.CENTER_Y_MAX)

    def compute_metrics(self) -> dict:
        recs = self._records
        if not recs:
            return {}

        xs = np.array([r['x'] for r in recs])
        ys = np.array([r['y'] for r in recs])
        speeds = np.array([r['speed'] for r in recs])
        headings = np.array([r['heading'] for r in recs])
        n = len(recs)

        # Center fraction
        in_center = [self._in_center(r['x'], r['y']) for r in recs]
        center_fraction = float(sum(in_center) / n)

        # Distance traveled
        distance_traveled = self._distance_traveled(xs, ys)

        # Path entropy across 6×6 grid
        path_entropy = self._path_entropy(xs, ys, self.arena_w, self.arena_h,
                                          grid_cols=6, grid_rows=6)

        # Thigmotaxis score — within WALL_MARGIN of any wall
        near_wall = sum(
            1 for r in recs if self._near_wall(r['x'], r['y'], self.WALL_MARGIN)
        )
        thigmotaxis_score = float(near_wall / n)

        # Velocity variance
        velocity_variance = float(np.var(speeds)) if n > 1 else 0.0

        # Mean angular velocity (mean |turn per step| as heading diff)
        if n > 1:
            d_heading = np.diff(headings)
            # Wrap to [-pi, pi]
            d_heading = (d_heading + math.pi) % (2 * math.pi) - math.pi
            mean_angular_velocity = float(np.mean(np.abs(d_heading)))
        else:
            mean_angular_velocity = 0.0

        return {
            'center_fraction': center_fraction,
            'distance_traveled': float(distance_traveled),
            'path_entropy': float(path_entropy),
            'thigmotaxis_score': thigmotaxis_score,
            'velocity_variance': velocity_variance,
            'mean_angular_velocity': mean_angular_velocity,
            'n_steps': n,
        }


# ---------------------------------------------------------------------------
# Assay Runner
# ---------------------------------------------------------------------------

def run_assay(brain, assay: BehavioralAssay,
              ckpt_mgr=None, ckpt_path: str = None,
              n_steps: int = 500, seed: int = 42) -> dict:
    """Run a headless assay episode and return computed metrics.

    Parameters
    ----------
    brain:
        A ZebrafishBrainV2 instance (or any object with a .step(obs, env) method
        returning {'turn': float, 'speed': float, 'goal': int, ...}).
    assay:
        A BehavioralAssay subclass instance (NovelTankTest, etc.).
    ckpt_mgr:
        Optional CheckpointManager. Used together with ckpt_path to load weights.
    ckpt_path:
        Optional path to a .pt checkpoint file. Loaded before the episode.
    n_steps:
        Number of environment steps to run.
    seed:
        RNG seed passed to env.reset().

    Returns
    -------
    dict
        Metrics dict from assay.compute_metrics().
    """
    # Load checkpoint if provided
    if ckpt_mgr is not None and ckpt_path is not None:
        if os.path.exists(ckpt_path):
            ckpt_mgr.load(brain, ckpt_path)

    # Create env (no food spawned by default — each assay sets up its own)
    env = ZebrafishPreyPredatorEnv(
        render_mode=None,
        n_food=0,
        max_steps=n_steps,
    )

    # Configure arena geometry and initial conditions
    assay.setup_env(env)

    # Reset env — seed is passed here; setup_env may override positions after reset
    obs, _info = env.reset(seed=seed)
    # Re-apply assay geometry after reset (reset may overwrite arena_w/h/fish_*)
    assay.setup_env(env)

    brain.reset()
    assay.reset()

    for _t in range(n_steps):
        # Signal current flee state to env (reduces render artifacts)
        is_flee = getattr(brain, 'current_goal', None) == 1
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(is_flee, panic_intensity=0.8 if is_flee else 0.0)

        # Compute retinal/sensory arrays from env geometry
        inject_sensory(env)

        # Brain step
        out = brain.step(obs, env)
        action = [out.get('turn', 0.0), out.get('speed', 0.0)]

        obs, _reward, terminated, truncated, _info = env.step(action)

        # Record assay data
        assay.record(
            fish_x=float(getattr(env, 'fish_x', 0.0)),
            fish_y=float(getattr(env, 'fish_y', 0.0)),
            fish_heading=float(getattr(env, 'fish_heading', 0.0)),
            speed=float(out.get('speed', 0.0)),
            goal=int(out.get('goal', 0)),
        )

        if terminated or truncated:
            break

    return assay.compute_metrics()
