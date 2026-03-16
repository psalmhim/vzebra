"""
Brain-Gym Bridge for Step 14.

Wraps the entire Step 13 brain pipeline as a Gymnasium-compatible agent.
Translates between gym pixel coordinates and world-centered coordinates,
then runs the full deliberative loop to produce actions.
"""
import os
import math
import datetime
import numpy as np
import torch
import torch.nn.functional as F

from zebra_v60.brain.zebrafish_snn_v60 import ZebrafishSNN_v60
from zebra_v60.brain.dopamine_v60 import DopamineSystem_v60
from zebra_v60.brain.basal_ganglia_v60 import BasalGanglia_v60
from zebra_v60.brain.optic_tectum_v60 import OpticTectum_v60
from zebra_v60.brain.thalamus_v60 import ThalamusRelay_v60
from zebra_v60.brain.goal_policy_v60 import (
    GoalPolicy_v60, SpikingGoalSelector, goal_to_behavior,
    GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, GOAL_SOCIAL,
)
from zebra_v60.brain.shoaling_v60 import ShoalingModule
from zebra_v60.brain.working_memory_v60 import WorkingMemory_v60, SpikingWorkingMemory
from zebra_v60.brain.habit_network_v60 import HabitNetwork_v60
from zebra_v60.brain.rl_critic_v60 import (
    RLCritic_v60, EFEWeightAdapter, EFELambdaAdapter)
from zebra_v60.brain.efe_engine_v60 import PreferredOutcomes, EFEEngine
from zebra_v60.brain.vae_world_model_v60 import VAEWorldModel
from zebra_v60.brain.place_cell_v60 import PlaceCellNetwork
from zebra_v60.brain.allostasis_v60 import AllostaticRegulator
from zebra_v60.brain.amygdala_v60 import Amygdala
from zebra_v60.brain.sleep_wake_v60 import SleepWakeRegulator
from zebra_v60.brain.hebbian_v60 import HebbianPlasticity
from zebra_v60.brain.device_util import get_device
from zebra_v60.world.world_env import WorldEnv
from zebra_v60.tests.step1_vision_pursuit import TurnSmoother


class GymWorldBridge:
    """Translates gym pixel coordinates to WorldEnv centered coordinates."""

    def __init__(self, arena_w=800, arena_h=600):
        self.arena_w = arena_w
        self.arena_h = arena_h

    def gym_to_world_pos(self, gx, gy):
        """Pixel coords → centered world coords (y-flip)."""
        wx = (gx / self.arena_w) * 400 - 200
        wy = -((gy / self.arena_h) * 300 - 150)
        return wx, wy

    def gym_to_world_heading(self, heading):
        """Gym heading → world heading (y-axis flip)."""
        return -heading

    def build_world(self, env):
        """Build a WorldEnv synced from gym env state.

        Returns:
            world: WorldEnv with synced entities
            fish_pos: np.array [2] in world coords
            world_heading: float in world coords
        """
        world = WorldEnv(xmin=-200, xmax=200, ymin=-150, ymax=150,
                         n_food=0, n_enemies=0, n_colleagues=0)

        # Sync foods
        for fx, fy in env.foods:
            wx, wy = self.gym_to_world_pos(fx, fy)
            world.foods.append((wx, wy))

        # Sync predator as enemy
        px, py = self.gym_to_world_pos(env.pred_x, env.pred_y)
        world.enemies.append((px, py))

        # Sync obstacles — prefer rock formation AABBs if available
        rock_formations = getattr(env, 'rock_formations', None)
        if rock_formations:
            scale_x = 400.0 / self.arena_w   # 800px → 400 world units
            scale_y = 300.0 / self.arena_h   # 600px → 300 world units
            for rock in rock_formations:
                for aabb in rock["aabbs"]:
                    wx, wy = self.gym_to_world_pos(aabb["x"], aabb["y"])
                    hw_world = aabb["hw"] * scale_x
                    hh_world = aabb["hh"] * scale_y
                    world.add_obstacle(wx, wy, hw_world, hh_world)
        else:
            # Fallback for old-style obstacles
            for obs in env.obstacles:
                wx, wy = self.gym_to_world_pos(obs["x"], obs["y"])
                hw = obs["r"] / 2.0
                hh = obs["r"] * 0.7 / 2.0
                world.add_obstacle(wx, wy, hw, hh)

        # Fish position and heading
        fish_pos = np.array(self.gym_to_world_pos(env.fish_x, env.fish_y))
        world_heading = self.gym_to_world_heading(env.fish_heading)

        return world, fish_pos, world_heading


class InteroceptiveEnergyModel:
    """Bayesian energy belief from motor efference + noisy interoception.

    The agent does not read env.fish_energy directly. Instead it:
      1. Predicts energy change from motor commands (speed → drain, eating → gain)
      2. Receives a noisy interoceptive observation (true energy + Gaussian noise)
      3. Blends prediction and observation via a fixed gain (w=0.2)

    The prediction error (|predicted - observed|) drives hunger urgency
    in the allostatic regulator.
    """

    def __init__(self, initial_energy=100.0, noise_std=3.0,
                 observation_weight=0.2,
                 drain_per_speed=0.08, gain_per_food=15.0):
        self.noise_std = noise_std
        self.observation_weight = observation_weight
        self.drain_per_speed = drain_per_speed
        self.gain_per_food = gain_per_food
        self._belief = initial_energy
        self._prediction = initial_energy
        self.prediction_error = 0.0

    def predict(self, speed, eaten):
        """Predict energy from motor efference copy.

        Args:
            speed: float — current movement speed [0, 1]
            eaten: int — food items eaten this step
        """
        delta = -self.drain_per_speed * speed + self.gain_per_food * eaten
        self._prediction = max(0.0, min(100.0, self._belief + delta))

    def observe(self, raw_energy):
        """Bayesian update with noisy interoceptive signal.

        Args:
            raw_energy: float — true energy from env (will be noised)
        """
        noisy = raw_energy + np.random.normal(0, self.noise_std)
        noisy = max(0.0, min(100.0, noisy))
        w = self.observation_weight
        self._belief = (1 - w) * self._prediction + w * noisy
        self._belief = max(0.0, min(100.0, self._belief))
        self.prediction_error = abs(self._prediction - noisy)

    def get_energy(self):
        """Return current energy belief."""
        return self._belief

    def reset(self, initial_energy=100.0):
        self._belief = initial_energy
        self._prediction = initial_energy
        self.prediction_error = 0.0


class BrainAgent:
    """Wraps the full brain pipeline as a gym-compatible agent."""

    def __init__(self, device="auto", cls_weights_path=None,
                 base_turn_gain=0.15, swim_speed=1.5, use_habit=True,
                 use_rl_critic=False, use_vae_planner=False,
                 world_model="none", use_allostasis=False,
                 use_sleep_cycle=False, use_active_inference=False):
        """
        Args:
            world_model: str — world model type for planning.
                "none": no world model (default)
                "vae": VAE latent space + transition model
                "place_cell": hippocampal place cell approach (future)
                Also accepts True/False for backwards compatibility
                with use_vae_planner.
        """
        self.device = get_device(device)
        self.base_turn_gain = base_turn_gain
        self.swim_speed = swim_speed
        self.use_habit = use_habit
        self.use_rl_critic = use_rl_critic

        # Resolve world_model option (use_vae_planner for backwards compat)
        if use_vae_planner and world_model == "none":
            world_model = "vae"
        self.world_model_type = world_model

        # SNN model
        self.model = ZebrafishSNN_v60(device=self.device)
        if cls_weights_path and os.path.exists(cls_weights_path):
            state = torch.load(cls_weights_path, weights_only=True,
                               map_location=self.device)
            self.model.load_saveable_state(state)
        self.model.reset()
        self.model.eval()

        # RL critic (Step 15) — must be created before goal_policy
        if use_rl_critic:
            self.critic = RLCritic_v60(
                state_dim=18, device=str(self.device))
            self.adapter = EFEWeightAdapter()
        else:
            self.critic = None
            self.adapter = None

        # Brain modules
        self.dopa_sys = DopamineSystem_v60()
        self.bg = BasalGanglia_v60(mode="exploratory")
        self.ot = OpticTectum_v60()
        self.thal = ThalamusRelay_v60()
        self.goal_policy = SpikingGoalSelector(
            n_goals=4, beta=2.0, device=str(self.device),
            weight_adapter=self.adapter)
        self.wm = SpikingWorkingMemory(
            n_goals=4, buffer_len=20, device=str(self.device))
        self.habit = HabitNetwork_v60()
        self.smoother = TurnSmoother(alpha=0.35)

        # Bridge
        self.bridge = GymWorldBridge()

        # State
        self.goal_vec = np.array([0.0, 0.0, 1.0, 0.0])  # start EXPLORE
        self.prev_oF = None
        self._eaten_buffer = 0
        self._prev_wm_state = None
        self._prev_reward = 0.0
        self._prev_done = False
        self._flee_burst_steps = 0
        self._prev_threat_arousal = 0.0
        self._saccade_active = False
        self._saccade_flash = 0
        self._enemy_pixels_total = 0
        self._enemy_pixels_L = 0
        self._enemy_pixels_R = 0

        # Patch memory: tracks food eaten per plankton patch
        self._patch_visit_counts = {}

        # Novelty detection: temporal-difference of type channels
        self._prev_typeL = None
        self._prev_typeR = None
        self._novelty_ema = 0.0

        # Per-pixel habituation (non-associative learning)
        self._habituation_L = np.zeros(400, dtype=np.float32)
        self._habituation_R = np.zeros(400, dtype=np.float32)

        # Shoaling (Step 20: social behavior)
        self.shoaling = ShoalingModule()

        # Allostasis (Step 18)
        self.use_allostasis = use_allostasis
        if use_allostasis:
            self.allostasis = AllostaticRegulator()
            self.amygdala = Amygdala()
        else:
            self.allostasis = None
            self.amygdala = None

        # Sleep/wake cycle (Step 23)
        self.use_sleep_cycle = use_sleep_cycle
        if use_sleep_cycle:
            self.sleep_regulator = SleepWakeRegulator()
        else:
            self.sleep_regulator = None

        # Hebbian plasticity (RPE-gated online learning)
        self.hebbian = HebbianPlasticity(eta=0.0003, decay=0.9999)

        # Active inference mode (all state inferred through generative models)
        self.use_active_inference = use_active_inference
        if use_active_inference:
            self.interoceptive = InteroceptiveEnergyModel()
            self._prev_enemy_px_total = 0.0
            self._prev_enemy_intensity_mean = 0.0
            self._retinal_features = {}
        else:
            self.interoceptive = None

        # World model (Step 16/17)
        self._prev_z = None
        self._prev_action_ctx = None
        if self.world_model_type == "vae":
            ctx_dim = 16 if use_allostasis else 13
            self.vae_world = VAEWorldModel(
                oF_dim=800, pool_dim=64, latent_dim=16,
                state_ctx_dim=ctx_dim,
                act_dim=6,  # 2 (turn, speed) + 4 (goal one-hot)
                device=str(self.device))
        elif self.world_model_type == "place_cell":
            self.place_cells = PlaceCellNetwork(n_cells=128)
            self.vae_world = None
        else:
            self.vae_world = None

        # Proper EFE engine (Step 25): active inference + VAE required
        if use_active_inference and self.vae_world is not None:
            self.preferred_outcomes = PreferredOutcomes()
            self.efe_engine = EFEEngine(
                self.vae_world, self.preferred_outcomes,
                horizon=5, gamma=0.9)
            self.lambda_adapter = EFELambdaAdapter()
        else:
            self.preferred_outcomes = None
            self.efe_engine = None
            self.lambda_adapter = None

        # Diagnostics (populated each step)
        self.last_diagnostics = {}

    def _compute_experience(self):
        """Compute normalized experience level (0=newbie, 1=expert).

        Combines three signals that grow with experience:
        - Place cell coverage (spatial knowledge)
        - VAE world model maturity (predictive model quality)
        - Habit repertoire size (behavioral learning)

        Weights are renormalized when a module is absent.
        """
        total_weight = 0.0
        experience = 0.0
        if hasattr(self, 'place_cells'):
            coverage = self.place_cells.n_allocated / self.place_cells.n_cells
            experience += 0.5 * coverage
            total_weight += 0.5
        if self.vae_world is not None:
            vae_maturity = self.vae_world._step / (self.vae_world._step + 300.0)
            experience += 0.3 * vae_maturity
            total_weight += 0.3
        if self.use_habit:
            n_habits = len(self.habit.habit_table)
            habit_maturity = n_habits / (n_habits + 5.0)
            experience += 0.2 * habit_maturity
            total_weight += 0.2
        if total_weight > 0:
            experience /= total_weight
        return float(experience)

    def _extract_retinal_features(self, out):
        """Extract structured features from SNN retinal type channels.

        Pure observation — no env access. Extracts enemy, food, boundary,
        and obstacle pixel statistics from left/right type channels.

        Args:
            out: dict from SNN forward pass

        Returns:
            dict with retinal feature keys
        """
        typeL = out["retL_full"][0, 400:].cpu().numpy()
        typeR = out["retR_full"][0, 400:].cpu().numpy()
        intL = out["retL_full"][0, :400].cpu().numpy()
        intR = out["retR_full"][0, :400].cpu().numpy()

        # --- Enemy pixels (type ≈ 0.5) ---
        enemy_mask_L = np.abs(typeL - 0.5) < 0.1
        enemy_mask_R = np.abs(typeR - 0.5) < 0.1
        enemy_px_L = float(np.sum(enemy_mask_L))
        enemy_px_R = float(np.sum(enemy_mask_R))
        enemy_px_total = enemy_px_L + enemy_px_R

        # Enemy intensity (distance proxy: intensity = exp(-d/80))
        enemy_int_vals = np.concatenate([
            intL[enemy_mask_L], intR[enemy_mask_R]])
        enemy_intensity_mean = (
            float(np.mean(enemy_int_vals)) if len(enemy_int_vals) > 0 else 0.0)

        # Lateral bias: (R - L) / (total + 1)
        enemy_lateral_bias = (enemy_px_R - enemy_px_L) / (enemy_px_total + 1)

        # Angular spread: reshape 400→20×20, find az range of enemy pixels
        enemy_spread = 0.0
        for mask in [enemy_mask_L, enemy_mask_R]:
            if np.sum(mask) > 0:
                grid = mask.reshape(20, 20)  # 20 azimuth × 20 elevation
                az_proj = np.any(grid, axis=1)
                az_indices = np.where(az_proj)[0]
                if len(az_indices) > 0:
                    spread = float(az_indices[-1] - az_indices[0])
                    enemy_spread = max(enemy_spread, spread)

        # Temporal dynamics (growth rate, intensity change)
        enemy_growth_rate = enemy_px_total - self._prev_enemy_px_total
        enemy_intensity_change = (
            enemy_intensity_mean - self._prev_enemy_intensity_mean)
        self._prev_enemy_px_total = enemy_px_total
        self._prev_enemy_intensity_mean = enemy_intensity_mean

        # --- Food pixels (type ≈ 1.0) ---
        food_mask_L = np.abs(typeL - 1.0) < 0.1
        food_mask_R = np.abs(typeR - 1.0) < 0.1
        food_px_L = float(np.sum(food_mask_L))
        food_px_R = float(np.sum(food_mask_R))
        food_px_total = food_px_L + food_px_R
        food_lateral_bias = (food_px_R - food_px_L) / (food_px_total + 1)
        food_int_vals = np.concatenate([
            intL[food_mask_L], intR[food_mask_R]])
        food_intensity_mean = (
            float(np.mean(food_int_vals)) if len(food_int_vals) > 0 else 0.0)

        # --- Boundary pixels (type ≈ 0.12) ---
        boundary_mask_L = np.abs(typeL - 0.12) < 0.08
        boundary_mask_R = np.abs(typeR - 0.12) < 0.08
        boundary_px_L = float(np.sum(boundary_mask_L))
        boundary_px_R = float(np.sum(boundary_mask_R))
        boundary_intensity_L = (
            float(np.mean(intL[boundary_mask_L]))
            if np.sum(boundary_mask_L) > 0 else 0.0)
        boundary_intensity_R = (
            float(np.mean(intR[boundary_mask_R]))
            if np.sum(boundary_mask_R) > 0 else 0.0)

        # --- Obstacle pixels (type ≈ 0.75) ---
        obs_mask_L = np.abs(typeL - 0.75) < 0.1
        obs_mask_R = np.abs(typeR - 0.75) < 0.1
        obstacle_px_L = float(np.sum(obs_mask_L))
        obstacle_px_R = float(np.sum(obs_mask_R))

        # --- Colleague pixels (type ≈ 0.25) ---
        coll_mask_L = np.abs(typeL - 0.25) < 0.1
        coll_mask_R = np.abs(typeR - 0.25) < 0.1
        colleague_px_L = float(np.sum(coll_mask_L))
        colleague_px_R = float(np.sum(coll_mask_R))

        return {
            "enemy_px_total": enemy_px_total,
            "enemy_px_L": enemy_px_L,
            "enemy_px_R": enemy_px_R,
            "enemy_intensity_mean": enemy_intensity_mean,
            "enemy_lateral_bias": enemy_lateral_bias,
            "enemy_spread": enemy_spread,
            "enemy_growth_rate": enemy_growth_rate,
            "enemy_intensity_change": enemy_intensity_change,
            "food_px_total": food_px_total,
            "food_px_L": food_px_L,
            "food_px_R": food_px_R,
            "food_lateral_bias": food_lateral_bias,
            "food_intensity_mean": food_intensity_mean,
            "boundary_px_L": boundary_px_L,
            "boundary_px_R": boundary_px_R,
            "boundary_intensity_L": boundary_intensity_L,
            "boundary_intensity_R": boundary_intensity_R,
            "obstacle_px_L": obstacle_px_L,
            "obstacle_px_R": obstacle_px_R,
            "colleague_px_L": colleague_px_L,
            "colleague_px_R": colleague_px_R,
        }

    def _infer_threat_from_belief(self, retinal_features, vae_threat=None):
        """Infer threat assessment from retinal features + VAE decoded threat.

        No env access. Blends raw retinal features with VAE-decoded
        threat assessment (when available and warmed up).

        Args:
            retinal_features: dict from _extract_retinal_features()
            vae_threat: numpy [4] or None — from VAEWorldModel.encode_threat()

        Returns:
            dict matching _compute_threat_assessment() interface
        """
        rf = retinal_features

        # Raw retinal estimates
        proximity_ret = min(1.0, rf["enemy_px_total"] / 50.0)
        lateral_ret = rf["enemy_lateral_bias"]
        growth = rf["enemy_growth_rate"]
        _exp_arg = max(-20.0, min(20.0, -growth * 5.0))
        approach_ret = 1.0 / (1.0 + math.exp(_exp_arg))
        facing_ret = min(1.0, rf["enemy_spread"] / 15.0)

        # Blend with VAE decoded threat if available
        if vae_threat is not None and self.vae_world is not None:
            w = self.vae_world.get_blend_weight()
        else:
            w = 0.0

        proximity = (1 - w) * proximity_ret + w * float(vae_threat[0]) if vae_threat is not None else proximity_ret
        lateral = (1 - w) * lateral_ret + w * float(vae_threat[1]) if vae_threat is not None else lateral_ret
        approach_rate = (1 - w) * approach_ret + w * float(vae_threat[2]) if vae_threat is not None else approach_ret
        facing_score = (1 - w) * facing_ret + w * float(vae_threat[3]) if vae_threat is not None else facing_ret

        # Derived quantities (same formulas as ground-truth version)
        pred_facing_fish = facing_score > 0.4

        # TTC from approach rate: approach_rate=0.5 means neutral
        closing_speed = max(0.0, (approach_rate - 0.5) * 2.0)
        headroom = max(0.0, 1.0 - proximity)
        if closing_speed > 0.05:
            ttc = (headroom / closing_speed) * 40.0
        else:
            ttc = 999.0

        # Panic level
        panic_level = 0.0
        if pred_facing_fish and ttc < 40:
            panic_level = facing_score * max(0.0, 1.0 - ttc / 40.0)
        elif ttc < 20:
            panic_level = 0.5 * max(0.0, 1.0 - ttc / 20.0)
        panic_level = min(1.0, panic_level)

        return {
            "pred_facing_score": facing_score,
            "pred_facing_fish": pred_facing_fish,
            "ttc": ttc,
            "closing_speed": closing_speed,
            "panic_level": panic_level,
            "cover_target": None,
            # Extra fields for active inference diagnostics
            "proximity": proximity,
            "lateral": lateral,
            "approach_rate": approach_rate,
        }

    def _estimate_position_from_retina(self, retinal_features, pi_pos, heading):
        """Estimate position correction from boundary pixel intensity.

        Uses boundary pixels as wall-distance proxy: intensity ≈ exp(-d/80).
        When boundary signal is strong, directly constrains position so the
        estimated wall distance is consistent with arena edges.

        Also uses total boundary pixel count on L vs R as a coarse bearing
        to the nearest wall, providing heading-independent triangulation.

        Args:
            retinal_features: dict from _extract_retinal_features()
            pi_pos: numpy [2] — current path-integrated position
            heading: float — current heading estimate

        Returns:
            corrected_pos: numpy [2] — corrected position estimate
        """
        rf = retinal_features
        corrected = pi_pos.copy()
        arena_w, arena_h = 800.0, 600.0

        # Direct wall-distance constraint: for each visible boundary side,
        # estimate distance to that wall and correct position accordingly.
        total_bnd = rf["boundary_px_L"] + rf["boundary_px_R"]

        for side, bnd_int, bnd_px in [
            ("L", rf["boundary_intensity_L"], rf["boundary_px_L"]),
            ("R", rf["boundary_intensity_R"], rf["boundary_px_R"]),
        ]:
            if bnd_px < 2 or bnd_int < 0.01:
                continue

            # Estimated distance to wall (gym px)
            est_dist = -80.0 * math.log(max(0.01, bnd_int)) * 2.0
            est_dist = max(5.0, min(400.0, est_dist))

            # Confidence: more pixels → more reliable
            confidence = min(1.0, bnd_px / 15.0) * 0.3

            # Wall direction in gym frame
            if side == "L":
                wall_angle = heading + math.pi / 2
            else:
                wall_angle = heading - math.pi / 2

            # Which arena wall is closest along this direction?
            cos_a = math.cos(wall_angle)
            sin_a = math.sin(wall_angle)

            # Project: how far to each wall along wall_angle
            best_wall_dist = 999.0
            wall_pos = None
            if abs(cos_a) > 0.1:
                if cos_a < 0:  # heading toward left wall (x=0)
                    d = pi_pos[0] / (-cos_a + 1e-8)
                else:  # heading toward right wall (x=800)
                    d = (arena_w - pi_pos[0]) / (cos_a + 1e-8)
                if 0 < d < best_wall_dist:
                    best_wall_dist = d
            if abs(sin_a) > 0.1:
                if sin_a < 0:  # heading toward top wall (y=0)
                    d = pi_pos[1] / (-sin_a + 1e-8)
                else:  # heading toward bottom wall (y=600)
                    d = (arena_h - pi_pos[1]) / (sin_a + 1e-8)
                if 0 < d < best_wall_dist:
                    best_wall_dist = d

            if best_wall_dist < 500:
                # Error = estimated wall distance - actual wall distance
                error = est_dist - best_wall_dist
                # Pull position along opposite direction to reduce error
                corrected[0] -= error * cos_a * confidence
                corrected[1] -= error * sin_a * confidence

        # Very strong walls: when boundary pixels are bright, clamp position
        # to be near the arena edge
        for bnd_int, bnd_px, dim_idx, dim_max in [
            (rf["boundary_intensity_L"], rf["boundary_px_L"], 0, arena_w),
            (rf["boundary_intensity_R"], rf["boundary_px_R"], 0, arena_w),
            (rf["boundary_intensity_L"], rf["boundary_px_L"], 1, arena_h),
            (rf["boundary_intensity_R"], rf["boundary_px_R"], 1, arena_h),
        ]:
            if bnd_int > 0.5 and bnd_px > 10:
                # Very close to a wall — estimate which edge
                est_d = -80.0 * math.log(max(0.01, bnd_int)) * 2.0
                if corrected[dim_idx] < dim_max * 0.3:
                    # Near low edge
                    target = max(5.0, est_d)
                    corrected[dim_idx] += 0.1 * (target - corrected[dim_idx])
                elif corrected[dim_idx] > dim_max * 0.7:
                    # Near high edge
                    target = min(dim_max - 5.0, dim_max - est_d)
                    corrected[dim_idx] += 0.1 * (target - corrected[dim_idx])

        # Clamp to arena
        corrected[0] = np.clip(corrected[0], 5, arena_w - 5)
        corrected[1] = np.clip(corrected[1], 5, arena_h - 5)

        return corrected

    def _compute_threat_assessment(self, env):
        """Compute predator gaze-awareness and time-to-contact.

        Returns:
            dict with keys:
                pred_facing_score: float [0, 1] — 1=facing fish dead-on
                pred_facing_fish: bool — |facing_diff| < 0.5 rad (~28 deg)
                ttc: float — time-to-contact in steps (999 if retreating)
                closing_speed: float — net closing speed (px/step)
                panic_level: float [0, 1] — high when facing + TTC low
                cover_target: None (populated later by cover-seeking)
        """
        dx = env.fish_x - env.pred_x
        dy = env.fish_y - env.pred_y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8

        # Angle from predator to fish
        angle_pred_to_fish = math.atan2(dy, dx)

        # How aligned is predator heading with direction to fish
        facing_diff = env.pred_heading - angle_pred_to_fish
        facing_diff = math.atan2(math.sin(facing_diff), math.cos(facing_diff))
        pred_facing_score = max(0.0, 1.0 - abs(facing_diff) / math.pi)
        pred_facing_fish = abs(facing_diff) < 0.5

        # Closing speed: predator velocity projected onto pred→fish axis
        pred_speed = getattr(env, 'pred_speed_current', 0.0)
        pred_vx = pred_speed * math.cos(env.pred_heading)
        pred_vy = pred_speed * math.sin(env.pred_heading)
        ux = dx / dist
        uy = dy / dist
        closing_speed = -(pred_vx * ux + pred_vy * uy)

        # Subtract fish retreat speed
        fish_retreat = env.fish_speed * max(0.0, math.cos(
            env.fish_heading - math.atan2(-dy, -dx)))
        net_closing = closing_speed - fish_retreat

        # TTC
        if net_closing > 0.1:
            ttc = dist / net_closing
        else:
            ttc = 999.0

        # Panic level: ramps up when predator faces fish AND TTC is short
        panic_level = 0.0
        if pred_facing_fish and ttc < 40:
            panic_level = pred_facing_score * max(0.0, 1.0 - ttc / 40.0)
        elif ttc < 20:
            panic_level = 0.5 * max(0.0, 1.0 - ttc / 20.0)
        panic_level = min(1.0, panic_level)

        return {
            "pred_facing_score": pred_facing_score,
            "pred_facing_fish": pred_facing_fish,
            "ttc": ttc,
            "closing_speed": net_closing,
            "panic_level": panic_level,
            "cover_target": None,
        }

    def _evaluate_food_prospects(self, env, fish_pos, world):
        """Active inference food prospection: evaluate cost-benefit per food.

        For each food item, compute:
          - metabolic_cost: energy needed to reach it (proportional to distance)
          - risk: danger from predator near that food location
          - urgency: how badly energy is needed
          - prospect_score: combined EFE-style score (lower = better target)

        Returns:
            list of dicts, sorted by prospect_score (best first), max 5 items.
        """
        energy = getattr(env, "fish_energy", 100.0)
        energy_ratio = energy / self.energy_max if hasattr(self, 'energy_max') else energy / 100.0
        urgency = max(0.0, 1.0 - energy_ratio)  # 0=full, 1=starving

        pred_wx, pred_wy = self.bridge.gym_to_world_pos(env.pred_x, env.pred_y)

        prospects = []
        for i, (fx, fy) in enumerate(env.foods):
            fwx, fwy = self.bridge.gym_to_world_pos(fx, fy)

            # Distance from fish to food (world coords)
            dx = fwx - fish_pos[0]
            dy = fwy - fish_pos[1]
            dist = math.sqrt(dx * dx + dy * dy) + 1e-8

            # Metabolic cost: energy drain proportional to distance
            # Roughly: drain_per_step * steps_to_reach ≈ 0.08 * (dist / speed)
            speed_est = max(0.5, 2.0)  # average speed in world units
            steps_to_reach = dist / speed_est
            metabolic_cost = 0.08 * steps_to_reach

            # Can we afford it?
            affordable = 1.0 if energy > metabolic_cost * 1.5 else 0.5

            # Risk: predator proximity to this food
            pdx = fwx - pred_wx
            pdy = fwy - pred_wy
            pred_dist_to_food = math.sqrt(pdx * pdx + pdy * pdy) + 1e-8
            # Risk is high when predator is close to food, normalized 0-1
            risk = max(0.0, 1.0 - pred_dist_to_food / 150.0)

            # Reward: energy gain from eating
            gain = 15.0

            # Patch productivity bonus: food in known-productive patches
            patch_bonus = 0.0
            patch_label = "none"
            patches = getattr(env, 'plankton_patches', [])
            for pi, patch in enumerate(patches):
                ddx = fx - patch["cx"]
                ddy = fy - patch["cy"]
                if math.sqrt(ddx * ddx + ddy * ddy) < patch.get("radius", 80):
                    patch_label = patch.get("label", f"patch_{pi}")
                    count = self._patch_visit_counts.get(patch_label, 0)
                    # More visits → higher confidence → bigger bonus
                    patch_bonus = min(0.15, count * 0.02)
                    break

            # Reachability: penalize food occluded by rocks (requires detour)
            occluded = 0.0
            if hasattr(env, '_has_rock_occlusion'):
                fish_gx, fish_gy = env.fish_x, env.fish_y
                # Check line-of-sight from CURRENT fish position (not spawn)
                _orig_center = (env.arena_w * 0.5, env.arena_h * 0.5)
                # Temporarily check from fish position
                for rock in getattr(env, 'rock_formations', []):
                    for aabb in rock["aabbs"]:
                        # Simple line-AABB intersection check
                        dx_f = fx - fish_gx
                        dy_f = fy - fish_gy
                        seg_len = math.sqrt(dx_f * dx_f + dy_f * dy_f)
                        if seg_len < 1e-6:
                            continue
                        # Check if AABB blocks the path
                        cx_a, cy_a = aabb["x"], aabb["y"]
                        hw_a, hh_a = aabb["hw"] + 8, aabb["hh"] + 8
                        # Parametric intersection with expanded AABB
                        t_min, t_max = 0.0, 1.0
                        for dim_f, dim_a, dim_hw in [
                            (fish_gx, cx_a, hw_a), (fish_gy, cy_a, hh_a)
                        ]:
                            d_val = (fx - fish_gx) if dim_f == fish_gx else (fy - fish_gy)
                            if abs(d_val) < 1e-6:
                                if abs(dim_f - dim_a) >= dim_hw:
                                    t_min = 2.0  # no intersection
                                    break
                            else:
                                t1 = (dim_a - dim_hw - dim_f) / d_val
                                t2 = (dim_a + dim_hw - dim_f) / d_val
                                if t1 > t2:
                                    t1, t2 = t2, t1
                                t_min = max(t_min, t1)
                                t_max = min(t_max, t2)
                                if t_min > t_max:
                                    break
                        if t_min <= t_max:
                            occluded = 1.5  # strong penalty for unreachable food
                            break
                    if occluded > 0:
                        break

            # EFE-style prospect score (lower = better)
            # cost term: metabolic cost weighted by energy state
            cost_weight = 0.3 + 0.5 * urgency  # more urgent → accept higher cost
            risk_weight = 0.4 - 0.2 * urgency  # reduced: foraging requires risk tolerance
            gain_weight = 0.6 + 0.4 * urgency  # more urgent → value gain more

            prospect_score = (
                cost_weight * (metabolic_cost / 20.0)  # normalized
                + risk_weight * risk
                + occluded  # penalty for food behind rocks
                - gain_weight * (gain / 20.0) * affordable
                - patch_bonus  # productive patches are preferred
            )

            prospects.append({
                "food_idx": i,
                "gym_pos": (fx, fy),
                "dist": dist,
                "metabolic_cost": metabolic_cost,
                "risk": risk,
                "pred_dist": pred_dist_to_food,
                "affordable": affordable,
                "urgency": urgency,
                "prospect_score": prospect_score,
            })

        prospects.sort(key=lambda p: p["prospect_score"])
        return prospects[:5]

    def act(self, obs, env):
        """Run full brain pipeline and produce gym action [turn_rate, speed].

        Args:
            obs: gym observation (not directly used — brain uses its own vision)
            env: the gym environment instance (for state access)

        Returns:
            action: np.array [2] — [turn_rate, speed_mod]
        """
        self._step_count = getattr(self, '_step_count', 0) + 1
        # 1. Build world from gym state
        world, fish_pos, world_heading = self.bridge.build_world(env)

        # 1a. Sync colleagues into world (Step 20)
        for c in getattr(env, 'colleagues', []):
            wx, wy = self.bridge.gym_to_world_pos(c["x"], c["y"])
            world.colleagues.append((wx, wy))

        # 1b. Active inference food prospection
        food_prospects = self._evaluate_food_prospects(env, fish_pos, world)

        # 2. Effective heading with eye position
        effective_heading = world_heading + self.ot.eye_pos * 0.25

        # 3. Forward pass through SNN (with depth shading + attention)
        goal_t = torch.tensor(
            self.goal_vec, dtype=torch.float32,
            device=self.device).unsqueeze(0)  # [1, 4]
        with torch.no_grad():
            out = self.model.forward(fish_pos, effective_heading, world,
                                     depth_shading=True,
                                     goal_probs=goal_t)

        # Store SNN output for neural activity visualization
        self._last_snn_out = out

        # 4. Classification
        cls_logits = out["cls"]
        cls_probs = F.softmax(cls_logits, dim=1)[0].cpu().numpy()

        # 4b. Count enemy pixels from the type channel (ground truth)
        typeL_raw = out["retL_full"][0, 400:].cpu().numpy()
        typeR_raw = out["retR_full"][0, 400:].cpu().numpy()
        ENEMY_TYPE_VAL = 0.5
        enemy_px_L = np.sum(np.abs(typeL_raw - ENEMY_TYPE_VAL) < 0.1)
        enemy_px_R = np.sum(np.abs(typeR_raw - ENEMY_TYPE_VAL) < 0.1)
        self._enemy_pixels_total = int(enemy_px_L + enemy_px_R)
        self._enemy_pixels_L = int(enemy_px_L)
        self._enemy_pixels_R = int(enemy_px_R)

        # 4d. Extract retinal features for active inference (Phase 1)
        if self.use_active_inference:
            self._retinal_features = self._extract_retinal_features(out)

        # 5. Free energy
        F_visual = self.model.compute_free_energy()

        # 6. Thalamic relay (simulated audio)
        F_audio_sim = 0.1 * abs(math.sin(0.05 * env.step_count))
        cms = self.thal.step(F_visual, F_audio_sim)
        self.dopa_sys.beta = self.thal.modulate_dopamine_gain()

        # 7. Dopamine system
        oL_mean = float(out["oL"].abs().mean())
        oR_mean = float(out["oR"].abs().mean())
        dopa, rpe, valL, valR = self.dopa_sys.step(
            F_visual, oL_mean, oR_mean, eaten=self._eaten_buffer)
        self._eaten_buffer = 0  # consumed

        # 7b. Allostatic interoception (Step 18)
        prev_speed = self.last_diagnostics.get("speed", 0.5)
        if self.use_active_inference:
            # Interoceptive energy model: predict from motor efference, observe noisily
            self.interoceptive.predict(prev_speed, self._eaten_buffer)
            self.interoceptive.observe(getattr(env, "fish_energy", 100.0))
            energy = self.interoceptive.get_energy()
            # Predator distance from retinal proxy (no env access)
            rf = self._retinal_features
            pred_dist_px = max(1.0,
                               (1.0 - min(1.0, rf["enemy_px_total"] / 50.0))
                               * 300.0)
        else:
            energy = getattr(env, "fish_energy", 100.0)
            pred_dist_px = math.sqrt(
                (env.fish_x - env.pred_x) ** 2
                + (env.fish_y - env.pred_y) ** 2)
        allo_state = None
        if self.allostasis is not None:
            p_colleague = cls_probs[3] if len(cls_probs) > 3 else 0.0
            allo_state = self.allostasis.step(
                energy, prev_speed, pred_dist_px, p_colleague=p_colleague)
            self.dopa_sys.beta = self.allostasis.modulate_dopamine_gain(
                self.dopa_sys.beta)

        # 7b2. Threat assessment — predator gaze, TTC, panic
        if self.use_active_inference:
            # Infer threat from retinal features + VAE decoded threat
            vae_threat = None
            if self.vae_world is not None and self._prev_z is not None:
                vae_threat = self.vae_world.encode_threat(self._prev_z)
            threat = self._infer_threat_from_belief(
                self._retinal_features, vae_threat)
        else:
            threat = self._compute_threat_assessment(env)

        # 7b3. Amygdala fear response — threat arousal from retinal + proximity + gaze
        threat_arousal = 0.0
        if self.amygdala is not None:
            stress_val = allo_state["stress"] if allo_state else 0.0
            threat_arousal = self.amygdala.step(
                self._enemy_pixels_total, pred_dist_px, stress_val,
                pred_facing_score=threat["pred_facing_score"])
            # Boost classifier p_enemy by threat arousal
            if threat_arousal > 0.05:
                cls_probs[2] = min(1.0, cls_probs[2] + 0.3 * threat_arousal)
                cls_probs /= cls_probs.sum() + 1e-8
            # Feed arousal back into allostatic stress
            if self.allostasis is not None:
                self.allostasis.stress = min(
                    1.0, self.allostasis.stress + 0.1 * threat_arousal)

        # 7b4. Final pixel-evidence cap on p_enemy
        #      The classifier + amygdala boost can hallucinate threats in
        #      complex multi-entity scenes.  Cap p_enemy based on actual
        #      enemy pixels in the type channel (ground truth).
        #      Need ~25 pixels for full confidence; 0 pixels → cap at 0.02.
        max_p_enemy = max(0.02, min(1.0, self._enemy_pixels_total / 25.0))
        if cls_probs[2] > max_p_enemy:
            cls_probs[2] = max_p_enemy
            cls_probs /= cls_probs.sum() + 1e-8

        # 7c. Sleep/wake modulation (Step 23)
        sleep_state = None
        if self.sleep_regulator is not None:
            fatigue = allo_state["fatigue"] if allo_state else 0.0
            stress_val = allo_state["stress"] if allo_state else 0.0
            pred_prox = max(0.0, 1.0 - pred_dist_px / 150.0)
            sleep_state = self.sleep_regulator.step(
                fatigue, stress_val, pred_prox)
            if sleep_state["is_sleeping"]:
                self.dopa_sys.beta *= sleep_state["dopa_beta_multiplier"]

        # 8. Working memory
        pi_OT = out["pi_OT"]
        pi_PC = out["pi_PC"]
        mem_state, alpha_eff, cls_summary = self.wm.step(
            cls_probs, self.goal_vec, dopa, cms, F_visual, pi_OT, pi_PC)

        # 8.5 VAE world model: encode, train, plan (Step 16b)
        if self.vae_world is not None:
            # Build 13-dim multi-modal state context for belief state
            if self.use_active_inference:
                # Phase 7: Use inferred state (path integration + interoceptive)
                pi_pos = (self.place_cells._pi_pos
                          if hasattr(self, 'place_cells')
                          else np.array([400.0, 300.0]))
                pi_heading = (self.place_cells._pi_heading
                              if hasattr(self, 'place_cells')
                              else world_heading)
                _energy_ctx = self.interoceptive.get_energy()
                state_ctx = np.array([
                    pi_pos[0] / 400.0,              # proprioceptive: x (PI)
                    pi_pos[1] / 300.0,              # proprioceptive: y (PI)
                    pi_heading / np.pi,              # proprioceptive: heading (PI)
                    _energy_ctx / 100.0,             # interoceptive: energy (belief)
                    dopa,
                    np.clip(rpe, -1, 1),
                    pi_OT,
                    pi_PC,
                    cls_probs[1],
                    cls_probs[2],
                    cls_probs[4],
                    cms,
                    np.clip(F_visual, 0, 2) / 2,
                ], dtype=np.float32)
            else:
                _energy = getattr(env, "fish_energy", 100.0)
                state_ctx = np.array([
                    fish_pos[0] / 200.0,
                    fish_pos[1] / 150.0,
                    world_heading / np.pi,
                    _energy / 100.0,
                    dopa,
                    np.clip(rpe, -1, 1),
                    pi_OT,
                    pi_PC,
                    cls_probs[1],
                    cls_probs[2],
                    cls_probs[4],
                    cms,
                    np.clip(F_visual, 0, 2) / 2,
                ], dtype=np.float32)

            # Extend state_ctx with allostasis (13 → 16 dims)
            if self.allostasis is not None:
                state_ctx = np.concatenate([
                    state_ctx,
                    self.allostasis.get_state_ctx_extension()])

            z_np, z_mu = self.vae_world.encode(out["oF"], state_ctx)
            self.vae_world.train_step(out["oF"], state_ctx)

            # Train threat decoder (active inference: self-supervised from retina)
            if self.use_active_inference and self._retinal_features:
                self.vae_world.update_threat_decoder(
                    z_np, self._retinal_features)

            if self._prev_z is not None and self._prev_action_ctx is not None:
                self.vae_world.update_transition(
                    self._prev_z, self._prev_action_ctx, z_np)

            last_act = np.array([
                self.last_diagnostics.get("turn_rate", 0.0),
                self.last_diagnostics.get("speed", 0.5)],
                dtype=np.float32)

            # Step 25: proper EFE engine replaces ad-hoc planning + allo bias
            if self.efe_engine is not None:
                _energy_bel = (self.interoceptive.get_energy()
                               if self.interoceptive else energy)
                _lambdas = (self.lambda_adapter.get_lambdas()
                            if self.lambda_adapter else None)
                efe_result = self.efe_engine.compute_efe(
                    z_np, self._retinal_features, allo_state,
                    _energy_bel, lambdas=_lambdas)
                self.goal_policy.set_efe_engine_result(efe_result)
                # Cache components for lambda adapter update
                if self.lambda_adapter is not None:
                    self.lambda_adapter.cache_components(
                        self.efe_engine._last_components)
            else:
                G_plan = self.vae_world.plan(z_np, last_act, dopa, cls_probs)
                self.goal_policy.set_plan_bonus(G_plan)

            self._prev_z = z_np

        # 8.5b Enhanced VAE training during sleep (Step 23)
        if (sleep_state is not None and sleep_state["is_sleeping"]
                and self.vae_world is not None):
            for _ in range(sleep_state["vae_train_multiplier"] - 1):
                self.vae_world.train_from_buffer()

        # 8.6 Place cell world model (Step 17)
        if hasattr(self, 'place_cells'):
            prev_turn = self.last_diagnostics.get("turn_rate", 0.0)
            prev_spd = self.last_diagnostics.get("speed", 0.5)
            if self.use_active_inference:
                # Initialize PI heading from first observation (legitimate:
                # derived from retinal input, not env state)
                if self._step_count == 1:
                    self.place_cells._pi_heading = -world_heading  # gym convention
                # Calibrate PI to match env kinematics:
                #   env heading: turn_rate * 0.15 rad/step
                #   PI heading: turn_rate * gain (0.8) → scale by 0.15/0.8
                #   env position: 2.0 * speed_mod px/step
                #   PI position: speed * 5.0 * gain (4.0) → scale by 2.0/4.0
                cal_turn = prev_turn * (0.15 / 0.8)
                cal_spd = prev_spd * (2.0 / 4.0)
                self.place_cells.path_integrate(cal_turn, cal_spd)
                # Boundary-based position correction (no env access)
                corrected = self._estimate_position_from_retina(
                    self._retinal_features,
                    self.place_cells._pi_pos,
                    self.place_cells._pi_heading)
                self.place_cells.correct_path_integration(
                    corrected, weight=0.15)
                gym_pos = self.place_cells._pi_pos.copy()
            else:
                self.place_cells.path_integrate(prev_turn, prev_spd)
                gym_pos = np.array(
                    [env.fish_x, env.fish_y], dtype=np.float32)
                self.place_cells.correct_path_integration(gym_pos)
            if (self.place_cells._step_count > 0
                    and self.place_cells._step_count
                    % self.place_cells.replay_interval == 0):
                self.place_cells.replay()
            G_plan = self.place_cells.plan(
                gym_pos, np.array([prev_turn, prev_spd]), dopa, cls_probs)
            self.goal_policy.set_plan_bonus(G_plan)

        # 8.6b Enhanced place cell replay during sleep (Step 23)
        if (sleep_state is not None and sleep_state["is_sleeping"]
                and hasattr(self, 'place_cells')):
            for _ in range(sleep_state["replay_multiplier"]):
                self.place_cells.replay_extended()

        # 8.7 Allostatic bias on top of planning bonus (Step 18)
        #     Skip when EFE engine is active — allostasis modulates precision
        #     inside the EFE computation instead of additive bias (Step 25).
        #     Note: read current bonus from VAE/place cell (set in 8.5/8.6).
        #     If no world model is active, start from zero to avoid accumulation.
        if self.allostasis is not None and self.efe_engine is None:
            allo_bias = self.allostasis.get_goal_prior_bias()
            if self.vae_world is not None or hasattr(self, 'place_cells'):
                current_bonus = self.goal_policy._external_efe_bonus.copy()
            else:
                current_bonus = np.zeros(self.goal_policy.n_goals, dtype=np.float64)
            # Pad allo_bias to match n_goals if needed
            if len(allo_bias) < self.goal_policy.n_goals:
                allo_bias = np.pad(allo_bias, (0, self.goal_policy.n_goals - len(allo_bias)))
            self.goal_policy.set_plan_bonus(current_bonus + allo_bias)

        # 8.8 Experience-based exploration bonus (explore first, exploit later)
        experience = self._compute_experience()
        explore_bonus = -0.15 * (1.0 - experience)
        if self.vae_world is not None or hasattr(self, 'place_cells') or self.allostasis is not None:
            current_bonus = self.goal_policy._external_efe_bonus.copy()
        else:
            current_bonus = np.zeros(self.goal_policy.n_goals, dtype=np.float64)
        current_bonus[GOAL_EXPLORE] += explore_bonus
        # Mirror partial bonus to FLEE so exploration doesn't suppress survival
        current_bonus[GOAL_FLEE] += explore_bonus * 0.5

        # 8.9 Energy-driven goal bias
        #     < 90%: prefer foraging (hungry fish seeks food)
        #     ≥ 90%: prefer socializing (well-fed fish joins conspecifics)
        _energy_r = energy / getattr(env, 'energy_max', 100.0)
        if _energy_r < 0.90:
            forage_drive = 0.7 * (0.90 - _energy_r) / 0.90  # 0→0.7 ramp
            current_bonus[GOAL_FORAGE] -= forage_drive       # lower EFE = preferred
            current_bonus[GOAL_SOCIAL] += forage_drive * 0.3  # suppress social
        else:
            social_drive = 0.6 * (_energy_r - 0.90) / 0.10  # 0→0.6 ramp
            current_bonus[GOAL_SOCIAL] -= social_drive       # lower EFE = preferred
            current_bonus[GOAL_FORAGE] += social_drive * 0.3  # suppress foraging

        self.goal_policy.set_plan_bonus(current_bonus)

        # 9. Goal policy selection
        _hunger = self.allostasis.hunger if self.allostasis else 0.0
        _hunger_err = self.allostasis.hunger_error if self.allostasis else 0.0
        _energy_ratio = energy / getattr(env, 'energy_max', 100.0)
        choice, self.goal_vec, posterior, confidence, efe_vec = \
            self.goal_policy.step(
                cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual,
                self.wm.get_mean(),
                hunger=_hunger, hunger_error=_hunger_err,
                pred_facing_score=threat["pred_facing_score"],
                ttc=threat["ttc"], energy_ratio=_energy_ratio)

        # 10. Habit shortcut
        if self.use_habit:
            # Sleep modulation: easier consolidation during sleep (Step 23)
            _saved_habit_threshold = None
            _saved_habit_decay = None
            if sleep_state is not None and sleep_state["is_sleeping"]:
                habit_mod = sleep_state["habit_modulation"]
                _saved_habit_threshold = self.habit.habit_threshold
                self.habit.habit_threshold = int(
                    self.habit.habit_threshold * habit_mod["threshold_mult"])
                if habit_mod["decay"] is not None:
                    _saved_habit_decay = self.habit.decay
                    self.habit.decay = habit_mod["decay"]

            effective_goal, shortcut_active = self.habit.step(
                cls_probs, choice, confidence, rpe)

            # Restore habit params after step
            if _saved_habit_threshold is not None:
                self.habit.habit_threshold = _saved_habit_threshold
            if _saved_habit_decay is not None:
                self.habit.decay = _saved_habit_decay
        else:
            effective_goal = choice
            shortcut_active = False

        # 11. Goal-conditioned behavior
        approach_gain, speed_mod_brain, explore_mod, turn_strategy = \
            goal_to_behavior(effective_goal, cls_probs, posterior, confidence,
                             pred_facing_score=threat["pred_facing_score"])

        # 11b. Shoaling integration (Step 20: social behavior)
        _shoal_diag = {}
        if effective_goal == GOAL_SOCIAL:
            colleagues = getattr(env, 'colleagues', [])
            if self.use_active_inference and hasattr(self, 'place_cells'):
                _sx = self.place_cells._pi_pos[0]
                _sy = self.place_cells._pi_pos[1]
                _sh = self.place_cells._pi_heading
            else:
                _sx, _sy, _sh = env.fish_x, env.fish_y, env.fish_heading
            shoal_turn, shoal_speed, _shoal_diag = self.shoaling.step(
                _sx, _sy, _sh, colleagues)
            approach_gain += shoal_turn * 0.5
            speed_mod_brain *= shoal_speed

        # 12. Retinal turn signal
        retL = out["retL"]
        retR = out["retR"]
        retR_sum = float(retR.sum())
        retL_sum = float(retL.sum())
        total = retR_sum + retL_sum + 1e-8
        raw_turn = (retR_sum - retL_sum) / total

        # 12b. Obstacle repulsion — steer away from rock-heavy side
        typeL_t = out["retL_full"][0, 400:].cpu().numpy()
        typeR_t = out["retR_full"][0, 400:].cpu().numpy()
        obs_px_L = float(np.sum(np.abs(typeL_t - 0.75) < 0.1))
        obs_px_R = float(np.sum(np.abs(typeR_t - 0.75) < 0.1))
        food_px_L = float(np.sum(np.abs(typeL_t - 1.0) < 0.1))
        food_px_R = float(np.sum(np.abs(typeR_t - 1.0) < 0.1))
        food_px_total = food_px_L + food_px_R
        obs_total = obs_px_L + obs_px_R
        if obs_total > 3:
            obs_repulsion = -1.2 * (obs_px_R - obs_px_L) / (obs_total + 1e-8)
            # Strong center-push when obstacle fills both eyes
            if obs_total > 15:
                obs_repulsion *= 1.5
            raw_turn = raw_turn + obs_repulsion

        # 12c. Food-directed navigation (FORAGE mode)
        #      Three regimes:
        #        a) Food on retina (food_px >= 3): steer toward food pixels
        #        b) Near a prospect but no retinal food: olfactory approach
        #        c) Far from prospects: wander + weak olfactory bias
        #      Olfactory turn is injected after the smoother (step 18)
        #      so it bypasses approach_gain damping.
        _olfactory_turn = 0.0
        _near_food = False
        _nearest_food_gym_dist = 999.0
        if food_prospects:
            best = food_prospects[0]
            fdx = best["gym_pos"][0] - env.fish_x
            fdy = best["gym_pos"][1] - env.fish_y
            _nearest_food_gym_dist = math.sqrt(fdx * fdx + fdy * fdy)
            _near_food = _nearest_food_gym_dist < 120

        if effective_goal == GOAL_FORAGE:
            # Compute olfactory heading cue toward best food prospect.
            if self.use_active_inference:
                # Use retinal food lateral bias as olfactory proxy
                rf = self._retinal_features
                _scaled_diff = np.clip(
                    -rf["food_lateral_bias"] * 2.0, -1.0, 1.0)
            elif food_prospects:
                fx, fy = food_prospects[0]["gym_pos"]
                food_angle = math.atan2(fy - env.fish_y,
                                        fx - env.fish_x)
                angle_diff = food_angle - env.fish_heading
                angle_diff = math.atan2(
                    math.sin(angle_diff), math.cos(angle_diff))
                _scaled_diff = np.clip(-angle_diff * 2.0, -1.0, 1.0)
            else:
                _scaled_diff = 0.0

            if food_px_total >= 3:
                # (a) Food visible — use food-pixel direction + strong
                #     olfactory cue.  Retinal food_turn helps when food
                #     is off-axis; olfactory cue dominates when ahead.
                food_turn = (food_px_R - food_px_L) / (food_px_total + 1e-8)
                raw_turn = raw_turn * 0.3 + food_turn * 0.7
                approach_gain = max(approach_gain, 1.2)
                _olfactory_turn = 0.7 * _scaled_diff
                # Keep decent speed near food — don't slow too much
                if _nearest_food_gym_dist < 50:
                    speed_mod_brain *= max(
                        0.6, _nearest_food_gym_dist / 50.0)
            elif _near_food:
                # (b) Food prospect nearby but off retina — strong olfactory
                approach_gain *= 0.5
                _olfactory_turn = 0.9 * _scaled_diff
            else:
                # (c) No nearby food — wander with moderate olfactory pull
                approach_gain *= 0.2
                explore_mod = max(explore_mod, 1.5)
                _olfactory_turn = 0.5 * _scaled_diff

        # 12d. Obstacle braking — when rocks densely cover both eyes, the
        #      directional repulsion cancels out. Add a forward-facing escape
        #      turn and reduce speed to prevent running into rock walls.
        #      Suppress braking/escape when foraging toward a known food target.
        obs_frac = obs_total / 800.0  # fraction of retina covered by rocks
        _foraging_with_target = (effective_goal == GOAL_FORAGE
                                 and abs(_olfactory_turn) > 0.05)
        if obs_frac > 0.5:
            if _foraging_with_target:
                # Foraging with olfactory target: minimal braking, no escape turn
                speed_mod_brain *= max(0.7, 1.0 - 0.2 * obs_frac)
            else:
                # No food target: full braking + escape turn
                speed_mod_brain *= max(0.3, 1.0 - obs_frac)
                if obs_px_L > obs_px_R:
                    raw_turn += 0.4 * obs_frac
                else:
                    raw_turn -= 0.4 * obs_frac

        # 12e. Cover-seeking during FLEE — steer toward the far side of
        #      a rock that shields from the predator (only when TTC allows
        #      deliberation and predator is looking at us).
        _cover_turn = 0.0
        if (effective_goal == GOAL_FLEE
                and 15 < threat["ttc"] < 120
                and threat["pred_facing_score"] > 0.3):
            best_score = 999.0
            best_target = None
            if self.use_active_inference:
                # Use path-integrated position + lateral threat as bearing
                rf = self._retinal_features
                _pi_pos = (self.place_cells._pi_pos
                           if hasattr(self, 'place_cells')
                           else np.array([env.fish_x, env.fish_y]))
                _pi_heading = (self.place_cells._pi_heading
                               if hasattr(self, 'place_cells')
                               else env.fish_heading)
                fish_cx, fish_cy = _pi_pos[0], _pi_pos[1]
                fish_hd = _pi_heading
                # Infer predator bearing from threat lateral signal
                pred_bearing = fish_hd + threat.get("lateral", 0.0) * math.pi / 2
            else:
                fish_cx, fish_cy = env.fish_x, env.fish_y
                fish_hd = env.fish_heading
                pred_bearing = math.atan2(env.pred_y - fish_cy,
                                          env.pred_x - fish_cx)
            for rock in getattr(env, 'rock_formations', []):
                rcx, rcy = rock["cx"], rock["cy"]
                rdx = rcx - fish_cx
                rdy = rcy - fish_cy
                rock_dist = math.sqrt(rdx * rdx + rdy * rdy) + 1e-8
                if rock_dist > 180 or rock_dist < rock["base_r"] + 5:
                    continue
                rock_angle = math.atan2(rdy, rdx)
                angle_to_pred = pred_bearing - rock_angle
                angle_to_pred = math.atan2(math.sin(angle_to_pred),
                                           math.cos(angle_to_pred))
                if abs(angle_to_pred) < math.pi / 4:
                    continue
                perp = abs(math.sin(angle_to_pred))
                score = rock_dist / 180.0 - 0.5 * perp
                if score < best_score:
                    best_score = score
                    if self.use_active_inference:
                        # Far side = opposite inferred predator direction
                        opp_x = -math.cos(pred_bearing) * rock["base_r"]
                        opp_y = -math.sin(pred_bearing) * rock["base_r"]
                        far_x = rcx + opp_x
                        far_y = rcy + opp_y
                    else:
                        far_x = rcx - (env.pred_x - rcx) / (rock_dist + 1e-8) * rock["base_r"]
                        far_y = rcy - (env.pred_y - rcy) / (rock_dist + 1e-8) * rock["base_r"]
                    best_target = (far_x, far_y)

            if best_target is not None:
                cover_angle = math.atan2(best_target[1] - fish_cy,
                                         best_target[0] - fish_cx)
                cover_diff = cover_angle - fish_hd
                cover_diff = math.atan2(math.sin(cover_diff),
                                        math.cos(cover_diff))
                _cover_turn = -np.clip(cover_diff * 1.5, -0.6, 0.6)
                threat["cover_target"] = best_target

        # 13. Smoothed turn
        turn = self.smoother.step(raw_turn * approach_gain)

        # 14. BG gating
        valL_eff = valL - 0.1 * turn
        valR_eff = valR + 0.1 * turn
        self.bg.noise = self.thal.modulate_bg_exploration() * explore_mod
        bg_gate = self.bg.step(valL_eff, valR_eff, dopa, rpe)

        # 14b. Extract type channels and compute novelty (needed for eye position)
        retL_intensity = retL[0, :].cpu().numpy()
        retR_intensity = retR[0, :].cpu().numpy()
        typeL = typeL_t
        typeR = typeR_t

        # Temporal-difference novelty with per-pixel habituation
        novelty_L = 0.0
        novelty_R = 0.0
        if self._prev_typeL is not None:
            type_diff_L = np.abs(typeL - self._prev_typeL)
            type_diff_R = np.abs(typeR - self._prev_typeR)

            # Habituation: unchanged pixels accumulate, changed pixels reset
            changed_L = type_diff_L > 0.05
            changed_R = type_diff_R > 0.05
            self._habituation_L = np.where(
                changed_L, 0.0, self._habituation_L + 1.0)
            self._habituation_R = np.where(
                changed_R, 0.0, self._habituation_R + 1.0)

            # Suppression: 1/(1 + hab/tau), tau=5 → half at 5 frames
            suppress_L = 1.0 / (1.0 + self._habituation_L / 5.0)
            suppress_R = 1.0 / (1.0 + self._habituation_R / 5.0)

            novelty_L = float(np.sum(type_diff_L * suppress_L))
            novelty_R = float(np.sum(type_diff_R * suppress_R))

        novelty_total = novelty_L + novelty_R
        # EMA decay: persistent stimuli → novelty decays
        self._novelty_ema = 0.3 * novelty_total + 0.7 * self._novelty_ema
        novelty = self._novelty_ema

        self._prev_typeL = typeL.copy()
        self._prev_typeR = typeR.copy()

        # 15. Eye position (with novelty-driven smooth bias)
        # Directional novelty: bias eye toward novel side
        if novelty > 1.0:
            novelty_dir = (novelty_R - novelty_L) / (novelty_total + 1e-8)
        else:
            novelty_dir = 0.0
        eye_pos = self.ot.step(valL_eff, valR_eff, F_visual, bg_gate, dopa,
                               novelty_drive=novelty_dir * novelty)

        # 16. Set vision strip on env for rendering
        # NOTE: SNN's retL = heading-offset = fish's RIGHT eye (computational label).
        # Swap for display so the user sees anatomically correct L/R.
        env.set_vision_strip(retR_intensity, retL_intensity, typeR, typeL)

        # 16b. Enemy saccade response (alarm boost already applied at step 4c)
        enemy_pixels_total = self._enemy_pixels_total

        # 16c. Novelty-driven saccade (fires before enemy check, but enemy overrides)
        NOVELTY_SACCADE_THRESH = 3.0  # ~3 pixels changed entity type
        novelty_saccade_fired = False
        if novelty > NOVELTY_SACCADE_THRESH and enemy_pixels_total < 3:
            direction = 1.0 if novelty_R > novelty_L else -1.0
            magnitude = min(0.4, novelty * 0.05)  # max 0.4 (weaker than enemy 0.6)
            novelty_saccade_fired = self.ot.trigger_saccade(
                direction, magnitude=magnitude)
            if novelty_saccade_fired:
                self._saccade_flash = 2  # short visual flash

        self._saccade_active = False
        if enemy_pixels_total >= 3:
            # Saccade toward predator (higher priority, magnitude 0.6)
            enemy_dir = (1.0 if self._enemy_pixels_R > self._enemy_pixels_L
                         else -1.0)
            saccade_fired = self.ot.trigger_saccade(enemy_dir)

            if saccade_fired:
                self._flee_burst_steps = 5 + int(5 * threat_arousal)
                # Only flash visual indicator for strong detections
                if enemy_pixels_total >= 15:
                    self._saccade_flash = 3

        # Saccade flash countdown (visual only, doesn't affect brain logic)
        if self._saccade_flash > 0:
            self._saccade_active = True
            self._saccade_flash -= 1

        # Pass saccade state to env for rendering
        env._saccade_active = self._saccade_active
        env._saccade_eye_pos = self.ot.eye_pos

        # Pass food prospects to env for target highlight rendering
        env._food_prospects = food_prospects

        # 17. Precision update — driven by prediction error (predictive coding)
        _precision_frozen = (sleep_state is not None
                             and sleep_state["is_sleeping"]
                             and sleep_state["precision_freeze"])
        if not _precision_frozen:
            pe_OT = self.model.OT_F.pred_error
            if pe_OT is not None:
                self.model.prec_OT.update_precision(pe_OT)
            pe_PC = self.model.PC_per.pred_error
            if pe_PC is not None:
                self.model.prec_PC.update_precision(pe_PC)
            with torch.no_grad():
                self.model.prec_OT.gamma.data += 0.008 * (dopa - 0.5)
                self.model.prec_PC.gamma.data += 0.008 * (dopa - 0.5)

        # 18. Compute gym action
        # Negate turn: brain computes in world coords (y-up), gym uses y-down
        brain_turn = (turn + 0.03 * bg_gate + 0.02 * eye_pos
                      + _olfactory_turn + _cover_turn)

        # Boundary avoidance: steer toward center when near walls
        margin = 80
        wall_urgency = 0.0
        if self.use_active_inference:
            _wx = (self.place_cells._pi_pos[0]
                   if hasattr(self, 'place_cells') else env.fish_x)
            _wy = (self.place_cells._pi_pos[1]
                   if hasattr(self, 'place_cells') else env.fish_y)
            _wh = (self.place_cells._pi_heading
                   if hasattr(self, 'place_cells') else env.fish_heading)
        else:
            _wx, _wy, _wh = env.fish_x, env.fish_y, env.fish_heading
        if _wx < margin:
            wall_urgency = max(wall_urgency, (margin - _wx) / margin)
        if _wx > env.arena_w - margin:
            wall_urgency = max(wall_urgency,
                               (_wx - (env.arena_w - margin)) / margin)
        if _wy < margin:
            wall_urgency = max(wall_urgency, (margin - _wy) / margin)
        if _wy > env.arena_h - margin:
            wall_urgency = max(wall_urgency,
                               (_wy - (env.arena_h - margin)) / margin)

        wall_turn = 0.0
        if wall_urgency > 0.01:
            center_dx = env.arena_w * 0.5 - _wx
            center_dy = env.arena_h * 0.5 - _wy
            angle_to_center = math.atan2(center_dy, center_dx)
            angle_diff = angle_to_center - _wh
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            wall_turn = wall_urgency * 0.8 * np.sign(angle_diff)

        # Blend: brain turn weight decreases near walls
        brain_weight = max(0.2, 1.0 - wall_urgency)
        turn_rate = np.clip(
            -brain_turn * brain_weight + wall_turn, -1.0, 1.0)

        speed = np.clip(
            speed_mod_brain * (0.8 + 0.4 * dopa), 0.0, 1.0)

        # Flee burst: temporary speed boost after saccade detection
        if self._flee_burst_steps > 0:
            speed = min(1.3, speed * 1.5)
            self._flee_burst_steps -= 1

        # Reduce speed when low energy (use inferred energy in AI mode)
        if self.use_active_inference:
            _energy_speed = self.interoceptive.get_energy()
        else:
            _energy_speed = getattr(env, "fish_energy", 100.0)
        if _energy_speed < 20:
            speed *= 0.5 + 0.5 * (_energy_speed / 20.0)

        # Panic sprint override: adrenaline restores speed during threat
        panic_level = threat["panic_level"]
        if panic_level > 0.1:
            speed = max(speed, 0.8 * panic_level + speed * (1 - panic_level))
            self._flee_burst_steps = max(
                self._flee_burst_steps, int(8 * panic_level))

        # Fatigue speed cap (Step 18) — partially overridden during panic
        if self.allostasis is not None:
            base_cap = self.allostasis.get_speed_cap()
            if panic_level > 0.3:
                cap = base_cap + (1.0 - base_cap) * 0.5 * panic_level
            else:
                cap = base_cap
            speed = min(speed, cap)

        # Sleep modulation: near-zero speed + fatigue recovery boost (Step 23)
        if sleep_state is not None and sleep_state["is_sleeping"]:
            speed *= sleep_state["speed_multiplier"]
            # Boost fatigue recovery during sleep
            if self.allostasis is not None:
                boost = sleep_state["fatigue_recovery_multiplier"] - 1.0
                self.allostasis.fatigue -= (
                    self.allostasis.fatigue_recovery * boost)
                self.allostasis.fatigue = max(0.0, self.allostasis.fatigue)

        # Signal flee state to env for energy cost multiplier
        env.set_flee_active(
            effective_goal == GOAL_FLEE or self._flee_burst_steps > 0,
            panic_intensity=panic_level)

        action = np.array([turn_rate, speed], dtype=np.float32)

        # Store action context for VAE transition training (Step 16)
        if self.vae_world is not None:
            goal_oh = np.zeros(self.goal_policy.n_goals, dtype=np.float32)
            goal_oh[min(effective_goal, len(goal_oh) - 1)] = 1.0
            self._prev_action_ctx = np.array([
                float(turn_rate), float(speed), *goal_oh],
                dtype=np.float32)

        # Populate diagnostics
        self.last_diagnostics = {
            "goal": effective_goal,
            "goal_efe": choice,
            "shortcut_active": shortcut_active,
            "cls_probs": cls_probs.copy(),
            "posterior": posterior.copy(),
            "confidence": confidence,
            "efe_vec": efe_vec.copy(),
            "dopa": dopa,
            "rpe": rpe,
            "F_visual": F_visual,
            "cms": cms,
            "bg_gate": bg_gate,
            "eye_pos": eye_pos,
            "energy": energy,
            "turn_rate": float(turn_rate),
            "speed": float(speed),
            "retL_max": float(retL.max()),
            "retR_max": float(retR.max()),
            "retL_intensity": retL_intensity,
            "retR_intensity": retR_intensity,
            "saccade_active": self._saccade_active,
            "novelty": novelty,
            "novelty_L": novelty_L,
            "novelty_R": novelty_R,
            "novelty_saccade": novelty_saccade_fired,
            "food_prospects": food_prospects,
            "flee_burst": self._flee_burst_steps,
            "wall_urgency": wall_urgency,
            "patch_memory": dict(self._patch_visit_counts),
            "habituation_mean": float(np.mean(np.concatenate([
                self._habituation_L, self._habituation_R]))),
            "experience": experience,
            "explore_bonus": explore_bonus,
            "threat": threat,
            # Predictive coding diagnostics
            "pe_OTF": out.get("pe_OTF", 0.0),
            "pe_PT": out.get("pe_PT", 0.0),
            "pe_PC_per": out.get("pe_PC_per", 0.0),
            "pe_PC_int": out.get("pe_PC_int", 0.0),
            "att_signals": out.get("att_signals", None),
        }

        # VAE diagnostics (Step 16)
        if self.vae_world is not None:
            self.last_diagnostics["vae"] = self.vae_world.get_diagnostics()

        # Place cell diagnostics (Step 17)
        if hasattr(self, 'place_cells'):
            self.last_diagnostics["place_cells"] = \
                self.place_cells.get_diagnostics()

        # Allostasis diagnostics (Step 18)
        if self.allostasis is not None:
            self.last_diagnostics["allostasis"] = \
                self.allostasis.get_diagnostics()

        # Amygdala diagnostics
        if self.amygdala is not None:
            self.last_diagnostics["amygdala"] = \
                self.amygdala.get_diagnostics()

        # EFE engine diagnostics (Step 25)
        if self.efe_engine is not None:
            self.last_diagnostics["efe_engine"] = \
                self.efe_engine.get_diagnostics()
            if self.lambda_adapter is not None:
                self.last_diagnostics["efe_lambdas"] = \
                    self.lambda_adapter.get_lambdas().tolist()

        # Active inference diagnostics
        if self.use_active_inference:
            ai_diag = {"retinal_features": self._retinal_features}
            if self.interoceptive is not None:
                ai_diag["energy_belief"] = self.interoceptive.get_energy()
                ai_diag["energy_pred_error"] = self.interoceptive.prediction_error
            if hasattr(self, 'place_cells'):
                ai_diag["pi_pos"] = self.place_cells._pi_pos.copy().tolist()
            self.last_diagnostics["active_inference"] = ai_diag

        # Shoaling diagnostics (Step 20)
        if _shoal_diag:
            self.last_diagnostics["shoaling"] = _shoal_diag

        # Sleep diagnostics (Step 23)
        if self.sleep_regulator is not None:
            self.last_diagnostics["sleep"] = \
                self.sleep_regulator.get_diagnostics()

        # Inference validation: compare inferred vs ground truth
        if (self.use_active_inference
                and os.environ.get("VZEBRA_VALIDATE_INFERENCE")):
            true_energy = getattr(env, "fish_energy", 100.0)
            true_x, true_y = env.fish_x, env.fish_y
            true_dist = math.sqrt(
                (true_x - env.pred_x) ** 2 + (true_y - env.pred_y) ** 2)
            inferred_energy = (self.interoceptive.get_energy()
                               if self.interoceptive else true_energy)
            pi_pos = (self.place_cells._pi_pos
                      if hasattr(self, 'place_cells')
                      else np.array([true_x, true_y]))
            pos_err = math.sqrt(
                (pi_pos[0] - true_x) ** 2 + (pi_pos[1] - true_y) ** 2)
            self.last_diagnostics["inference_validation"] = {
                "energy_error": abs(inferred_energy - true_energy),
                "position_error": pos_err,
                "true_pred_dist": true_dist,
                "inferred_pred_dist_px": pred_dist_px,
            }

        # Pass diagnostics to env for monitoring panel
        env.set_brain_diagnostics(self.last_diagnostics)

        # RL critic update (Step 15)
        if self.critic is not None:
            wm_state = self.wm.m.copy()

            # Push transition from PREVIOUS step: (s_prev, r_prev, s_now, done_prev)
            # The true next-state is the current wm state (updated this act() call)
            if self._prev_wm_state is not None:
                self.critic.push(
                    self._prev_wm_state, self._prev_reward,
                    wm_state, self._prev_done)

            critic_value = self.critic.predict(wm_state)
            td_err = self.critic.update()
            if td_err is not None and self.adapter is not None:
                self.adapter.update(
                    td_err, posterior, confidence, shortcut_active)
            # Step 25: update lambda adapter with TD error
            if td_err is not None and self.lambda_adapter is not None:
                self.lambda_adapter.update(td_err, effective_goal)

            self.last_diagnostics["critic_value"] = critic_value
            self.last_diagnostics["td_error"] = td_err if td_err is not None else 0.0
            self.last_diagnostics["efe_deltas"] = self.adapter.get_deltas() if self.adapter else {}
            self.last_diagnostics["critic_update_count"] = self.critic.get_stats()["update_count"]

            self._prev_wm_state = wm_state

        # Online learning for spiking modules (RPE-gated Hebbian/RL)
        _rpe = self.dopa_sys.rpe
        _dopa = self.dopa_sys.dopa
        _reward = self._prev_reward
        if hasattr(self.wm, 'learn'):
            self.wm.learn(_rpe, _dopa)
        if hasattr(self.goal_policy, 'learn'):
            self.goal_policy.learn(_rpe, _dopa, _reward)

        # Hebbian plasticity: RPE-gated weight updates for SNN pathway
        # Boost learning 3x during high-threat for escape learning
        _threat_boost = 1.0
        if self.amygdala is not None:
            _ta = self.amygdala.threat_arousal
            if _ta > 0.3:
                _threat_boost = 1.0 + 2.0 * _ta  # up to 3x
        self.hebbian.update(self.model, _rpe, _dopa, threat_boost=_threat_boost)
        # PE-driven feedback learning (every step, independent of RPE)
        if not getattr(self, '_skip_fb_update', False):
            self.hebbian.update_feedback(self.model)
        hebb_stats = self.hebbian.get_stats()
        self.last_diagnostics["hebb_updates"] = hebb_stats["hebb_updates"]
        self.last_diagnostics["hebb_dw_norm"] = hebb_stats["hebb_dw_norm"]
        self.last_diagnostics["hebb_fb_dw_norm"] = hebb_stats["hebb_fb_dw_norm"]
        self.last_diagnostics["escape_successes"] = hebb_stats["escape_successes"]
        self.last_diagnostics["escape_failures"] = hebb_stats["escape_failures"]
        self.last_diagnostics["threat_boost"] = _threat_boost

        return action

    def update_post_step(self, info, reward=0.0, done=False, env=None):
        """Feed actual eaten count to dopamine system and save reward for critic."""
        eaten = info.get("food_eaten_this_step", 0)
        self._eaten_buffer = eaten

        # Patch memory: attribute eaten food to nearest patch
        if eaten > 0 and env is not None:
            fish_x, fish_y = info.get("fish_pos", (0, 0))
            patches = getattr(env, 'plankton_patches', [])
            for patch in patches:
                dx = fish_x - patch["cx"]
                dy = fish_y - patch["cy"]
                if math.sqrt(dx * dx + dy * dy) < patch.get("radius", 80):
                    label = patch.get("label", "unknown")
                    self._patch_visit_counts[label] = (
                        self._patch_visit_counts.get(label, 0) + eaten)
                    break

        # VAE memory update (Step 16)
        if self.vae_world is not None and self._prev_z is not None and env is not None:
            if self.use_active_inference:
                # Phase 6: risk from retinal proximity (no env access)
                rf = self._retinal_features
                retinal_prox = min(1.0, rf.get("enemy_px_total", 0) / 50.0)
                pred_dist = max(1.0, (1.0 - retinal_prox) * 300.0)
            else:
                pred_dist = 150.0
                fish_x, fish_y = info.get("fish_pos", (0, 0))
                pred_x = getattr(env, "pred_x", fish_x)
                pred_y = getattr(env, "pred_y", fish_y)
                dx = fish_x - pred_x
                dy = fish_y - pred_y
                pred_dist = math.sqrt(dx * dx + dy * dy)
            self.vae_world.update_memory(self._prev_z, eaten, pred_dist)

        # Place cell memory update (Step 17)
        if hasattr(self, 'place_cells') and env is not None:
            if self.use_active_inference:
                # Phase 6: use path-integrated position + retinal risk
                gym_pos = self.place_cells._pi_pos.copy()
                rf = self._retinal_features
                risk_signal = min(1.0, rf.get("enemy_px_total", 0) / 50.0)
            else:
                fish_x, fish_y = info.get("fish_pos", (0, 0))
                gym_pos = np.array([fish_x, fish_y], dtype=np.float32)
                pred_x = getattr(env, "pred_x", fish_x)
                pred_y = getattr(env, "pred_y", fish_y)
                dx = fish_x - pred_x
                dy = fish_y - pred_y
                pred_dist = math.sqrt(dx * dx + dy * dy)
                risk_signal = max(0.0, 1.0 - pred_dist / 150.0)
            self.place_cells.update(gym_pos, float(eaten), risk_signal)

            # Fear conditioning: amygdala → place cell risk
            if self.amygdala is not None:
                ta = self.amygdala.threat_arousal
                if ta > 0.1:
                    self.place_cells.update_fear(gym_pos, ta, alpha=0.3)
                self.place_cells.check_fear_memory(ta)
            self.place_cells.decay_risk(0.999)

        # Escape learning: track success/failure transitions
        if self.amygdala is not None:
            current_arousal = self.amygdala.threat_arousal
            if self._prev_threat_arousal > 0.5 and current_arousal < 0.1:
                self.hebbian.record_escape_success()
            self._prev_threat_arousal = current_arousal
        if done and self._prev_threat_arousal > 0.3:
            self.hebbian.record_escape_failure()

        # Save reward/done for next act() call which pushes the transition
        if self.critic is not None:
            self._prev_reward = reward
            self._prev_done = done

    def reset(self):
        """Reset brain for a new episode, preserving learned state."""
        # SNN: reset activation voltages only (weights preserved)
        self.model.reset()
        # Transient modules: full reset (no learned state)
        self.dopa_sys.reset()
        self.bg.reset()
        self.ot.reset()
        self.thal.reset()
        self.goal_policy.reset()
        self.wm.reset()
        # Learned modules: episode reset (preserve learned state)
        self.habit.reset_episode()
        self.smoother.smoothed = 0.0
        self.goal_vec = np.zeros(self.goal_policy.n_goals)
        self.goal_vec[GOAL_EXPLORE] = 1.0
        self.prev_oF = None
        self._step_count = 0
        self._eaten_buffer = 0
        self._prev_wm_state = None
        self._prev_reward = 0.0
        self._prev_done = False
        self._flee_burst_steps = 0
        self._saccade_active = False
        self._saccade_flash = 0
        self._enemy_pixels_total = 0
        self._enemy_pixels_L = 0
        self._enemy_pixels_R = 0
        self._patch_visit_counts = {}
        self._prev_typeL = None
        self._prev_typeR = None
        self._novelty_ema = 0.0
        self._habituation_L = np.zeros(400, dtype=np.float32)
        self._habituation_R = np.zeros(400, dtype=np.float32)
        self.last_diagnostics = {}
        if self.interoceptive is not None:
            self.interoceptive.reset()
        if self.use_active_inference:
            self._prev_enemy_px_total = 0.0
            self._prev_enemy_intensity_mean = 0.0
            self._retinal_features = {}
        if self.critic is not None:
            self.critic.reset_episode()
        if self.adapter is not None:
            self.adapter.reset_episode()
        if self.efe_engine is not None:
            self.efe_engine.reset()
        if self.preferred_outcomes is not None:
            self.preferred_outcomes.reset()
        if self.lambda_adapter is not None:
            self.lambda_adapter.reset_episode()
        if self.vae_world is not None:
            self.vae_world.reset_episode()
            self._prev_z = None
            self._prev_action_ctx = None
        if hasattr(self, 'place_cells'):
            self.place_cells.reset_episode()
        if self.allostasis is not None:
            self.allostasis.reset()
        if self.amygdala is not None:
            self.amygdala.reset()
        if self.sleep_regulator is not None:
            self.sleep_regulator.reset()

    def save_checkpoint(self, path):
        """Save all learned parameters to a single checkpoint file.

        Args:
            path: file path for the checkpoint (.pt)
        """
        checkpoint = {
            "version": 2,
            "timestamp": datetime.datetime.now().isoformat(),
            "snn": self.model.get_saveable_state(),
            "habit": self.habit.get_saveable_state(),
        }
        if self.critic is not None:
            checkpoint["critic"] = self.critic.get_saveable_state()
        if self.adapter is not None:
            checkpoint["adapter"] = self.adapter.get_saveable_state()
        if self.vae_world is not None:
            checkpoint["vae_world"] = self.vae_world.get_saveable_state()
        if self.lambda_adapter is not None:
            checkpoint["lambda_adapter"] = self.lambda_adapter.get_saveable_state()
        if hasattr(self, 'place_cells'):
            checkpoint["place_cells"] = self.place_cells.get_saveable_state()
        if hasattr(self.wm, 'get_saveable_state'):
            checkpoint["spiking_wm"] = self.wm.get_saveable_state()
        if hasattr(self.goal_policy, 'get_saveable_state'):
            checkpoint["spiking_goal"] = self.goal_policy.get_saveable_state()
        checkpoint["hebbian"] = self.hebbian.get_saveable_state()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(checkpoint, path)
        print(f"[checkpoint] Saved to {path}")

    def load_checkpoint(self, path):
        """Load learned parameters from a checkpoint file.

        Args:
            path: file path to the checkpoint (.pt)
        """
        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=False)
        version = checkpoint.get("version", 0)
        print(f"[checkpoint] Loading v{version} from {path} "
              f"(saved {checkpoint.get('timestamp', 'unknown')})")

        self.model.load_saveable_state(checkpoint["snn"])
        self.habit.load_saveable_state(checkpoint["habit"])

        if "critic" in checkpoint and self.critic is not None:
            self.critic.load_saveable_state(checkpoint["critic"])
        if "adapter" in checkpoint and self.adapter is not None:
            self.adapter.load_saveable_state(checkpoint["adapter"])
        if "vae_world" in checkpoint and self.vae_world is not None:
            self.vae_world.load_saveable_state(checkpoint["vae_world"])
        if ("lambda_adapter" in checkpoint
                and self.lambda_adapter is not None):
            self.lambda_adapter.load_saveable_state(
                checkpoint["lambda_adapter"])
        if "place_cells" in checkpoint and hasattr(self, 'place_cells'):
            self.place_cells.load_saveable_state(checkpoint["place_cells"])
        if ("spiking_wm" in checkpoint
                and hasattr(self.wm, 'load_saveable_state')):
            self.wm.load_saveable_state(checkpoint["spiking_wm"])
        if ("spiking_goal" in checkpoint
                and hasattr(self.goal_policy, 'load_saveable_state')):
            self.goal_policy.load_saveable_state(checkpoint["spiking_goal"])
        if "hebbian" in checkpoint:
            self.hebbian.load_saveable_state(checkpoint["hebbian"])
        print("[checkpoint] Loaded successfully")
