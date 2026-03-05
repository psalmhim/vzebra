"""
Brain-Gym Bridge for Step 14.

Wraps the entire Step 13 brain pipeline as a Gymnasium-compatible agent.
Translates between gym pixel coordinates and world-centered coordinates,
then runs the full deliberative loop to produce actions.
"""
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

from zebra_v60.brain.zebrafish_snn_v60 import ZebrafishSNN_v60
from zebra_v60.brain.dopamine_v60 import DopamineSystem_v60
from zebra_v60.brain.basal_ganglia_v60 import BasalGanglia_v60
from zebra_v60.brain.optic_tectum_v60 import OpticTectum_v60
from zebra_v60.brain.thalamus_v60 import ThalamusRelay_v60
from zebra_v60.brain.goal_policy_v60 import (
    GoalPolicy_v60, goal_to_behavior,
    GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE,
)
from zebra_v60.brain.working_memory_v60 import WorkingMemory_v60
from zebra_v60.brain.habit_network_v60 import HabitNetwork_v60
from zebra_v60.brain.rl_critic_v60 import RLCritic_v60, EFEWeightAdapter
from zebra_v60.brain.vae_world_model_v60 import VAEWorldModel
from zebra_v60.brain.place_cell_v60 import PlaceCellNetwork
from zebra_v60.brain.allostasis_v60 import AllostaticRegulator
from zebra_v60.brain.amygdala_v60 import Amygdala
from zebra_v60.brain.sleep_wake_v60 import SleepWakeRegulator
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


class BrainAgent:
    """Wraps the full brain pipeline as a gym-compatible agent."""

    def __init__(self, device="auto", cls_weights_path=None,
                 base_turn_gain=0.15, swim_speed=1.5, use_habit=True,
                 use_rl_critic=False, use_vae_planner=False,
                 world_model="none", use_allostasis=False,
                 use_sleep_cycle=False):
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
            self.model.load_state_dict(
                torch.load(cls_weights_path, weights_only=True,
                           map_location=self.device))
        self.model.reset()
        self.model.eval()

        # RL critic (Step 15) — must be created before goal_policy
        if use_rl_critic:
            self.critic = RLCritic_v60(device=str(self.device))
            self.adapter = EFEWeightAdapter()
        else:
            self.critic = None
            self.adapter = None

        # Brain modules
        self.dopa_sys = DopamineSystem_v60()
        self.bg = BasalGanglia_v60(mode="exploratory")
        self.ot = OpticTectum_v60()
        self.thal = ThalamusRelay_v60()
        self.goal_policy = GoalPolicy_v60(
            n_goals=3, beta=2.0, persist_steps=8,
            weight_adapter=self.adapter)
        self.wm = WorkingMemory_v60(n_latent=16, n_goals=3, buffer_len=20)
        self.habit = HabitNetwork_v60()
        self.smoother = TurnSmoother(alpha=0.35)

        # Bridge
        self.bridge = GymWorldBridge()

        # State
        self.goal_vec = np.array([0.0, 0.0, 1.0])  # start EXPLORE
        self.prev_oF = None
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

        # Patch memory: tracks food eaten per plankton patch
        self._patch_visit_counts = {}

        # Novelty detection: temporal-difference of type channels
        self._prev_typeL = None
        self._prev_typeR = None
        self._novelty_ema = 0.0

        # Per-pixel habituation (non-associative learning)
        self._habituation_L = np.zeros(400, dtype=np.float32)
        self._habituation_R = np.zeros(400, dtype=np.float32)

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

        # World model (Step 16/17)
        self._prev_z = None
        self._prev_action_ctx = None
        if self.world_model_type == "vae":
            ctx_dim = 16 if use_allostasis else 13
            self.vae_world = VAEWorldModel(
                oF_dim=800, pool_dim=64, latent_dim=16,
                state_ctx_dim=ctx_dim,
                device=str(self.device))
        elif self.world_model_type == "place_cell":
            self.place_cells = PlaceCellNetwork(n_cells=128)
            self.vae_world = None
        else:
            self.vae_world = None

        # Diagnostics (populated each step)
        self.last_diagnostics = {}

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

            # EFE-style prospect score (lower = better)
            # cost term: metabolic cost weighted by energy state
            cost_weight = 0.3 + 0.5 * urgency  # more urgent → accept higher cost
            risk_weight = 0.8 - 0.3 * urgency  # more urgent → accept more risk
            gain_weight = 0.6 + 0.4 * urgency  # more urgent → value gain more

            prospect_score = (
                cost_weight * (metabolic_cost / 20.0)  # normalized
                + risk_weight * risk
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
        # 1. Build world from gym state
        world, fish_pos, world_heading = self.bridge.build_world(env)

        # 1b. Active inference food prospection
        food_prospects = self._evaluate_food_prospects(env, fish_pos, world)

        # 2. Effective heading with eye position
        effective_heading = world_heading + self.ot.eye_pos * 0.25

        # 3. Forward pass through SNN (with depth shading)
        with torch.no_grad():
            out = self.model.forward(fish_pos, effective_heading, world,
                                     depth_shading=True)

        # Store SNN output for neural activity visualization
        self._last_snn_out = out

        # 4. Classification
        cls_logits = out["cls"]
        cls_probs = F.softmax(cls_logits, dim=1)[0].cpu().numpy()

        # 4b. Obstacle-aware correction: large rocks flood the type channel
        #     with obstacle pixels (0.75), which the classifier may confuse
        #     with enemy signal.  Count ground-truth type pixels to discount
        #     false enemy classification when rocks dominate the retina.
        typeL_raw = out["retL_full"][0, 400:].cpu().numpy()
        typeR_raw = out["retR_full"][0, 400:].cpu().numpy()
        OBSTACLE_TYPE_VAL = 0.75
        ENEMY_TYPE_VAL_CHECK = 0.5
        obs_px = (np.sum(np.abs(typeL_raw - OBSTACLE_TYPE_VAL) < 0.1)
                  + np.sum(np.abs(typeR_raw - OBSTACLE_TYPE_VAL) < 0.1))
        enemy_px = (np.sum(np.abs(typeL_raw - ENEMY_TYPE_VAL_CHECK) < 0.1)
                    + np.sum(np.abs(typeR_raw - ENEMY_TYPE_VAL_CHECK) < 0.1))
        if obs_px > 10 and enemy_px < 5:
            # Heavy obstacle presence with few real enemy pixels → false alarm
            discount = min(0.35, obs_px * 0.008)
            cls_probs[2] = max(0.02, cls_probs[2] - discount)
            cls_probs[4] += discount * 0.7   # shift mass to environment
            cls_probs[0] += discount * 0.3   # rest to nothing
            cls_probs /= cls_probs.sum() + 1e-8

        # 4c. Retinal alarm: boost enemy probability from direct pixel evidence
        #     (must run BEFORE goal selection at step 9 so FLEE can trigger)
        ENEMY_TYPE_VAL = 0.5
        enemy_px_L = np.sum(np.abs(typeL_raw - ENEMY_TYPE_VAL) < 0.1)
        enemy_px_R = np.sum(np.abs(typeR_raw - ENEMY_TYPE_VAL) < 0.1)
        self._enemy_pixels_total = int(enemy_px_L + enemy_px_R)
        self._enemy_pixels_L = int(enemy_px_L)
        self._enemy_pixels_R = int(enemy_px_R)
        if self._enemy_pixels_total >= 3:
            alarm_boost = min(0.5, self._enemy_pixels_total * 0.05)
            cls_probs[2] = min(1.0, cls_probs[2] + alarm_boost)
            cls_probs /= cls_probs.sum() + 1e-8

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
        energy = getattr(env, "fish_energy", 100.0)
        prev_speed = self.last_diagnostics.get("speed", 0.5)
        pred_dist_px = math.sqrt(
            (env.fish_x - env.pred_x) ** 2 + (env.fish_y - env.pred_y) ** 2)
        allo_state = None
        if self.allostasis is not None:
            allo_state = self.allostasis.step(energy, prev_speed, pred_dist_px)
            self.dopa_sys.beta = self.allostasis.modulate_dopamine_gain(
                self.dopa_sys.beta)

        # 7b2. Amygdala fear response — threat arousal from retinal + proximity
        threat_arousal = 0.0
        if self.amygdala is not None:
            stress_val = allo_state["stress"] if allo_state else 0.0
            threat_arousal = self.amygdala.step(
                self._enemy_pixels_total, pred_dist_px, stress_val)
            # Boost classifier p_enemy by threat arousal
            if threat_arousal > 0.05:
                cls_probs[2] = min(1.0, cls_probs[2] + 0.3 * threat_arousal)
                cls_probs /= cls_probs.sum() + 1e-8
            # Feed arousal back into allostatic stress
            if self.allostasis is not None:
                self.allostasis.stress = min(
                    1.0, self.allostasis.stress + 0.1 * threat_arousal)

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
            _energy = getattr(env, "fish_energy", 100.0)
            state_ctx = np.array([
                fish_pos[0] / 200.0,           # proprioceptive: x [-1, 1]
                fish_pos[1] / 150.0,           # proprioceptive: y [-1, 1]
                world_heading / np.pi,          # proprioceptive: heading [-1, 1]
                _energy / 100.0,                # interoceptive: energy [0, 1]
                dopa,                           # neuromodulatory: dopamine [0, 1]
                np.clip(rpe, -1, 1),            # neuromodulatory: RPE [-1, 1]
                pi_OT,                          # precision: optic tectum [0, 1]
                pi_PC,                          # precision: pred coding [0, 1]
                cls_probs[1],                   # exteroceptive: p_food
                cls_probs[2],                   # exteroceptive: p_enemy
                cls_probs[4],                   # exteroceptive: p_environ
                cms,                            # cognitive: cross-modal surprise
                np.clip(F_visual, 0, 2) / 2,    # cognitive: visual free energy [0,1]
            ], dtype=np.float32)

            # Extend state_ctx with allostasis (13 → 16 dims)
            if self.allostasis is not None:
                state_ctx = np.concatenate([
                    state_ctx,
                    self.allostasis.get_state_ctx_extension()])

            z_np, z_mu = self.vae_world.encode(out["oF"], state_ctx)
            self.vae_world.train_step(out["oF"], state_ctx)

            if self._prev_z is not None and self._prev_action_ctx is not None:
                self.vae_world.update_transition(
                    self._prev_z, self._prev_action_ctx, z_np)

            last_act = np.array([
                self.last_diagnostics.get("turn_rate", 0.0),
                self.last_diagnostics.get("speed", 0.5)],
                dtype=np.float32)
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
            gym_pos = np.array([env.fish_x, env.fish_y], dtype=np.float32)
            prev_turn = self.last_diagnostics.get("turn_rate", 0.0)
            prev_spd = self.last_diagnostics.get("speed", 0.5)
            self.place_cells.path_integrate(prev_turn, prev_spd)
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
        #     Note: read current bonus from VAE/place cell (set in 8.5/8.6).
        #     If no world model is active, start from zero to avoid accumulation.
        if self.allostasis is not None:
            allo_bias = self.allostasis.get_goal_prior_bias()
            if self.vae_world is not None or hasattr(self, 'place_cells'):
                current_bonus = self.goal_policy._external_efe_bonus.copy()
            else:
                current_bonus = np.zeros(3, dtype=np.float64)
            self.goal_policy.set_plan_bonus(current_bonus + allo_bias)

        # 9. Goal policy selection
        _hunger = self.allostasis.hunger if self.allostasis else 0.0
        _hunger_err = self.allostasis.hunger_error if self.allostasis else 0.0
        choice, self.goal_vec, posterior, confidence, efe_vec = \
            self.goal_policy.step(
                cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual,
                self.wm.get_mean(),
                hunger=_hunger, hunger_error=_hunger_err)

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
            goal_to_behavior(effective_goal, cls_probs, posterior, confidence)

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
        obs_total = obs_px_L + obs_px_R
        if obs_total > 5:
            obs_repulsion = -0.6 * (obs_px_R - obs_px_L) / obs_total
            raw_turn = raw_turn + obs_repulsion

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

        # 17. Precision update (freeze gamma during sleep — Step 23)
        oF = out["oF"]
        _precision_frozen = (sleep_state is not None
                             and sleep_state["is_sleeping"]
                             and sleep_state["precision_freeze"])
        if self.prev_oF is not None and not _precision_frozen:
            error_OT = oF - self.prev_oF
            self.model.prec_OT.update_precision(error_OT)
            self.model.prec_PC.update_precision(
                torch.tensor([[F_visual]], device=self.device))
            with torch.no_grad():
                self.model.prec_OT.gamma.data += 0.008 * (dopa - 0.5)
                self.model.prec_PC.gamma.data += 0.008 * (dopa - 0.5)
        self.prev_oF = oF.clone()

        # 18. Compute gym action
        # Negate turn: brain computes in world coords (y-up), gym uses y-down
        brain_turn = turn + 0.03 * bg_gate + 0.02 * eye_pos

        # Boundary avoidance: steer toward center when near walls
        margin = 80
        wall_urgency = 0.0
        if env.fish_x < margin:
            wall_urgency = max(wall_urgency, (margin - env.fish_x) / margin)
        if env.fish_x > env.arena_w - margin:
            wall_urgency = max(wall_urgency,
                               (env.fish_x - (env.arena_w - margin)) / margin)
        if env.fish_y < margin:
            wall_urgency = max(wall_urgency, (margin - env.fish_y) / margin)
        if env.fish_y > env.arena_h - margin:
            wall_urgency = max(wall_urgency,
                               (env.fish_y - (env.arena_h - margin)) / margin)

        wall_turn = 0.0
        if wall_urgency > 0.01:
            center_dx = env.arena_w * 0.5 - env.fish_x
            center_dy = env.arena_h * 0.5 - env.fish_y
            angle_to_center = math.atan2(center_dy, center_dx)
            angle_diff = angle_to_center - env.fish_heading
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            # Strong push: up to 0.8 at wall, overriding brain turn
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

        # Reduce speed when low energy
        energy = getattr(env, "fish_energy", 100.0)
        if energy < 20:
            speed *= 0.5 + 0.5 * (energy / 20.0)

        # Fatigue speed cap (Step 18)
        if self.allostasis is not None:
            speed = min(speed, self.allostasis.get_speed_cap())

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
            effective_goal == GOAL_FLEE or self._flee_burst_steps > 0)

        action = np.array([turn_rate, speed], dtype=np.float32)

        # Store action context for VAE transition training (Step 16)
        if self.vae_world is not None:
            goal_oh = np.zeros(3, dtype=np.float32)
            goal_oh[effective_goal] = 1.0
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

        # Sleep diagnostics (Step 23)
        if self.sleep_regulator is not None:
            self.last_diagnostics["sleep"] = \
                self.sleep_regulator.get_diagnostics()

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

            self.last_diagnostics["critic_value"] = critic_value
            self.last_diagnostics["td_error"] = td_err if td_err is not None else 0.0
            self.last_diagnostics["efe_deltas"] = self.adapter.get_deltas() if self.adapter else {}
            self.last_diagnostics["critic_update_count"] = self.critic.get_stats()["update_count"]

            self._prev_wm_state = wm_state

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
            pred_dist = 150.0  # default safe distance
            fish_x, fish_y = info.get("fish_pos", (0, 0))
            pred_x = getattr(env, "pred_x", fish_x)
            pred_y = getattr(env, "pred_y", fish_y)
            dx = fish_x - pred_x
            dy = fish_y - pred_y
            pred_dist = math.sqrt(dx * dx + dy * dy)
            self.vae_world.update_memory(self._prev_z, eaten, pred_dist)

        # Place cell memory update (Step 17)
        if hasattr(self, 'place_cells') and env is not None:
            fish_x, fish_y = info.get("fish_pos", (0, 0))
            gym_pos = np.array([fish_x, fish_y], dtype=np.float32)
            pred_x = getattr(env, "pred_x", fish_x)
            pred_y = getattr(env, "pred_y", fish_y)
            dx = fish_x - pred_x
            dy = fish_y - pred_y
            pred_dist = math.sqrt(dx * dx + dy * dy)
            risk_signal = max(0.0, 1.0 - pred_dist / 150.0)
            self.place_cells.update(gym_pos, float(eaten), risk_signal)

        # Save reward/done for next act() call which pushes the transition
        if self.critic is not None:
            self._prev_reward = reward
            self._prev_done = done

    def reset(self):
        """Reset all brain modules."""
        self.model.reset()
        self.dopa_sys.reset()
        self.bg.reset()
        self.ot.reset()
        self.thal.reset()
        self.goal_policy.reset()
        self.wm.reset()
        self.habit.reset()
        self.smoother.smoothed = 0.0
        self.goal_vec = np.array([0.0, 0.0, 1.0])
        self.prev_oF = None
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
        if self.critic is not None:
            self.critic.reset()
        if self.adapter is not None:
            self.adapter.reset()
        if self.vae_world is not None:
            self.vae_world.reset()
            self._prev_z = None
            self._prev_action_ctx = None
        if hasattr(self, 'place_cells'):
            self.place_cells.reset()
        if self.allostasis is not None:
            self.allostasis.reset()
        if self.amygdala is not None:
            self.amygdala.reset()
        if self.sleep_regulator is not None:
            self.sleep_regulator.reset()
