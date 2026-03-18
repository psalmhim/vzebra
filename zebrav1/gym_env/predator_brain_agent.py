"""
Predator Brain Agent — SNN-based predator with goal-directed hunting.

Mirrors the fish BrainAgent architecture but with predator-specific goals:
  HUNT (0): chase the fish using visual tracking
  PATROL (1): search when fish not visible
  AMBUSH (2): wait near food clusters to intercept fish
  REST (3): recover stamina when exhausted

Reuses existing brain modules (SNN, dopamine, BG, OT, goal selector, WM).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from zebrav1.brain.zebrafish_snn import ZebrafishSNN
from zebrav1.brain.dopamine import DopamineSystem
from zebrav1.brain.basal_ganglia import BasalGanglia
from zebrav1.brain.optic_tectum import OpticTectum
from zebrav1.brain.goal_policy import SpikingGoalSelector
from zebrav1.brain.working_memory import SpikingWorkingMemory
from zebrav1.brain.device_util import get_device
from zebrav1.world.world_env import WorldEnv
from zebrav1.tests.step1_vision_pursuit import TurnSmoother

GOAL_HUNT = 0
GOAL_PATROL = 1
GOAL_AMBUSH = 2
GOAL_REST = 3
GOAL_NAMES = ["HUNT", "PATROL", "AMBUSH", "REST"]

PREY_TYPE_VAL = 0.38


class PredatorWorldBridge:
    """Translates gym pixel coords to WorldEnv for predator's perspective.

    Key difference from fish's GymWorldBridge: the fish becomes prey,
    and the predator itself is not added to enemies.
    """

    def __init__(self, arena_w=800, arena_h=600):
        self.arena_w = arena_w
        self.arena_h = arena_h

    def gym_to_world_pos(self, gx, gy):
        wx = (gx / self.arena_w) * 400 - 200
        wy = -((gy / self.arena_h) * 300 - 150)
        return wx, wy

    def gym_to_world_heading(self, heading):
        return -heading

    def build_world(self, env):
        """Build WorldEnv from predator's perspective.

        - Fish (the prey target) goes into world.prey
        - Colleagues also go into world.prey (multiple hunt targets)
        - Food items go into world.foods (predator can see them for ambush)
        - No enemies (predator is the apex predator)
        """
        world = WorldEnv(xmin=-200, xmax=200, ymin=-150, ymax=150,
                         n_food=0, n_enemies=0, n_colleagues=0)

        # Fish is the predator's primary prey
        fx, fy = self.gym_to_world_pos(env.fish_x, env.fish_y)
        world.prey.append((fx, fy))

        # Colleagues are also prey targets (multiple fish to hunt)
        for c in getattr(env, 'colleagues', []):
            wx, wy = self.gym_to_world_pos(c["x"], c["y"])
            world.prey.append((wx, wy))

        # Food items visible (for ambush strategy)
        for food in env.foods:
            food_x, food_y = food[0], food[1]
            sz = food[2] if len(food) > 2 else "small"
            wx, wy = self.gym_to_world_pos(food_x, food_y)
            world.foods.append({"x": wx, "y": wy, "size": sz})

        # Obstacles
        rock_formations = getattr(env, 'rock_formations', None)
        if rock_formations:
            scale_x = 400.0 / self.arena_w
            scale_y = 300.0 / self.arena_h
            for rock in rock_formations:
                for aabb in rock["aabbs"]:
                    wx, wy = self.gym_to_world_pos(aabb["x"], aabb["y"])
                    hw_world = aabb["hw"] * scale_x
                    hh_world = aabb["hh"] * scale_y
                    world.add_obstacle(wx, wy, hw_world, hh_world)
        else:
            for obs in getattr(env, 'obstacles', []):
                wx, wy = self.gym_to_world_pos(obs["x"], obs["y"])
                hw = obs["r"] / 2.0
                hh = obs["r"] * 0.7 / 2.0
                world.add_obstacle(wx, wy, hw, hh)

        # Predator position/heading
        pred_pos = np.array(self.gym_to_world_pos(env.pred_x, env.pred_y))
        world_heading = self.gym_to_world_heading(env.pred_heading)

        return world, pred_pos, world_heading


class PredatorBrainAgent:
    """SNN-based predator brain with goal-directed hunting behavior."""

    def __init__(self, device="auto"):
        self.device = get_device(device)

        # SNN model (same architecture, fresh weights — no classifier needed)
        self.model = ZebrafishSNN(device=self.device)
        self.model.reset()
        self.model.eval()

        # Brain modules
        self.dopa_sys = DopamineSystem(alpha=0.08, beta=2.5, decay=0.97)
        self.bg = BasalGanglia(mode="exploratory")
        self.ot = OpticTectum()
        self.goal_policy = SpikingGoalSelector(
            n_goals=4, beta=2.0, device=str(self.device))
        self.wm = SpikingWorkingMemory(
            n_goals=4, buffer_len=20, device=str(self.device))
        self.smoother = TurnSmoother(alpha=0.35)

        # Bridge
        self.bridge = PredatorWorldBridge()

        # State
        self.goal_vec = np.array([0.0, 1.0, 0.0, 0.0])  # start PATROL
        self.prev_oF = None
        self._catch_buffer = 0
        self._step_count = 0
        self._prey_pixels_total = 0
        self._prey_pixels_L = 0
        self._prey_pixels_R = 0

        # Diagnostics
        self.last_diagnostics = {}

    def act(self, env):
        """Run predator brain pipeline and produce [turn_rate, speed_mod].

        Args:
            env: the gym environment instance

        Returns:
            turn_rate: float in [-1, 1]
            speed: float in [0, 1]
        """
        self._step_count += 1

        # 1. Build world from predator's viewpoint
        world, pred_pos, world_heading = self.bridge.build_world(env)

        # 2. Effective heading with eye position
        effective_heading = world_heading + self.ot.eye_pos * 0.25

        # 3. SNN forward pass (extended vision range for apex predator)
        #    depth_scale=160 so distant prey doesn't fade too quickly
        with torch.no_grad():
            out = self.model.forward(pred_pos, effective_heading, world,
                                     depth_shading=True, depth_scale=160.0,
                                     max_dist=400)

        # 4. Classification (repurposed: detect prey instead of food)
        cls_probs = F.softmax(out["cls"], dim=1)[0].cpu().numpy()

        # 4b. Prey detection from retinal type channel
        typeL_raw = out["retL_full"][0, 400:].cpu().numpy()
        typeR_raw = out["retR_full"][0, 400:].cpu().numpy()

        prey_px_L = np.sum(np.abs(typeL_raw - PREY_TYPE_VAL) < 0.1)
        prey_px_R = np.sum(np.abs(typeR_raw - PREY_TYPE_VAL) < 0.1)
        self._prey_pixels_total = int(prey_px_L + prey_px_R)
        self._prey_pixels_L = int(prey_px_L)
        self._prey_pixels_R = int(prey_px_R)

        # Boost "prey detected" signal in cls_probs
        # Map prey detection onto cls index 1 (food class in original SNN)
        if self._prey_pixels_total >= 2:
            prey_boost = min(0.7, self._prey_pixels_total * 0.06)
            cls_probs[1] = min(1.0, cls_probs[1] + prey_boost)
            cls_probs /= cls_probs.sum() + 1e-8

        # 5. Free energy
        F_visual = self.model.compute_free_energy()

        # 6. Dopamine: reward from tracking prey + catching
        oL_mean = float(out["oL"].abs().mean())
        oR_mean = float(out["oR"].abs().mean())
        dopa, rpe, valL, valR = self.dopa_sys.step(
            F_visual, oL_mean, oR_mean, eaten=self._catch_buffer)
        self._catch_buffer = 0

        # 7. Precision
        pi_OT = out["pi_OT"]
        pi_PC = out["pi_PC"]

        # 8. Working memory
        cms = 0.1 * abs(F_visual)
        m_out, alpha_eff, cls_summary = self.wm.step(
            cls_probs, self.goal_vec, dopa, cms, F_visual, pi_OT, pi_PC)

        # 9. Goal selection with predator-specific drives
        p_prey = cls_probs[1]  # prey probability
        stamina = env.pred_stamina
        hunger = env.pred_hunger

        # Construct predator-adapted input for goal selector
        # Override the EFE computation with direct drives
        prey_visible = self._prey_pixels_total >= 1

        # Compute proximity from retinal intensity
        intL = out["retL_full"][0, :400].cpu().numpy()
        intR = out["retR_full"][0, :400].cpu().numpy()
        prey_mask_L = np.abs(typeL_raw - PREY_TYPE_VAL) < 0.1
        prey_mask_R = np.abs(typeR_raw - PREY_TYPE_VAL) < 0.1
        prey_int_vals = np.concatenate([
            intL[prey_mask_L], intR[prey_mask_R]])
        prey_intensity = (float(np.mean(prey_int_vals))
                          if len(prey_int_vals) > 0 else 0.0)

        # Emergency overrides for goal selector
        # HUNT: when prey is visible and we have stamina
        # REST: when stamina critically low
        # AMBUSH: when hungry but prey not visible and food clusters exist
        hunt_drive = 0.0
        if prey_visible and stamina > 0.1:
            hunt_drive = 5.0 + 3.0 * prey_intensity
        elif prey_visible:
            hunt_drive = 3.0  # still hunt even when tired

        rest_drive = 0.0
        if stamina < 0.05:
            rest_drive = 6.0
        elif stamina < 0.15:
            rest_drive = 2.0

        ambush_drive = 0.0
        if not prey_visible and hunger > 0.5:
            food_px_L = np.sum(np.abs(typeL_raw - 1.0) < 0.1)
            food_px_R = np.sum(np.abs(typeR_raw - 1.0) < 0.1)
            if food_px_L + food_px_R > 5:
                ambush_drive = 2.0 + hunger

        # Strong patrol drive when no prey visible (active explorer)
        patrol_drive = 0.0
        if not prey_visible:
            patrol_drive = 3.0 + 1.0 * hunger  # hungrier = more motivated

        # Inject drives as external EFE bonus (lower = preferred)
        efe_bonus = np.array([
            -hunt_drive,    # HUNT: lower EFE = preferred when prey visible
            -patrol_drive,  # PATROL: active when no prey (never idle)
            -ambush_drive,  # AMBUSH: active near food when hungry
            -rest_drive,    # REST: active when exhausted
        ])
        self.goal_policy.set_plan_bonus(efe_bonus)

        # Step goal selector
        choice, self.goal_vec, posterior, confidence, efe_vec = \
            self.goal_policy.step(
                cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual,
                self.wm.get_mean(),
                hunger=hunger, hunger_error=0.0,
                pred_facing_score=0.0, ttc=999.0,
                energy_ratio=stamina)

        # 10. Goal-conditioned behavior
        approach_gain, speed_mod, turn_bias = self._goal_to_behavior(
            choice, prey_visible, prey_intensity, stamina, hunger,
            typeL_raw, typeR_raw)

        # 11. Retinal turn signal — steer toward prey
        retL = out["retL"]
        retR = out["retR"]
        retR_sum = float(retR.sum())
        retL_sum = float(retL.sum())
        total = retR_sum + retL_sum + 1e-8
        raw_turn = (retR_sum - retL_sum) / total

        # 11b. Prey-directed turn (when hunting)
        if choice == GOAL_HUNT and self._prey_pixels_total >= 1:
            prey_turn = (prey_px_R - prey_px_L) / (self._prey_pixels_total + 1e-8)
            raw_turn = raw_turn * 0.3 + prey_turn * 0.7

        # 11c. Food cluster approach (when ambushing)
        if choice == GOAL_AMBUSH:
            food_px_L = np.sum(np.abs(typeL_raw - 1.0) < 0.1)
            food_px_R = np.sum(np.abs(typeR_raw - 1.0) < 0.1)
            food_total = food_px_L + food_px_R
            if food_total > 3:
                food_turn = (food_px_R - food_px_L) / (food_total + 1e-8)
                raw_turn = raw_turn * 0.4 + food_turn * 0.6

        # 11d. Obstacle repulsion
        obs_px_L = float(np.sum(np.abs(typeL_raw - 0.75) < 0.1))
        obs_px_R = float(np.sum(np.abs(typeR_raw - 0.75) < 0.1))
        obs_total = obs_px_L + obs_px_R
        if obs_total > 5:
            obs_repulsion = -0.6 * (obs_px_R - obs_px_L) / obs_total
            raw_turn += obs_repulsion

        # 12. Smoothed turn
        turn = self.smoother.step(raw_turn * approach_gain)

        # 13. BG gating
        valL_eff = valL - 0.1 * turn
        valR_eff = valR + 0.1 * turn
        bg_gate = self.bg.step(valL_eff, valR_eff, dopa, rpe)

        # 14. Eye position
        eye_pos = self.ot.step(valL_eff, valR_eff, F_visual, bg_gate, dopa)

        # 14b. Prey saccade (lock onto prey)
        if self._prey_pixels_total >= 5:
            prey_dir = 1.0 if self._prey_pixels_R > self._prey_pixels_L else -1.0
            self.ot.trigger_saccade(prey_dir, magnitude=0.5)

        # 15. Precision update
        oF = out["oF"]
        if self.prev_oF is not None:
            error_OT = oF - self.prev_oF
            self.model.prec_OT.update_precision(error_OT)
            self.model.prec_PC.update_precision(
                torch.tensor([[F_visual]], device=self.device))
        self.prev_oF = oF.clone()

        # 16. Compute action
        brain_turn = turn + 0.03 * bg_gate + 0.02 * eye_pos + turn_bias

        # Boundary avoidance
        margin = 60
        wall_turn = 0.0
        wall_urgency = 0.0
        px, py = env.pred_x, env.pred_y
        if px < margin:
            wall_urgency = max(wall_urgency, (margin - px) / margin)
        if px > env.arena_w - margin:
            wall_urgency = max(wall_urgency,
                               (px - (env.arena_w - margin)) / margin)
        if py < margin:
            wall_urgency = max(wall_urgency, (margin - py) / margin)
        if py > env.arena_h - margin:
            wall_urgency = max(wall_urgency,
                               (py - (env.arena_h - margin)) / margin)

        if wall_urgency > 0.01:
            center_dx = env.arena_w * 0.5 - px
            center_dy = env.arena_h * 0.5 - py
            angle_to_center = math.atan2(center_dy, center_dx)
            angle_diff = angle_to_center - env.pred_heading
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            wall_turn = wall_urgency * 0.8 * np.sign(angle_diff)

        brain_weight = max(0.2, 1.0 - wall_urgency)
        turn_rate = float(np.clip(
            -brain_turn * brain_weight + wall_turn, -1.0, 1.0))

        speed = float(np.clip(
            speed_mod * (0.85 + 0.3 * dopa), 0.0, 1.0))

        # Stamina modulates speed (softer penalty during hunt)
        if choice == GOAL_HUNT:
            speed *= max(0.4, stamina)  # hunt at 40% even when exhausted
        else:
            speed *= max(0.1, stamina)

        # Hunger boost (hungrier = much more aggressive)
        speed *= 1.0 + 0.4 * hunger

        speed = float(np.clip(speed, 0.0, 1.0))

        # 17. Diagnostics
        self.last_diagnostics = {
            "goal": choice,
            "goal_name": GOAL_NAMES[choice],
            "confidence": confidence,
            "prey_visible": prey_visible,
            "prey_pixels": self._prey_pixels_total,
            "prey_intensity": prey_intensity,
            "dopa": dopa,
            "rpe": rpe,
            "stamina": stamina,
            "hunger": hunger,
            "speed": speed,
            "F_visual": F_visual,
        }

        # Hebbian learning
        _rpe = self.dopa_sys.rpe
        _dopa = self.dopa_sys.dopa
        if hasattr(self.wm, 'learn'):
            self.wm.learn(_rpe, _dopa)
        if hasattr(self.goal_policy, 'learn'):
            self.goal_policy.learn(_rpe, _dopa, 0.0)

        return turn_rate, speed

    def _goal_to_behavior(self, goal, prey_visible, prey_intensity,
                          stamina, hunger, typeL, typeR):
        """Convert goal to motor parameters.

        Returns:
            approach_gain: float — turn gain multiplier
            speed_mod: float — speed multiplier
            turn_bias: float — additional turn bias
        """
        if goal == GOAL_HUNT:
            # Aggressive pursuit: max gain, sprint speed
            approach_gain = 2.5 + 1.0 * prey_intensity
            speed_mod = 1.0
            turn_bias = 0.0
            return approach_gain, speed_mod, turn_bias

        elif goal == GOAL_PATROL:
            # Active searching: high speed, wide scanning sweeps
            approach_gain = 1.2
            speed_mod = 0.85
            # Wide sinusoidal scanning + periodic sharp turns for area coverage
            scan_phase = math.sin(self._step_count * 0.05) * 0.35
            # Sharp direction change every ~80 steps
            if self._step_count % 80 < 10:
                scan_phase += 0.5 * math.sin(self._step_count * 0.3)
            turn_bias = scan_phase
            return approach_gain, speed_mod, turn_bias

        elif goal == GOAL_AMBUSH:
            # Move to food cluster and wait
            approach_gain = 0.8
            speed_mod = 0.4
            turn_bias = 0.0
            return approach_gain, speed_mod, turn_bias

        elif goal == GOAL_REST:
            # Minimal movement, recover stamina
            approach_gain = 0.2
            speed_mod = 0.1
            turn_bias = 0.0
            return approach_gain, speed_mod, turn_bias

        # Fallback
        return 0.5, 0.4, 0.0

    def update_post_step(self, caught):
        """Feed catch events to dopamine system.

        Args:
            caught: bool — whether the predator caught the fish this step
        """
        if caught:
            self._catch_buffer = 1

    def reset(self):
        """Reset brain for a new episode."""
        self.model.reset()
        self.dopa_sys.reset()
        self.bg.reset()
        self.ot.reset()
        self.goal_policy.reset()
        self.wm.reset()
        self.smoother.smoothed = 0.0
        self.goal_vec = np.zeros(4)
        self.goal_vec[GOAL_PATROL] = 1.0
        self.prev_oF = None
        self._step_count = 0
        self._catch_buffer = 0
        self._prey_pixels_total = 0
        self._prey_pixels_L = 0
        self._prey_pixels_R = 0
        self.last_diagnostics = {}
