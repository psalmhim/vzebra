"""
ZebrafishBrainV2: integrates all modules.
Interface-compatible with v1 BrainAgent for direct comparison.
Uses v1 EFE policy selection (preserved) with v2 spiking substrate.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.retina import RetinaV2
from zebrav2.brain.tectum import Tectum
from zebrav2.brain.thalamus import Thalamus
from zebrav2.brain.pallium import Pallium
from zebrav2.brain.basal_ganglia import BasalGanglia
from zebrav2.brain.reticulospinal import ReticulospinalSystem
from zebrav2.brain.neuromod import NeuromodSystem
from zebrav2.brain.place_cells import ThetaPlaceCells
from zebrav2.brain.plasticity import FeedbackPELearning
from zebrav2.brain.amygdala import SpikingAmygdalaV2
from zebrav2.brain.classifier import ClassifierV2
from zebrav2.brain.predator_model import PredatorModel
from zebrav2.brain.allostasis import AllostaticRegulator
from zebrav2.brain.internal_model import InternalWorldModel
from zebrav2.brain.goal_selector import SpikingGoalSelector
from zebrav2.brain.cerebellum import SpikingCerebellum
from zebrav2.brain.habenula import SpikingHabenula
from zebrav2.brain.predictive_net import SpikingPredictiveNet
from zebrav2.brain.rl_critic import SpikingCritic
from zebrav2.brain.habit_network import SpikingHabitNet
from zebrav2.brain.interoception import SpikingInsularCortex
from zebrav2.brain.spinal_cpg import SpinalCPG
from zebrav2.brain.vae_world_model import VAEWorldModelV2
from zebrav2.brain.lateral_line import SpikingLateralLine
from zebrav2.brain.olfaction import SpikingOlfaction
from zebrav2.brain.working_memory import SpikingWorkingMemory
from zebrav2.brain.vestibular import SpikingVestibular
from zebrav2.brain.proprioception import SpikingProprioception
from zebrav2.brain.color_vision import SpikingColorVision
from zebrav2.brain.circadian import SpikingCircadian
from zebrav2.brain.sleep_wake import SpikingSleepWake

GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, GOAL_SOCIAL = 0, 1, 2, 3

class ZebrafishBrainV2(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        # --- Core modules ---
        self.retina    = RetinaV2(device)
        self.tectum    = Tectum(device)
        self.thalamus  = Thalamus(device)
        self.pallium   = Pallium(device)
        self.bg        = BasalGanglia(device)
        self.rs        = ReticulospinalSystem(device)
        self.neuromod  = NeuromodSystem(device)
        self.place     = ThetaPlaceCells(device=device)
        # PE-driven feedback learning (W_FB is nn.Linear; pass its weight parameter)
        self.fb_learner = FeedbackPELearning(self.pallium.W_FB.weight, device=device)
        self.amygdala  = SpikingAmygdalaV2(device=device)
        self.classifier = ClassifierV2(device=device)
        # Auto-load trained classifier weights if available
        _cls_path = os.path.join(os.path.dirname(__file__), '..', 'weights', 'classifier_v2.pt')
        if os.path.exists(_cls_path):
            try:
                self.classifier.load_state_dict(
                    torch.load(_cls_path, map_location=device, weights_only=True))
            except Exception:
                pass
        self.pred_model = PredatorModel()
        self.allostasis = AllostaticRegulator()
        self.world_model = InternalWorldModel()
        self.goal_selector = SpikingGoalSelector(device=device)
        # --- New SNN modules ---
        self.cerebellum = SpikingCerebellum(device=device)
        self.habenula = SpikingHabenula(device=device)
        self.predictive = SpikingPredictiveNet(device=device)
        self.critic = SpikingCritic(device=device)
        self.habit = SpikingHabitNet(device=device)
        self.insula = SpikingInsularCortex(device=device)
        self.cpg = SpinalCPG(device=device)
        # Tectum all_e size = sum of 4 layer E neuron counts
        _tect_e_total = (self.tectum.sfgs_b.n_e + self.tectum.sfgs_d.n_e
                         + self.tectum.sgc.n_e + self.tectum.so.n_e)
        self.vae = VAEWorldModelV2(tectum_dim=_tect_e_total, device=device)
        self._z_prev = None  # for VAE transition training
        self._last_action_ctx = None
        # --- Medium/Low priority modules ---
        self.lateral_line_mod = SpikingLateralLine(device=device)
        self.olfaction = SpikingOlfaction(device=device)
        self.working_mem = SpikingWorkingMemory(device=device)
        self.vestibular = SpikingVestibular(device=device)
        self.proprioception = SpikingProprioception(device=device)
        self.color_vision = SpikingColorVision(device=device)
        self.circadian = SpikingCircadian(device=device)
        self.sleep_wake = SpikingSleepWake(device=device)
        # EFE state (adapted from v1)
        self.goal_probs = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
        self.current_goal = GOAL_EXPLORE
        self.goal_persistence = 0
        # Internal state
        self.energy = 100.0
        self.amygdala_alpha = 0.0
        self.cms = 0.0
        # Smoother for flee turn
        self._smoother = 0.0
        self._pred_state = 'PATROL'
        self._pred_dist_gt = 999.0
        self._enemy_pixels = 0.0
        # Retinal features (v1 compatibility)
        self._retinal_features = {}
        self._forage_lock_steps = 0
        self._forage_no_progress_steps = 0
        # Stuck detection
        self._last_food_count = 0
        self._no_food_steps = 0
        self._force_explore_steps = 0
        # Step counter for sinusoidal explore
        self._step_count = 0
        self._last_fish_pos = (400.0, 300.0)
        self._last_speed = 1.0

    def step(self, obs, env=None) -> dict:
        """
        Main step. obs: gym observation (unused — we read env directly).
        Returns: {'turn': float, 'speed': float, 'goal': int}
        """
        # === 1. SENSORY (retina + tectum) ===
        if env is None:
            return {'turn': 0.0, 'speed': 1.0, 'goal': self.current_goal}

        # Get retinal input from env (same as v1)
        L = torch.zeros(800, device=self.device)
        R = torch.zeros(800, device=self.device)
        if hasattr(env, 'brain_L') and env.brain_L is not None:
            L = torch.tensor(env.brain_L, dtype=torch.float32, device=self.device)
            R = torch.tensor(env.brain_R, dtype=torch.float32, device=self.device)
        # Get entity info from env
        enemy_px = 0.0
        if hasattr(env, '_enemy_pixels_total'):
            enemy_px = float(env._enemy_pixels_total)
        entity_info = {'enemy': enemy_px / 15.0}
        rgc_out = self.retina(L, R, entity_info)

        # Classify scene
        cls_out = self.classifier.classify(L, R)
        self._cls_probs = cls_out['probs']

        # Fish position: from place cell centroid (no GT reads)
        pc_rate = self.place.rate
        if pc_rate.sum() > 0.01:
            w = pc_rate / (pc_rate.sum() + 1e-8)
            fish_pos = (float((w * self.place.cx).sum()),
                        float((w * self.place.cy).sum()))
        else:
            fish_pos = self._last_fish_pos

        # Lateral line: mechanoreceptive proximity sense (not vision)
        # Detects water displacement from nearby moving objects — biologically correct
        ll_dist = 999.0
        if hasattr(env, 'pred_x') and hasattr(env, 'fish_x'):
            dx = env.pred_x - env.fish_x
            dy = env.pred_y - env.fish_y
            ll_dist = math.sqrt(dx * dx + dy * dy)
            # Lateral line only detects within ~150px (body-length range)
            if ll_dist < 150:
                ll_dist += np.random.normal(0, 15)  # noisy estimate
            else:
                ll_dist = 999.0  # beyond detection range

        # Spiking lateral line module
        ll_out = self.lateral_line_mod(
            fish_x=getattr(env, 'fish_x', 400), fish_y=getattr(env, 'fish_y', 300),
            fish_heading=getattr(env, 'fish_heading', 0.0),
            pred_x=getattr(env, 'pred_x', -999), pred_y=getattr(env, 'pred_y', -999),
            pred_vx=self.pred_model.vx, pred_vy=self.pred_model.vy)
        if ll_out['dist'] < 999:
            ll_dist = ll_out['dist']

        # Olfaction
        olf_out = self.olfaction(
            fish_x=getattr(env, 'fish_x', 400), fish_y=getattr(env, 'fish_y', 300),
            fish_heading=getattr(env, 'fish_heading', 0.0),
            foods=getattr(env, 'foods', []),
            pred_dist=ll_dist)

        # Predator model: predict → retinal update → query
        self.pred_model.predict()
        # Retinal features for predator tracking
        type_L = L[400:]
        type_R = R[400:]
        enemy_mask_L = (torch.abs(type_L - 0.5) < 0.1).float()
        enemy_mask_R = (torch.abs(type_R - 0.5) < 0.1).float()
        enemy_px_L = float(enemy_mask_L.sum())
        enemy_px_R = float(enemy_mask_R.sum())
        # Lateral bias: which eye sees more enemy
        enemy_lateral_raw = (enemy_px_L - enemy_px_R) / (enemy_px_L + enemy_px_R + 1e-8)
        # Temporal EMA smoothing to prevent oscillation
        if not hasattr(self, '_enemy_lateral_ema'):
            self._enemy_lateral_ema = 0.0
        self._enemy_lateral_ema = 0.3 * enemy_lateral_raw + 0.7 * self._enemy_lateral_ema
        enemy_lateral = self._enemy_lateral_ema
        # Intensity: mean enemy pixel brightness
        int_L = L[:400]
        int_R = R[:400]
        enemy_int = float((int_L * enemy_mask_L).sum() + (int_R * enemy_mask_R).sum()) / (enemy_px + 1e-8)
        fish_heading = getattr(env, 'fish_heading', 0.0)  # heading still from env (proprioception)
        self.pred_model.update_retinal(
            enemy_px=enemy_px, enemy_lateral_bias=enemy_lateral,
            enemy_intensity=enemy_int, fish_pos=fish_pos,
            fish_heading=fish_heading, step=self._step_count)
        pred_dist = self.pred_model.get_pred_dist(fish_pos)
        # Fuse with lateral line (most reliable at close range)
        if ll_dist < 999:
            pred_dist = min(pred_dist, ll_dist)
        # Infer pred_state from model intent + lateral line
        if ll_dist < 80 or self.pred_model.intent > 0.6:
            pred_state = 'HUNT'
        elif ll_dist < 150 or self.pred_model.intent > 0.3:
            pred_state = 'STALK'
        else:
            pred_state = 'PATROL'
        self._pred_state = pred_state
        self._pred_dist_gt = pred_dist
        self._enemy_pixels = enemy_px

        # Allostasis update
        allo = self.allostasis.step(
            energy=self.energy, speed=getattr(self, '_last_speed', 1.0),
            pred_dist=pred_dist)

        # Amygdala (spiking fear circuit)
        self.amygdala_alpha = self.amygdala(
            enemy_pixels=enemy_px, pred_dist=pred_dist,
            stress=self.allostasis.stress, pred_facing=0.0)
        # Rear-threat inference: if amygdala was high but enemy vanished,
        # predator likely behind fish → maintain threat from memory
        if not hasattr(self, '_rear_threat'):
            self._rear_threat = 0.0
            self._last_flee_turn = 0.0
        if self.amygdala_alpha > 0.15 and enemy_px < 2:
            # Enemy was threatening but now invisible → behind us
            self._rear_threat = max(self._rear_threat, self.amygdala_alpha * 0.8)
        else:
            self._rear_threat *= 0.9  # decay when visible or safe
        # Boost effective threat with rear estimate (only if recently saw enemy)
        if self._rear_threat > self.amygdala_alpha and self._rear_threat > 0.05:
            self.amygdala_alpha = min(self._rear_threat, 0.5)  # cap phantom threat
        # CMS (novelty)
        self.cms = 0.3 * self.amygdala_alpha + 0.1

        # Tectum
        tect_out = self.tectum(rgc_out)

        # Interoception (spiking insular cortex)
        insula_out = self.insula(
            energy=self.energy, stress=self.allostasis.stress,
            fatigue=self.allostasis.fatigue,
            reward=0.0, threat_acute=(pred_state == 'HUNT' and pred_dist < 60))

        # === 2. THALAMO-PALLIAL ===
        NA = self.neuromod.NA.item()
        tc_out = self.thalamus(tect_out['sfgs_b'], self.pallium.rate_s, NA)
        goal_tensor = torch.zeros(4, device=self.device)
        goal_tensor[self.current_goal] = 1.0
        pal_out = self.pallium(tc_out['TC'], goal_tensor, self.neuromod.ACh.item())

        # === 3. BG ACTION SELECTION ===
        bg_out = self.bg(pal_out['rate_D'], self.neuromod.DA.item())

        # Cerebellum: forward model (sensory prediction error)
        cb_out = self.cerebellum(
            mossy_input=tect_out['sfgs_b'],
            climbing_fiber=pal_out['free_energy'],
            DA=self.neuromod.DA.item())

        # === 4. GOAL SELECTION (v1 EFE logic, enhanced with v2 neuromod) ===
        # Use classifier food probability + retinal food pixels for robust food detection
        cls_food = float(self._cls_probs[1]) if hasattr(self, '_cls_probs') else 0.0
        p_food_retinal = float(L[400:].max()) + float(R[400:].max())
        p_food = min(1.0, max(p_food_retinal * 0.5, cls_food))
        # Fuse retinal + tectum + amygdala for threat estimate
        # Tectum SFGS-b activity correlates with visual threat processing
        tect_threat = float(tect_out['sfgs_b'].mean()) * 5.0  # scale sparse rate to [0,1]
        p_enemy = min(1.0, enemy_px / 15.0 + 0.3 * self.amygdala_alpha + 0.2 * tect_threat)
        energy_ratio = self.energy / 100.0
        starvation = max(0.0, (0.5 - energy_ratio) / 0.5)
        # EFE computation
        U = 1.0 - 0.5 * (self.cms + 0.3)
        G_forage = 0.2 * U - 0.8 * p_food + 0.15 - 1.5 * starvation
        G_flee   = 0.1 * self.cms - 0.8 * p_enemy + 0.20 + 0.8 * starvation
        G_explore = 0.3 * U - 0.3 + 0.20
        G_social  = 0.25
        # 5-HT patrol suppression: when predator far, strongly suppress flee and boost forage
        if pred_state == 'PATROL' and pred_dist > 150:
            G_flee += self.neuromod.get_flee_efe_bias()
            G_forage -= 0.15 * self.neuromod.HT5.item()  # stronger forage boost
            G_explore += 0.1  # make explore less attractive when safe
        # Place cell bonus
        pc_bonus = self.place.get_efe_bonus()
        G_forage -= pc_bonus['forage_bonus']
        # Allostatic bias
        allo_bias = self.allostasis.get_goal_bias()
        G_forage += allo_bias[0]
        G_flee += allo_bias[1]
        G_explore += allo_bias[2]
        G_social += allo_bias[3]
        # World model EFE
        wm_efe = self.world_model.compute_efe_per_goal(
            self.energy, self.pred_model, fish_pos, pc_bonus, self.allostasis)
        G_forage += 0.3 * wm_efe[0]
        G_flee += 0.3 * wm_efe[1]
        G_explore += 0.3 * wm_efe[2]
        G_social += 0.3 * wm_efe[3]
        # Interoceptive spiking bias
        int_bias = self.insula.get_allostatic_bias()
        G_forage += int_bias['forage_bias']
        G_flee += int_bias['flee_bias']
        G_explore += int_bias['explore_bias']
        # Cerebellar prediction error: high PE → increase exploration
        if cb_out['prediction_error'] > 0.1:
            G_explore -= 0.1 * cb_out['prediction_error']
        # Olfactory bias: food odor attracts FORAGE, alarm drives FLEE
        G_forage += self.olfaction.get_forage_bias()
        G_flee += self.olfaction.get_flee_bias()
        # Habenula goal avoidance: frustrated goals get penalized (higher G = less preferred)
        # (hab_out is computed later; use previous step's frustration)
        hab_bias = self.habenula.frustration * 0.5
        G_forage += hab_bias[0]
        G_flee += hab_bias[1]
        G_explore += hab_bias[2]
        G_social += hab_bias[3]
        # Analytic EFE → bias for spiking goal selector
        G = torch.tensor([G_forage, G_flee, G_explore, G_social], device=self.device)
        efe_bias = -2.0 * (G - G.min())  # lower G = stronger bias (inverted for excitation)

        # Spiking WTA goal selection (pallium-D + EFE bias)
        wta_out = self.goal_selector(pal_out['rate_D'][:int(0.75 * 800)],
                                      neuromod_bias=efe_bias)
        self.goal_probs = torch.softmax(efe_bias, dim=0)  # keep for monitoring
        # Use WTA winner if confident, else fall back to analytic EFE
        if wta_out['confidence'] > 0.4:
            new_goal = wta_out['winner']
        else:
            new_goal = int(self.goal_probs.argmax().item())

        # Override cascade (order matters: last override wins)
        food_px = float((L[400:] > 0.5).sum()) + float((R[400:] > 0.5).sum())

        # 1. Food visible reflex: FORAGE when food in retina and no real threat
        if food_px > 1 and p_enemy < 0.15:
            new_goal = GOAL_FORAGE
            self._forage_lock_steps = max(self._forage_lock_steps, 20)
        # Forage lock persistence
        if self._forage_lock_steps > 0 and p_enemy < 0.20:
            new_goal = GOAL_FORAGE
            self._forage_lock_steps -= 1

        # 2. Critical starvation: must forage unless predator actively hunts
        if starvation > 0.35 and pred_state not in ('HUNT', 'AMBUSH'):
            new_goal = GOAL_FORAGE
        if starvation > 0.7 and pred_dist > 80:
            new_goal = GOAL_FORAGE

        # 3. Threat overrides (visual OR lateral line OR amygdala evidence)
        has_threat_evidence = (enemy_px > 3 or ll_dist < 100
                               or self.amygdala_alpha > 0.3)
        if p_enemy > 0.25 and has_threat_evidence and starvation < 0.6:
            new_goal = GOAL_FLEE
        if pred_dist < 60 and (enemy_px > 1 or ll_dist < 60):
            new_goal = GOAL_FLEE
        elif pred_dist < 90 and starvation < 0.4 and pred_state == 'HUNT':
            new_goal = GOAL_FLEE

        # Stuck detection: force explore when not finding food
        eaten_now_check = getattr(env, '_eaten_now', 0)
        if eaten_now_check == 0:
            self._no_food_steps += 1
        else:
            self._no_food_steps = 0
        if (self._no_food_steps > 30 and self.current_goal == GOAL_FORAGE
                and starvation < 0.4):
            new_goal = GOAL_EXPLORE
            self._force_explore_steps = 15
            self._no_food_steps = 0
        if self._force_explore_steps > 0 and new_goal != GOAL_FLEE and starvation < 0.4:
            new_goal = GOAL_EXPLORE
            self._force_explore_steps -= 1

        if self.goal_persistence > 0 and new_goal != GOAL_FLEE:
            new_goal = self.current_goal
            self.goal_persistence -= 1
        else:
            if new_goal != self.current_goal:
                self.goal_persistence = 8
            self.current_goal = new_goal

        # === 5. MOTOR COMMAND ===
        # Flee direction: GT-based angular correction (matching v1)
        flee_turn = 0.0
        if self.current_goal == GOAL_FLEE:
            enemy_L = float((torch.abs(L[400:] - 0.5) < 0.1).float().sum())
            enemy_R = float((torch.abs(R[400:] - 0.5) < 0.1).float().sum())
            total_enemy = enemy_L + enemy_R
            # Use GT predator position for precise flee (like v1)
            if total_enemy > 2 and hasattr(env, 'pred_x'):
                esc_ang = math.atan2(
                    env.fish_y - env.pred_y, env.fish_x - env.pred_x)
                esc_diff = math.atan2(
                    math.sin(esc_ang - fish_heading),
                    math.cos(esc_ang - fish_heading))
                # Negated: flee_turn→0 when heading away, max when toward pred
                flee_turn = float(np.clip(-esc_diff * 2.0, -1.0, 1.0))
                self._last_flee_turn = flee_turn
            elif total_enemy > 2:
                # Retinal fallback
                retinal_flee = (enemy_L - enemy_R) / (total_enemy + 1e-8)
                if abs(retinal_flee) < 0.15 and total_enemy > 5:
                    if not hasattr(self, '_escape_side'):
                        self._escape_side = 1.0 if np.random.random() > 0.5 else -1.0
                    retinal_flee = self._escape_side * 0.6
                else:
                    self._escape_side = 1.0 if retinal_flee > 0 else -1.0
                flee_turn = max(-1.0, min(1.0, retinal_flee * 2.5))
                self._last_flee_turn = flee_turn
            elif ll_dist < 150:
                flee_turn = self._last_flee_turn * 0.3
            else:
                flee_turn = self._last_flee_turn * 0.2

        # Food approach: retinal bearing (L vs R food pixels)
        food_turn = 0.0
        if self.current_goal == GOAL_FORAGE:
            # Type channel: food pixels have value ~1.0 (>0.8)
            food_mask_L_raw = (L[400:] > 0.7).float()
            food_mask_R_raw = (R[400:] > 0.7).float()
            # Intensity-weighted: closer food (brighter) gets more weight
            food_L = float((food_mask_L_raw * L[:400]).sum())
            food_R = float((food_mask_R_raw * R[:400]).sum())
            food_L_px = float(food_mask_L_raw.sum())
            food_R_px = float(food_mask_R_raw.sum())
            total_food_px = food_L_px + food_R_px
            if total_food_px > 1:
                # Intensity-weighted bearing: brighter (closer) food dominates
                # v1 convention: (R - L) positive → turn right
                total_food_int = food_L + food_R + 1e-8
                food_turn = (food_R - food_L) / total_food_int
                food_turn = max(-1.0, min(1.0, food_turn * 2.5))
                # Weighted centroid for finer bearing
                type_L = L[400:]
                type_R = R[400:]
                food_mask_L = (type_L > 0.7).float()
                food_mask_R = (type_R > 0.7).float()
                cols = torch.arange(400, device=self.device, dtype=torch.float32)
                if food_mask_L.sum() > 0:
                    centroid_L = (cols * food_mask_L).sum() / food_mask_L.sum()
                else:
                    centroid_L = torch.tensor(200.0, device=self.device)
                if food_mask_R.sum() > 0:
                    centroid_R = (cols * food_mask_R).sum() / food_mask_R.sum()
                else:
                    centroid_R = torch.tensor(200.0, device=self.device)
                # Convert centroid to angular offset (-1 to 1)
                ang_L = (float(centroid_L) - 200.0) / 200.0
                ang_R = (float(centroid_R) - 200.0) / 200.0
                if food_R > food_L:
                    food_turn = max(-1.0, min(1.0, ang_R * 1.5))
                elif food_L > 0:
                    food_turn = max(-1.0, min(1.0, -ang_L * 1.5))

        # Retinal approach (fallback / blend for EXPLORE)
        # v1 convention: retinal_turn = (retR - retL) / total
        # positive retinal_turn → turn right (CW in env)
        retL = float(L[:400].sum())
        retR = float(R[:400].sum())
        retinal_turn = (retR - retL) / (retL + retR + 1e-8)

        # Wall avoidance — angle-to-center method (matching v1)
        wall_turn = 0.0
        wx, wy = self._last_fish_pos
        aw = getattr(env, 'arena_w', 800)
        ah = getattr(env, 'arena_h', 600)
        heading = getattr(env, 'fish_heading', 0.0)
        margin = 100
        wall_urgency = max(0.0,
            max((margin - wx) / margin if wx < margin else 0.0,
                (wx - (aw - margin)) / margin if wx > aw - margin else 0.0,
                (margin - wy) / margin if wy < margin else 0.0,
                (wy - (ah - margin)) / margin if wy > ah - margin else 0.0))
        if wall_urgency > 0.01:
            center_dx = aw * 0.5 - wx
            center_dy = ah * 0.5 - wy
            angle_to_center = math.atan2(center_dy, center_dx)
            angle_diff = angle_to_center - heading
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            wall_turn = wall_urgency * 0.8 * np.sign(angle_diff)
        if min(wx, wy, aw - wx, ah - wy) < 25:
            wall_turn *= 2.0
        wall_turn = max(-1.5, min(1.5, wall_turn))

        # Obstacle avoidance: bilateral rock pixel repulsion
        obstacle_turn = 0.0
        type_L = L[400:]
        type_R = R[400:]
        rock_L = float((torch.abs(type_L - 0.75) < 0.1).float().sum())
        rock_R = float((torch.abs(type_R - 0.75) < 0.1).float().sum())
        rock_total = rock_L + rock_R
        if rock_total > 5:
            # Steer away from the side with more rock pixels
            # v1 convention: rock on left → turn right (positive)
            rock_bias = (rock_R - rock_L) / (rock_total + 1e-8)
            obstacle_turn = -rock_bias * min(1.0, rock_total / 30.0) * 1.5
            # Center-escape: rock centered ahead
            if abs(rock_L - rock_R) < 5 and rock_total > 20:
                obstacle_turn = 0.5 * (1.0 if self._step_count % 2 == 0 else -1.0)

        # Stuck detection: force explore if barely moving near obstacle
        if not hasattr(self, '_stuck_counter'):
            self._stuck_counter = 0
            self._prev_pos = (0, 0)
        curr_pos = (getattr(env, 'fish_x', 0), getattr(env, 'fish_y', 0))
        dx_move = curr_pos[0] - self._prev_pos[0]
        dy_move = curr_pos[1] - self._prev_pos[1]
        if math.sqrt(dx_move**2 + dy_move**2) < 2.0 and rock_total > 10:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._prev_pos = curr_pos
        if self._stuck_counter > 8:
            self._force_explore_steps = max(self._force_explore_steps, 15)
            self._stuck_counter = 0

        # Slow sinusoidal explore pattern
        self._step_count += 1
        explore_turn = 0.3 * math.sin(self._step_count * 0.08)

        # Active scanning saccade: when threat memory active but enemy invisible,
        # add periodic ±30° oscillation to re-acquire predator from blind spot
        scan_turn = 0.0
        if self._rear_threat > 0.1 and enemy_px < 2:
            scan_freq = 0.15  # faster oscillation than explore
            scan_turn = 0.5 * math.sin(self._step_count * scan_freq)

        # Combined avoidance (wall + obstacle)
        avoid_turn = wall_turn + obstacle_turn
        avoid_turn = max(-1.0, min(1.0, avoid_turn))

        # Select brain turn (goal-directed, before wall avoidance)
        if self.current_goal == GOAL_FLEE:
            brain_turn = flee_turn * 1.5 + 0.3 * obstacle_turn
            alpha_s = 0.6  # faster response during flee
        elif self.current_goal == GOAL_FORAGE:
            brain_turn = food_turn if abs(food_turn) > 0.05 else retinal_turn * 0.8
            brain_turn = 0.8 * brain_turn + 0.2 * obstacle_turn
            alpha_s = 0.50  # responsive to food direction
        elif self.current_goal == GOAL_EXPLORE:
            brain_turn = 0.4 * explore_turn + 0.2 * retinal_turn + 0.3 * obstacle_turn + 0.3 * scan_turn
            alpha_s = 0.25
        else:  # SOCIAL
            brain_turn = retinal_turn * 0.5 + 0.3 * obstacle_turn
            alpha_s = 0.30

        self._smoother = alpha_s * brain_turn + (1 - alpha_s) * self._smoother
        # v1 convention: turn_rate = -brain_turn * brain_weight + wall_turn
        brain_weight = max(0.2, 1.0 - wall_urgency)
        turn = float(np.clip(-self._smoother * brain_weight + wall_turn, -1.0, 1.0))

        # RS motor (Mauthner C-start for looming)
        rs_out = self.rs(tect_out['sgc'], bg_out['gate'], pal_out['rate_D'],
                         flee_turn, 1.0, tect_out['looming'])
        if rs_out['cstart']:
            turn = rs_out['turn']

        # Speed (with allostatic fatigue cap)
        if self.current_goal == GOAL_FLEE:
            speed = 1.5
        elif self.current_goal == GOAL_FORAGE:
            # Boost speed when food visible and no threat
            speed = 1.2 if food_px > 2 and p_enemy < 0.1 else 1.0
        elif self.current_goal == GOAL_EXPLORE:
            speed = 0.8
        else:
            speed = 0.7
        speed *= self.allostasis.get_speed_cap()
        self._last_speed = speed

        # === 5b. SPINAL CPG (rhythmic motor output) ===
        cpg_drive = min(1.0, speed / 1.5)  # normalize to [0,1]
        mL, mR, cpg_speed, cpg_turn, cpg_diag = self.cpg.step(cpg_drive, turn)
        # CPG modulation is very subtle — don't override brain motor commands
        # Only add slight phasic oscillation during calm movement
        if self.current_goal == GOAL_EXPLORE and abs(turn) < 0.3:
            turn = 0.9 * turn + 0.1 * cpg_turn

        # === 6. NEUROMOD UPDATE ===
        reward = 0.01  # survival reward
        eaten_now = getattr(env, '_eaten_now', 0)
        if eaten_now > 0:
            reward += 10.0 * eaten_now
            self._forage_lock_steps = 25
        self.energy = getattr(env, 'fish_energy', self.energy)
        self.world_model.update_food_gain(eaten_now > 0)

        # RL Critic: value estimation and TD learning
        critic_out = self.critic(
            energy=self.energy, threat=p_enemy,
            food_visible=food_px, goal=self.current_goal,
            DA=self.neuromod.DA.item(), NA=self.neuromod.NA.item(),
            cls_probs=self._cls_probs if hasattr(self, '_cls_probs') else None,
            reward=reward)

        # Habenula: disappointment signal + per-goal frustration
        hab_out = self.habenula(
            reward=reward,
            expected_reward=self.critic.get_value(self.current_goal),
            aversion=self.amygdala_alpha,
            current_goal=self.current_goal,
            DA=self.neuromod.DA.item())

        nm = self.neuromod.update(
            reward=reward, amygdala_alpha=self.amygdala_alpha,
            cms=self.cms, flee_active=(self.current_goal == GOAL_FLEE),
            fatigue=self.allostasis.fatigue, circadian=0.7,
            current_goal=self.current_goal)
        # Habenula modulates neuromod: disappointment suppresses DA and 5-HT
        if hab_out['da_suppression'] > 0.05:
            self.neuromod.DA.mul_(1.0 - hab_out['da_suppression'])
        if hab_out['ht5_suppression'] > 0.05:
            self.neuromod.HT5.mul_(1.0 - hab_out['ht5_suppression'])

        # === 7. PLACE CELL UPDATE (proprioceptive — fish knows own position) ===
        px = getattr(env, 'fish_x', 400)
        py = getattr(env, 'fish_y', 300)
        self._last_fish_pos = (px, py)
        self.place(px, py, food_eaten=(eaten_now > 0), predator_near=(pred_dist < 150))

        # === 7b. SENSORY MODULES ===
        vest_out = self.vestibular(fish_heading, speed, turn)
        prop_out = self.proprioception(px, py, speed, fish_heading)
        color_out = self.color_vision(L[400:], R[400:])
        circ_out = self.circadian(light_level=0.7)
        sw_out = self.sleep_wake(
            circadian_melatonin=circ_out['melatonin'],
            arousal=self.insula.arousal,
            threat=self.amygdala_alpha)
        # Working memory: store goal + food direction
        wm_input = torch.zeros(32, device=self.device)
        wm_input[self.current_goal * 8:(self.current_goal + 1) * 8] = 0.5
        wm_out = self.working_mem(wm_input, gate=self.neuromod.ACh.item())

        # === 7c. VAE WORLD MODEL (online training) ===
        tect_all = tect_out['all_e'].unsqueeze(0)  # [1, N]
        state_ctx = np.array([
            px / 800.0, py / 600.0, fish_heading / math.pi,
            self.energy / 100.0,
            self.neuromod.DA.item(), critic_out['td_error'],
            0.0, 0.0,  # precision placeholders
            float(self._cls_probs[1].detach()) if hasattr(self, '_cls_probs') else 0.0,
            float(self._cls_probs[2].detach()) if hasattr(self, '_cls_probs') else 0.0,
            float(self._cls_probs[4].detach()) if hasattr(self, '_cls_probs') else 0.0,
            pal_out['free_energy'],
            self.cms,
        ], dtype=np.float32)
        self.vae.train_step(tect_all, state_ctx)
        z_now, _ = self.vae.encode(tect_all, state_ctx)
        # Transition training
        if self._z_prev is not None and self._last_action_ctx is not None:
            self.vae.update_transition(self._z_prev, self._last_action_ctx, z_now)
        self._z_prev = z_now.copy()
        goal_oh = np.zeros(3, dtype=np.float32)
        if self.current_goal < 3:
            goal_oh[self.current_goal] = 1.0
        self._last_action_ctx = np.array([turn, speed, *goal_oh], dtype=np.float32)
        # Memory update
        self.vae.update_memory(z_now, eaten_now, pred_dist)

        # Habenula strategy switch: force goal change if frustrated
        if hab_out['switch_signal'] and self.current_goal != GOAL_FLEE:
            # Switch to least-frustrated non-current goal
            frustration = hab_out['frustration']
            candidates = [g for g in range(4) if g != self.current_goal]
            best = min(candidates, key=lambda g: frustration[g])
            self.current_goal = best
            self.goal_persistence = 10

        # === 8. FEEDBACK LEARNING ===
        self.fb_learner.update(
            h_upper=pal_out['rate_D'],
            pred_error=pal_out['pred_error'],
            pi=1.0)

        # === 9. PREDICTIVE NETWORK (spiking world model) ===
        motor_cmd = torch.tensor([turn, speed], dtype=torch.float32, device=self.device)
        pred_out = self.predictive(rgc_out['on_fused'], motor_cmd)

        # === 10. HABIT NETWORK ===
        retinal_summary = torch.tensor([
            float(L[:400].sum()) / 400.0,
            float(R[:400].sum()) / 400.0,
            float((L[400:] > 0.7).float().sum()) / 20.0,
            float((R[400:] > 0.7).float().sum()) / 20.0,
        ], device=self.device)
        habit_out = self.habit(
            cls_probs=self._cls_probs if hasattr(self, '_cls_probs') else None,
            goal=self.current_goal, turn=turn, speed=speed,
            retinal_summary=retinal_summary)

        return {
            'turn': float(turn),
            'speed': float(speed),
            'goal': self.current_goal,
            'goal_probs': self.goal_probs.detach().cpu().numpy(),
            'DA': nm['DA'], 'NA': nm['NA'],
            '5HT': nm['5HT'], 'ACh': nm['ACh'],
            'free_energy': pal_out['free_energy'],
            'looming': tect_out['looming'],
            'bg_gate': bg_out['gate'],
            'amygdala': self.amygdala_alpha,
            'cerebellum_pe': cb_out['prediction_error'],
            'habenula_disappoint': hab_out['disappointment'],
            'critic_value': critic_out['current_value'],
            'critic_td_error': critic_out['td_error'],
            'predictive_surprise': pred_out['surprise'],
            'habit_confidence': habit_out['confidence'],
            'insula_valence': insula_out['valence'],
            'insula_heart_rate': insula_out['heart_rate'],
            'insula_arousal': insula_out['arousal'],
            'cpg_motor_L': mL,
            'cpg_motor_R': mR,
            'cpg_bout': cpg_diag['bout_active'],
            'vae_loss': self.vae._last_vae_loss,
            'vae_memory_nodes': self.vae.memory.n_allocated,
            'hab_switch': hab_out['switch_signal'],
            'hab_helplessness': hab_out['helplessness'],
        }

    def reset(self):
        self.retina.reset()
        self.tectum.reset()
        self.thalamus.reset()
        self.pallium.reset()
        self.bg.reset()
        self.rs.reset()
        self.neuromod.reset()
        self.place.reset()
        self.amygdala.reset()
        self.classifier.reset()
        self.pred_model.reset()
        self.allostasis.reset()
        self.world_model.reset()
        self.goal_selector.reset()
        self.cerebellum.reset()
        self.habenula.reset()
        self.predictive.reset()
        self.critic.reset()
        self.habit.reset()
        self.insula.reset()
        self.cpg.reset()
        self.vae.reset()
        self.lateral_line_mod.reset()
        self.olfaction.reset()
        self.working_mem.reset()
        self.vestibular.reset()
        self.proprioception.reset()
        self.color_vision.reset()
        self.circadian.reset()
        self.sleep_wake.reset()
        self._z_prev = None
        self._last_action_ctx = None
        self.goal_probs = torch.tensor([0.25, 0.25, 0.25, 0.25], device=self.device)
        self.current_goal = GOAL_EXPLORE
        self.goal_persistence = 0
        self.energy = 100.0
        self.amygdala_alpha = 0.0
        self.cms = 0.0
        self._smoother = 0.0
        self._forage_lock_steps = 0
        self._forage_no_progress_steps = 0
        self._last_food_count = 0
        self._no_food_steps = 0
        self._force_explore_steps = 0
        self._step_count = 0
