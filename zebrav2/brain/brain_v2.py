"""
ZebrafishBrainV2: integrates all modules.
Interface-compatible with v1 BrainAgent for direct comparison.
Uses v1 EFE policy selection (preserved) with v2 spiking substrate.
"""
import os
import math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, N_TC
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.retina import RetinaV2
from zebrav2.brain.tectum import Tectum
from zebrav2.brain.thalamus import Thalamus
from zebrav2.brain.pallium import Pallium
from zebrav2.brain.basal_ganglia import BasalGanglia
from zebrav2.brain.reticulospinal import ReticulospinalSystem
from zebrav2.brain.neuromod import NeuromodSystem
from zebrav2.brain.place_cells import ThetaPlaceCells
from zebrav2.brain.plasticity import FeedbackPELearning, EligibilitySTDP
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
from zebrav2.brain.saccade import SpikingSaccade
from zebrav2.brain.geographic_model import GeographicModel
from zebrav2.brain.binocular_depth import BinocularDepth
from zebrav2.brain.shoaling import ShoalingModule
from zebrav2.brain.prey_capture import PreyCaptureKinematics
from zebrav2.brain.personality import get_personality
from zebrav2.brain.meta_goal import MetaGoalWeights
from zebrav2.brain.active_motor import ActiveInferenceMotor
from zebrav2.brain.social_memory import SocialMemory
from zebrav2.brain.hpa_axis import HPAAxis
from zebrav2.brain.oxytocin import OxytocinSystem
from zebrav2.brain.pretectum import SpikingPretectum
from zebrav2.brain.ipn import SpikingIPN
from zebrav2.brain.raphe import SpikingRaphe
from zebrav2.brain.locus_coeruleus import SpikingLocusCoeruleus
from zebrav2.brain.habituation import SynapticDepression
from zebrav2.brain.pectoral_fin import PectoralFinMotor

GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, GOAL_SOCIAL = 0, 1, 2, 3

class ZebrafishBrainV2(nn.Module):
    def __init__(self, device=DEVICE, personality=None, brain_config=None, body_config=None):
        super().__init__()
        self.device = device
        # Config objects (lazy import to avoid circular deps)
        from zebrav2.config.brain_config import BrainConfig
        from zebrav2.config.body_config import BodyConfig
        self.cfg = brain_config if brain_config is not None else BrainConfig()
        self.body_cfg = body_config if body_config is not None else BodyConfig()
        self._ablated = self.cfg.get_ablated_set()
        # Perturbation manager (set externally via VirtualZebrafish or directly)
        self._perturbations = None  # PerturbationManager, set via set_perturbations()
        # Personality profile (config overrides parameter)
        _pers_name = personality if personality is not None else self.cfg.personality
        if isinstance(_pers_name, dict):
            self.personality = _pers_name
        else:
            self.personality = get_personality(_pers_name if _pers_name else 'default')
        # --- Core modules ---
        self.retina    = RetinaV2(device)
        self.tectum    = Tectum(device)
        _sfgs_b_half_n_e = self.tectum.sfgs_b_L.n_e   # 450 per hemisphere
        self.thalamus_L = Thalamus(device, sfgs_b_n_e=_sfgs_b_half_n_e, n_tc=150)  # L_tectum → TC_L
        self.thalamus_R = Thalamus(device, sfgs_b_n_e=_sfgs_b_half_n_e, n_tc=150)  # R_tectum → TC_R
        self.pallium   = Pallium(device)
        self.bg        = BasalGanglia(device)
        self.rs        = ReticulospinalSystem(device)
        self.neuromod  = NeuromodSystem(device)
        self.place     = ThetaPlaceCells(device=device)
        # PE-driven feedback learning (W_FB is nn.Linear; pass its weight parameter)
        self.fb_learner = FeedbackPELearning(self.pallium.W_FB.weight, device=device)
        # --- Three-factor STDP: tectum→thalamus→pallium visual pathway ---
        sfgs_b_n_e = self.tectum.sfgs_b_L.n_e  # 450 per hemisphere
        pal_s_n_e  = self.pallium.pal_s.n_e    # 1200
        pal_d_n_e  = self.pallium.pal_d.n_e    # 600
        _p_tt = self.cfg.plasticity.tect_tc
        _p_tp = self.cfg.plasticity.tc_pal
        _p_pd = self.cfg.plasticity.pal_d
        self.stdp_tect_tc_L = EligibilitySTDP(
            self.thalamus_L.W_tect_tc.weight, device=device,
            A_plus=_p_tt.a_plus, A_minus=_p_tt.a_minus, w_max=_p_tt.w_max, w_min=_p_tt.w_min)
        self.stdp_tect_tc_R = EligibilitySTDP(
            self.thalamus_R.W_tect_tc.weight, device=device,
            A_plus=_p_tt.a_plus, A_minus=_p_tt.a_minus, w_max=_p_tt.w_max, w_min=_p_tt.w_min)
        self.stdp_tc_pal = EligibilitySTDP(
            self.pallium.W_tc_pals.weight, device=device,
            A_plus=_p_tp.a_plus, A_minus=_p_tp.a_minus, w_max=_p_tp.w_max, w_min=_p_tp.w_min)
        self.stdp_pal_d = EligibilitySTDP(
            self.pallium.W_pals_pald.weight, device=device,
            A_plus=_p_pd.a_plus, A_minus=_p_pd.a_minus, w_max=_p_pd.w_max, w_min=_p_pd.w_min)
        # Top-down attention: pallium-S → tectum (one-step delayed prediction)
        _pal_s_n_e = self.pallium.pal_s.n_e   # 1200
        _sfgs_b_n_e = self.tectum.sfgs_b_L.n_e  # 450
        self.W_pal_tect_L = nn.Linear(_pal_s_n_e, _sfgs_b_n_e, bias=False)
        self.W_pal_tect_R = nn.Linear(_pal_s_n_e, _sfgs_b_n_e, bias=False)
        _td_gain = self.cfg.plasticity.top_down_gain
        nn.init.xavier_uniform_(self.W_pal_tect_L.weight, gain=_td_gain)
        nn.init.xavier_uniform_(self.W_pal_tect_R.weight, gain=_td_gain)
        self.W_pal_tect_L.to(device)
        self.W_pal_tect_R.to(device)
        # Previous-step pallium rate (for top-down prediction, 1-step delay)
        self.register_buffer('_prev_pal_rate_s', torch.zeros(_pal_s_n_e, device=device))
        # Thalamic activity from previous step: used as "prediction" for spatial novelty gaze
        # tc_combined = cat([TC_R (left visual), TC_L (right visual)]) = 300 neurons
        self.register_buffer('_prev_tc', torch.zeros(N_TC, device=device))
        self._da_phasic_steps = 0  # phasic DA burst counter
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
        _cpg_cfg = self.body_cfg.motor.cpg
        self.cpg = SpinalCPG(
            tau_m=_cpg_cfg.tau_m, tau_syn=_cpg_cfg.tau_syn,
            v_thresh=_cpg_cfg.v_thresh, v_reset=_cpg_cfg.v_reset,
            w_v2a_mn=_cpg_cfg.w_v2a_mn, w_v2a_v0d=_cpg_cfg.w_v2a_v0d,
            w_v0d_cross=_cpg_cfg.w_v0d_cross, w_v2a_di6=_cpg_cfg.w_v2a_di6,
            w_di6_v2a=_cpg_cfg.w_di6_v2a, w_di6_mn=_cpg_cfg.w_di6_mn,
            w_mn_rsh=_cpg_cfg.w_mn_rsh, w_rsh_v2a=_cpg_cfg.w_rsh_v2a,
            noise=_cpg_cfg.noise, device=device)
        # Tectum all_e size = sum of all 8 half-layer E neuron counts (2 hemispheres)
        _tect_e_total = (self.tectum.sfgs_b_L.n_e + self.tectum.sfgs_d_L.n_e +
                         self.tectum.sgc_L.n_e    + self.tectum.so_L.n_e    +
                         self.tectum.sfgs_b_R.n_e + self.tectum.sfgs_d_R.n_e +
                         self.tectum.sgc_R.n_e    + self.tectum.so_R.n_e)
        self.vae = VAEWorldModelV2(tectum_dim=_tect_e_total, device=device)
        self._z_prev = None  # for VAE transition training
        self._last_action_ctx = None
        # World learning state
        self._novelty_ema = 1.0  # starts high (unknown world)
        self._surprise_history = []
        self._exploration_phase = True  # True until world is learned
        # --- Medium/Low priority modules ---
        self.lateral_line_mod = SpikingLateralLine(device=device)
        _ll_cfg = self.body_cfg.sensory.lateral_line
        self.lateral_line_mod.SN_RANGE = _ll_cfg.sn_range
        self.lateral_line_mod.CN_RANGE = _ll_cfg.cn_range
        self.lateral_line_mod.PREY_RANGE = _ll_cfg.prey_range
        self.olfaction = SpikingOlfaction(device=device)
        self.working_mem = SpikingWorkingMemory(device=device)
        self.vestibular = SpikingVestibular(device=device)
        self.proprioception = SpikingProprioception(device=device)
        self.active_motor = ActiveInferenceMotor(device=device)
        self.color_vision = SpikingColorVision(device=device)
        self.circadian = SpikingCircadian(device=device)
        self.sleep_wake = SpikingSleepWake(device=device)
        self.saccade = SpikingSaccade(device=device)
        self.geo_model = GeographicModel()
        self.binocular = BinocularDepth()
        self.shoaling = ShoalingModule()
        self.prey_capture = PreyCaptureKinematics()
        # Learned EFE weights + social inference
        self.meta_goal = MetaGoalWeights(device=device)
        self.social_mem = SocialMemory()
        # Stress hormone (HPA/HPI axis) + social bonding (oxytocin/vasopressin)
        self.hpa = HPAAxis()
        self.oxt = OxytocinSystem()
        # Pretectum: optokinetic response (OKR) from direction-selective retinal input
        self.pretectum = SpikingPretectum(device=device)
        # Interpeduncular nucleus: habenula relay → behavioral inhibition + DA feedback
        self.ipn = SpikingIPN(device=device)
        # Spiking raphe: population-coded 5-HT (overrides scalar neuromod.HT5)
        self.raphe = SpikingRaphe(device=device)
        # Spiking locus coeruleus: population-coded NA (overrides scalar neuromod.NA)
        self.lc = SpikingLocusCoeruleus(device=device)
        # Pectoral fin motor neurons: slow-turn kinematics (non-flee)
        self.pectoral_fin = PectoralFinMotor(device=device)
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
        # Motor / threat state (pre-initialized to avoid hasattr guards in step())
        self._enemy_lateral_ema = 0.0
        self._rear_threat = 0.0
        self._last_flee_turn = 0.0
        self._escape_side = 1.0
        self._stuck_counter = 0
        self._prev_pos = (0.0, 0.0)
        # Classifier probs cache
        self._cls_probs = torch.zeros(5, device=device)
        # Sensory module state — previous-step values fed into EFE / motor
        self._last_activity_drive = 1.0   # circadian: 1.0=day, 0.3=night
        self._last_td_error = 0.0          # critic TD error from previous step
        # Action-perception cycle state
        self._prev_heading = 0.0           # previous heading for delta
        self._last_ai_motor = None         # last active inference motor output
        self._prop_cache = None            # cached proprioception from previous step
        self._last_pred_surprise = 0.0    # predictive network surprise from previous step
        # Free energy gradient: tracks dF/dt for closed-loop goal modulation
        # Rising F → current goal failing → switch.  Falling F → goal working → persist.
        self._fe_history = deque(maxlen=10)
        self._fe_gradient = 0.0
        # Spontaneity: anti-FEP mechanism (see _compute_spontaneity)
        # Real organisms don't always minimize F — they explore, play, get frustrated
        self._spontaneity = 0.0
        # Ablation set loaded from self.cfg.ablation (habenula + insula off by default)
        # Spatial registry: assign atlas-derived positions to all neurons
        from zebrav2.brain.spatial_registry import SpatialRegistry
        self._spatial = SpatialRegistry(device=device)
        self._spatial.assign_to_brain(self)
        # Modulate STDP weight matrices by inter-region distance
        self._apply_spatial_priors()
        # Apply personality to neuromod baselines and thresholds
        self._apply_personality()

    def _apply_spatial_priors(self):
        """Modulate STDP pathway weights by atlas-derived inter-region distance.

        Nearby neurons get stronger initial connections, distant ones weaker.
        Uses exponential decay: w *= (1-s) + s*exp(-dist/lambda).
        """
        sr = self._spatial
        # Tectum → Thalamus (retinotopic, short-range: lambda=80um)
        sr.apply_distance_weights(
            self.thalamus_L.W_tect_tc.weight,
            'tectum.sfgs_b_L', 'thalamus.tc_L',
            lambda_um=80.0, strength=0.4)
        sr.apply_distance_weights(
            self.thalamus_R.W_tect_tc.weight,
            'tectum.sfgs_b_R', 'thalamus.tc_R',
            lambda_um=80.0, strength=0.4)
        # Thalamus → Pallium-S (longer range: lambda=120um)
        sr.apply_distance_weights(
            self.pallium.W_tc_pals.weight,
            'thalamus.tc_L', 'pallium.pal_s',
            lambda_um=120.0, strength=0.3)
        # Pallium-S → Pallium-D (local: lambda=60um)
        sr.apply_distance_weights(
            self.pallium.W_pals_pald.weight,
            'pallium.pal_s', 'pallium.pal_d',
            lambda_um=60.0, strength=0.3)
        # Top-down attention: Pallium → Tectum (long-range: lambda=150um)
        sr.apply_distance_weights(
            self.W_pal_tect_L.weight,
            'pallium.pal_s', 'tectum.sfgs_b_L',
            lambda_um=150.0, strength=0.2)
        sr.apply_distance_weights(
            self.W_pal_tect_R.weight,
            'pallium.pal_s', 'tectum.sfgs_b_R',
            lambda_um=150.0, strength=0.2)

    def _apply_personality(self):
        """Set neuromodulator baselines and thresholds from personality profile."""
        p = self.personality
        self.neuromod.DA.fill_(p['DA_baseline'])
        self.neuromod.HT5.fill_(p['HT5_baseline'])
        self.neuromod.NA.fill_(p['NA_baseline'])
        self.neuromod.ACh.fill_(p['ACh_baseline'])
        # Amygdala gain: scales fear sensitivity
        self.amygdala.retinal_gain = 0.08 * p['amy_gain']
        # Habenula frustration threshold
        self.habenula.threshold = p['habenula_threshold']
        # CPG noise
        self.cpg.noise = p['cpg_noise']
        # Store flee threshold for use in step()
        self._flee_threshold = p.get('flee_threshold', 0.25)
        self._explore_bias = p.get('explore_bias', 0.0)
        self._social_bias = p.get('social_bias', 0.0)

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
        # Cache intensity / type channel slices — used 7+ times in step()
        L_int, L_type = L[:400], L[400:]
        R_int, R_type = R[:400], R[400:]
        # Binocular depth from retinal overlap
        if L.sum() > 0:
            bino_out = self.binocular.estimate(
                L_type.cpu().numpy(), R_type.cpu().numpy(),
                L_int.cpu().numpy(), R_int.cpu().numpy())
        else:
            bino_out = {'food_distance': 999, 'enemy_distance': 999,
                        'food_confidence': 0, 'enemy_confidence': 0, 'stereo_correlation': 0}

        # Get entity info from env
        enemy_px = 0.0
        if hasattr(env, '_enemy_pixels_total'):
            enemy_px = float(env._enemy_pixels_total)
        entity_info = {'enemy': enemy_px / 15.0}
        rgc_out = self.retina(L, R, entity_info)

        # Classify scene
        cls_out = self.classifier.classify(L, R)
        self._cls_probs = cls_out['probs'].detach()

        # Pretectum: optokinetic response from direction-selective RGC input
        # DS rates are mean direction-selective activity per eye
        _ds_L = float(rgc_out['L_ds'].mean()) if rgc_out['L_ds'].numel() > 0 else 0.0
        _ds_R = float(rgc_out['R_ds'].mean()) if rgc_out['R_ds'].numel() > 0 else 0.0
        _eye_vel = self.saccade.gaze_offset * 0.5  # rough eye velocity proxy
        pretect_out = self.pretectum(_ds_L, _ds_R, eye_velocity=_eye_vel)

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

        # Spiking lateral line module — multi-source: predator + food + conspecifics
        _ll_conspecifics = []
        if hasattr(env, 'all_fish'):
            for _i, _f in enumerate(env.all_fish):
                if _i == 0 or not _f.get('alive', True):
                    continue
                _ll_conspecifics.append({'x': _f['x'], 'y': _f['y'],
                                         'speed': _f.get('speed', 0.5)})
        ll_out = self.lateral_line_mod(
            fish_x=getattr(env, 'fish_x', 400), fish_y=getattr(env, 'fish_y', 300),
            fish_heading=getattr(env, 'fish_heading', 0.0),
            pred_x=getattr(env, 'pred_x', -999), pred_y=getattr(env, 'pred_y', -999),
            pred_vx=self.pred_model.vx, pred_vy=self.pred_model.vy,
            foods=getattr(env, 'foods', []),
            conspecifics=_ll_conspecifics)
        if ll_out['dist'] < 999:
            ll_dist = ll_out['dist']

        # Olfaction — detect if any conspecific is close to the predator
        _pred_x = getattr(env, 'pred_x', -9999)
        _pred_y = getattr(env, 'pred_y', -9999)
        _conspc_dist = ll_out.get('conspecific_dist', 999.0)
        _conspecific_injured = False
        for _cf in _ll_conspecifics:
            try:
                _cdx = float(_cf['x']) - _pred_x
                _cdy = float(_cf['y']) - _pred_y
                if math.sqrt(_cdx * _cdx + _cdy * _cdy) < 40:
                    _conspecific_injured = True
                    break
            except (KeyError, TypeError):
                pass

        olf_out = self.olfaction(
            fish_x=getattr(env, 'fish_x', 400), fish_y=getattr(env, 'fish_y', 300),
            fish_heading=getattr(env, 'fish_heading', 0.0),
            foods=getattr(env, 'foods', []),
            conspecific_injured=_conspecific_injured,
            pred_dist=ll_dist,
            conspc_dist=_conspc_dist)

        # Predator model: predict → retinal update → query
        self.pred_model.predict()
        # Retinal features for predator tracking (use cached slices)
        enemy_mask_L = (torch.abs(L_type - 0.5) < 0.1).float()
        enemy_mask_R = (torch.abs(R_type - 0.5) < 0.1).float()
        enemy_px_L = float(enemy_mask_L.sum())
        enemy_px_R = float(enemy_mask_R.sum())
        # Lateral bias: which eye sees more enemy
        enemy_lateral_raw = (enemy_px_L - enemy_px_R) / (enemy_px_L + enemy_px_R + 1e-8)
        # Temporal EMA smoothing to prevent oscillation
        self._enemy_lateral_ema = 0.3 * enemy_lateral_raw + 0.7 * self._enemy_lateral_ema
        enemy_lateral = self._enemy_lateral_ema
        # Intensity: mean enemy pixel brightness
        enemy_int = float((L_int * enemy_mask_L).sum() + (R_int * enemy_mask_R).sum()) / (enemy_px + 1e-8)
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
        if self._active('amygdala'):
            self.amygdala_alpha = self.amygdala(
                enemy_pixels=enemy_px, pred_dist=pred_dist,
                stress=self.allostasis.stress, pred_facing=0.0)
        else:
            self.amygdala_alpha = 0.0
        # Rear-threat inference: if amygdala was high but enemy vanished,
        # predator likely behind fish → maintain threat from memory
        if self.amygdala_alpha > 0.15 and enemy_px < 2:
            # Enemy was threatening but now invisible → behind us
            self._rear_threat = max(self._rear_threat, self.amygdala_alpha * 0.8)
        else:
            self._rear_threat *= 0.9  # decay when visible or safe
        # Boost effective threat with rear estimate (only if recently saw enemy)
        if self._rear_threat > self.amygdala_alpha and self._rear_threat > 0.05:
            self.amygdala_alpha = min(self._rear_threat, 0.5)  # cap phantom threat
        # HPA axis: update cortisol from sustained amygdala activation
        self.hpa.update(self.amygdala_alpha)
        # Allostatic sensitization: chronic cortisol amplifies fear
        self.amygdala_alpha = min(1.0, self.amygdala_alpha * self.hpa.amygdala_sensitization())
        # CMS (novelty)
        self.cms = 0.3 * self.amygdala_alpha + 0.1

        # Tectum
        if self._active('tectum'):
            # Top-down prediction from previous step's pallium-S
            with torch.no_grad():
                I_td_L = self.W_pal_tect_L(self._prev_pal_rate_s).clamp(-3.0, 3.0)
                I_td_R = self.W_pal_tect_R(self._prev_pal_rate_s).clamp(-3.0, 3.0)
            tect_out = self.tectum(rgc_out, I_topdown_L=I_td_L, I_topdown_R=I_td_R)
        else:
            tect_out = {
                'sfgs_b': torch.zeros(self.tectum.sfgs_b_L.n_e * 2, device=self.device),
                'sfgs_d': torch.zeros(self.tectum.sfgs_d_L.n_e * 2, device=self.device),
                'sgc':    torch.zeros(self.tectum.sgc_L.n_e    * 2, device=self.device),
                'so':     torch.zeros(self.tectum.so_L.n_e     * 2, device=self.device),
                'sfgs_b_L': torch.zeros(self.tectum.sfgs_b_L.n_e, device=self.device),
                'sfgs_b_R': torch.zeros(self.tectum.sfgs_b_R.n_e, device=self.device),
                'sgc_L':    torch.zeros(self.tectum.sgc_L.n_e,    device=self.device),
                'sgc_R':    torch.zeros(self.tectum.sgc_R.n_e,    device=self.device),
                'sgc_L_mean': 0.0, 'sgc_R_mean': 0.0,
            }

        # Interoception (spiking insular cortex)
        if self._active('insula'):
            insula_out = self.insula(
                energy=self.energy, stress=self.allostasis.stress,
                fatigue=self.allostasis.fatigue,
                reward=0.0, threat_acute=(pred_state == 'HUNT' and pred_dist < 60))
        else:
            insula_out = {'heart_rate': 2.0, 'arousal': 0.5, 'valence': 0.0}

        # === 2. THALAMO-PALLIAL ===
        NA = self.neuromod.NA.item()
        if self._active('thalamus'):
            tc_L_out = self.thalamus_L(tect_out['sfgs_b_L'], self.pallium.rate_s, NA)
            tc_R_out = self.thalamus_R(tect_out['sfgs_b_R'], self.pallium.rate_s, NA)
            # cat([TC_R, TC_L]): TC_R carries left visual field, TC_L carries right visual field.
            # This ordering makes pal_d[:half]=left_visual, pal_d[half:]=right_visual,
            # so RS voluntary_turn = (pal_R - pal_L) = (right_visual - left_visual) → correct sign.
            tc_combined = torch.cat([tc_R_out['TC'], tc_L_out['TC']])
        else:
            tc_L_out = {'TC': torch.zeros(self.thalamus_L.TC.n, device=self.device),
                        'TRN': torch.zeros(self.thalamus_L.TRN.n, device=self.device)}
            tc_R_out = {'TC': torch.zeros(self.thalamus_R.TC.n, device=self.device),
                        'TRN': torch.zeros(self.thalamus_R.TRN.n, device=self.device)}
            tc_combined = torch.zeros(self.thalamus_L.TC.n + self.thalamus_R.TC.n, device=self.device)
        # Thalamic delta: |tc_now − tc_prev| → spatial novelty for gaze
        # [:n_tc_half] = TC_R (left visual), [n_tc_half:] = TC_L (right visual)
        _tc_delta = (tc_combined.detach() - self._prev_tc).abs()
        self._prev_tc.copy_(tc_combined.detach())
        _n_tc_half = self.thalamus_L.TC.n   # 150
        goal_tensor = torch.zeros(4, device=self.device)
        goal_tensor[self.current_goal] = 1.0
        if self._active('pallium'):
            pal_out = self.pallium(tc_combined, goal_tensor, self.neuromod.ACh.item())
        else:
            pal_out = {'rate_S': torch.zeros(self.pallium.pal_s.n_e, device=self.device),
                       'rate_D': torch.zeros(self.pallium.pal_d.n_e, device=self.device),
                       'pred_error': torch.zeros(self.pallium.pal_s.n_e, device=self.device),
                       'free_energy': 0.0}

        # STDP trace update — only during elevated DA (phasic burst or above baseline 0.55)
        # Updating traces unconditionally during low-DA periods causes uncorrelated Hebbian drift
        _da_for_stdp = self.neuromod.DA.item()
        if _da_for_stdp > 0.55 or self._da_phasic_steps > 0:
            if self._active('tectum') and self._active('thalamus'):
                self.stdp_tect_tc_L.update_traces(tect_out['sfgs_b_L'], tc_L_out['TC'])
                self.stdp_tect_tc_R.update_traces(tect_out['sfgs_b_R'], tc_R_out['TC'])
            if self._active('thalamus') and self._active('pallium'):
                self.stdp_tc_pal.update_traces(tc_combined, pal_out['rate_S'])
            if self._active('pallium'):
                self.stdp_pal_d.update_traces(pal_out['rate_S'], pal_out['rate_D'])

        # === 3. BG ACTION SELECTION ===
        if self._active('bg'):
            bg_out = self.bg(pal_out['rate_D'], self.neuromod.DA.item())
        else:
            bg_out = {'gate': torch.zeros(4, device=self.device), 'd1_rate': torch.zeros(400, device=self.device), 'd2_rate': torch.zeros(300, device=self.device)}

        # Cerebellum: forward model (sensory prediction error)
        if self._active('cerebellum'):
            cb_out = self.cerebellum(
                mossy_input=tect_out['sfgs_b'],
                climbing_fiber=pal_out.get('free_energy', 0.0),
                DA=self.neuromod.DA.item())
        else:
            cb_out = {'dcn_rate': torch.zeros(200, device=self.device),
                      'gc_sparsity': 0.0, 'pc_rate_mean': 0.0, 'prediction_error': 0.0}

        # === 4. GOAL SELECTION (v1 EFE logic, enhanced with v2 neuromod) ===
        # Use classifier food probability + retinal food pixels for robust food detection
        cls_food = float(self._cls_probs[1])
        p_food_retinal = float(L_type.max()) + float(R_type.max())
        # UV prey channel: UV cones detect small dark food items against bright background
        _uv_prey = float(rgc_out.get('uv_prey_L', 0.0)) + float(rgc_out.get('uv_prey_R', 0.0))
        _tc = self.cfg.threat
        p_food = min(1.0, max(p_food_retinal * _tc.p_food_retinal_weight, cls_food) + _uv_prey * _tc.p_food_uv_prey_weight)
        # Fuse retinal + tectum + amygdala for threat estimate
        tect_threat = float(tect_out['sfgs_b'].mean()) * _tc.tect_threat_scale
        p_enemy = min(1.0, enemy_px / _tc.enemy_pixel_normalization + _tc.amygdala_weight_in_fusion * self.amygdala_alpha + _tc.tect_threat_weight * tect_threat)
        energy_ratio = self.energy / 100.0
        _ec = self.cfg.efe
        starvation = max(0.0, (_ec.starvation_threshold - energy_ratio) / _ec.starvation_threshold)
        # EFE computation — energy trajectory prediction
        U = 1.0 - 0.5 * (self.cms + 0.3)
        # Predict future energy: current drain rate × horizon
        speed_now = getattr(self, '_last_speed', 1.0)
        energy_drain_per_step = 0.2 * speed_now  # matches env drain
        predicted_energy_10 = max(0, self.energy - energy_drain_per_step * 10)
        predicted_energy_30 = max(0, self.energy - energy_drain_per_step * 30)
        # Future energy crisis: will I die in 30 steps if I don't eat?
        future_crisis = max(0.0, 1.0 - predicted_energy_30 / 30.0)
        # Middle-range energy anxiety: worry about trajectory before reaching crisis.
        # Peaks at ~50% energy (fish is concerned but not yet panicking).
        # Fires when within 200 steps of critical (energy < 25) — starts at ~65% energy.
        # Distinct from starvation panic: this is anticipatory, not reactive.
        # The parabolic middle_factor ensures it's QUIET at full/empty and LOUDEST at 50%.
        steps_to_critical = (self.energy - _ec.critical_energy) / (energy_drain_per_step + 1e-8)
        middle_factor = _ec.middle_factor_amplitude * energy_ratio * (1.0 - energy_ratio)
        energy_anxiety = max(0.0, (_ec.energy_anxiety_steps - steps_to_critical) / _ec.energy_anxiety_steps) * middle_factor * _ec.energy_anxiety_scale
        # Combined urgency: current + predicted + anticipatory (scaled by learned urgency weight)
        _mw = self.meta_goal.scales().detach().cpu().numpy()  # [8] floats
        _mc = np.zeros(8, dtype=np.float32)   # modulation contribution magnitudes
        energy_urgency = (starvation * starvation * _ec.starvation_urgency_weight + future_crisis * _ec.future_crisis_weight + energy_anxiety) * float(_mw[7])
        _mc[7] = starvation * starvation * _ec.starvation_urgency_weight + future_crisis * _ec.future_crisis_weight + energy_anxiety
        G_forage = _ec.forage_uncertainty_weight * U + _ec.forage_food_weight * p_food + _ec.forage_offset - energy_urgency
        G_flee   = _ec.flee_cms_weight * self.cms + _ec.flee_enemy_weight * p_enemy + _ec.flee_offset
        G_explore = 0.3 * U - 0.3 + _ec.explore_offset + starvation * _ec.explore_starvation_weight + future_crisis * _ec.explore_future_crisis_weight
        G_social  = _ec.social_offset + starvation * _ec.social_starvation_weight + future_crisis * _ec.social_future_crisis_weight
        # 5-HT patrol suppression: when predator far, strongly suppress flee and boost forage
        if pred_state == 'PATROL' and pred_dist > _tc.patrol_threshold_distance:
            G_flee += self.neuromod.get_flee_efe_bias()
            G_forage -= _ec.ht5_forage_suppression * self.neuromod.HT5.item()
            G_explore += _ec.safe_explore_boost
        # Place cell bonus
        pc_bonus = self.place.get_efe_bonus()
        G_forage -= pc_bonus['forage_bonus']
        # Allostatic bias
        allo_bias = self.allostasis.get_goal_bias()
        G_forage += allo_bias[0]
        G_flee += allo_bias[1]
        G_explore += allo_bias[2]
        G_social += allo_bias[3]
        # World model EFE (mod_w[0])
        wm_efe = self.world_model.compute_efe_per_goal(
            self.energy, self.pred_model, fish_pos, pc_bonus, self.allostasis)
        _wm_scale = _ec.world_model_efe_scale * float(_mw[0])
        G_forage += _wm_scale * wm_efe[0]
        G_flee += _wm_scale * wm_efe[1]
        G_explore += _wm_scale * wm_efe[2]
        G_social += _wm_scale * wm_efe[3]
        _mc[0] = _wm_scale * sum(abs(float(x)) for x in wm_efe)
        # Interoceptive spiking bias (gated — ablation showed insula hurts performance)
        if self._active('insula'):
            int_bias = self.insula.get_allostatic_bias()
            G_forage += _ec.insula_forage_coupling * int_bias['forage_bias']
            G_flee += _ec.insula_flee_coupling * int_bias['flee_bias']
            G_explore += _ec.insula_explore_coupling * int_bias['explore_bias']
        # Cerebellar prediction error: high PE → increase exploration
        if cb_out['prediction_error'] > _ec.cerebellum_pe_threshold:
            G_explore -= _ec.cerebellum_pe_explore_weight * cb_out['prediction_error']
        # Olfactory bias: food odor attracts FORAGE, alarm drives FLEE
        G_forage += self.olfaction.get_forage_bias()
        G_flee += self.olfaction.get_flee_bias()
        # Lateral line prey detection: water disturbance from nearby food → attract FORAGE
        if ll_out['prey_detected']:
            G_forage -= _ec.ll_prey_forage_weight * ll_out['low_freq_prey']
        # Personality bias on EFE
        G_explore += self._explore_bias
        G_social += self._social_bias
        G_social += self.oxt.social_efe_bias()  # OXT makes social goal more preferred
        # Shoaling: social cues from conspecifics (mod_w[1] + learned social weights)
        colleagues = []
        if hasattr(env, 'all_fish'):
            for i, f in enumerate(env.all_fish):
                if i == 0 or not f.get('alive', True):
                    continue
                colleagues.append({'x': f['x'], 'y': f['y'],
                                   'heading': f['heading'],
                                   'speed': f.get('speed', 0.5)})
        fish_px, fish_py = self._last_fish_pos
        self.shoaling.step(fish_px, fish_py, fish_heading, colleagues)
        self.shoaling.observe_social_cues(fish_px, fish_py, colleagues)
        social_bias = self.shoaling.get_efe_bias()
        # Update social memory state tracker
        self.social_mem.update_states(fish_px, fish_py, colleagues)
        # Oxytocin/vasopressin: compute proximity counts from cached fish states
        _n_nearby = sum(1 for s in self.social_mem._fish_states.values() if s['dist'] < 150)
        _n_crowd  = sum(1 for s in self.social_mem._fish_states.values()
                        if s['dist'] < 100 and s['eating'])
        self.oxt.update(_n_nearby, _n_crowd)
        # OXT fear buffering: social contact reduces acute fear expression
        if _n_nearby > 0:
            self.amygdala_alpha = max(0.0, self.amygdala_alpha
                                      - self.oxt.fear_extinction_factor() * 0.3)
        # Learned alarm replaces fixed shoaling alarm coefficient
        learned_alarm = self.social_mem.get_social_alarm(self.shoaling.social_alarm)
        _social_scale = float(_mw[1])
        G_forage += social_bias['social_forage'] * _social_scale
        G_flee += -learned_alarm * 0.2 * _social_scale  # learned alarm weight
        G_explore += social_bias['social_explore'] * _social_scale
        # Social food cue: steer toward where conspecifics are eating (uses cached _fish_states)
        G_forage += self.social_mem.get_food_cue_efe(fish_px, fish_py, fish_heading) * _social_scale
        # Competition penalty: avoid crowded food patches
        _competition_penalty = self.social_mem.get_competition_penalty(fish_px, fish_py)
        G_forage += _competition_penalty * _social_scale
        _mc[1] = abs(social_bias['social_forage']) + learned_alarm * 0.2 + abs(social_bias['social_explore'])
        # Error-driven social weight updates — called unconditionally (window-based, not single-snapshot)
        self.social_mem.update_alarm_outcome(pred_dist < 100)
        self.social_mem.update_food_cue_outcome(getattr(env, '_eaten_now', 0) > 0)
        # Geographic model EFE bias (mod_w[2])
        geo_bias = self.geo_model.get_efe_bias(self._last_fish_pos[0], self._last_fish_pos[1])
        _geo_scale = float(_mw[2])
        G_forage += geo_bias['forage_bias'] * _geo_scale
        G_flee += geo_bias['flee_bias'] * _geo_scale
        G_explore += geo_bias['explore_bias'] * _geo_scale
        _mc[2] = (abs(geo_bias['forage_bias']) + abs(geo_bias['flee_bias']) + abs(geo_bias['explore_bias']))
        # World learning: novelty drives exploration in unfamiliar environments
        # VAE surprise + place cell visit count → novelty estimate
        vae_novelty, _ = self.vae.memory.query_epistemic(self._z_prev if self._z_prev is not None else np.zeros(16, dtype=np.float32))
        place_visits = float(self.place.visit_count.mean()) if hasattr(self.place, 'visit_count') else 0
        place_novelty = 1.0 / (1.0 + place_visits * 0.1)  # high when few visits
        novelty = 0.5 * vae_novelty + 0.5 * place_novelty
        self._novelty_ema = 0.98 * self._novelty_ema + 0.02 * novelty
        # High novelty → explore more (lower G_explore = more preferred) (mod_w[3])
        if self._novelty_ema > 0.5:
            _novelty_contrib = 0.3 * (self._novelty_ema - 0.5) * float(_mw[3])
            G_explore -= _novelty_contrib
            self._exploration_phase = True
            _mc[3] = _novelty_contrib
        else:
            self._exploration_phase = False
        # VAE world model planning: 3-goal latent horizon simulation (mod_w[4])
        # plan() returns G[0..2] for FORAGE/FLEE/EXPLORE; blends in gradually after warmup
        if self._z_prev is not None and self._last_action_ctx is not None:
            vae_G = self.vae.plan(self._z_prev, self._last_action_ctx)
            _vae_scale = 0.2 * float(_mw[4])
            G_forage  += _vae_scale * float(vae_G[0])
            G_flee    += _vae_scale * float(vae_G[1])
            G_explore += _vae_scale * float(vae_G[2])
            _mc[4] = _vae_scale * sum(abs(float(x)) for x in vae_G[:3])
        # Habenula goal avoidance (gated — ablation showed habenula hurts performance)
        if self._active('habenula'):
            hab_frustration = self.habenula.frustration  # numpy (4,)
            hab_gate = np.clip(hab_frustration - _ec.habenula_frustration_threshold, 0.0, None) / _ec.habenula_frustration_gain
            hab_bias = hab_gate * _ec.habenula_goal_bias_weight
            G_forage += hab_bias[0]
            G_flee += hab_bias[1]
            G_explore += hab_bias[2]
            G_social += hab_bias[3]
        # Circadian EFE bias: daytime (activity_drive~1.0) → prefer forage + explore (mod_w[5])
        # Uses previous step's value — 1-step lag is biologically fine for circadian
        circ_efe = (self._last_activity_drive - _ec.circadian_activity_threshold) * _ec.circadian_efe_gain
        _circ_scale = float(_mw[5])
        G_explore -= circ_efe * _circ_scale
        G_forage -= circ_efe * _ec.circadian_forage_reduction * _circ_scale
        _mc[5] = abs(circ_efe)
        # Predictive surprise: high surprise last step → boost exploration this step (mod_w[6])
        if self._last_pred_surprise > _ec.surprise_threshold:
            _cb_contrib = _ec.surprise_explore_weight * (self._last_pred_surprise - _ec.surprise_threshold) * float(_mw[6])
            G_explore -= _cb_contrib
            _mc[6] = _cb_contrib
        # --- Action-perception cycle: F-gradient goal modulation ---
        # If total free energy is RISING under current goal, that goal is
        # failing to reduce surprise → penalize it in EFE (Friston 2010:
        # "action and perception are both in the service of minimizing F").
        # Conversely, falling F → current goal is working → no penalty.
        _fe_goal_penalty = 0.0
        if self._fe_gradient > 0.05 and self._step_count > 10:
            _fe_goal_penalty = 0.3 * self._fe_gradient
            _goals = [G_forage, G_flee, G_explore, G_social]
            _goals[self.current_goal] += _fe_goal_penalty  # higher G = less preferred
            G_forage, G_flee, G_explore, G_social = _goals
        # Analytic EFE → bias for spiking goal selector
        # Add learned goal_bias (autograd flows for REINFORCE)
        G_base = torch.tensor([G_forage, G_flee, G_explore, G_social], device=self.device)
        G = G_base + self.meta_goal.goal_bias  # gradient through goal_bias
        efe_bias = _ec.efe_bias_amplitude * (G - G.min())

        # Spiking WTA goal selection — full pallium-D (no 75% truncation)
        # Detach efe_bias for goal_selector to avoid spurious gradients through spiking ops
        wta_out = self.goal_selector(pal_out['rate_D'],
                                      neuromod_bias=efe_bias.detach())
        _goal_probs_grad = torch.softmax(efe_bias, dim=0)  # with gradient for REINFORCE
        self.goal_probs = _goal_probs_grad.detach()         # detached for monitoring
        # Use WTA winner if confident, else fall back to analytic EFE
        _gs = self.cfg.goal_selection
        if wta_out['confidence'] > _gs.wta_confidence_threshold:
            new_goal = wta_out['winner']
        else:
            new_goal = int(self.goal_probs.argmax().item())

        # Override cascade (order matters: last override wins)
        food_px = float((L_type > 0.5).sum()) + float((R_type > 0.5).sum())

        # 1. Food visible reflex: FORAGE when food in retina and no real threat
        # Tighter p_enemy threshold (0.10 vs 0.15) improves structured decision accuracy
        if food_px > _gs.food_pixels_threshold and p_enemy < _gs.food_reflex_threat_gate:
            new_goal = GOAL_FORAGE
            self._forage_lock_steps = max(self._forage_lock_steps, _gs.forage_lock_duration)
        if self._forage_lock_steps > 0 and p_enemy < _gs.forage_exit_threat:
            new_goal = GOAL_FORAGE
            self._forage_lock_steps -= 1

        # 2. Critical starvation: must forage unless predator actively hunts
        # Raised threshold (0.45 vs 0.35) — was firing too early in decision scenarios
        if starvation > _gs.starvation_panic_threshold_1 and pred_state not in ('HUNT', 'AMBUSH'):
            new_goal = GOAL_FORAGE
        if starvation > _gs.starvation_panic_threshold_2 and pred_dist > _gs.starvation_panic_distance:
            new_goal = GOAL_FORAGE

        # 3. Threat overrides (visual OR lateral line OR amygdala evidence)
        ll_proximity = self.lateral_line_mod.proximity  # 0-1, high = close
        has_threat_evidence = (enemy_px > _gs.enemy_pixels_threat or ll_proximity > _gs.ll_proximity_threshold
                               or self.amygdala_alpha > _gs.amygdala_threat_threshold)
        # Adjust flee threshold by hunger: hungry fish takes more risk
        effective_flee_threshold = self._flee_threshold + starvation * 0.15

        # Predictive flee: anticipate predator intercept 8 steps ahead
        # Uses pred_model Kalman velocity estimate — no GT reads
        if (self.pred_model.visible or self.pred_model.steps_since_seen < 20):
            pfx = self.pred_model.x + 8 * self.pred_model.vx
            pfy = self.pred_model.y + 8 * self.pred_model.vy
            fx, fy = self._last_fish_pos
            intercept_dist = math.sqrt((pfx - fx) ** 2 + (pfy - fy) ** 2)
            if intercept_dist < _tc.intercept_flee_distance and pred_dist < _tc.intercept_flee_range and starvation < _gs.starvation_hunger_gate:
                # Predator trajectory converges — treat as soft threat
                p_enemy = max(p_enemy, effective_flee_threshold * _gs.threat_evidence_confidence)
                has_threat_evidence = True
        if p_enemy > effective_flee_threshold and has_threat_evidence and starvation < 0.6:
            new_goal = GOAL_FLEE
        # Close proximity flee: lateral line detects predator nearby
        if ll_proximity > _gs.ll_close_threshold or (pred_dist < _gs.pred_close_distance and enemy_px > 1):
            new_goal = GOAL_FLEE
        # Lateral line + amygdala: predator close even if not visible
        elif ll_proximity > _gs.ll_proximity_threshold and self.amygdala_alpha > _gs.amygdala_moderate_threshold and starvation < 0.5:
            new_goal = GOAL_FLEE

        # Stuck detection: force explore when not finding food
        eaten_now_check = getattr(env, '_eaten_now', 0)
        if eaten_now_check == 0:
            self._no_food_steps += 1
        else:
            self._no_food_steps = 0
        if (self._no_food_steps > _gs.no_food_timeout and self.current_goal == GOAL_FORAGE
                and starvation < _gs.starvation_explore_gate):
            new_goal = GOAL_EXPLORE
            self._force_explore_steps = _gs.force_explore_duration
            self._no_food_steps = 0
        if self._force_explore_steps > 0 and new_goal != GOAL_FLEE and starvation < _gs.starvation_explore_gate:
            new_goal = GOAL_EXPLORE
            self._force_explore_steps -= 1

        if self.goal_persistence > 0 and new_goal != GOAL_FLEE:
            # Strongly negative TD error: current goal yielding worse-than-expected outcome
            # → shorten persistence to allow faster recovery
            if self._last_td_error < _gs.td_error_shortening_threshold:
                self.goal_persistence = max(_gs.persistence_minimum, self.goal_persistence - _gs.td_error_shortening_amount)
            new_goal = self.current_goal
            self.goal_persistence -= 1
        else:
            if new_goal != self.current_goal:
                self.goal_persistence = _gs.new_goal_persistence
            self.current_goal = new_goal

        # Record step for REINFORCE update at episode end
        self.meta_goal.record_step(self.current_goal, _goal_probs_grad, _mc)

        # === 5. MOTOR COMMAND ===
        # Flee direction: pred_model Kalman estimate (no GT reads)
        flee_turn = 0.0
        if self.current_goal == GOAL_FLEE:
            enemy_L = float((torch.abs(L_type - 0.5) < 0.1).float().sum())
            enemy_R = float((torch.abs(R_type - 0.5) < 0.1).float().sum())
            total_enemy = enemy_L + enemy_R
            # Use pred_model Kalman position — biologically realistic (no GT cheat)
            # Falls back to retinal bearing when model confidence is low
            fx, fy = self._last_fish_pos
            pred_visible = self.pred_model.visible or self.pred_model.steps_since_seen < 10
            if total_enemy > 2 and pred_visible:
                esc_ang = math.atan2(fy - self.pred_model.y, fx - self.pred_model.x)
                esc_diff = math.atan2(
                    math.sin(esc_ang - fish_heading),
                    math.cos(esc_ang - fish_heading))
                flee_turn = float(np.clip(-esc_diff * 2.0, -1.0, 1.0))
                self._last_flee_turn = flee_turn
            elif total_enemy > 2:
                # Retinal fallback
                retinal_flee = (enemy_L - enemy_R) / (total_enemy + 1e-8)
                if abs(retinal_flee) < 0.15 and total_enemy > 5:
                    retinal_flee = self._escape_side * 0.6
                else:
                    self._escape_side = 1.0 if retinal_flee > 0 else -1.0
                flee_turn = max(-1.0, min(1.0, retinal_flee * 2.5))
                self._last_flee_turn = flee_turn
            elif ll_dist < 150:
                flee_turn = self._last_flee_turn * 0.3
            else:
                flee_turn = self._last_flee_turn * 0.2

        # Tectum lateral asymmetry → escape direction (correct biological source)
        # sgc_L_mean > sgc_R_mean: L_tectum active = R_eye input = RIGHT visual field = threat RIGHT
        _tect_flee = (tect_out['sgc_L_mean'] - tect_out['sgc_R_mean']) * 2.0
        flee_dir = 0.6 * flee_turn + 0.4 * _tect_flee  # blend retinal + tectal
        flee_dir = max(-1.0, min(1.0, flee_dir))
        if self.current_goal == GOAL_FLEE:
            flee_turn = flee_dir

        # Food approach: retinal bearing (L vs R food pixels)
        food_turn = 0.0
        if self.current_goal == GOAL_FORAGE:
            # Type channel: food pixels have value ~1.0 (>0.8) — use cached slices
            food_mask_L_raw = (L_type > 0.7).float()
            food_mask_R_raw = (R_type > 0.7).float()
            # Intensity-weighted: closer food (brighter) gets more weight
            food_L = float((food_mask_L_raw * L_int).sum())
            food_R = float((food_mask_R_raw * R_int).sum())
            food_L_px = float(food_mask_L_raw.sum())
            food_R_px = float(food_mask_R_raw.sum())
            total_food_px = food_L_px + food_R_px
            if total_food_px > 1:
                # Intensity-weighted bearing: brighter (closer) food dominates
                # v1 convention: (R - L) positive → turn right
                total_food_int = food_L + food_R + 1e-8
                food_turn = (food_R - food_L) / total_food_int
                food_turn = max(-1.0, min(1.0, food_turn * 2.5))
                # Weighted centroid for finer bearing (reuse cached slices)
                food_mask_L = (L_type > 0.7).float()
                food_mask_R = (R_type > 0.7).float()
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
                    food_turn = max(-1.0, min(1.0, -ang_R * 1.5))  # negate: food right → positive turn_rate
                elif food_L > 0:
                    food_turn = max(-1.0, min(1.0, -ang_L * 1.5))

        # Retinal approach (fallback / blend for EXPLORE)
        # v1 convention: retinal_turn = (retR - retL) / total
        # positive retinal_turn → turn right (CW in env)
        retL = float(L_int.sum())
        retR = float(R_int.sum())
        retinal_turn = (retR - retL) / (retL + retR + 1e-8)

        # Wall avoidance — angle-to-center method (matching v1)
        wall_turn = 0.0
        wx, wy = self._last_fish_pos
        aw = getattr(env, 'arena_w', 800)
        ah = getattr(env, 'arena_h', 600)
        heading = getattr(env, 'fish_heading', 0.0)
        margin = 120  # wider margin for earlier wall avoidance
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
            wall_turn = wall_urgency * 1.2 * np.sign(angle_diff)  # stronger wall avoidance
        if min(wx, wy, aw - wx, ah - wy) < 25:
            wall_turn *= 2.0
        wall_turn = max(-1.5, min(1.5, wall_turn))

        # Obstacle avoidance: bilateral rock pixel repulsion (use cached slices)
        obstacle_turn = 0.0
        rock_L = float((torch.abs(L_type - 0.75) < 0.1).float().sum())
        rock_R = float((torch.abs(R_type - 0.75) < 0.1).float().sum())
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
            # Shoaling turn bias when in social mode
            shoal_turn = self.shoaling.turn_bias if self.shoaling.n_neighbours > 0 else 0.0
            brain_turn = 0.4 * shoal_turn + 0.3 * retinal_turn + 0.3 * obstacle_turn
            alpha_s = 0.30

        # Pectoral fin: slow-turn kinematics (suppressed during flee)
        if self._active('pectoral_fin'):
            _food_bearing = 0.0
            if self.current_goal == GOAL_FORAGE and food_px > 0:
                _food_bearing = (float((R_type > 0.7).float().sum()) -
                                 float((L_type > 0.7).float().sum())) / (food_px + 1e-8)
            pect_out = self.pectoral_fin(
                food_bearing=_food_bearing,
                turn_request=brain_turn,
                goal=self.current_goal,
                goal_speed=self._last_speed)
            if pect_out['active']:
                brain_turn = 0.85 * brain_turn + 0.15 * pect_out['fin_turn']
        else:
            pect_out = {'fin_turn': 0.0, 'fin_L_rate': 0.0,
                        'fin_R_rate': 0.0, 'active': False}

        self._smoother = alpha_s * brain_turn + (1 - alpha_s) * self._smoother
        # v1 convention: turn_rate = -brain_turn * brain_weight + wall_turn
        brain_weight = max(0.2, 1.0 - wall_urgency)
        turn = float(np.clip(-self._smoother * brain_weight + wall_turn, -1.0, 1.0))

        # RS motor: C-start for looming + voluntary pallium-D asymmetry for normal movement
        _mc_cfg = self.body_cfg.motor
        rs_out = self.rs(tect_out['sgc'], bg_out['gate'], pal_out['rate_D'],
                         flee_turn, 1.0, tect_out['looming'])
        if rs_out['cstart']:
            turn = rs_out['turn']
        else:
            # RS voluntary turn: pallium-D L/R asymmetry gated by BG — activate when BG engaged
            # As pallium-D improves through STDP, this path gets better automatically
            bg_gate_val = float(bg_out['gate'])
            if bg_gate_val > _mc_cfg.bg_gate_threshold_rs and self.current_goal != GOAL_FLEE:
                turn = float(np.clip((1 - _mc_cfg.rs_weight) * turn + _mc_cfg.rs_weight * rs_out['turn'], -1.0, 1.0))

        # Speed (with allostatic fatigue cap)
        if self.current_goal == GOAL_FLEE:
            speed = _mc_cfg.speed_flee
        elif self.current_goal == GOAL_FORAGE:
            speed = _mc_cfg.speed_forage_food_visible if food_px > 2 and p_enemy < 0.1 else _mc_cfg.speed_forage
        elif self.current_goal == GOAL_EXPLORE:
            speed = _mc_cfg.speed_explore
        else:
            speed = _mc_cfg.speed_social
        speed *= self.allostasis.get_speed_cap()
        # Binocular depth: slow down for precise food approach
        speed *= self.binocular.get_approach_gain()

        # Prey capture kinematics: J-turn → approach → strike
        food_lateral = 0.0
        if food_px > 0:
            food_L_px = float((L_type > 0.7).float().sum())
            food_R_px = float((R_type > 0.7).float().sum())
            food_lateral = (food_R_px - food_L_px) / (food_L_px + food_R_px + 1e-8)
        food_dist_est = bino_out.get('food_distance', 999) if bino_out['food_confidence'] > 0.1 else (200.0 / (food_px + 1e-8))
        capture_result = self.prey_capture.update(
            self.current_goal, food_px, food_dist_est, food_lateral, rock_total)
        if capture_result is not None:
            turn, speed = capture_result

        self._last_speed = speed

        # === 5a. ACTIVE INFERENCE MOTOR (action-perception cycle) ===
        # Compute heading delta (proprioceptive feedback)
        heading_delta = fish_heading - self._prev_heading
        heading_delta = math.atan2(math.sin(heading_delta), math.cos(heading_delta))
        self._prev_heading = fish_heading

        # Food/enemy bearing for predictions
        _ai_food_bearing = 0.0
        if food_px > 0:
            _ai_food_bearing = (float((R_type > 0.7).float().sum()) -
                                float((L_type > 0.7).float().sum())) / (food_px + 1e-8)
        _ai_enemy_bearing = 0.0
        if enemy_px > 0:
            _ai_enemy_bearing = (float((torch.abs(R_type - 0.5) < 0.1).float().sum()) -
                                 float((torch.abs(L_type - 0.5) < 0.1).float().sum())) / (enemy_px + 1e-8)

        ai_motor = self.active_motor.step(
            goal=self.current_goal,
            food_bearing=_ai_food_bearing,
            enemy_bearing=_ai_enemy_bearing,
            wall_proximity=wall_urgency,
            food_visible=food_px > 2,
            enemy_visible=enemy_px > 2,
            gaze_target=self.saccade.gaze_offset,
            explore_phase=math.sin(self._step_count * 0.08),
            DA=self.neuromod.DA.item(),
            NA=self.neuromod.NA.item(),
            HT5=self.neuromod.HT5.item(),
            ACh=self.neuromod.ACh.item(),
            actual_speed=self._prop_cache['actual_speed'] if self._prop_cache else 0.0,
            heading_delta=heading_delta,
            tail_L=self.cpg.motor_L,
            tail_R=self.cpg.motor_R,
            gaze_offset=self.saccade.gaze_offset,
            collision=self.proprioception.collision,
            turn_rate=turn,
        )
        self._last_ai_motor = ai_motor

        # --- Adaptive active inference blend ---
        # The blend is NOT fixed — it depends on the brain's current state:
        #   High precision → trust PE-driven motor more (proper active inference)
        #   High spontaneity → less mechanistic (anti-FEP exploration/play)
        # This implements the tension between exploiting a generative model
        # (active inference) and breaking free of it (spontaneity).
        _mean_pi = float(self.active_motor.column.precision.mean())
        self._spontaneity = self._compute_spontaneity()
        _base_blend = self.active_motor.ai_blend  # 0.3 default
        _precision_bonus = 0.4 * _mean_pi          # 0 to ~0.4 at full precision
        _spontaneity_penalty = self._spontaneity * 0.3
        _blend = _base_blend + _precision_bonus - _spontaneity_penalty
        _blend = max(0.1, min(0.8, _blend))

        turn = (1.0 - _blend) * turn + _blend * ai_motor['turn']
        speed = (1.0 - _blend) * speed + _blend * (ai_motor['speed'] * 1.5)

        # Spontaneous motor perturbation: when spontaneity is high, inject
        # stochastic noise that violates FE minimization.  Biological basis:
        # default mode network, hippocampal replay → random exploration,
        # habenula frustration → try something new (Hikosaka 2010).
        if self._spontaneity > 0.3:
            _noise_scale = (self._spontaneity - 0.3) * 0.7
            turn += (np.random.random() - 0.5) * _noise_scale
            speed += (np.random.random() - 0.3) * _noise_scale * 0.5

        turn = float(np.clip(turn, -1.0, 1.0))
        speed = max(0.0, speed)

        # === 5b. SPINAL CPG (rhythmic motor output) ===
        # CPG drive: blend reactive + active inference predictions
        cpg_drive = min(1.0, speed / 1.5)  # normalize to [0,1]
        cpg_turn_input = (1.0 - _blend) * turn + _blend * ai_motor['cpg_bias']
        mL, mR, cpg_speed, cpg_turn, cpg_diag = self.cpg.step(cpg_drive, cpg_turn_input)
        # CPG modulation is very subtle — don't override brain motor commands
        # Only add slight phasic oscillation during calm movement
        if self.current_goal == GOAL_EXPLORE and abs(turn) < 0.3:
            turn = 0.9 * turn + 0.1 * cpg_turn

        # === 6. NEUROMOD UPDATE ===
        _nm = self.cfg.neuromod
        reward = _nm.survival_reward
        eaten_now = getattr(env, '_eaten_now', 0)
        if eaten_now > 0:
            reward += _nm.food_reward_gain * eaten_now
            self._forage_lock_steps = _nm.forage_lock_on_food
            self._da_phasic_steps = _nm.da_phasic_burst_duration
        self.energy = getattr(env, 'fish_energy', self.energy)
        self.world_model.update_food_gain(eaten_now > 0)

        # RL Critic: value estimation and TD learning
        critic_out = self.critic(
            energy=self.energy, threat=p_enemy,
            food_visible=food_px, goal=self.current_goal,
            DA=self.neuromod.DA.item(), NA=self.neuromod.NA.item(),
            cls_probs=self._cls_probs,
            reward=reward)
        self._last_td_error = critic_out['td_error']   # used in goal_persistence next step

        # Habenula: disappointment signal + per-goal frustration (gated — ablated by default)
        _hab_null = {'da_suppression': 0.0, 'ht5_suppression': 0.0, 'switch_signal': False,
                     'frustration': np.zeros(4), 'disappointment': 0.0, 'helplessness': 0.0}
        if self._active('habenula'):
            hab_out = self.habenula(
                reward=reward,
                expected_reward=self.critic.get_value(self.current_goal),
                aversion=self.amygdala_alpha,
                current_goal=self.current_goal,
                DA=self.neuromod.DA.item())
        else:
            hab_out = _hab_null

        # IPN: habenula relay → behavioral inhibition + DA/5-HT modulation
        # Receives MHb (aversion) and LHb (disappointment) rates from habenula
        _ipn_null = {'behavioral_inhibition': 0.0, 'da_feedback': 0.0,
                     'raphe_drive': 0.0, 'aversion_memory': 0.0,
                     'vipn_rate': 0.0, 'dipn_rate': 0.0, 'speed_multiplier': 1.0}
        if self._active('habenula'):
            ipn_out = self.ipn(
                mhb_rate=hab_out.get('mhb_rate', 0.0),
                lhb_rate=hab_out.get('lhb_rate', 0.0),
                aversion=self.amygdala_alpha)
        else:
            ipn_out = _ipn_null

        nm = self.neuromod.update(
            reward=reward, amygdala_alpha=self.amygdala_alpha,
            cms=self.cms, flee_active=(self.current_goal == GOAL_FLEE),
            fatigue=self.allostasis.fatigue, circadian=0.7,
            current_goal=self.current_goal)
        # HPA cortisol suppresses DA (anhedonia under chronic stress)
        self.neuromod.DA.mul_(self.hpa.da_suppression())
        self.neuromod.DA.clamp_(0.0, 1.0)
        # Phasic DA burst: override neuromod EMA for reward window
        if self._da_phasic_steps > 0:
            self.neuromod.DA.fill_(_nm.da_phasic_baseline + _nm.da_phasic_modulation * self._da_phasic_steps / float(_nm.da_phasic_burst_duration))
            self._da_phasic_steps -= 1
        elif self._active('habenula'):
            # Habenula suppression only outside phasic window — don't mute food reward
            if hab_out['da_suppression'] > _nm.habenula_da_suppression_gate:
                self.neuromod.DA.mul_(1.0 - hab_out['da_suppression'])
            if hab_out['ht5_suppression'] > _nm.habenula_ht5_suppression_gate:
                self.neuromod.HT5.mul_(1.0 - hab_out['ht5_suppression'])
            # IPN DA feedback: sustained aversion → additional DA suppression
            if ipn_out['da_feedback'] < -0.05:
                self.neuromod.DA.mul_(1.0 + ipn_out['da_feedback'])  # da_feedback is negative
                self.neuromod.DA.clamp_(0.0, 1.0)

        # ── Spiking raphe: override scalar 5-HT with population-coded output ──
        if self._active('raphe'):
            raphe_out = self.raphe(
                lhb_rate=hab_out.get('lhb_rate', 0.0),
                ipn_raphe_drive=ipn_out.get('raphe_drive', 0.0),
                amygdala_stress=self.amygdala_alpha,
                circadian=self._last_activity_drive,
                flee_active=(self.current_goal == GOAL_FLEE))
            self.neuromod.HT5.fill_(raphe_out['ht5_level'])
        else:
            raphe_out = {'ht5_level': self.neuromod.HT5.item(), 'dr_rate': 0.0,
                         'mr_rate': 0.0, 'sensory_gain': 1.0, 'patience': 0.5}

        # ── Spiking locus coeruleus: override scalar NA with population-coded output ──
        if self._active('locus_coeruleus'):
            lc_out = self.lc(
                amygdala_alpha=self.amygdala_alpha,
                insula_arousal=insula_out.get('arousal', 0.5),
                circadian=self._last_activity_drive,
                cms=self.cms)
            self.neuromod.NA.fill_(lc_out['na_level'])
        else:
            lc_out = {'na_level': self.neuromod.NA.item(), 'phasic': False,
                      'lc_rate': 0.0, 'wake_gate': 1.0, 'attention': 0.5}

        # ── Perturbation: apply drug multipliers to neuromodulators ──
        if self._perturbations is not None:
            _mults = self._perturbations.get_neuromod_multipliers()
            self.neuromod.DA.mul_(_mults.get('DA', 1.0))
            self.neuromod.NA.mul_(_mults.get('NA', 1.0))
            self.neuromod.HT5.mul_(_mults.get('5HT', 1.0))
            self.neuromod.ACh.mul_(_mults.get('ACh', 1.0))
            self.neuromod.DA.clamp_(0.0, 1.0)
            self.neuromod.NA.clamp_(0.0, 1.0)
            self.neuromod.HT5.clamp_(0.0, 1.0)
            self.neuromod.ACh.clamp_(0.0, 1.0)
            self._perturbations.step()

        # Goal selector reward-modulated learning (three-factor Hebbian)
        DA_now = self.neuromod.DA.item()
        if eaten_now > 0:
            # Food eaten → reinforce FORAGE channel
            self.goal_selector.reinforce(GOAL_FORAGE, pal_out['rate_D'], DA=DA_now)
        elif (self.current_goal == GOAL_FLEE and pred_dist > 150
              and self._pred_dist_gt < 100):
            # Predator successfully evaded (was close, now far) → reinforce FLEE
            self.goal_selector.reinforce(GOAL_FLEE, pal_out['rate_D'], DA=DA_now, eta=1e-4)

        # === 7. PLACE CELL UPDATE (proprioceptive — fish knows own position) ===
        px = getattr(env, 'fish_x', 400)
        py = getattr(env, 'fish_y', 300)
        self._last_fish_pos = (px, py)
        self.place(px, py, food_eaten=(eaten_now > 0), predator_near=(pred_dist < 150))

        # Saccade: gaze shift for active vision
        food_bearing = self.olfaction.food_gradient_dir if hasattr(self, 'olfaction') else 0.0
        enemy_bearing = math.atan2(
            getattr(env, 'pred_y', 300) - py,
            getattr(env, 'pred_x', 400) - px) - fish_heading if enemy_px > 1 else 0.0
        # Prediction-error per hemifield: DS motion + loom + thalamic novelty signal.
        # Thalamic delta |tc_now − tc_prev| cleanly separates left vs right visual field:
        #   _tc_delta[:n_tc_half] = TC_R change (left visual field, right thalamus)
        #   _tc_delta[n_tc_half:] = TC_L change (right visual field, left thalamus)
        # Novel input on right → TC_L fires unexpectedly → _tc_delta[n_tc_half:] high → _pe_R↑ → gaze right.
        # (Pallium pred_error cannot be used directly because W_tc_pals is dense/random,
        #  making pred_error spatially unstructured across pallium-S neurons.)
        _pe_L = (float(rgc_out['L_ds'].mean()) + float(rgc_out['L_loom'].mean())
                 + float(_tc_delta[:_n_tc_half].mean()) * 5.0)   # TC_R = left visual
        _pe_R = (float(rgc_out['R_ds'].mean()) + float(rgc_out['R_loom'].mean())
                 + float(_tc_delta[_n_tc_half:].mean()) * 5.0)   # TC_L = right visual
        saccade_out = self.saccade(
            food_bearing=food_bearing, enemy_bearing=enemy_bearing,
            current_goal=self.current_goal,
            salience_L=float(L_int.sum()), salience_R=float(R_int.sum()),
            pe_L=_pe_L, pe_R=_pe_R)

        # Geographic model update
        self.geo_model.update(
            px, py, eaten_now, pred_dist,
            pred_x=getattr(env, 'pred_x', None),
            pred_y=getattr(env, 'pred_y', None))

        # === 7b. SENSORY MODULES ===
        # Vestibular: pass efference copy (predicted motor output) for FEP
        _ai_pred_turn = ai_motor['turn'] if self._last_ai_motor else 0.0
        _ai_pred_speed = ai_motor['speed'] if self._last_ai_motor else 0.0
        vest_out = self.vestibular(fish_heading, speed, turn,
                                   predicted_turn=_ai_pred_turn,
                                   predicted_speed=_ai_pred_speed)
        # Set proprioceptive predictions from active motor (descending commands)
        self.proprioception.set_predictions(
            predicted_speed=ai_motor['speed'] if ai_motor else None,
            predicted_heading_delta=ai_motor['turn'] * 0.1 if ai_motor else None)
        prop_out = self.proprioception(px, py, speed, fish_heading)
        self._prop_cache = prop_out  # cached for next step's active inference motor
        color_out = self.color_vision(L_type, R_type)
        circ_out = self.circadian(light_level=0.7)
        sw_out = self.sleep_wake(
            circadian_melatonin=circ_out['melatonin'],
            arousal=self.insula.arousal,
            threat=self.amygdala_alpha)
        # Working memory: store goal + food direction
        wm_input = torch.zeros(32, device=self.device)
        wm_input[self.current_goal * 8:(self.current_goal + 1) * 8] = 0.5
        wm_out = self.working_mem(wm_input, gate=self.neuromod.ACh.item())

        # === 7c. SENSORY MODULE INTEGRATION ===
        # Apply sensory module outputs to the already-computed motor commands
        # Vestibular postural correction: counteract over-rotation during sharp turns
        turn += vest_out['postural_correction'] * 0.15
        # Pretectum OKR: compensatory gaze stabilization (image slip → counter-rotation)
        # Small contribution (0.05) to avoid overriding intentional turns
        turn += pretect_out['okr_velocity'] * 0.05
        # Saccade gaze micro-adjustment: active vision (only during non-flee)
        if saccade_out['saccade_active'] and self.current_goal != GOAL_FLEE:
            turn += saccade_out['gaze_offset'] * 0.1
        turn = float(np.clip(turn, -1.0, 1.0))
        # Store gaze offset on env so sensory_bridge uses it next step
        try:
            env.gaze_offset = saccade_out['gaze_offset']
        except Exception:
            pass
        # Sleep-wake: reduce responsiveness when drowsy (melatonin-driven)
        speed *= sw_out['responsiveness']
        # IPN behavioral inhibition: aversion → slow down (not during active flee)
        if self.current_goal != GOAL_FLEE and self._active('habenula'):
            speed *= ipn_out['speed_multiplier']
        speed = max(_mc_cfg.speed_min_clamp, min(_mc_cfg.speed_max_clamp, speed))
        # Proprioception: collision detected → trigger stuck recovery
        if prop_out['collision']:
            self._stuck_counter += 3
        # Update previous-step state for next step's EFE
        self._last_activity_drive = circ_out['activity_drive']

        # === 7d. VAE WORLD MODEL (online training) ===
        tect_all = tect_out['all_e'].unsqueeze(0)  # [1, N]
        state_ctx = np.array([
            px / 800.0, py / 600.0, fish_heading / math.pi,
            self.energy / 100.0,
            self.neuromod.DA.item(), critic_out['td_error'],
            0.0, 0.0,  # precision placeholders
            float(self._cls_probs[1].detach()),
            float(self._cls_probs[2].detach()),
            float(self._cls_probs[4].detach()),
            pal_out['free_energy'],
            self.cms,
        ], dtype=np.float32)
        # VAE training every 10 steps — backprop every step is wasteful (125k steps/50 rounds)
        if self._step_count % self.cfg.plasticity.vae_training_every == 0:
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

        # Habenula strategy switch: force goal change if frustrated (gated)
        if self._active('habenula') and hab_out['switch_signal'] and self.current_goal != GOAL_FLEE:
            # Switch to least-frustrated non-current goal
            frustration = hab_out['frustration']
            candidates = [g for g in range(4) if g != self.current_goal]
            best = min(candidates, key=lambda g: frustration[g])
            self.current_goal = best
            self.goal_persistence = 10

        # === 8. FEEDBACK LEARNING + STDP CONSOLIDATION ===
        self.fb_learner.update(
            h_upper=pal_out['rate_D'],
            pred_error=pal_out['pred_error'],
            pi=1.0)
        # Three-factor STDP consolidation every 5 steps — only when DA elevated
        # Low-DA consolidation corrupts weights with uncorrelated Hebbian noise
        _plast = self.cfg.plasticity
        if self._step_count % _plast.stdp_consolidation_every == 0:
            DA_now = self.neuromod.DA.item()
            ACh_now = self.neuromod.ACh.item()
            # Homeostatic scaling always (normalizes firing rates)
            self.stdp_tect_tc_L.homeostatic_scale(tc_L_out['TC'])
            self.stdp_tect_tc_R.homeostatic_scale(tc_R_out['TC'])
            self.stdp_tc_pal.homeostatic_scale(pal_out['rate_S'])
            self.stdp_pal_d.homeostatic_scale(pal_out['rate_D'])
            # Weight update only during elevated DA (reward signal present)
            if DA_now > _plast.da_consolidation_threshold or self._da_phasic_steps > 0:
                self.stdp_tect_tc_L.consolidate(DA_now, ACh_now, eta=_plast.tect_tc.consolidation_eta)
                self.stdp_tect_tc_R.consolidate(DA_now, ACh_now, eta=_plast.tect_tc.consolidation_eta)
                self.stdp_tc_pal.consolidate(DA_now, ACh_now, eta=_plast.tc_pal.consolidation_eta)
                self.stdp_pal_d.consolidate(DA_now, ACh_now, eta=_plast.pal_d.consolidation_eta)
        # Online Hebbian classifier fine-tuning: food confirmation → reinforce food class
        if eaten_now > 0 and self.classifier._last_hidden is not None:
            with torch.no_grad():
                target = torch.zeros(1, 5, device=self.device)
                target[0, 1] = 1.0  # class 1 = food
                cls_now = self._cls_probs.detach().unsqueeze(0)
                err = target - cls_now
                h = self.classifier._last_hidden  # (1, 128)
                eta_cls = _plast.classifier_eta * self.neuromod.ACh.item()
                self.classifier.W_out.weight.data.add_(eta_cls * err.T @ h)
                self.classifier.W_out.weight.data.clamp_(-3.0, 3.0)
        # Enemy class reinforcement: near-miss → strengthen class 2 (enemy) detection
        # NA (noradrenaline) drives threat memory consolidation
        if (pred_dist < 60 and self.amygdala_alpha > 0.3
                and self.classifier._last_hidden is not None):
            with torch.no_grad():
                target_e = torch.zeros(1, 5, device=self.device)
                target_e[0, 2] = 1.0  # class 2 = enemy
                cls_now_e = self._cls_probs.detach().unsqueeze(0)
                err_e = target_e - cls_now_e
                h_e = self.classifier._last_hidden
                eta_enemy = _plast.enemy_class_eta * self.neuromod.NA.item()
                self.classifier.W_out.weight.data.add_(eta_enemy * err_e.T @ h_e)
                self.classifier.W_out.weight.data.clamp_(-3.0, 3.0)

        # === 9. PREDICTIVE NETWORK (spiking world model) ===
        motor_cmd = torch.tensor([turn, speed], dtype=torch.float32, device=self.device)
        pred_out = self.predictive(rgc_out['on_fused'], motor_cmd)
        self._last_pred_surprise = pred_out['surprise']  # used in EFE next step

        # === 10. HABIT NETWORK ===
        retinal_summary = torch.tensor([
            float(L_int.sum()) / 400.0,
            float(R_int.sum()) / 400.0,
            float((L_type > 0.7).float().sum()) / 20.0,
            float((R_type > 0.7).float().sum()) / 20.0,
        ], device=self.device)
        habit_out = self.habit(
            cls_probs=self._cls_probs,
            goal=self.current_goal, turn=turn, speed=speed,
            retinal_summary=retinal_summary)

        # Store pallium-S rate for top-down attention next step
        self._prev_pal_rate_s.copy_(pal_out['rate_S'].detach())

        # --- Action-perception cycle: track free energy gradient ---
        # dF/dt > 0 → current policy failing (surprise rising)
        # dF/dt < 0 → current policy working (surprise falling)
        # Used NEXT step for F-based goal modulation and spontaneity
        _total_fe = self._aggregate_free_energy(pal_out, ai_motor, saccade_out)
        self._fe_history.append(_total_fe)
        if len(self._fe_history) >= 2:
            self._fe_gradient = self._fe_history[-1] - self._fe_history[-2]

        return {
            'turn': float(turn),
            'speed': float(speed),
            'goal': self.current_goal,
            'goal_probs': self.goal_probs.detach().cpu().numpy(),
            'DA': self.neuromod.DA.item(), 'NA': nm['NA'],
            '5HT': nm['5HT'], 'ACh': nm['ACh'],
            'free_energy': pal_out['free_energy'],
            'total_free_energy': _total_fe,
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
            'novelty': self._novelty_ema,
            'exploration_phase': self._exploration_phase,
            # Action-perception cycle diagnostics
            'ai_free_energy': ai_motor['free_energy'],
            'ai_blend': _blend,
            'ai_convergence': ai_motor.get('inference_convergence', 0.0),
            'fe_gradient': self._fe_gradient,
            'spontaneity': self._spontaneity,
            'gaze_free_energy': saccade_out.get('gaze_free_energy', 0.0),
            'gaze_precision': saccade_out.get('gaze_precision', 1.0),
            # Pretectum OKR
            'okr_velocity': pretect_out['okr_velocity'],
            'pretectum_dsi': pretect_out['dsi'],
            'retinal_slip': pretect_out['retinal_slip'],
            # IPN behavioral inhibition
            'ipn_inhibition': ipn_out['behavioral_inhibition'],
            'ipn_aversion_memory': ipn_out['aversion_memory'],
            # Gap junction state
            'gap_state_L': rs_out.get('gap_state_L', 0.0),
            'gap_state_R': rs_out.get('gap_state_R', 0.0),
            # Spiking raphe (5-HT)
            'raphe_ht5': raphe_out['ht5_level'],
            'raphe_dr_rate': raphe_out['dr_rate'],
            'raphe_patience': raphe_out.get('patience', 0.5),
            # Spiking LC (NA)
            'lc_na': lc_out['na_level'],
            'lc_phasic': lc_out['phasic'],
            'lc_attention': lc_out.get('attention', 0.5),
            # Pectoral fin
            'pect_fin_turn': pect_out['fin_turn'],
            'pect_fin_active': pect_out['active'],
        }

    # ------------------------------------------------------------------
    # Brain-level free energy aggregation
    # ------------------------------------------------------------------

    def _aggregate_free_energy(self, pal_out: dict, ai_motor: dict,
                                saccade_out: dict) -> float:
        """Sum variational free energy across all TwoCompColumn modules.

        Each module computes F = Σ πᵢ εᵢ² from within-neuron PE and
        learned precision (Lee, Lee & Park 2026).  The brain-level
        total is the global surprise signal available for monitoring,
        disorder assays, and homeostatic regulation.
        """
        fe = 0.0
        fe += self.retina.free_energy
        fe += self.olfaction.free_energy
        fe += self.lateral_line_mod.free_energy
        fe += self.proprioception.free_energy
        fe += self.vestibular.free_energy
        fe += self.color_vision.free_energy
        fe += self.thalamus_L.free_energy
        fe += self.thalamus_R.free_energy
        fe += pal_out.get('free_energy', 0.0)
        fe += ai_motor.get('free_energy', 0.0)
        fe += saccade_out.get('gaze_free_energy', 0.0)
        return float(fe)

    def get_module_free_energies(self) -> dict:
        """Return per-module free energy breakdown (for diagnostics)."""
        return {
            'retina':        self.retina.free_energy,
            'olfaction':     self.olfaction.free_energy,
            'lateral_line':  self.lateral_line_mod.free_energy,
            'proprioception':self.proprioception.free_energy,
            'vestibular':    self.vestibular.free_energy,
            'color_vision':  self.color_vision.free_energy,
            'thalamus_L':    self.thalamus_L.free_energy,
            'thalamus_R':    self.thalamus_R.free_energy,
            'pallium':       self.pallium.pc.free_energy,
            'active_motor':  self.active_motor.column.free_energy,
        }

    # ------------------------------------------------------------------
    # Spontaneity: anti-FEP mechanism
    # ------------------------------------------------------------------

    def _compute_spontaneity(self) -> float:
        """Compute spontaneity factor that opposes pure FE minimization.

        Real organisms don't always minimize free energy.  Exploration,
        play, frustration, curiosity, and neural noise all produce behavior
        that TEMPORARILY INCREASES prediction error.  This is not a bug —
        it's essential for avoiding local minima in the FE landscape and
        for discovering new behavioral strategies.

        Biological basis:
          - Default mode network: spontaneous mentation during disengagement
          - Hippocampal sharp-wave ripples: random replay → novel actions
          - Habenula: frustration / learned helplessness → try something new
            (Hikosaka 2010, "The habenula: from stress evasion to value-based
            decision-making")
          - Dopamine: phasic DA bursts encourage exploration beyond FE minimum
            (Friston et al. 2012, "Dopamine, affordance and active inference")
          - Play behavior: juvenile zebrafish exhibit spontaneous non-functional
            swimming patterns (Dreosti et al. 2015)

        Returns:
            float [0, 1]: 0 = fully mechanistic (pure FEP), 1 = fully spontaneous
        """
        s = 0.0

        # 1. Habenula frustration → break the action-perception loop
        #    When goals repeatedly fail (high helplessness), stop trying
        #    to minimize F and explore randomly instead.
        hab_help = float(getattr(self.habenula, 'helplessness', 0.0))
        s += hab_help * 0.4

        # 2. Boredom: F is stable (flat gradient) + low novelty
        #    The world is predictable but not rewarding → spontaneous exploration
        if abs(self._fe_gradient) < 0.02 and self._novelty_ema < 0.3:
            s += 0.25

        # 3. High DA → exploration beyond current FE minimum
        #    Phasic DA signals "try something new" — overrides FEP
        da = self.neuromod.DA.item()
        if da > 0.6:
            s += (da - 0.6) * 0.8

        # 4. Social context → playful / less mechanistic behavior
        if self.current_goal == GOAL_SOCIAL:
            s += 0.15

        # 5. Neural noise baseline: intrinsic stochasticity in all circuits
        s += 0.05

        return min(1.0, max(0.0, s))

    def set_perturbations(self, pm):
        """Attach a PerturbationManager for drug/stimulation effects."""
        self._perturbations = pm

    def set_region_enabled(self, region_name, enabled):
        """Enable/disable a brain region for ablation studies."""
        if enabled:
            self._ablated.discard(region_name)
        else:
            self._ablated.add(region_name)

    def _active(self, region_name):
        """Check if a region is active (not ablated)."""
        return region_name not in self._ablated

    def reset(self):
        self.retina.reset()
        self.tectum.reset()
        self.thalamus_L.reset()
        self.thalamus_R.reset()
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
        self.pretectum.reset()
        self.ipn.reset()
        self.raphe.reset()
        self.lc.reset()
        self.pectoral_fin.reset()
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
        self.active_motor.reset()
        self.color_vision.reset()
        self.circadian.reset()
        self.sleep_wake.reset()
        self.saccade.reset()
        self.geo_model.reset()
        self.binocular.reset()
        self.shoaling.reset()
        self.prey_capture.reset()
        self.stdp_tect_tc_L.reset()
        self.stdp_tect_tc_R.reset()
        self.stdp_tc_pal.reset()
        self.stdp_pal_d.reset()
        self._prev_pal_rate_s.zero_()
        self._prev_tc.zero_()
        self._da_phasic_steps = 0
        self._z_prev = None
        self._last_action_ctx = None
        self._prev_heading = 0.0
        self._last_ai_motor = None
        self._prop_cache = None
        self._novelty_ema = 1.0
        self._surprise_history = []
        self._exploration_phase = True
        self._fe_history.clear()
        self._fe_gradient = 0.0
        self._spontaneity = 0.0
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
        # Reset inter-episode carry-over state
        self._enemy_lateral_ema = 0.0
        self._rear_threat = 0.0
        self._last_flee_turn = 0.0
        self._escape_side = 1.0
        self._stuck_counter = 0
        self._prev_pos = (0.0, 0.0)
        self._last_fish_pos = (400.0, 300.0)
        self._last_speed = 1.0
        self._last_activity_drive = 1.0
        self._last_td_error = 0.0
        self._last_pred_surprise = 0.0
        self._pred_state = 'PATROL'
        self._pred_dist_gt = 999.0
        self._enemy_pixels = 0.0
        # Reset per-episode state (learned weights persist across episodes)
        self.social_mem.reset()
        self.meta_goal.clear_episode()
        self.hpa.reset()
        self.oxt.reset()  # discard any stale log_probs from aborted episodes

    def on_episode_end(self, fitness: float, goal_counts: dict = None):
        """
        Called at end of each episode to update all online-learned parameters.
        Trainer calls this after run_round() with the episode fitness score.

        Updates:
          - MetaGoalWeights via REINFORCE (goal_bias) + fitness correlation (mod_w)
          - SocialMemory episode reset (weights persist across episodes)
        """
        self.meta_goal.episode_update(fitness)
        self.social_mem.reset()
