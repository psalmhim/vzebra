"""
BrainConfig: neural architecture, EFE goal selection, neuromodulation, plasticity.

Extracts 200+ hardcoded parameters from brain_v2.py into a structured,
JSON-serializable configuration that researchers can modify without touching code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# EFE (Expected Free Energy) goal selection
# ---------------------------------------------------------------------------

@dataclass
class EFEConfig:
    """Active-inference goal selection coefficients."""
    # Base offsets per goal
    forage_offset: float = 0.15
    flee_offset: float = 0.35
    explore_offset: float = 0.20
    social_offset: float = 0.10

    # Prediction weights (negative = suppress when predicted)
    forage_food_weight: float = -0.8
    forage_uncertainty_weight: float = 0.2
    flee_enemy_weight: float = -0.8
    flee_cms_weight: float = 0.1
    explore_starvation_weight: float = 0.3
    explore_future_crisis_weight: float = 0.3
    social_starvation_weight: float = 0.2
    social_future_crisis_weight: float = 0.2

    # Energy / urgency
    starvation_threshold: float = 0.75   # energy ratio below which hunger activates
    critical_energy: float = 25.0
    energy_anxiety_steps: float = 200.0
    energy_anxiety_scale: float = 0.4
    starvation_urgency_weight: float = 3.0
    future_crisis_weight: float = 2.0
    middle_factor_amplitude: float = 4.0

    # Interoception coupling
    insula_forage_coupling: float = 0.5
    insula_flee_coupling: float = 0.5
    insula_explore_coupling: float = 0.5

    # Cerebellar PE → explore
    cerebellum_pe_threshold: float = 0.1
    cerebellum_pe_explore_weight: float = 0.1

    # Lateral line prey detection
    ll_prey_forage_weight: float = 0.12

    # Habenula frustration gating
    habenula_frustration_threshold: float = 0.65
    habenula_frustration_gain: float = 0.35
    habenula_goal_bias_weight: float = 0.15

    # Circadian modulation
    circadian_activity_threshold: float = 0.65
    circadian_efe_gain: float = 0.15
    circadian_forage_reduction: float = 0.5

    # Predictive surprise → explore
    surprise_threshold: float = 0.1
    surprise_explore_weight: float = 0.15

    # 5-HT patrol suppression
    ht5_forage_suppression: float = 0.15
    safe_explore_boost: float = 0.1

    # World model coupling
    world_model_efe_scale: float = 0.15

    # EFE → motor bias
    efe_bias_amplitude: float = -2.0

    # Place cell bonus
    place_cell_forage_weight: float = 1.0  # subtracted from G_forage


@dataclass
class GoalSelectionConfig:
    """Goal persistence, overrides, and state machine."""
    # WTA confidence
    wta_confidence_threshold: float = 0.4

    # Food-visible reflex
    food_pixels_threshold: int = 1
    food_reflex_threat_gate: float = 0.10
    forage_lock_duration: int = 20

    # Forage persistence
    forage_exit_threat: float = 0.12

    # Starvation overrides
    starvation_panic_threshold_1: float = 0.45
    starvation_panic_threshold_2: float = 0.70
    starvation_panic_distance: float = 80.0

    # Threat evidence integration
    enemy_pixels_threat: int = 3
    ll_proximity_threshold: float = 0.15
    amygdala_threat_threshold: float = 0.25
    starvation_hunger_gate: float = 0.55
    threat_evidence_confidence: float = 0.85
    ll_close_threshold: float = 0.4
    pred_close_distance: float = 60.0
    amygdala_moderate_threshold: float = 0.2

    # Stuck detection → force explore
    no_food_timeout: int = 30
    starvation_explore_gate: float = 0.4
    force_explore_duration: int = 15

    # Goal persistence state machine
    td_error_shortening_threshold: float = -0.5
    td_error_shortening_amount: int = 3
    persistence_minimum: int = 2
    new_goal_persistence: int = 8


# ---------------------------------------------------------------------------
# Neuromodulation
# ---------------------------------------------------------------------------

@dataclass
class NeuromodConfig:
    """Neuromodulator dynamics and coupling."""
    # Reward computation
    survival_reward: float = 0.01
    food_reward_gain: float = 10.0
    forage_lock_on_food: int = 25
    da_phasic_burst_duration: int = 6

    # DA phasic
    da_phasic_baseline: float = 0.90
    da_phasic_modulation: float = 0.02

    # Habenula suppression
    habenula_da_suppression_gate: float = 0.05
    habenula_ht5_suppression_gate: float = 0.05


# ---------------------------------------------------------------------------
# Plasticity / STDP
# ---------------------------------------------------------------------------

@dataclass
class STDPPathwayConfig:
    """STDP parameters for one synaptic pathway."""
    a_plus: float = 0.002
    a_minus: float = 0.001
    w_max: float = 0.8
    w_min: float = 0.0
    consolidation_eta: float = 1e-4


@dataclass
class PlasticityConfig:
    """All learning-rate and plasticity parameters."""
    # Visual pathway STDP
    tect_tc: STDPPathwayConfig = field(
        default_factory=lambda: STDPPathwayConfig(
            a_plus=0.002, a_minus=0.001, consolidation_eta=1e-4))
    tc_pal: STDPPathwayConfig = field(
        default_factory=lambda: STDPPathwayConfig(
            a_plus=0.002, a_minus=0.001, consolidation_eta=1e-4))
    pal_d: STDPPathwayConfig = field(
        default_factory=lambda: STDPPathwayConfig(
            a_plus=0.001, a_minus=0.001, consolidation_eta=5e-5))

    # Classifier learning
    classifier_eta: float = 8e-6
    enemy_class_eta: float = 6e-6

    # Goal selector reinforcement
    goal_selector_eta: float = 1e-4

    # Top-down attention
    top_down_gain: float = 0.05

    # DA threshold for consolidation
    da_consolidation_threshold: float = 0.55

    # STDP consolidation frequency
    stdp_consolidation_every: int = 5

    # Synaptic dropout during STDP consolidation (0 = off, 0.1 = 10% synapses
    # silenced per step).  Builds fault-tolerant distributed representations.
    stdp_dropout_p: float = 0.10

    # VAE training frequency
    vae_training_every: int = 10


# ---------------------------------------------------------------------------
# Threat processing
# ---------------------------------------------------------------------------

@dataclass
class ThreatConfig:
    """Threat detection and fear memory parameters."""
    # Enemy probability fusion
    enemy_pixel_normalization: float = 15.0
    tect_threat_scale: float = 5.0
    tect_threat_weight: float = 0.2
    amygdala_weight_in_fusion: float = 0.15

    # Food probability fusion
    p_food_retinal_weight: float = 0.5
    p_food_uv_prey_weight: float = 0.1

    # CMS (novelty/fear state)
    cms_amygdala_weight: float = 0.3
    cms_baseline: float = 0.1

    # Rear threat memory
    amygdala_rear_threshold: float = 0.15
    rear_threat_memory_weight: float = 0.8
    rear_threat_decay: float = 0.9
    enemy_lateral_ema_blend: float = 0.7

    # Lateral line thresholds
    ll_hunt_threshold: float = 80.0
    ll_stalk_threshold: float = 150.0
    ll_distance_noise_std: float = 15.0
    ll_max_detection_range: float = 150.0
    olfaction_conspecific_distance: float = 40.0

    # Predator state
    patrol_threshold_distance: float = 150.0
    intercept_flee_distance: float = 110.0
    intercept_flee_range: float = 280.0
    predictive_flee_lookahead: int = 8


# ---------------------------------------------------------------------------
# Novelty / world learning
# ---------------------------------------------------------------------------

@dataclass
class NoveltyConfig:
    """World learning and novelty-seeking parameters."""
    place_visit_novelty_gain: float = 0.1
    place_novelty_weight: float = 0.5
    vae_novelty_weight: float = 0.5
    novelty_ema_old: float = 0.98
    novelty_ema_new: float = 0.02


# ---------------------------------------------------------------------------
# Module enable/disable (for ablation studies)
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Enable/disable individual brain modules for ablation studies."""
    habenula: bool = False     # ablated by default (performance gain)
    insula: bool = False       # ablated by default (performance gain)
    cerebellum: bool = True
    amygdala: bool = True
    place_cells: bool = True
    working_memory: bool = True
    predictive_net: bool = True
    habit_network: bool = True
    circadian: bool = True
    sleep_wake: bool = True
    lateral_line: bool = True
    olfaction: bool = True
    vestibular: bool = True
    proprioception: bool = True
    color_vision: bool = True
    shoaling: bool = True
    binocular_depth: bool = True
    social_memory: bool = True
    hpa_axis: bool = True
    oxytocin: bool = True
    meta_goal: bool = True
    vae_world_model: bool = True
    geographic_model: bool = True
    saccade: bool = True
    rl_critic: bool = True
    prey_capture: bool = True
    classifier: bool = True
    predator_model: bool = True
    internal_model: bool = True
    spinal_cpg: bool = True
    goal_selector: bool = True
    allostasis: bool = True
    raphe: bool = True
    locus_coeruleus: bool = True
    habituation: bool = True
    pectoral_fin: bool = True
    hypothalamus: bool = True
    pineal: bool = True
    inferior_olive: bool = True
    dl_pallium: bool = True
    vagus_nerve: bool = True
    pituitary: bool = True
    area_postrema: bool = True
    nts: bool = True
    ll_efferent: bool = True


# ---------------------------------------------------------------------------
# Top-level brain config
# ---------------------------------------------------------------------------

@dataclass
class BrainConfig:
    """Complete brain configuration for the virtual zebrafish."""
    efe: EFEConfig = field(default_factory=EFEConfig)
    goal_selection: GoalSelectionConfig = field(default_factory=GoalSelectionConfig)
    neuromod: NeuromodConfig = field(default_factory=NeuromodConfig)
    plasticity: PlasticityConfig = field(default_factory=PlasticityConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    # Personality preset
    personality: str = 'default'  # 'default', 'bold', 'shy', 'explorer', 'social'

    # Fidelity level
    fidelity: str = 'spiking'  # 'spiking', 'rate_coded', 'minimal'

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> BrainConfig:
        efe = EFEConfig(**d.pop('efe', {}))
        goal_sel = GoalSelectionConfig(**d.pop('goal_selection', {}))
        neuromod = NeuromodConfig(**d.pop('neuromod', {}))

        # Plasticity has nested STDPPathwayConfig
        plast_d = d.pop('plasticity', {})
        tect_tc = STDPPathwayConfig(**plast_d.pop('tect_tc', {}))
        tc_pal = STDPPathwayConfig(**plast_d.pop('tc_pal', {}))
        pal_d = STDPPathwayConfig(**plast_d.pop('pal_d', {}))
        plasticity = PlasticityConfig(
            tect_tc=tect_tc, tc_pal=tc_pal, pal_d=pal_d, **plast_d)

        threat = ThreatConfig(**d.pop('threat', {}))
        novelty = NoveltyConfig(**d.pop('novelty', {}))
        ablation = AblationConfig(**d.pop('ablation', {}))

        return cls(
            efe=efe, goal_selection=goal_sel, neuromod=neuromod,
            plasticity=plasticity, threat=threat, novelty=novelty,
            ablation=ablation, **d)

    @classmethod
    def from_json(cls, s: str) -> BrainConfig:
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> BrainConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def get_ablated_set(self) -> set[str]:
        """Return set of module names that are disabled."""
        return {name for name, enabled in asdict(self.ablation).items()
                if not enabled}
