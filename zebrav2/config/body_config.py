"""
BodyConfig: sensory channels, motor output, and metabolism.

Parameters governing the physical body of the virtual zebrafish,
independent of brain implementation (spiking vs rate-coded).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Sensory channel configs
# ---------------------------------------------------------------------------

@dataclass
class RetinaConfig:
    """Retinal ganglion cell parameters."""
    n_per_type: int = 400        # ON/OFF per eye
    n_loom: int = 100            # looming detectors per eye
    n_ds: int = 100              # direction-selective per eye
    dog_center_sigma: float = 3.0
    dog_surround_sigma: float = 9.0
    off_ema_weight: float = 0.5
    loom_expand_gain: float = 20.0
    loom_size_threshold: float = 0.02


@dataclass
class LateralLineConfig:
    """Mechanoreceptor (neuromast) parameters."""
    sn_range: float = 200.0      # superficial neuromast range (px)
    cn_range: float = 150.0      # canal neuromast range (px)
    prey_range: float = 100.0
    sn_decay_scale: float = 50.0
    cn_decay_scale: float = 30.0
    sn_tonic_current: float = -2.0
    cn_tonic_current: float = -2.0


@dataclass
class OlfactionConfig:
    """Odorant detection parameters."""
    n_alarm: int = 10
    n_food: int = 10
    n_bilateral: int = 8
    lambda_food: float = 70.0         # food odor diffusion length (px)
    lambda_alarm: float = 100.0       # alarm substance diffusion length
    diffusion_denominator: float = 50.0
    alarm_tonic_current: float = -1.0
    food_tonic_current: float = -2.0
    bilateral_tonic_current: float = -2.0
    naris_offset: float = 5.0


@dataclass
class VestibularConfig:
    """Balance and acceleration sensing."""
    n_neurons: int = 6
    yaw_gain: float = 10.0
    forward_accel_gain: float = 5.0
    decel_threshold: float = 0.5
    decel_gain: float = 5.0
    tilt_scale: float = 0.5
    tilt_gain: float = 8.0
    postural_correction_gain: float = 0.1
    tonic_current: float = -1.0


@dataclass
class ProprioceptionConfig:
    """Body-state sensing."""
    n_neurons: int = 8
    wall_margin: float = 50.0
    collision_speed_threshold: float = 1.0
    collision_expected_speed: float = 0.5
    speed_gain: float = 5.0
    decel_max: float = 3.0
    decel_gain: float = 3.0
    heading_gain: float = 3.0
    wall_proximity_gain: float = 12.0
    collision_gain: float = 15.0
    tonic_current: float = -1.0


@dataclass
class ColorVisionConfig:
    """Spectral processing (UV/B/G/R cones)."""
    n_per_channel: int = 8
    food_spectrum: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.8, 0.4])
    enemy_spectrum: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.3, 0.7])
    conspecific_spectrum: list[float] = field(
        default_factory=lambda: [0.8, 0.5, 0.3, 0.2])
    rock_spectrum: list[float] = field(
        default_factory=lambda: [0.2, 0.3, 0.3, 0.3])
    drive_scale: float = 10.0
    noise_std: float = 0.3
    tonic_current: float = -2.0


@dataclass
class SensoryConfig:
    """All sensory channel parameters."""
    retina: RetinaConfig = field(default_factory=RetinaConfig)
    lateral_line: LateralLineConfig = field(default_factory=LateralLineConfig)
    olfaction: OlfactionConfig = field(default_factory=OlfactionConfig)
    vestibular: VestibularConfig = field(default_factory=VestibularConfig)
    proprioception: ProprioceptionConfig = field(default_factory=ProprioceptionConfig)
    color_vision: ColorVisionConfig = field(default_factory=ColorVisionConfig)


# ---------------------------------------------------------------------------
# Motor output configs
# ---------------------------------------------------------------------------

@dataclass
class CPGConfig:
    """Spinal central pattern generator parameters."""
    n_v2a: int = 16        # rhythm generator per side
    n_v0d: int = 8         # commissural inhibitory per side
    n_di6: int = 4         # burst-shaping per side
    n_mn: int = 12         # motor neurons per side
    n_rsh: int = 8         # Renshaw inhibitory per side
    tau_m: float = 0.5     # LIF membrane time constant
    tau_syn: float = 0.6   # synaptic time constant
    v_thresh: float = 0.5  # spike threshold
    v_reset: float = 0.0
    # Synaptic weights
    w_v2a_mn: float = 0.7
    w_v2a_v0d: float = 0.6
    w_v0d_cross: float = 0.9  # half-centre coupling
    w_v2a_di6: float = 0.4
    w_di6_v2a: float = 0.3
    w_di6_mn: float = 0.2
    w_mn_rsh: float = 0.5
    w_rsh_v2a: float = 0.6
    noise: float = 0.15


@dataclass
class MotorConfig:
    """Motor output and locomotion parameters."""
    cpg: CPGConfig = field(default_factory=CPGConfig)
    n_reticulospinal: int = 21

    # Speed by goal (multiplier on fish_speed_base)
    speed_flee: float = 1.5
    speed_forage: float = 1.0
    speed_forage_food_visible: float = 1.2
    speed_explore: float = 0.8
    speed_social: float = 0.7
    speed_min_clamp: float = 0.3
    speed_max_clamp: float = 2.0

    # Turn parameters
    turn_max_normal: float = 0.15   # rad/step
    turn_max_flee: float = 0.45

    # Motor smoothing (EMA alpha per goal)
    alpha_flee: float = 0.6
    alpha_forage: float = 0.50
    alpha_explore: float = 0.25
    alpha_social: float = 0.30

    # RS voluntary integration
    bg_gate_threshold_rs: float = 0.3
    rs_weight: float = 0.15

    # Exploration oscillation
    explore_amplitude: float = 0.3
    explore_frequency: float = 0.08
    scan_amplitude: float = 0.5
    scan_frequency: float = 0.15

    # Flee turn scaling
    flee_gain: float = 2.0
    retinal_flee_gain: float = 2.5
    flee_brain_weight: float = 1.5
    flee_obstacle_weight: float = 0.3

    # Forage turn scaling
    forage_turn_gain: float = 2.5
    forage_brain_weight: float = 0.8
    forage_obstacle_weight: float = 0.2

    # Wall avoidance
    wall_margin: float = 120.0
    wall_urgency_gain: float = 1.2
    wall_critical_distance: float = 25.0
    wall_critical_gain: float = 2.0
    wall_turn_limit: float = 1.5

    # Obstacle avoidance
    rock_obstacle_gain: float = 1.5
    rock_normalization: float = 30.0

    # Vestibular correction
    vestibular_motor_correction: float = 0.15


# ---------------------------------------------------------------------------
# Metabolism configs
# ---------------------------------------------------------------------------

@dataclass
class MetabolismConfig:
    """Energy system and survival parameters."""
    energy_start: float = 100.0
    energy_max: float = 100.0
    energy_per_food: float = 8.0
    energy_drain_base: float = 0.05
    speed_cost_scale: float = 0.03
    critical_energy: float = 25.0

    # Inverted-U motility: motility = max(min_motility, 4*e*(1-e))
    # where e = energy / energy_max
    min_motility: float = 0.15


# ---------------------------------------------------------------------------
# Top-level body config
# ---------------------------------------------------------------------------

@dataclass
class BodyConfig:
    """Complete body configuration for the virtual zebrafish."""
    sensory: SensoryConfig = field(default_factory=SensoryConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)

    # Fish body
    fish_speed_base: float = 3.0    # px/step base multiplier
    eat_radius: float = 35.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> BodyConfig:
        sensory = d.pop('sensory', {})
        motor = d.pop('motor', {})
        metabolism = d.pop('metabolism', {})

        sensory_cfg = SensoryConfig(
            retina=RetinaConfig(**sensory.get('retina', {})),
            lateral_line=LateralLineConfig(**sensory.get('lateral_line', {})),
            olfaction=OlfactionConfig(**sensory.get('olfaction', {})),
            vestibular=VestibularConfig(**sensory.get('vestibular', {})),
            proprioception=ProprioceptionConfig(**sensory.get('proprioception', {})),
            color_vision=ColorVisionConfig(**sensory.get('color_vision', {})),
        )
        cpg_cfg = CPGConfig(**motor.pop('cpg', {}))
        motor_cfg = MotorConfig(cpg=cpg_cfg, **motor)
        metab_cfg = MetabolismConfig(**metabolism)
        return cls(sensory=sensory_cfg, motor=motor_cfg, metabolism=metab_cfg, **d)

    @classmethod
    def from_json(cls, s: str) -> BodyConfig:
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> BodyConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
