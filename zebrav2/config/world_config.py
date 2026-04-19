"""
WorldConfig: arena geometry, entity spawning, predator AI, and physics.

All parameters JSON-serializable for web dashboard communication.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class ArenaConfig:
    """Arena geometry and boundaries."""
    width: int = 800
    height: int = 600
    wall_collision_buffer: int = 10

    # Food patches (3 default patches with Gaussian clusters)
    n_food: int = 20
    food_respawn: bool = True
    food_respawn_min: int = 5
    food_per_patch_fraction: float = 0.75      # fraction of food in patches
    food_cluster_spread: float = 25.0          # Gaussian sigma within patch
    food_patch_jitter: float = 30.0            # random offset from patch center

    # Rocks / obstacles
    rocks_enabled: bool = True
    rocks_per_patch: int = 8
    rock_gap_count: int = 2
    rock_distance_from_center: float = 60.0
    rock_base_radius_min: float = 15.0
    rock_base_radius_max: float = 35.0
    scattered_rocks_min: int = 5
    scattered_rocks_max: int = 10
    scattered_rock_radius_min: float = 20.0
    scattered_rock_radius_max: float = 45.0
    rock_min_distance_to_patches: float = 80.0


@dataclass
class PredatorConfig:
    """Predator AI behavior parameters."""
    enabled: bool = True
    ai_mode: str = 'intelligent'  # 'simple', 'intelligent', 'none'

    # Energy
    energy_start: float = 80.0
    energy_drain_passive: float = 0.08
    energy_per_catch: float = 40.0
    speed_cost_scale: float = 0.0005

    # Stamina
    stamina_start: float = 1.0
    stamina_drain_hunt: float = 0.02
    stamina_regen_base: float = 0.008
    stamina_regen_stalk_factor: float = 0.5
    burst_threshold: float = 0.3

    # Speeds (px/step)
    speed_patrol: float = 1.5
    speed_stalk: float = 1.0
    speed_hunt: float = 4.0
    speed_rest: float = 0.3
    speed_ambush: float = 0.1
    hunt_speed_hunger_gain: float = 0.4  # speed *= (1 + gain * hunger)

    # Detection
    detect_range_base: float = 250.0
    detect_range_hunger_gain: float = 0.5
    stalk_range_base: float = 150.0
    stalk_range_hunger_gain: float = 0.5
    catch_radius: float = 20.0

    # Timers
    hunt_duration_base: int = 30
    hunt_duration_hunger_gain: float = 0.67
    rest_duration: int = 20
    ambush_patience: int = 40

    # Intelligence
    navigation_noise: float = 8.0
    distraction_chance_base: float = 0.03
    distraction_hunger_suppress: float = 0.83
    prediction_lookahead: int = 3

    # Place cells
    place_cell_count: int = 64
    place_cell_sigma: float = 80.0
    place_cell_learning_rate: float = 0.05
    place_cell_prey_decay: float = 0.998
    place_cell_catch_decay: float = 0.999


@dataclass
class WorldConfig:
    """Complete world/environment configuration."""
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    predator: PredatorConfig = field(default_factory=PredatorConfig)

    # Episode
    max_steps: int = 500
    seed: int | None = None

    # Vision/sensory physics
    fov_degrees: float = 200.0
    max_vision_distance: float = 500.0
    retinal_intensity_scale: float = 15.0

    # Visual type codes (retinal encoding of object identity)
    type_code_predator: float = 0.5
    type_code_food: float = 0.8
    type_code_rock: float = 0.3
    type_code_conspecific: float = 0.25

    # Visual radii (for retinal projection)
    visual_radius_predator: float = 20.0
    visual_radius_food: float = 12.0
    visual_radius_rock: float = 30.0
    visual_radius_conspecific: float = 8.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> WorldConfig:
        arena = ArenaConfig(**d.pop('arena', {}))
        predator = PredatorConfig(**d.pop('predator', {}))
        return cls(arena=arena, predator=predator, **d)

    @classmethod
    def from_json(cls, s: str) -> WorldConfig:
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> WorldConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
