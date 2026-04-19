"""
Hierarchical configuration system for the Virtual Zebrafish platform.

Three config levels:
    WorldConfig  — arena, entities, stimuli, physics
    BodyConfig   — sensory channels, motor output, metabolism
    BrainConfig  — neural architecture, EFE, neuromodulation, plasticity
"""
from zebrav2.config.world_config import WorldConfig
from zebrav2.config.body_config import BodyConfig
from zebrav2.config.brain_config import BrainConfig

__all__ = ['WorldConfig', 'BodyConfig', 'BrainConfig']
