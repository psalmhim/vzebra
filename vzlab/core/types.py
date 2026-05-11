"""
Shared data types for the vzlab platform.
All types are pure numpy / dataclass — no torch dependency in core.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


# ── Physical world state ──────────────────────────────────────────────────────

@dataclass
class AgentBody:
    agent_id: int
    position: np.ndarray        # (2,) or (3,) world coords
    orientation: float          # radians
    velocity: np.ndarray        # (2,) or (3,)
    angular_velocity: float = 0.0
    alive: bool = True
    tags: dict = field(default_factory=dict)  # species-specific extras


@dataclass
class WorldState:
    t: int                                      # timestep
    agents: list[AgentBody] = field(default_factory=list)
    food_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    predator_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    obstacle_map: np.ndarray | None = None      # binary 2-D grid
    extras: dict = field(default_factory=dict)  # arena-specific state


# ── Raw physical signals (species-agnostic) ───────────────────────────────────

@dataclass
class PhotonField:
    """Light intensity per wavelength per angular bin."""
    angles: np.ndarray          # (N_bins,) radians
    wavelengths: np.ndarray     # (N_wl,) nm
    intensity: np.ndarray       # (N_bins, N_wl)


@dataclass
class ChemField:
    """Chemical concentration map at agent position."""
    species: list[str]          # e.g. ['food', 'alarm', 'predator']
    concentrations: np.ndarray  # (len(species),)
    gradient: np.ndarray        # (len(species), 2) — spatial gradient direction


@dataclass
class PressureField:
    """Hydrodynamic flow around agent (lateral line input)."""
    positions: np.ndarray       # (N_neuromasts, 2) relative positions
    flow_vectors: np.ndarray    # (N_neuromasts, 2)


@dataclass
class BodyState:
    """Proprioceptive / interoceptive body state."""
    position: np.ndarray
    velocity: np.ndarray
    orientation: float
    angular_velocity: float
    in_contact: bool = False
    energy: float = 1.0
    heart_rate: float = 1.0


@dataclass
class SocialField:
    """Visible conspecifics and their states."""
    positions: np.ndarray       # (N_visible, 2)
    orientations: np.ndarray    # (N_visible,)
    velocities: np.ndarray      # (N_visible, 2)
    alarm_states: np.ndarray    # (N_visible,) bool


@dataclass
class SensorySignals:
    """All raw physical signals delivered to one agent each timestep."""
    photon: PhotonField | None = None
    chem: ChemField | None = None
    pressure: PressureField | None = None
    body: BodyState | None = None
    social: SocialField | None = None
    extras: dict = field(default_factory=dict)


# ── Motor output ──────────────────────────────────────────────────────────────

@dataclass
class MotorOutput:
    agent_id: int
    turn: float = 0.0           # signed angular velocity
    speed: float = 1.0          # forward speed multiplier
    extras: dict = field(default_factory=dict)


# ── Hierarchical state (6 levels) ─────────────────────────────────────────────

@dataclass
class HierarchicalState:
    t: int
    agent_id: int
    L1_synaptic: dict = field(default_factory=dict)
    L2_neuron: dict = field(default_factory=dict)
    L3_circuit: dict = field(default_factory=dict)
    L4_region: dict = field(default_factory=dict)
    L5_behaviour: dict = field(default_factory=dict)
    L6_social: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "t": self.t,
            "agent_id": self.agent_id,
            "L1": self.L1_synaptic,
            "L2": self.L2_neuron,
            "L3": self.L3_circuit,
            "L4": self.L4_region,
            "L5": self.L5_behaviour,
            "L6": self.L6_social,
        }


# ── Stimuli (injectable) ──────────────────────────────────────────────────────

@dataclass
class Stimulus:
    kind: str                   # 'looming' | 'odorant' | 'alarm' | 'dark_flash' | 'electric'
    onset_t: int
    duration: int = 1
    params: dict = field(default_factory=dict)


@dataclass
class LoomingDisc(Stimulus):
    def __init__(self, onset_t: int, expansion_rate: float = 0.2,
                 duration: int = 20, position: tuple = (400, 400)):
        super().__init__('looming', onset_t, duration,
                         {'expansion_rate': expansion_rate, 'position': position})


@dataclass
class OdorantPulse(Stimulus):
    def __init__(self, onset_t: int, chemical: str = 'food',
                 concentration: float = 1.0, duration: int = 30):
        super().__init__('odorant', onset_t, duration,
                         {'chemical': chemical, 'concentration': concentration})


@dataclass
class AlarmSubstance(Stimulus):
    def __init__(self, onset_t: int, source_position: tuple = (400, 400),
                 duration: int = 50):
        super().__init__('alarm', onset_t, duration,
                         {'source_position': source_position})


@dataclass
class DarkFlash(Stimulus):
    def __init__(self, onset_t: int, duration: int = 5):
        super().__init__('dark_flash', onset_t, duration, {})


# ── Experiment result ─────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    episode: int
    steps_survived: int
    food_eaten: int
    predation_events: int
    total_reward: float
    hierarchical_log: list[HierarchicalState] = field(default_factory=list)
    extras: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    label: str
    species: str
    protocol: str
    episodes: list[EpisodeResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def mean_survival(self) -> float:
        return float(np.mean([e.steps_survived for e in self.episodes]))

    @property
    def mean_food(self) -> float:
        return float(np.mean([e.food_eaten for e in self.episodes]))

    @property
    def mean_predation(self) -> float:
        return float(np.mean([e.predation_events for e in self.episodes]))

    def summary(self) -> dict:
        return {
            "label": self.label,
            "species": self.species,
            "protocol": self.protocol,
            "n_episodes": len(self.episodes),
            "mean_survival": round(self.mean_survival, 1),
            "mean_food": round(self.mean_food, 2),
            "mean_predation": round(self.mean_predation, 2),
        }
