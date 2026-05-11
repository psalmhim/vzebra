"""
Four core ABCs for the vzlab platform.

  Connectome          — species connectivity data
  BrainModule         — neural computation
  Environment         — physical world + sensory transduction
  BehavioralProtocol  — experimental protocol + scoring

Every species implementation must satisfy these contracts.
The platform engine (VirtualOrganism, ExperimentRunner) only
depends on these ABCs — never on species-specific code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from .types import (
    WorldState, SensorySignals, MotorOutput,
    HierarchicalState, Stimulus, ExperimentResult, EpisodeResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONNECTOME
# ─────────────────────────────────────────────────────────────────────────────

class Connectome(ABC):
    """Species connectivity database.

    Provides neuron populations and synaptic projections derived from
    empirical data (EM reconstruction, calcium imaging atlas, etc.).
    """

    @property
    @abstractmethod
    def species(self) -> str:
        """Species identifier, e.g. 'danio_rerio', 'c_elegans'."""

    @property
    @abstractmethod
    def n_neurons(self) -> int:
        """Total neuron count in this connectome."""

    @property
    @abstractmethod
    def regions(self) -> list[str]:
        """All named brain regions in this connectome."""

    @abstractmethod
    def region_neurons(self, name: str) -> dict:
        """Return neuron population descriptor for a named region.

        Returns dict with at least:
          'n': int             — neuron count
          'cell_types': list   — cell type labels
          'positions': ndarray — (n, 3) 3-D atlas coordinates
        """

    @abstractmethod
    def projection(self, src: str, tgt: str) -> np.ndarray:
        """Return weight matrix (n_src, n_tgt) for a named projection.

        Returns zero matrix if projection is absent.
        """

    @abstractmethod
    def neuromodulatory_targets(self, modulator: str) -> list[str]:
        """Regions that receive a given neuromodulator (DA/NA/5HT/ACh)."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Connectome":
        """Load connectome from file(s) at path."""


# ─────────────────────────────────────────────────────────────────────────────
# 2. BRAIN MODULE
# ─────────────────────────────────────────────────────────────────────────────

class BrainModule(ABC):
    """A single brain region or the whole-brain neural computation unit.

    Receives raw SensorySignals each timestep and returns MotorOutput.
    Also exposes its full HierarchicalState for the platform logger.
    """

    @property
    @abstractmethod
    def species(self) -> str:
        """Species this module is implemented for."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Module identifier, e.g. 'zebrafish_brain_v2'."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all transient state for a new episode."""

    @abstractmethod
    def step(self, signals: SensorySignals, t: int) -> MotorOutput:
        """Process one timestep of sensory input → motor command."""

    @abstractmethod
    def get_hierarchical_state(self, t: int) -> HierarchicalState:
        """Return full 6-level state snapshot at time t."""

    # ── Intervention API ──────────────────────────────────────────────────────

    @abstractmethod
    def ablate(self, region: str) -> None:
        """Permanently silence a named region for this episode."""

    @abstractmethod
    def restore(self, region: str) -> None:
        """Re-enable a previously ablated region."""

    @abstractmethod
    def set_param(self, path: str, value: float) -> None:
        """Set a named parameter (dot-path notation).

        Examples:
          set_param('efe.beta_epistemic', 3.0)
          set_param('tectum.na_gain', 0.0)
          set_param('shoaling.gamma_s', 0.5)
        """

    @abstractmethod
    def get_param(self, path: str) -> float:
        """Read a named parameter."""

    @abstractmethod
    def list_regions(self) -> list[str]:
        """All ablatable region names."""

    @abstractmethod
    def list_params(self) -> list[str]:
        """All settable parameter paths."""


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class Environment(ABC):
    """Physical world simulation.

    Produces species-agnostic raw physical signals (photon field,
    chemical gradient, pressure field, body state, social field).
    Does NOT perform sensory transduction — that is the brain module's job.
    """

    @property
    @abstractmethod
    def species(self) -> str:
        """Primary species this environment is designed for.
        Other species may also run in it via the same interface.
        """

    @abstractmethod
    def reset(self, n_agents: int = 1, seed: int | None = None) -> WorldState:
        """Reset the world and return initial state."""

    @abstractmethod
    def step(self, motor_outputs: list[MotorOutput]) -> WorldState:
        """Advance physics one timestep given agent motor commands."""

    # ── Sensory signal accessors (raw physics, not transduced) ────────────────

    @abstractmethod
    def get_sensory_signals(self, agent_id: int) -> SensorySignals:
        """Return all raw physical signals for one agent at current state."""

    # ── Stimulus injection ────────────────────────────────────────────────────

    @abstractmethod
    def inject(self, stimulus: Stimulus) -> None:
        """Schedule a stimulus to be applied at its onset_t."""

    @abstractmethod
    def clear_stimuli(self) -> None:
        """Remove all scheduled stimuli."""

    # ── Multi-agent ───────────────────────────────────────────────────────────

    @abstractmethod
    def add_agent(self, agent_id: int, position: np.ndarray | None = None) -> None:
        """Register a new agent in the environment."""

    @abstractmethod
    def remove_agent(self, agent_id: int) -> None:
        """Remove an agent from the environment."""

    @property
    @abstractmethod
    def current_state(self) -> WorldState:
        """Most recent WorldState without advancing physics."""


# ─────────────────────────────────────────────────────────────────────────────
# 4. BEHAVIORAL PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralProtocol(ABC):
    """Experimental protocol: stimuli, episode structure, scoring.

    Protocols are species-agnostic — the same looming-disc protocol
    can be run on a zebrafish avatar and a Drosophila avatar.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol identifier, e.g. 'predator_avoidance'."""

    @property
    @abstractmethod
    def episode_length(self) -> int:
        """Number of timesteps per episode."""

    @abstractmethod
    def setup(self, env: Environment, brain: BrainModule) -> None:
        """Configure environment and brain parameters for this protocol."""

    @abstractmethod
    def get_stimuli(self) -> list[Stimulus]:
        """Return all stimuli to be injected during an episode."""

    @abstractmethod
    def score(self, result: EpisodeResult, world_log: list[WorldState]) -> dict:
        """Compute protocol-specific metrics from one episode.

        Returns dict of metric_name → float, e.g.:
          {'flee_latency': 4.2, 'escape_success': 0.84, 'distance_travelled': 310}
        """

    @abstractmethod
    def is_done(self, world_state: WorldState, t: int) -> bool:
        """Return True to terminate episode early (e.g., agent eaten)."""
