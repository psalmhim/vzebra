"""
AbstractBrain: protocol for brain implementations (spiking, rate-coded, minimal).

All brain implementations must conform to this interface so that
VirtualZebrafish can swap between fidelity levels transparently.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AbstractBrain(Protocol):
    """Protocol that all brain implementations must satisfy."""

    def step(self, obs: dict, env: Any = None) -> dict:
        """Process one timestep of sensory input and return motor commands.

        Args:
            obs: sensory observation dict (retinal_L/R, fish_pos, etc.)
            env: optional environment reference for reading state

        Returns:
            dict with at least 'turn' (float) and 'speed' (float)
        """
        ...

    def reset(self) -> None:
        """Reset all transient state for a new episode."""
        ...

    @property
    def current_goal(self) -> int:
        """Current active goal (0=FORAGE, 1=FLEE, 2=EXPLORE, 3=SOCIAL)."""
        ...

    @property
    def energy(self) -> float:
        """Current energy level."""
        ...

    def set_region_enabled(self, region_name: str, enabled: bool) -> None:
        """Enable/disable a brain region for ablation studies."""
        ...
