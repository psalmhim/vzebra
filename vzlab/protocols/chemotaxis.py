"""
ChemotaxisProtocol — standardised odorant gradient navigation assay.

Measures chemotaxis index (CI = fraction of time near attractant).
Works for C. elegans on AgarPlate; can also run for zebrafish olfaction.
"""
from __future__ import annotations
import numpy as np
from ..core.interfaces import BehavioralProtocol, Environment, BrainModule
from ..core.types import EpisodeResult, WorldState


class ChemotaxisProtocol(BehavioralProtocol):

    @property
    def name(self) -> str:
        return "chemotaxis"

    @property
    def episode_length(self) -> int:
        return 1000

    def setup(self, env: Environment, brain: BrainModule) -> None:
        pass

    def get_stimuli(self) -> list:
        return []   # gradient is always-on in AgarPlate

    def score(self, result: EpisodeResult,
              world_log: list[WorldState]) -> dict:
        ci_values = [
            s.extras.get("chemotaxis_index", 0.0)
            for s in world_log if s.extras
        ]
        mean_ci = float(np.mean(ci_values)) if ci_values else 0.0
        return {
            "chemotaxis_index": round(mean_ci, 3),
            "steps": result.steps_survived,
        }

    def is_done(self, world_state: WorldState, t: int) -> bool:
        return False
