"""
OlfactoryConditioningProtocol — Drosophila olfactory learning assay.

Episode length: 500 steps (configurable).

Scoring:
  approach_index    mean approach-to-CS+ index across episode (higher = better)
  cs_plus_visits    number of steps spent in CS+ reward zone
  cs_minus_avoidance fraction of time the agent is away from CS- zone

Setup ensures both CS+ and CS- are active simultaneously.
"""
from __future__ import annotations

import numpy as np

from ..core.interfaces import BehavioralProtocol, Environment, BrainModule
from ..core.types import EpisodeResult, WorldState


class OlfactoryConditioningProtocol(BehavioralProtocol):

    def __init__(self, n_steps: int = 500):
        self._n_steps = n_steps

    @property
    def name(self) -> str:
        return "olfactory_conditioning"

    @property
    def episode_length(self) -> int:
        return self._n_steps

    def setup(self, env: Environment, brain: BrainModule) -> None:
        """Ensure both CS+ and CS- are present (OlfactoryArena handles this by default)."""
        pass

    def get_stimuli(self) -> list:
        return []   # odor gradients are always-on in OlfactoryArena

    def score(self, result: EpisodeResult, world_log: list[WorldState]) -> dict:
        approach_values = []
        cs_plus_visits  = 0
        cs_minus_steps  = 0
        total_steps     = max(1, len(world_log))

        for state in world_log:
            if not state.extras:
                continue
            ai = state.extras.get("approach_index", 0.0)
            approach_values.append(ai)

            if state.extras.get("in_cs_plus_zone", False):
                cs_plus_visits += 1

            if state.extras.get("in_cs_minus_zone", False):
                cs_minus_steps += 1

        mean_approach       = float(np.mean(approach_values)) if approach_values else 0.0
        cs_minus_avoidance  = float(1.0 - cs_minus_steps / total_steps)

        return {
            "approach_index":      round(mean_approach, 3),
            "cs_plus_visits":      cs_plus_visits,
            "cs_minus_avoidance":  round(cs_minus_avoidance, 3),
            "steps":               result.steps_survived,
        }

    def is_done(self, world_state: WorldState, t: int) -> bool:
        return False
