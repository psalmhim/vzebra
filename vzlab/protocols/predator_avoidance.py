"""
PredatorAvoidanceProtocol — standardised escape assay.

Injects a looming disc at t=100, measures flee latency and escape success.
Works identically for any species avatar run in AquaticArena.
"""
from __future__ import annotations
import numpy as np
from ..core.interfaces import BehavioralProtocol, Environment, BrainModule
from ..core.types import EpisodeResult, WorldState, LoomingDisc


class PredatorAvoidanceProtocol(BehavioralProtocol):

    @property
    def name(self) -> str:
        return "predator_avoidance"

    @property
    def episode_length(self) -> int:
        return 500

    def setup(self, env: Environment, brain: BrainModule) -> None:
        pass

    def get_stimuli(self) -> list:
        return [LoomingDisc(onset_t=100, expansion_rate=0.25, duration=30)]

    def score(self, result: EpisodeResult,
              world_log: list[WorldState]) -> dict:
        survived = result.steps_survived
        predated = result.predation_events
        # flee latency: first step where agent changes direction sharply
        flee_latency = survived  # default: never fled
        for i, state in enumerate(world_log):
            if i < 95:
                continue
            for agent in state.agents:
                if abs(agent.tags.get("turn", 0.0)) > 0.3:
                    flee_latency = i - 100
                    break
            if flee_latency < survived:
                break
        return {
            "steps_survived": survived,
            "predation_events": predated,
            "escape_success": float(predated == 0),
            "flee_latency": float(max(0, flee_latency)),
        }

    def is_done(self, world_state: WorldState, t: int) -> bool:
        return world_state.extras.get("predation_events", 0) > 0
