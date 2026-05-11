"""
AlarmPropagationProtocol — social alarm spread assay (multi-agent).

Injects alarm substance at t=50; measures fraction of group fleeing
within 10 steps and mean alarm propagation latency.
"""
from __future__ import annotations
import numpy as np
from ..core.interfaces import BehavioralProtocol, Environment, BrainModule
from ..core.types import EpisodeResult, WorldState, AlarmSubstance


class AlarmPropagationProtocol(BehavioralProtocol):

    @property
    def name(self) -> str:
        return "alarm_propagation"

    @property
    def episode_length(self) -> int:
        return 300

    def setup(self, env: Environment, brain: BrainModule) -> None:
        pass

    def get_stimuli(self) -> list:
        return [AlarmSubstance(onset_t=50, duration=20)]

    def score(self, result: EpisodeResult,
              world_log: list[WorldState]) -> dict:
        fleeing_at_60 = []
        for state in world_log:
            if state.t < 50 or state.t > 70:
                continue
            n_agents = len(state.agents)
            if n_agents == 0:
                continue
            n_alarmed = sum(
                1 for a in state.agents if a.tags.get("alarmed", False)
            )
            fleeing_at_60.append(n_alarmed / n_agents)
        mean_flee = float(np.mean(fleeing_at_60)) if fleeing_at_60 else 0.0
        return {
            "flee_fraction": round(mean_flee, 3),
            "alarm_onset_t": 50,
        }

    def is_done(self, world_state: WorldState, t: int) -> bool:
        return False
