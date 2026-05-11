"""
ExplorationProtocol — novel environment exploration assay.

Measures spatial coverage (fraction of arena visited) and time-to-first-food.
Useful for sweeping epistemic value weight (beta_epistemic).
"""
from __future__ import annotations
import numpy as np
from ..core.interfaces import BehavioralProtocol, Environment, BrainModule
from ..core.types import EpisodeResult, WorldState


class ExplorationProtocol(BehavioralProtocol):

    def __init__(self, grid_resolution: int = 20):
        self._grid = grid_resolution
        self._visited: set[tuple] = set()
        self._arena_w = 800
        self._arena_h = 600

    @property
    def name(self) -> str:
        return "exploration"

    @property
    def episode_length(self) -> int:
        return 500

    def setup(self, env: Environment, brain: BrainModule) -> None:
        self._visited.clear()
        try:
            self._arena_w = env.W
            self._arena_h = env.H
        except AttributeError:
            pass

    def get_stimuli(self) -> list:
        return []

    def score(self, result: EpisodeResult,
              world_log: list[WorldState]) -> dict:
        visited_cells: set[tuple] = set()
        time_to_food = result.steps_survived  # default: never found food
        for state in world_log:
            for agent in state.agents:
                cx = int(agent.position[0] / self._arena_w * self._grid)
                cy = int(agent.position[1] / self._arena_h * self._grid)
                visited_cells.add((cx, cy))
            if (state.extras.get("food_eaten", 0) > 0
                    and time_to_food == result.steps_survived):
                time_to_food = state.t

        total_cells = self._grid ** 2
        coverage = len(visited_cells) / total_cells
        return {
            "spatial_coverage": round(coverage, 3),
            "unique_cells_visited": len(visited_cells),
            "time_to_first_food": time_to_food,
            "food_eaten": result.food_eaten,
        }

    def is_done(self, world_state: WorldState, t: int) -> bool:
        return False
