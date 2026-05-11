"""
TwoSpotAssayProtocol — choice assay between two attractant sources.

Classic C. elegans assay: worm starts at plate centre, two odour spots
are placed at opposite sides. Preference Index (PI) = fraction near spot A
minus fraction near spot B after 60 minutes (modelled as 1000 steps).

Useful for:
  - Testing which circuit dominates when two attractants compete
  - Sweeping chemotaxis.gain or bilateral_diff_gain
  - Ablating AIY vs AIB to see which arm is preferred
"""
from __future__ import annotations

import numpy as np
from ..core.interfaces import BehavioralProtocol, Environment, BrainModule
from ..core.types import EpisodeResult, WorldState


class TwoSpotAssayProtocol(BehavioralProtocol):

    def __init__(self, n_steps: int = 1000):
        self._n_steps = n_steps
        self._arena_w = 100.0
        self._pi_history: list[float] = []

    @property
    def name(self) -> str:
        return "two_spot_assay"

    @property
    def episode_length(self) -> int:
        return self._n_steps

    def setup(self, env: Environment, brain: BrainModule) -> None:
        self._pi_history.clear()
        try:
            self._arena_w = env.W
        except AttributeError:
            pass
        # Require two-spot mode
        if hasattr(env, "mode") and env.mode != "two_spot":
            env.mode = "two_spot"

    def get_stimuli(self) -> list:
        return []

    def score(self, result: EpisodeResult, world_log: list[WorldState]) -> dict:
        pi_vals: list[float] = []
        ci_a_vals: list[float] = []
        ci_b_vals: list[float] = []

        for state in world_log:
            pi  = state.extras.get("preference_index", 0.0)
            ci_a = state.extras.get("ci_spot_a", 0.0)
            ci_b = state.extras.get("ci_spot_b", 0.0)
            pi_vals.append(pi)
            ci_a_vals.append(ci_a)
            ci_b_vals.append(ci_b)

        final_pi = float(pi_vals[-1]) if pi_vals else 0.0
        mean_pi  = float(np.mean(pi_vals)) if pi_vals else 0.0

        # Time to first preference: first step where |PI| > 0.3
        tpref = result.steps_survived
        for i, pi in enumerate(pi_vals):
            if abs(pi) > 0.3:
                tpref = i
                break

        return {
            "preference_index":      round(final_pi, 3),
            "mean_preference_index": round(mean_pi,  3),
            "ci_spot_a":  round(float(ci_a_vals[-1]) if ci_a_vals else 0.0, 3),
            "ci_spot_b":  round(float(ci_b_vals[-1]) if ci_b_vals else 0.0, 3),
            "time_to_preference": tpref,
        }

    def is_done(self, world_state: WorldState, t: int) -> bool:
        return False
