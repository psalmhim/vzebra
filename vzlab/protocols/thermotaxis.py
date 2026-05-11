"""
ThermotaxisProtocol — AFD-driven temperature gradient navigation.

The worm is placed on an AgarPlate in "thermal" mode with a linear
temperature gradient (cold left, warm right by default). Worm's
cultivation temperature is set to T_cult (°C). The goal is to reach
and maintain position at T_cult.

Thermotaxis Index (TI):
  TI = 1 - |T_worm - T_cult| / (T_warm - T_cold)
  TI = 1.0 means worm is exactly at cultivation temp.
  TI = 0.0 means worm is at the extreme end.

Useful for:
  - Sweeping circuit.tau to see effect on gradient integration
  - Ablating AFD neurons (AFDL/AFDR) to abolish thermotaxis
  - Sweeping neuromod.serotonin_tone (serotonin impairs thermotaxis)
"""
from __future__ import annotations

import numpy as np
from ..core.interfaces import BehavioralProtocol, Environment, BrainModule
from ..core.types import EpisodeResult, WorldState


class ThermotaxisProtocol(BehavioralProtocol):

    def __init__(self, t_cultivation: float = 20.0, n_steps: int = 1000):
        self._t_cult  = t_cultivation
        self._n_steps = n_steps
        self._t_cold  = 15.0
        self._t_warm  = 25.0

    @property
    def name(self) -> str:
        return "thermotaxis"

    @property
    def episode_length(self) -> int:
        return self._n_steps

    def setup(self, env: Environment, brain: BrainModule) -> None:
        # Switch environment to thermal mode
        if hasattr(env, "mode"):
            env.mode = "thermal"
        if hasattr(env, "temp_cold"):
            self._t_cold = env.temp_cold
        if hasattr(env, "temp_warm"):
            self._t_warm = env.temp_warm

    def get_stimuli(self) -> list:
        return []

    def score(self, result: EpisodeResult, world_log: list[WorldState]) -> dict:
        ti_vals: list[float] = []
        temp_range = self._t_warm - self._t_cold

        for state in world_log:
            for agent in state.agents:
                if temp_range > 0:
                    # Infer worm temperature from x-position on plate
                    t_worm = (self._t_cold
                              + (self._t_warm - self._t_cold)
                              * agent.position[0] / 100.0)
                    ti = 1.0 - abs(t_worm - self._t_cult) / temp_range
                    ti_vals.append(max(0.0, ti))

        mean_ti  = float(np.mean(ti_vals)) if ti_vals else 0.0
        final_ti = float(ti_vals[-1])      if ti_vals else 0.0

        # Thermotaxis accuracy: fraction of steps with TI > 0.7
        accuracy = float(np.mean([v > 0.7 for v in ti_vals])) if ti_vals else 0.0

        return {
            "thermotaxis_index":       round(mean_ti,  3),
            "final_thermotaxis_index": round(final_ti, 3),
            "thermotaxis_accuracy":    round(accuracy, 3),
            "cultivation_temp":        self._t_cult,
        }

    def is_done(self, world_state: WorldState, t: int) -> bool:
        return False
