"""
ExperimentRunner — scripted intervention engine with hierarchical logging.

Usage:
    runner = ExperimentRunner(organism, protocol, log_levels=[1,2,3,4,5,6])
    runner.ablate("amygdala")
    runner.set_param("efe.beta_epistemic", 3.0)
    result = runner.run(n_episodes=100)
    runner.compare(baseline_result, result)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .organism import VirtualOrganism
from ..core.interfaces import BehavioralProtocol
from ..core.types import (
    ExperimentResult, EpisodeResult, HierarchicalState, Stimulus,
)


@dataclass
class _Intervention:
    kind: str        # 'ablate' | 'restore' | 'param' | 'stimulus'
    args: tuple
    kwargs: dict = field(default_factory=dict)


class ExperimentRunner:
    """Manages ablations, parameter sweeps, stimulus injection, and logging."""

    def __init__(
        self,
        organism: VirtualOrganism,
        protocol: BehavioralProtocol,
        log_levels: list[int] | None = None,
        seed: int = 0,
    ):
        self.organism = organism
        self.protocol = protocol
        self.log_levels = set(log_levels or [1, 2, 3, 4, 5, 6])
        self.seed = seed
        self._interventions: list[_Intervention] = []

    # ── Intervention builder API (chainable) ──────────────────────────────────

    def ablate(self, region: str) -> "ExperimentRunner":
        self._interventions.append(_Intervention("ablate", (region,)))
        return self

    def restore(self, region: str) -> "ExperimentRunner":
        self._interventions.append(_Intervention("restore", (region,)))
        return self

    def set_param(self, path: str, value: float) -> "ExperimentRunner":
        self._interventions.append(_Intervention("param", (path, value)))
        return self

    def inject(self, stimulus: Stimulus) -> "ExperimentRunner":
        self._interventions.append(_Intervention("stimulus", (stimulus,)))
        return self

    def clear_interventions(self) -> "ExperimentRunner":
        self._interventions.clear()
        return self

    # ── Apply all queued interventions ────────────────────────────────────────

    def _apply_interventions(self) -> None:
        for iv in self._interventions:
            if iv.kind == "ablate":
                self.organism.ablate(iv.args[0])
            elif iv.kind == "restore":
                self.organism.restore(iv.args[0])
            elif iv.kind == "param":
                self.organism.set_param(iv.args[0], iv.args[1])
            elif iv.kind == "stimulus":
                self.organism.inject(iv.args[0])

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self, n_episodes: int = 10, label: str = "experiment",
            verbose: bool = True) -> ExperimentResult:

        result = ExperimentResult(
            label=label,
            species=self.organism.species,
            protocol=self.protocol.name,
            metadata={
                "log_levels": sorted(self.log_levels),
                "interventions": [
                    {"kind": iv.kind, "args": iv.args}
                    for iv in self._interventions
                ],
                "n_episodes": n_episodes,
            },
        )

        for ep in range(n_episodes):
            ep_seed = self.seed + ep
            world_log = []
            hier_log = []
            food_eaten = 0
            predation_events = 0
            total_reward = 0.0

            # setup
            self.organism.env.clear_stimuli()
            self.protocol.setup(self.organism.env, self.organism.brain)
            for stim in self.protocol.get_stimuli():
                self.organism.env.inject(stim)
            self._apply_interventions()

            world_state = self.organism.reset(seed=ep_seed)
            world_log.append(world_state)

            for t in range(self.protocol.episode_length):
                motor, hier_state = self.organism.step()
                world_state = self.organism.env.current_state

                # filter log levels
                filtered = _filter_levels(hier_state, self.log_levels)
                hier_log.append(filtered)
                world_log.append(world_state)

                # track episode metrics from world state
                food_eaten = world_state.extras.get("food_eaten", food_eaten)
                predation_events = world_state.extras.get("predation_events", predation_events)
                total_reward += world_state.extras.get("reward", 0.0)

                if self.protocol.is_done(world_state, t):
                    break

            ep_result = EpisodeResult(
                episode=ep,
                steps_survived=t + 1,
                food_eaten=food_eaten,
                predation_events=predation_events,
                total_reward=total_reward,
                hierarchical_log=hier_log,
            )

            # protocol-specific scoring
            scores = self.protocol.score(ep_result, world_log)
            ep_result.extras["scores"] = scores

            result.episodes.append(ep_result)

            if verbose:
                print(
                    f"  ep {ep+1:3d}/{n_episodes}  "
                    f"survived={ep_result.steps_survived:4d}  "
                    f"food={food_eaten:3d}  "
                    f"scores={scores}"
                )

        return result

    # ── Parameter sweep ───────────────────────────────────────────────────────

    def sweep(
        self,
        param_path: str,
        values: list[float],
        n_episodes: int = 10,
        verbose: bool = True,
    ) -> list[ExperimentResult]:
        """Run n_episodes for each value in values, returning one result per value."""
        results = []
        baseline_params = [iv for iv in self._interventions
                           if iv.kind != "param" or iv.args[0] != param_path]
        for val in values:
            self._interventions = baseline_params + [
                _Intervention("param", (param_path, val))
            ]
            label = f"{param_path}={val}"
            if verbose:
                print(f"\n── sweep: {label} ──")
            results.append(self.run(n_episodes=n_episodes, label=label,
                                    verbose=verbose))
        return results

    # ── Comparison report ─────────────────────────────────────────────────────

    @staticmethod
    def compare(baseline: ExperimentResult,
                intervention: ExperimentResult,
                metrics: list[str] | None = None) -> dict:
        """Compute delta and effect size between two ExperimentResult objects."""
        metrics = metrics or ["mean_survival", "mean_food", "mean_predation"]
        report = {
            "baseline": baseline.label,
            "intervention": intervention.label,
            "deltas": {},
        }
        for m in metrics:
            b = getattr(baseline, m, None)
            i = getattr(intervention, m, None)
            if b is None or i is None:
                continue
            delta = i - b
            pct = 100 * delta / (abs(b) + 1e-9)
            report["deltas"][m] = {
                "baseline": round(b, 3),
                "intervention": round(i, 3),
                "delta": round(delta, 3),
                "pct_change": round(pct, 1),
            }
        return report

    def save(self, result: ExperimentResult, path: str) -> None:
        """Save result summary to JSON."""
        data = result.summary()
        data["episodes"] = [
            {
                "episode": e.episode,
                "steps_survived": e.steps_survived,
                "food_eaten": e.food_eaten,
                "predation_events": e.predation_events,
                "total_reward": round(e.total_reward, 3),
                "scores": e.extras.get("scores", {}),
            }
            for e in result.episodes
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")


def _filter_levels(state: HierarchicalState,
                   levels: set[int]) -> HierarchicalState:
    return HierarchicalState(
        t=state.t,
        agent_id=state.agent_id,
        L1_synaptic=state.L1_synaptic if 1 in levels else {},
        L2_neuron=state.L2_neuron if 2 in levels else {},
        L3_circuit=state.L3_circuit if 3 in levels else {},
        L4_region=state.L4_region if 4 in levels else {},
        L5_behaviour=state.L5_behaviour if 5 in levels else {},
        L6_social=state.L6_social if 6 in levels else {},
    )
