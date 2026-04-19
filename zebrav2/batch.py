"""
Batch experiment runner: parameter sweeps and multi-seed experiments.

Usage:
    from zebrav2.batch import BatchRunner, ParameterSweep

    sweep = ParameterSweep('brain.efe.forage_offset', [0.1, 0.15, 0.2, 0.25])
    runner = BatchRunner(sweeps=[sweep], n_seeds=5, steps=1000)
    results = runner.run()
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any

from zebrav2.config import BrainConfig, BodyConfig, WorldConfig


@dataclass
class ParameterSweep:
    """A single parameter to sweep over."""
    path: str          # dot-separated path: 'brain.efe.forage_offset'
    values: list[Any]  # values to test


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    params: dict[str, Any]
    seed: int
    survived: int = 0
    food_eaten: int = 0
    energy_final: float = 0.0
    goals: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _set_nested(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute by dot-path on dataclass objects."""
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class BatchRunner:
    """Run parameter sweep experiments."""

    def __init__(
        self,
        sweeps: list[ParameterSweep] | None = None,
        n_seeds: int = 3,
        steps: int = 500,
        base_brain: BrainConfig | None = None,
        base_body: BodyConfig | None = None,
        base_world: WorldConfig | None = None,
        lesions: list[str] | None = None,
    ):
        self.sweeps = sweeps or []
        self.n_seeds = n_seeds
        self.steps = steps
        self.base_brain = base_brain or BrainConfig()
        self.base_body = base_body or BodyConfig()
        self.base_world = base_world or WorldConfig()
        self.lesions = lesions or []
        self.results: list[ExperimentResult] = []

    def _make_configs(self, param_combo: dict[str, Any]) -> tuple:
        """Create config copies with parameter overrides applied."""
        brain = BrainConfig.from_json(self.base_brain.to_json())
        body = BodyConfig.from_json(self.base_body.to_json())
        world = WorldConfig.from_json(self.base_world.to_json())

        for path, value in param_combo.items():
            section = path.split('.')[0]
            subpath = '.'.join(path.split('.')[1:])
            if section == 'brain':
                _set_nested(brain, subpath, value)
            elif section == 'body':
                _set_nested(body, subpath, value)
            elif section == 'world':
                _set_nested(world, subpath, value)

        for region in self.lesions:
            if hasattr(brain.ablation, region):
                setattr(brain.ablation, region, False)

        return brain, body, world

    def _get_combinations(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations from sweeps."""
        if not self.sweeps:
            return [{}]
        names = [s.path for s in self.sweeps]
        value_lists = [s.values for s in self.sweeps]
        combos = []
        for values in itertools.product(*value_lists):
            combos.append(dict(zip(names, values)))
        return combos

    def run(self, verbose: bool = True) -> list[ExperimentResult]:
        """Run all experiments and return results."""
        from zebrav2.virtual_fish import VirtualZebrafish

        combos = self._get_combinations()
        total = len(combos) * self.n_seeds
        self.results = []

        for i, combo in enumerate(combos):
            for seed in range(self.n_seeds):
                brain, body, world = self._make_configs(combo)
                world.seed = seed
                world.max_steps = self.steps

                fish = VirtualZebrafish(world=world, body=body, brain=brain)
                run_result = fish.run(steps=self.steps)

                result = ExperimentResult(
                    params=combo, seed=seed,
                    survived=run_result.get('survived', 0),
                    food_eaten=run_result.get('food_eaten', 0),
                    energy_final=run_result.get('energy_final', 0.0),
                    goals=run_result.get('goals', {}),
                )
                self.results.append(result)

                if verbose:
                    idx = i * self.n_seeds + seed + 1
                    print(f"[{idx}/{total}] params={combo} seed={seed} "
                          f"survived={result.survived}")

        return self.results

    def summary(self) -> dict[str, Any]:
        """Aggregate results by parameter combination."""
        from collections import defaultdict
        import statistics

        groups: dict[str, list[ExperimentResult]] = defaultdict(list)
        for r in self.results:
            key = json.dumps(r.params, sort_keys=True)
            groups[key].append(r)

        summary = []
        for key, results in groups.items():
            params = json.loads(key)
            survived = [r.survived for r in results]
            summary.append({
                'params': params,
                'n_seeds': len(results),
                'survived_mean': statistics.mean(survived),
                'survived_std': statistics.stdev(survived) if len(survived) > 1 else 0,
                'survived_min': min(survived),
                'survived_max': max(survived),
                'food_mean': statistics.mean(r.food_eaten for r in results),
                'energy_mean': statistics.mean(r.energy_final for r in results),
            })
        return {'experiments': summary, 'total_runs': len(self.results)}

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
