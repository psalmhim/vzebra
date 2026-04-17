"""
Multi-seed runner: execute a brain condition across N seeds and return
mean ± SEM for a set of metrics.

Usage:
    results = run_multi_seed(
        condition_fn=lambda brain: apply_disorder(brain, 'asd'),
        assay=NovelTankTest(),
        n_seeds=20,
        ckpt_path='zebrav2/checkpoints/ckpt_round_0005.pt',
    )
    # results: {'top_fraction': SeedResult(mean=0.31, sem=0.04, values=[...]), ...}
"""
from __future__ import annotations

import gc
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Optional scipy — fall back to manual SEM if unavailable
# ---------------------------------------------------------------------------
try:
    from scipy.stats import sem as _scipy_sem
    def _sem(values: List[float]) -> float:
        return float(_scipy_sem(values)) if len(values) > 1 else 0.0
except ImportError:
    def _sem(values: List[float]) -> float:  # type: ignore[misc]
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance / len(values))


# ---------------------------------------------------------------------------
# Core data class
# ---------------------------------------------------------------------------
@dataclass
class SeedResult:
    mean: float
    sem: float          # standard error of mean
    std: float
    n: int
    values: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.mean:.3f} ± {self.sem:.3f}"

    def cohens_d(self, other: 'SeedResult') -> float:
        """Cohen's d effect size between this and another condition."""
        pooled_std = math.sqrt((self.std ** 2 + other.std ** 2) / 2)
        if pooled_std < 1e-8:
            return 0.0
        return (self.mean - other.mean) / pooled_std


def _build_seed_result(values: List[float]) -> SeedResult:
    n = len(values)
    if n == 0:
        return SeedResult(mean=float('nan'), sem=0.0, std=0.0, n=0, values=[])
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return SeedResult(mean=mean, sem=_sem(values), std=std, n=n, values=list(values))


# ---------------------------------------------------------------------------
# Main multi-seed runner
# ---------------------------------------------------------------------------
def run_multi_seed(
    condition_fn: Callable,
    assay,
    n_seeds: int = 20,
    ckpt_path: Optional[str] = None,
    n_steps: int = 500,
    personality_name: str = 'default',
    verbose: bool = True,
) -> Dict[str, SeedResult]:
    """
    Run a single condition (brain manipulation) across N seeds.

    Parameters
    ----------
    condition_fn : callable(brain) -> None
        Applied to the fresh brain after checkpoint load.
        Pass ``lambda brain: None`` for the wildtype / control condition.
    assay : BehavioralAssay
        Instance from ``zebrav2.gym_env.assay_arenas``.
    n_seeds : int
        Number of independent seeds to average over.
    ckpt_path : str or None
        Path to a ``.pt`` checkpoint produced by ``CheckpointManager.save()``.
        If ``None`` or the path does not exist, the brain runs with random
        initial weights (useful for ablation / untrained baseline runs).
    n_steps : int
        Simulation steps per seed.
    personality_name : str
        Personality profile applied to every fresh brain instance.
    verbose : bool
        Print per-seed progress.

    Returns
    -------
    dict[str, SeedResult]
        One entry per metric reported by the assay.
    """
    # Deferred imports so individual failures produce clear errors
    try:
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
    except ImportError as exc:
        raise ImportError(f"Cannot import ZebrafishBrainV2: {exc}") from exc

    try:
        from zebrav2.brain.personality import get_personality
    except ImportError as exc:
        raise ImportError(f"Cannot import get_personality: {exc}") from exc

    try:
        from zebrav2.spec import DEVICE
    except ImportError as exc:
        raise ImportError(f"Cannot import DEVICE from zebrav2.spec: {exc}") from exc

    try:
        from zebrav2.gym_env.assay_arenas import run_assay
    except ImportError as exc:
        raise ImportError(f"Cannot import run_assay from assay_arenas: {exc}") from exc

    # Optional checkpoint manager
    ckpt_manager = None
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            from zebrav2.engine.checkpoint import CheckpointManager
            ckpt_manager = CheckpointManager.__new__(CheckpointManager)
            # minimal init — we only need .load()
            ckpt_manager.save_dir = os.path.dirname(ckpt_path) or '.'
        except ImportError:
            if verbose:
                print("[warn] CheckpointManager unavailable; running without checkpoint.")
    elif ckpt_path:
        if verbose:
            print(f"[warn] Checkpoint not found: {ckpt_path}  — running without checkpoint.")

    per_metric: Dict[str, List[float]] = {}

    for seed in range(n_seeds):
        if verbose:
            print(f"  seed {seed + 1:02d}/{n_seeds}", end="\r", flush=True)

        try:
            # 1. Fresh brain
            personality = get_personality(personality_name)
            brain = ZebrafishBrainV2(device=DEVICE, personality=personality)

            # 2. Load checkpoint weights
            if ckpt_manager is not None:
                try:
                    from zebrav2.engine.checkpoint import CheckpointManager as _CM
                    real_mgr = _CM(ckpt_manager.save_dir)
                    real_mgr.load(brain, ckpt_path)
                except Exception as exc:
                    if verbose:
                        print(f"\n  [warn] seed {seed}: checkpoint load failed: {exc}")

            # 3. Apply condition (disorder / manipulation)
            condition_fn(brain)

            # 4. Reset assay state
            assay.reset()

            # 5. Run assay
            metrics: dict = run_assay(brain, assay, n_steps=n_steps, seed=seed)

            # 6. Accumulate per-metric
            for key, val in metrics.items():
                per_metric.setdefault(key, []).append(float(val))

        except Exception as exc:
            if verbose:
                print(f"\n  [error] seed {seed}: {exc}")
        finally:
            # Free brain + MPS cache to prevent OOM across many seeds
            del brain
            gc.collect()
            try:
                import torch
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except Exception:
                pass

    if verbose:
        print()  # newline after \r progress

    # Aggregate
    results: Dict[str, SeedResult] = {
        metric: _build_seed_result(values)
        for metric, values in per_metric.items()
    }

    if verbose:
        print(f"  Completed {n_seeds} seeds.  Metrics: {list(results.keys())}")
        for metric, sr in results.items():
            print(f"    {metric:30s}  {sr}")

    return results


# ---------------------------------------------------------------------------
# Multi-condition comparison
# ---------------------------------------------------------------------------
def compare_conditions(
    conditions: Dict[str, Callable],
    assay,
    n_seeds: int = 20,
    ckpt_path: Optional[str] = None,
    n_steps: int = 500,
    personality_name: str = 'default',
    verbose: bool = True,
) -> Dict[str, Dict[str, SeedResult]]:
    """
    Run multiple conditions and return ``results[condition_name][metric]``.

    Parameters
    ----------
    conditions : dict[str, callable]
        Mapping from condition label → ``condition_fn(brain)``.
        Example::

            {
                'wildtype':     lambda brain: None,
                'anxiety':      lambda brain: apply_disorder(brain, 'anxiety'),
            }
    assay, n_seeds, ckpt_path, n_steps, personality_name, verbose
        Forwarded to :func:`run_multi_seed`.

    Returns
    -------
    dict[str, dict[str, SeedResult]]
    """
    results: Dict[str, Dict[str, SeedResult]] = {}
    n_conditions = len(conditions)

    for idx, (name, fn) in enumerate(conditions.items()):
        if verbose:
            print(f"[{idx + 1}/{n_conditions}] condition: {name}")
        results[name] = run_multi_seed(
            condition_fn=fn,
            assay=assay,
            n_seeds=n_seeds,
            ckpt_path=ckpt_path,
            n_steps=n_steps,
            personality_name=personality_name,
            verbose=verbose,
        )

    return results


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(
    results: Dict[str, Dict[str, SeedResult]],
    metrics: Optional[List[str]] = None,
    col_width: int = 16,
) -> None:
    """
    Print a formatted table of mean ± SEM values.

    Parameters
    ----------
    results : dict[str, dict[str, SeedResult]]
        Output of :func:`compare_conditions`.
    metrics : list[str] or None
        Subset of metrics to display.  If ``None``, all metrics found in the
        first condition are shown.
    col_width : int
        Column width for each metric cell.
    """
    if not results:
        print("(no results)")
        return

    # Determine metric list
    all_conditions = list(results.keys())
    if metrics is None:
        first = all_conditions[0] if all_conditions else None
        metrics = list(results[first].keys()) if first else []

    cond_col = max(len(c) for c in all_conditions)
    cond_col = max(cond_col, 12)

    # Header
    header = f"{'Condition':<{cond_col}}"
    for m in metrics:
        header += f"  {m[:col_width - 2]:>{col_width - 2}}"
    print(header)
    print("-" * len(header))

    # Rows
    for cond in all_conditions:
        row = f"{cond:<{cond_col}}"
        for m in metrics:
            sr = results[cond].get(m)
            if sr is None or sr.n == 0:
                cell = "n/a"
            else:
                cell = str(sr)
            row += f"  {cell:>{col_width - 2}}"
        print(row)


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("multi_seed_runner: SeedResult smoke test")
    sr1 = _build_seed_result([0.1, 0.2, 0.3, 0.4, 0.5])
    sr2 = _build_seed_result([0.4, 0.5, 0.6, 0.7, 0.8])
    print(f"  sr1 = {sr1}")
    print(f"  sr2 = {sr2}")
    print(f"  Cohen's d (sr1 vs sr2) = {sr1.cohens_d(sr2):.3f}")

    fake_results = {
        'wildtype': {'top_fraction': sr1, 'dark_fraction': sr2},
        'anxiety':  {'top_fraction': sr2, 'dark_fraction': sr1},
    }
    print()
    print_comparison_table(fake_results, metrics=['top_fraction', 'dark_fraction'])
