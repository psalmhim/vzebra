"""Core data structures for the evaluation platform."""
from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import operator, time

PASS_OPS = {'>': operator.gt, '>=': operator.ge, '<': operator.lt,
            '<=': operator.le, '==': operator.eq, '!=': operator.ne}

@dataclass
class EvalResult:
    name: str
    tier: int
    tags: list[str]
    purpose: str
    method: str
    reference: str
    metrics: dict[str, float]          # {name: value}
    pass_criteria: dict[str, tuple]    # {name: (op_str, threshold)}
    passed: bool
    skipped: bool = False
    skip_reason: str = ''
    duration_s: float = 0.0
    artifacts: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    notes: str = ''

    @classmethod
    def skipped_result(cls, plugin: 'EvalPlugin', reason: str) -> 'EvalResult':
        return cls(
            name=plugin.name, tier=plugin.tier, tags=plugin.tags,
            purpose=plugin.purpose, method=plugin.method, reference=plugin.reference,
            metrics={}, pass_criteria={}, passed=False,
            skipped=True, skip_reason=reason)

    def criteria_detail(self) -> str:
        lines = []
        for m, (op, thresh) in self.pass_criteria.items():
            val = self.metrics.get(m, 'N/A')
            ok = '✓' if (isinstance(val, float) and PASS_OPS[op](val, thresh)) else '✗'
            lines.append(f'  {ok} {m} {op} {thresh:.4g}  (got {val:.4g if isinstance(val,float) else val})')
        return '\n'.join(lines)


@dataclass
class EvalContext:
    brain: Any           # ZebrafishBrainV2 or None for unit tests
    device: str
    output_dir: Path
    artifacts: dict[str, Any] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)


class EvalPlugin(ABC):
    """Abstract base for one evaluation station on the pipeline."""
    tier: int = 0
    name: str = ''
    tags: list[str] = []
    requires: list[str] = []      # names of upstream plugins whose results must pass
    purpose: str = ''
    method: str = ''
    reference: str = ''

    @abstractmethod
    def run(self, ctx: EvalContext) -> EvalResult: ...

    def _result(self, metrics: dict, pass_criteria: dict,
                artifacts: dict = None, figures: list = None, notes: str = '') -> EvalResult:
        passed = all(
            PASS_OPS[op](metrics.get(k, float('-inf')), thresh)
            for k, (op, thresh) in pass_criteria.items()
        )
        return EvalResult(
            name=self.name, tier=self.tier, tags=list(self.tags),
            purpose=self.purpose, method=self.method, reference=self.reference,
            metrics=metrics, pass_criteria=pass_criteria, passed=passed,
            artifacts=artifacts or {}, figures=figures or [], notes=notes)
