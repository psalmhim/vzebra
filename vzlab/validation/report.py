"""
ValidationReport — aggregates all three tiers into a single graded report.

Grade logic:
  A    — all three tiers pass
  B    — tier1 + tier2 pass
  C    — only tier1 passes
  FAIL — tier1 fails
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .tier1 import Tier1BehaviouralBattery
from .tier2 import Tier2AtlasCorrespondence
from .tier3 import Tier3LesionReplication
from .tier4 import Tier4Robustness


@dataclass
class ValidationReport:
    species: str
    tier1: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    tier3: dict = field(default_factory=dict)
    tier4: dict = field(default_factory=dict)

    # ── Pass/fail helpers ─────────────────────────────────────────────────────

    def passed_tier1(self) -> bool:
        if not self.tier1:
            return False
        return all(
            v.get("passed", False)
            for k, v in self.tier1.items()
            if not k.startswith("_")
        )

    def passed_tier2(self) -> bool:
        if not self.tier2:
            return False
        return all(
            v.get("passed", False)
            for k, v in self.tier2.items()
            if not k.startswith("_")
        )

    def passed_tier3(self) -> bool:
        if not self.tier3:
            return False
        return all(
            v.get("passed", False)
            for k, v in self.tier3.items()
            if not k.startswith("_")
        )

    def passed_tier4(self) -> bool:
        if not self.tier4:
            return False
        return self.tier4.get("all_passed", False)

    def robustness_index(self) -> float:
        return float(self.tier4.get("robustness_index", 0.0))

    # ── Grading ───────────────────────────────────────────────────────────────

    def overall_grade(self) -> str:
        """
        A+  — T1 + T2 + T3 + T4 all pass
        A   — T1 + T2 + T3 pass  (canonical grade)
        B   — T1 + T2 pass
        C   — T1 only passes
        FAIL— T1 fails
        """
        t1 = self.passed_tier1()
        t2 = self.passed_tier2()
        t3 = self.passed_tier3()
        t4 = self.passed_tier4()

        if not t1:
            return "FAIL"
        if t1 and not t2 and not t3:
            return "C"
        if t1 and t2 and not t3:
            return "B"
        if t1 and t2 and t3 and t4:
            return "A+"
        if t1 and t2 and t3:
            return "A"
        return "C"

    # ── Display ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  VALIDATION REPORT — {self.species}")
        print(f"{'='*60}")

        self._print_tier("TIER 1 — Behavioural Battery", self.tier1, self._t1_row)
        self._print_tier("TIER 2 — Atlas Correspondence", self.tier2, self._t2_row)
        self._print_tier("TIER 3 — Lesion Replication", self.tier3, self._t3_row)
        self._print_tier("TIER 4 — Robustness / Graceful Degradation", self.tier4, self._t4_row)

        grade = self.overall_grade()
        t1_str = "PASS" if self.passed_tier1() else "FAIL"
        t2_str = "PASS" if self.passed_tier2() else "FAIL"
        t3_str = "PASS" if self.passed_tier3() else "FAIL"
        t4_str = f"PASS (RI={self.robustness_index():.3f})" if self.passed_tier4() else "FAIL"

        print(f"\n{'─'*60}")
        print(f"  T1: {t1_str}  |  T2: {t2_str}  |  T3: {t3_str}  |  T4: {t4_str}")
        print(f"  Overall Grade: {grade}")
        print(f"{'='*60}\n")

    def _print_tier(self, title: str, data: dict, row_fn) -> None:
        print(f"\n  {title}")
        print(f"  {'─'*56}")
        if not data:
            print("    (no results)")
            return
        for name, result in data.items():
            if name.startswith("_"):
                continue
            print(f"    {row_fn(name, result)}")

    @staticmethod
    def _t1_row(name: str, r: dict) -> str:
        status = "PASS" if r.get("passed") else "FAIL"
        val = r.get("value", 0.0)
        thr = r.get("threshold", 0.0)
        return f"[{status}] {name:<35s}  val={val:.4f}  thr={thr:.4f}"

    @staticmethod
    def _t2_row(name: str, r: dict) -> str:
        status = "PASS" if r.get("passed") else "FAIL"
        corr = r.get("correlation", 0.0)
        return f"[{status}] {name:<35s}  r_s={corr:.4f}  (>0.5)"

    @staticmethod
    def _t3_row(name: str, r: dict) -> str:
        status = "PASS" if r.get("passed") else "FAIL"
        chg = r.get("change_pct", 0.0)
        direction = r.get("expected_direction", "?")
        return f"[{status}] {name:<35s}  chg={chg:+.1f}%  ({direction})"

    @staticmethod
    def _t4_row(name: str, r: dict) -> str:
        if name in ("robustness_index", "all_passed"):
            return f"         {name:<35s}  {r}"
        status = "PASS" if r.get("passed") else "FAIL"
        ret = r.get("retention", 0.0)
        thr = r.get("threshold", 0.0)
        return f"[{status}] {name:<35s}  ret={ret:.3f}  thr={thr:.2f}"

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> dict:
        return {
            "species": self.species,
            "grade": self.overall_grade(),
            "tier1_passed": self.passed_tier1(),
            "tier2_passed": self.passed_tier2(),
            "tier3_passed": self.passed_tier3(),
            "tier4_passed": self.passed_tier4(),
            "robustness_index": self.robustness_index(),
            "tier1": self.tier1,
            "tier2": self.tier2,
            "tier3": self.tier3,
            "tier4": self.tier4,
        }


# ── Top-level convenience ─────────────────────────────────────────────────────

def run_full_validation(
    species: str,
    verbose: bool = True,
    n_episodes_t1: int = 10,
    n_steps_t2: int = 50,
) -> ValidationReport:
    """Run all three validation tiers and return a ValidationReport."""

    if verbose:
        print(f"\n[vzlab] Starting validation for species='{species}' ...")

    # Tier 1
    if verbose:
        print("\n[Tier 1] Behavioural battery ...")
    t1 = Tier1BehaviouralBattery()
    tier1_results = t1.run(species, n_episodes=n_episodes_t1)

    # Tier 2
    if verbose:
        print("\n[Tier 2] Atlas correspondence ...")
    t2 = Tier2AtlasCorrespondence()
    tier2_results = t2.run(species, n_steps_per_condition=n_steps_t2)

    # Tier 3
    if verbose:
        print("\n[Tier 3] Lesion replication ...")
    t3 = Tier3LesionReplication()
    tier3_results = t3.run(species)

    # Tier 4
    if verbose:
        print("\n[Tier 4] Robustness / graceful degradation ...")
    t4 = Tier4Robustness()
    tier4_results = t4.run(species)

    return ValidationReport(
        species=species,
        tier1=tier1_results,
        tier2=tier2_results,
        tier3=tier3_results,
        tier4=tier4_results,
    )
