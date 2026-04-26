"""
Tier 4 — Robustness (Graceful Degradation under Neuron Loss).

Simulates biological neuron death by applying independent Bernoulli
dropout to each sensory unit *before* the brain's step(), so the brain
genuinely makes decisions from degraded input.  Motor output is measured
as the behavioral score: speed × directionality.

Biological reference (Rakic 2002; Bhanu 2022):
  neural circuits tolerate ~30% cell loss with < 50% functional decline.

Pass thresholds (minimum retention of baseline score):
  10% dropout → retain ≥ 80%
  30% dropout → retain ≥ 50%
  50% dropout → retain ≥ 20%

Robustness index (RI): area under the retention curve [0–1], trapezoid
over fractions [0, 0.1, 0.3, 0.5].  RI = 1.0 means full retention at
every dropout level.

Zebrafish-specific note: the "cells-to-social-behaviors" cascade is
tested by applying dropout at the sensory (L1) boundary and measuring
whether the behavioral (L5) output is preserved — the full causal chain.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from ..core.registry import build
from ..core.types import SensorySignals, MotorOutput
from ..engine.organism import VirtualOrganism

# ── Constants ─────────────────────────────────────────────────────────────────

FRACTIONS  = [0.10, 0.30, 0.50]
THRESHOLDS = {0.10: 0.80, 0.30: 0.50, 0.50: 0.20}
N_STEPS    = 40     # steps per measurement window
N_SEEDS    = 5      # seeds averaged to reduce stochastic variance


# ── Public class ──────────────────────────────────────────────────────────────

class Tier4Robustness:
    """Measure behavioral retention under graded sensory neuron dropout."""

    def run(self, species: str) -> dict:
        """
        Returns:
          {
            "dropout_10pct": {passed, baseline, retained, retention, threshold},
            "dropout_30pct": {...},
            "dropout_50pct": {...},
            "robustness_index": float,
            "all_passed": bool,
          }
        """
        baseline = self._mean_score(species, p=0.0)
        if baseline < 1e-6:
            return {"_error": {
                "passed": False,
                "note": "Baseline behavioral score near zero; cannot assess retention.",
            }}

        results: dict = {}
        retentions: list[float] = []
        all_passed = True

        for p in FRACTIONS:
            try:
                score = self._mean_score(species, p)
                retention = float(score / baseline)
                threshold = THRESHOLDS[p]
                passed = retention >= threshold
                all_passed = all_passed and passed
                retentions.append(retention)
                results[f"dropout_{int(p * 100)}pct"] = {
                    "passed":    passed,
                    "baseline":  round(baseline, 4),
                    "retained":  round(score, 4),
                    "retention": round(retention, 4),
                    "threshold": threshold,
                }
            except Exception as exc:
                all_passed = False
                retentions.append(0.0)
                results[f"dropout_{int(p * 100)}pct"] = {
                    "passed": False, "baseline": 0.0, "retained": 0.0,
                    "retention": 0.0, "threshold": THRESHOLDS[p],
                    "error": f"{type(exc).__name__}: {exc}",
                }

        # AUC of retention curve, normalised so 1.0 = full retention everywhere
        xs = [0.0] + FRACTIONS
        ys = [1.0] + retentions
        ri = float(np.trapezoid(ys, x=xs) / 0.5)
        results["robustness_index"] = round(min(ri, 1.0), 4)
        results["all_passed"] = all_passed
        return results

    # ── Score infrastructure ──────────────────────────────────────────────────

    def _mean_score(self, species: str, p: float) -> float:
        return float(np.mean([
            self._episode_score(species, p, seed)
            for seed in range(N_SEEDS)
        ]))

    def _episode_score(self, species: str, p: float, seed: int) -> float:
        """Run one episode with dropout p and return mean behavioral score."""
        _, brain, env = build(species)
        organism = VirtualOrganism(brain, env)
        organism.reset(seed=seed)
        self._position_agent(species, env)

        rng = np.random.default_rng(seed + 9999)
        step_scores: list[float] = []

        for _ in range(N_STEPS):
            signals = env.get_sensory_signals(organism.agent_id)
            if p > 0.0:
                signals = _apply_sensory_dropout(signals, p, rng)
            motor = brain.step(signals, organism._t)
            motor.agent_id = organism.agent_id
            env.step([motor])
            organism._t += 1
            step_scores.append(_behavioral_score(species, motor))

        return float(np.mean(step_scores))

    # ── Species-specific agent positioning ───────────────────────────────────

    def _position_agent(self, species: str, env) -> None:
        """Place agent to produce active, measurable baseline behavior."""
        if species == "xenopus_laevis":
            # Left wall → touch signal initiates CPG
            env._agents[0].position = np.array([1.0, 10.0], dtype=np.float32)

        elif species == "drosophila_melanogaster":
            # Near CS+ odour → PN driven, approach behaviour active
            env._agents[0].position = np.array([145.0, 100.0], dtype=np.float32)

        elif species == "c_elegans":
            # 15 mm from attractant → chemical gradient sensed by AWC
            apos = getattr(env, "attractant_pos",
                           np.array([50.0, 50.0], dtype=np.float32))
            env._agents[0].position = np.array(
                [apos[0] - 15.0, apos[1]], dtype=np.float32
            )
        # danio_rerio: default arena start position is already active


# ── Sensory dropout ───────────────────────────────────────────────────────────

def _apply_sensory_dropout(
    signals: SensorySignals,
    p: float,
    rng: np.random.Generator,
) -> SensorySignals:
    """
    Return a copy of signals with each sensory unit independently zeroed
    with probability p.  Simulates neuron death in sensory epithelia:
      - ChemField concentrations  (olfactory / gustatory receptor neurons)
      - BodyState.in_contact      (mechanosensory / Rohon-Beard neurons)
      - PhotonField intensity     (photoreceptors / retinal ganglion cells)
      - PressureField flow_vectors (lateral-line hair cells)
    """
    s = deepcopy(signals)

    # Olfactory / chemosensory receptor neurons
    if s.chem is not None:
        mask = rng.random(len(s.chem.concentrations)) >= p
        s.chem.concentrations = s.chem.concentrations * mask.astype(np.float32)
        if s.chem.gradient.ndim == 2:
            s.chem.gradient = s.chem.gradient * mask[:, None].astype(np.float32)

    # Mechanosensory / tactile neurons (Rohon-Beard in xenopus; body touch)
    if s.body is not None and rng.random() < p:
        s.body.in_contact = False

    # Photoreceptors / retinal ganglion cells
    if s.photon is not None:
        mask = rng.random(s.photon.intensity.shape) >= p
        s.photon.intensity = s.photon.intensity * mask.astype(np.float32)

    # Lateral-line hair cells
    if s.pressure is not None:
        mask = rng.random(len(s.pressure.flow_vectors)) >= p
        s.pressure.flow_vectors = (
            s.pressure.flow_vectors * mask[:, None].astype(np.float32)
        )

    return s


# ── Behavioral score from motor output ───────────────────────────────────────

def _behavioral_score(species: str, motor: MotorOutput) -> float:
    """
    Scalar [0, 1] combining locomotor speed and directional control.

    Speed captures whether the animal is moving at all (primary proxy for
    circuit integrity).  Turn penalty flags wild, uncontrolled rotation,
    except for c_elegans which uses pirouettes in its normal repertoire.
    """
    speed_score = float(np.clip(motor.speed, 0.0, 1.0))

    if species == "c_elegans":
        # Worms always turn; score speed only
        return speed_score

    # Penalise excessively large turns (sign of degraded motor control)
    max_turn = math.pi / 4
    turn_penalty = float(np.clip(abs(motor.turn) / max_turn, 0.0, 1.0))
    direction_score = float(np.clip(1.0 - 0.4 * turn_penalty, 0.0, 1.0))

    return 0.65 * speed_score + 0.35 * direction_score
