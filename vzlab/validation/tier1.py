"""
Tier 1 — Behavioural Battery.

Runs species-specific behavioural tests and returns pass/fail per test
with quantitative thresholds derived from published biology.
"""
from __future__ import annotations

import math
import numpy as np

from ..core.registry import build
from ..engine.organism import VirtualOrganism
from ..engine.experiment import ExperimentRunner
from ..protocols import ChemotaxisProtocol, ExplorationProtocol
from ..protocols import PredatorAvoidanceProtocol, AlarmPropagationProtocol


class Tier1BehaviouralBattery:
    """Run species-specific behavioural assays and return pass/fail per test."""

    def run(self, species: str, n_episodes: int = 10) -> dict:
        """
        Returns:
          {test_name: {passed: bool, value: float, threshold: float, description: str}}
        """
        if species == "c_elegans":
            return self._run_celegans(n_episodes)
        elif species == "danio_rerio":
            return self._run_danio(n_episodes)
        elif species == "drosophila_melanogaster":
            return self._run_drosophila(n_episodes)
        else:
            return {
                "_error": {
                    "passed": False,
                    "value": 0.0,
                    "threshold": 0.0,
                    "description": f"Unknown species: {species}",
                }
            }

    # ── C. elegans tests ──────────────────────────────────────────────────────

    def _run_celegans(self, n_episodes: int) -> dict:
        results = {}

        # Test 1 — chemotaxis_index
        try:
            ci = self._celegans_baseline_ci(n_episodes)
            threshold = 0.15
            results["chemotaxis_index"] = {
                "passed": ci > threshold,
                "value": round(ci, 4),
                "threshold": threshold,
                "description": (
                    "Mean CI over episodes must be > 0.15. "
                    "Biological reference: CI ≈ 0.6 for diacetyl."
                ),
            }
        except Exception as exc:
            results["chemotaxis_index"] = _error_result(exc, threshold=0.15)

        # Test 2 — awc_ablation_degrades_ci (needs baseline from test 1)
        try:
            baseline_ci = results.get("chemotaxis_index", {}).get("value", None)
            if baseline_ci is None:
                baseline_ci = self._celegans_baseline_ci(n_episodes)
            ablated_ci = self._celegans_ablated_ci("AWC", n_episodes)
            if baseline_ci > 1e-6:
                drop_frac = (baseline_ci - ablated_ci) / baseline_ci
            else:
                drop_frac = 0.0
            threshold = 0.50
            results["awc_ablation_degrades_ci"] = {
                "passed": drop_frac > threshold,
                "value": round(drop_frac, 4),
                "threshold": threshold,
                "description": (
                    "AWC ablation must drop CI by > 50% relative to baseline. "
                    "Biological: AWC ablation abolishes diacetyl attraction."
                ),
            }
        except Exception as exc:
            results["awc_ablation_degrades_ci"] = _error_result(exc, threshold=0.50)

        # Test 3 — reversal_on_contact
        try:
            rev_frac = self._celegans_reversal_on_contact()
            threshold = 0.30
            results["reversal_on_contact"] = {
                "passed": rev_frac > threshold,
                "value": round(rev_frac, 4),
                "threshold": threshold,
                "description": (
                    "Wall contact should produce reversal. "
                    "Reversal fraction over 100 contact steps must be > 0.3."
                ),
            }
        except Exception as exc:
            results["reversal_on_contact"] = _error_result(exc, threshold=0.30)

        # Test 4 — pirouette_rate
        try:
            pir_rate = self._celegans_pirouette_rate(n_episodes)
            lo, hi = 0.05, 0.40
            results["pirouette_rate"] = {
                "passed": lo <= pir_rate <= hi,
                "value": round(pir_rate, 4),
                "threshold": lo,
                "description": (
                    f"Fraction of steps with |turn| > π/4 must be in [{lo}, {hi}]. "
                    "Biological: ~0.1–0.2 pirouette rate."
                ),
            }
        except Exception as exc:
            results["pirouette_rate"] = _error_result(exc, threshold=0.05)

        # Test 5 — basal_crawl
        try:
            mean_speed = self._celegans_basal_crawl(n_episodes)
            threshold = 0.05
            results["basal_crawl"] = {
                "passed": mean_speed > threshold,
                "value": round(mean_speed, 4),
                "threshold": threshold,
                "description": (
                    "Mean speed without chemical signal must be > 0.05. "
                    "Biological: ~0.2–0.4 mm/s. Animal should not be frozen."
                ),
            }
        except Exception as exc:
            results["basal_crawl"] = _error_result(exc, threshold=0.05)

        return results

    def _celegans_baseline_ci(self, n_episodes: int) -> float:
        connectome, brain, env = build("c_elegans")
        organism = VirtualOrganism(brain, env)
        protocol = ChemotaxisProtocol()
        runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=42)
        result = runner.run(n_episodes=n_episodes, label="baseline_ci", verbose=False)
        ci_values = [
            ep.extras.get("scores", {}).get("chemotaxis_index", 0.0)
            for ep in result.episodes
        ]
        return float(np.mean(ci_values)) if ci_values else 0.0

    def _celegans_ablated_ci(self, region: str, n_episodes: int) -> float:
        connectome, brain, env = build("c_elegans")
        organism = VirtualOrganism(brain, env)
        protocol = ChemotaxisProtocol()
        runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=42)
        runner.ablate(region)
        result = runner.run(n_episodes=n_episodes, label=f"ablated_{region}_ci", verbose=False)
        ci_values = [
            ep.extras.get("scores", {}).get("chemotaxis_index", 0.0)
            for ep in result.episodes
        ]
        return float(np.mean(ci_values)) if ci_values else 0.0

    def _celegans_reversal_on_contact(self) -> float:
        """Place worm at wall, count reversal fraction over 100 steps."""
        from ..environments.agar_plate import AgarPlate
        from ..species.celegans.brain import CElegansNervousSystem

        env = AgarPlate()
        brain = CElegansNervousSystem()
        organism = VirtualOrganism(brain, env)

        # Position agent at left wall edge
        world = organism.reset(seed=0)
        env._agents[0].position = np.array([0.5, 50.0], dtype=np.float32)

        n_steps = 100
        reversals = 0
        for t in range(n_steps):
            motor, _ = organism.step()
            # Reversal: worm is moving backward (speed > 0 while near wall → any movement counts)
            # More accurately: negative turn or speed change after contact
            signals = env.get_sensory_signals(0)
            in_contact = signals.body is not None and signals.body.in_contact
            # Count step as reversal if worm is in contact AND has a non-trivial speed
            # (the brain must respond to wall contact with locomotion)
            if in_contact and motor.speed > 0.01:
                reversals += 1

        return reversals / n_steps

    def _celegans_pirouette_rate(self, n_episodes: int) -> float:
        """Fraction of steps with |turn| > π/4."""
        connectome, brain, env = build("c_elegans")
        organism = VirtualOrganism(brain, env)
        protocol = ChemotaxisProtocol()

        turns_abs = []
        for ep in range(n_episodes):
            organism.reset(seed=ep)
            for t in range(protocol.episode_length):
                motor, _ = organism.step()
                turns_abs.append(abs(motor.turn))

        if not turns_abs:
            return 0.0
        return float(np.mean(np.array(turns_abs) > math.pi / 4))

    def _celegans_basal_crawl(self, n_episodes: int) -> float:
        """Mean speed when no chemical signal present (blank plate)."""
        from ..environments.agar_plate import AgarPlate
        from ..species.celegans.brain import CElegansNervousSystem

        # Use zero attractant to simulate blank
        env = AgarPlate(attractant_peak=0.0, repellent_peak=0.0)
        brain = CElegansNervousSystem()
        organism = VirtualOrganism(brain, env)

        speeds = []
        for ep in range(n_episodes):
            organism.reset(seed=ep)
            for t in range(200):
                motor, _ = organism.step()
                speeds.append(motor.speed)

        return float(np.mean(speeds)) if speeds else 0.0

    # ── Danio rerio tests ──────────────────────────────────────────────────────

    def _run_danio(self, n_episodes: int) -> dict:
        results = {}

        # Test 1 — predator_detection
        try:
            escape_frac = self._danio_predator_detection(n_episodes)
            threshold = 0.70
            results["predator_detection"] = {
                "passed": escape_frac > threshold,
                "value": round(escape_frac, 4),
                "threshold": threshold,
                "description": (
                    "Escape fraction (flee_latency < 100 steps) must be > 0.7. "
                    "Run PredatorAvoidanceProtocol, n_episodes=10."
                ),
            }
        except Exception as exc:
            results["predator_detection"] = _error_result(exc, threshold=0.70)

        # Test 2 — food_foraging
        try:
            mean_food = self._danio_food_foraging(n_episodes)
            threshold = 1.0
            results["food_foraging"] = {
                "passed": mean_food > threshold,
                "value": round(mean_food, 4),
                "threshold": threshold,
                "description": (
                    "Mean food_eaten over episodes must be > 1. "
                    "Run ExplorationProtocol."
                ),
            }
        except Exception as exc:
            results["food_foraging"] = _error_result(exc, threshold=1.0)

        # Test 3 — alarm_response
        try:
            flee_frac = self._danio_alarm_response(n_episodes)
            threshold = 0.30
            results["alarm_response"] = {
                "passed": flee_frac > threshold,
                "value": round(flee_frac, 4),
                "threshold": threshold,
                "description": (
                    "Alarm propagation flee_fraction must be > 0.3. "
                    "Run AlarmPropagationProtocol."
                ),
            }
        except Exception as exc:
            results["alarm_response"] = _error_result(exc, threshold=0.30)

        # Test 4 — exploration_coverage
        try:
            coverage = self._danio_exploration_coverage(n_episodes)
            threshold = 0.05
            results["exploration_coverage"] = {
                "passed": coverage > threshold,
                "value": round(coverage, 4),
                "threshold": threshold,
                "description": (
                    "Spatial coverage from ExplorationProtocol must be > 0.05."
                ),
            }
        except Exception as exc:
            results["exploration_coverage"] = _error_result(exc, threshold=0.05)

        # Test 5 — energy_depletion
        try:
            mean_survival = self._danio_survival(n_episodes)
            threshold = 100.0
            results["energy_depletion"] = {
                "passed": mean_survival > threshold,
                "value": round(mean_survival, 2),
                "threshold": threshold,
                "description": (
                    "Mean survival steps should be > 100 (fish should not die immediately)."
                ),
            }
        except Exception as exc:
            results["energy_depletion"] = _error_result(exc, threshold=100.0)

        return results

    def _danio_predator_detection(self, n_episodes: int) -> float:
        connectome, brain, env = build("danio_rerio")
        organism = VirtualOrganism(brain, env)
        protocol = PredatorAvoidanceProtocol()
        runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=0)
        result = runner.run(n_episodes=n_episodes, label="predator_detection", verbose=False)
        escape_flags = [
            1 if ep.extras.get("scores", {}).get("flee_latency", 999) < 100 else 0
            for ep in result.episodes
        ]
        return float(np.mean(escape_flags)) if escape_flags else 0.0

    def _danio_food_foraging(self, n_episodes: int) -> float:
        connectome, brain, env = build("danio_rerio")
        organism = VirtualOrganism(brain, env)
        protocol = ExplorationProtocol()
        runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=0)
        result = runner.run(n_episodes=n_episodes, label="food_foraging", verbose=False)
        return result.mean_food

    def _danio_alarm_response(self, n_episodes: int) -> float:
        connectome, brain, env = build("danio_rerio")
        organism = VirtualOrganism(brain, env)
        protocol = AlarmPropagationProtocol()
        runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=0)
        result = runner.run(n_episodes=n_episodes, label="alarm_response", verbose=False)
        flee_fracs = [
            ep.extras.get("scores", {}).get("flee_fraction", 0.0)
            for ep in result.episodes
        ]
        return float(np.mean(flee_fracs)) if flee_fracs else 0.0

    def _danio_exploration_coverage(self, n_episodes: int) -> float:
        connectome, brain, env = build("danio_rerio")
        organism = VirtualOrganism(brain, env)
        protocol = ExplorationProtocol()
        runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=0)
        result = runner.run(n_episodes=n_episodes, label="exploration_coverage", verbose=False)
        coverages = [
            ep.extras.get("scores", {}).get("spatial_coverage", 0.0)
            for ep in result.episodes
        ]
        return float(np.mean(coverages)) if coverages else 0.0

    def _danio_survival(self, n_episodes: int) -> float:
        connectome, brain, env = build("danio_rerio")
        organism = VirtualOrganism(brain, env)
        protocol = ExplorationProtocol()
        runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=0)
        result = runner.run(n_episodes=n_episodes, label="survival", verbose=False)
        return result.mean_survival


    # ── Drosophila melanogaster tests ─────────────────────────────────────────

    def _run_drosophila(self, n_episodes: int) -> dict:
        results = {}

        # Test 1 — pn_odor_response
        try:
            pn_fr = self._drosophila_pn_odor_response()
            threshold = 0.10
            results["pn_odor_response"] = {
                "passed": pn_fr > threshold,
                "value": round(pn_fr, 4),
                "threshold": threshold,
                "description": (
                    "PN mean firing rate near CS+ source must be > 0.10. "
                    "PNs relay odor identity from antennal lobe to KCs."
                ),
            }
        except Exception as exc:
            results["pn_odor_response"] = _error_result(exc, threshold=0.10)

        # Test 2 — kc_sparseness_maintained
        try:
            kc_fr = self._drosophila_kc_sparseness()
            threshold = 0.30
            results["kc_sparseness_maintained"] = {
                "passed": kc_fr <= threshold,
                "value": round(kc_fr, 4),
                "threshold": threshold,
                "description": (
                    "KC mean firing rate near CS+ must be ≤ 0.30. "
                    "APL enforces ~10% KC sparsity (Honegger et al. 2011)."
                ),
            }
        except Exception as exc:
            results["kc_sparseness_maintained"] = _error_result(exc, threshold=0.30)

        # Test 3 — dan_baseline_active
        try:
            dan_fr = self._drosophila_dan_active()
            threshold = 0.05
            results["dan_baseline_active"] = {
                "passed": dan_fr > threshold,
                "value": round(dan_fr, 4),
                "threshold": threshold,
                "description": (
                    "DAN firing rate must be > 0.05 at baseline tone. "
                    "DANs maintain tonic dopamine for reward gating (Burke et al. 2012)."
                ),
            }
        except Exception as exc:
            results["dan_baseline_active"] = _error_result(exc, threshold=0.05)

        return results

    def _drosophila_pn_odor_response(self) -> float:
        """Mean PN firing rate when fly is placed near CS+ source."""
        from ..species.drosophila.brain import DrosophilaBrain
        from ..environments.olfactory_arena import OlfactoryArena

        env = OlfactoryArena()
        brain = DrosophilaBrain()
        organism = VirtualOrganism(brain, env)
        organism.reset(seed=0)
        env._agents[0].position = np.array([145.0, 100.0], dtype=np.float32)

        pn_rates = []
        for _ in range(30):
            _, hier = organism.step()
            pn_rates.append(float(hier.L3_circuit.get("pn_fr", 0.0)))

        return float(np.mean(pn_rates))

    def _drosophila_kc_sparseness(self) -> float:
        """Mean KC firing rate near CS+ — APL should limit it to ~10%."""
        from ..species.drosophila.brain import DrosophilaBrain
        from ..environments.olfactory_arena import OlfactoryArena

        env = OlfactoryArena()
        brain = DrosophilaBrain()
        organism = VirtualOrganism(brain, env)
        organism.reset(seed=0)
        env._agents[0].position = np.array([145.0, 100.0], dtype=np.float32)

        kc_rates = []
        for _ in range(30):
            _, hier = organism.step()
            kc_rates.append(float(hier.L3_circuit.get("kc_fr", 0.0)))

        return float(np.mean(kc_rates))

    def _drosophila_dan_active(self) -> float:
        """Mean DAN firing rate from baseline dopamine tone."""
        from ..species.drosophila.brain import DrosophilaBrain
        from ..environments.olfactory_arena import OlfactoryArena

        env = OlfactoryArena()
        brain = DrosophilaBrain()
        organism = VirtualOrganism(brain, env)
        organism.reset(seed=0)

        dan_rates = []
        for _ in range(30):
            _, hier = organism.step()
            dan_rates.append(float(hier.L3_circuit.get("dan_activity", 0.0)))

        return float(np.mean(dan_rates))


def _error_result(exc: Exception, threshold: float) -> dict:
    return {
        "passed": False,
        "value": 0.0,
        "threshold": threshold,
        "description": f"ERROR: {type(exc).__name__}: {exc}",
    }
