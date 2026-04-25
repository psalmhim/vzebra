"""
Tier 3 — Lesion Replication.

Run known biological interventions and verify they produce the expected
directional effects reported in the literature.
"""
from __future__ import annotations

import math
import numpy as np

from ..core.registry import build
from ..engine.organism import VirtualOrganism
from ..engine.experiment import ExperimentRunner
from ..protocols import (
    ChemotaxisProtocol,
    ExplorationProtocol,
    AlarmPropagationProtocol,
)


class Tier3LesionReplication:
    """Run biological interventions and check for expected effects."""

    def run(self, species: str) -> dict:
        """
        Returns:
          {test_name: {passed: bool, baseline: float, intervention: float,
                       change_pct: float, expected_direction: str}}
        """
        if species == "c_elegans":
            return self._run_celegans()
        elif species == "danio_rerio":
            return self._run_danio()
        elif species == "drosophila_melanogaster":
            return self._run_drosophila()
        else:
            return {
                "_error": {
                    "passed": False,
                    "baseline": 0.0,
                    "intervention": 0.0,
                    "change_pct": 0.0,
                    "expected_direction": "unknown",
                    "description": f"Unknown species: {species}",
                }
            }

    # ── C. elegans interventions ──────────────────────────────────────────────

    def _run_celegans(self) -> dict:
        results = {}

        # 1. AWC ablation → CI drops by ≥ 50%
        try:
            results["awc_ablation_ci_drop"] = self._celegans_ablation_ci(
                "AWC",
                expected_direction="decrease",
                threshold_pct=50.0,
                n_episodes=5,
            )
        except Exception as exc:
            results["awc_ablation_ci_drop"] = _error_intervention(exc)

        # 2. AIY ablation → pirouette rate increases ≥ 50%
        try:
            results["aiy_ablation_pirouette"] = self._celegans_ablation_pirouette(
                "AIY",
                expected_direction="increase",
                threshold_pct=50.0,
                n_episodes=5,
            )
        except Exception as exc:
            results["aiy_ablation_pirouette"] = _error_intervention(exc)

        # 3. AVA ablation → reversal rate drops ≥ 50%
        try:
            results["ava_ablation_reversal"] = self._celegans_ablation_reversal(
                "AVA",
                expected_direction="decrease",
                threshold_pct=50.0,
                n_episodes=5,
            )
        except Exception as exc:
            results["ava_ablation_reversal"] = _error_intervention(exc)

        # 4. Serotonin high → speed decreases ≥ 30%
        try:
            results["serotonin_slows_locomotion"] = self._celegans_serotonin_speed(
                serotonin_level=0.8,
                expected_direction="decrease",
                threshold_pct=30.0,
                n_episodes=5,
            )
        except Exception as exc:
            results["serotonin_slows_locomotion"] = _error_intervention(exc)

        # 5. Hebbian learning → AWC→AIY weight increases ≥ 0.1
        try:
            results["hebbian_potentiation"] = self._celegans_hebbian_weight(
                hebb_lr=0.05,
                n_steps=200,
                threshold_delta=0.1,
            )
        except Exception as exc:
            results["hebbian_potentiation"] = _error_intervention(exc)

        return results

    def _celegans_ablation_ci(
        self,
        region: str,
        expected_direction: str,
        threshold_pct: float,
        n_episodes: int,
    ) -> dict:
        baseline_ci = self._celegans_run_ci(ablated=None, n_episodes=n_episodes)
        ablated_ci  = self._celegans_run_ci(ablated=region, n_episodes=n_episodes)

        if baseline_ci > 1e-9:
            change_pct = 100.0 * (ablated_ci - baseline_ci) / baseline_ci
        else:
            change_pct = 0.0

        passed = change_pct <= -threshold_pct  # must decrease by threshold_pct
        return {
            "passed": passed,
            "baseline": round(baseline_ci, 4),
            "intervention": round(ablated_ci, 4),
            "change_pct": round(change_pct, 2),
            "expected_direction": expected_direction,
        }

    def _celegans_run_ci(self, ablated: str | None, n_episodes: int) -> float:
        connectome, brain, env = build("c_elegans")
        organism = VirtualOrganism(brain, env)
        protocol = ChemotaxisProtocol()
        runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=1)
        if ablated:
            runner.ablate(ablated)
        result = runner.run(n_episodes=n_episodes, label="ci_run", verbose=False)
        ci_values = [
            ep.extras.get("scores", {}).get("chemotaxis_index", 0.0)
            for ep in result.episodes
        ]
        return float(np.mean(ci_values)) if ci_values else 0.0

    def _celegans_ablation_pirouette(
        self,
        region: str,
        expected_direction: str,
        threshold_pct: float,
        n_episodes: int,
    ) -> dict:
        baseline_rate = self._celegans_pirouette_rate(ablated=None, n_episodes=n_episodes)
        ablated_rate  = self._celegans_pirouette_rate(ablated=region, n_episodes=n_episodes)

        if baseline_rate > 1e-9:
            change_pct = 100.0 * (ablated_rate - baseline_rate) / baseline_rate
        else:
            change_pct = 0.0

        passed = change_pct >= threshold_pct  # must increase
        return {
            "passed": passed,
            "baseline": round(baseline_rate, 4),
            "intervention": round(ablated_rate, 4),
            "change_pct": round(change_pct, 2),
            "expected_direction": expected_direction,
        }

    def _celegans_pirouette_rate(self, ablated: str | None, n_episodes: int) -> float:
        from ..environments.agar_plate import AgarPlate
        from ..species.celegans.brain import CElegansNervousSystem

        env = AgarPlate()
        brain = CElegansNervousSystem()
        if ablated:
            brain.ablate(ablated)
        organism = VirtualOrganism(brain, env)
        protocol = ChemotaxisProtocol()

        turns_abs = []
        for ep in range(n_episodes):
            organism.reset(seed=ep + 10)
            for t in range(200):
                motor, _ = organism.step()
                turns_abs.append(abs(motor.turn))

        if not turns_abs:
            return 0.0
        return float(np.mean(np.array(turns_abs) > math.pi / 4))

    def _celegans_ablation_reversal(
        self,
        region: str,
        expected_direction: str,
        threshold_pct: float,
        n_episodes: int,
    ) -> dict:
        baseline_rate = self._celegans_reversal_rate(ablated=None, n_episodes=n_episodes)
        ablated_rate  = self._celegans_reversal_rate(ablated=region, n_episodes=n_episodes)

        if baseline_rate > 1e-9:
            change_pct = 100.0 * (ablated_rate - baseline_rate) / baseline_rate
        else:
            change_pct = 0.0

        passed = change_pct <= -threshold_pct  # must decrease
        return {
            "passed": passed,
            "baseline": round(baseline_rate, 4),
            "intervention": round(ablated_rate, 4),
            "change_pct": round(change_pct, 2),
            "expected_direction": expected_direction,
        }

    def _celegans_reversal_rate(self, ablated: str | None, n_episodes: int) -> float:
        """Fraction of steps where AVA voltage exceeds reversal threshold."""
        from ..species.celegans.brain import CElegansNervousSystem
        from ..environments.agar_plate import AgarPlate

        env = AgarPlate()
        brain = CElegansNervousSystem()
        if ablated:
            brain.ablate(ablated)
        organism = VirtualOrganism(brain, env)

        reversal_count = 0
        total = 0
        for ep in range(n_episodes):
            organism.reset(seed=ep + 20)
            for t in range(200):
                motor, hier = organism.step()
                # Reversal indicated by negative-like speed or backward motion
                # Use AVA voltage > reversal_threshold as proxy
                ava_v = hier.L3_circuit.get("ava_v", 0.0)
                rev_thr = brain.get_param("motor.reversal_threshold")
                if ava_v > rev_thr:
                    reversal_count += 1
                total += 1

        return reversal_count / total if total > 0 else 0.0

    def _celegans_serotonin_speed(
        self,
        serotonin_level: float,
        expected_direction: str,
        threshold_pct: float,
        n_episodes: int,
    ) -> dict:
        baseline_speed = self._celegans_mean_speed(serotonin=0.0, n_episodes=n_episodes)
        treated_speed  = self._celegans_mean_speed(serotonin=serotonin_level, n_episodes=n_episodes)

        if baseline_speed > 1e-9:
            change_pct = 100.0 * (treated_speed - baseline_speed) / baseline_speed
        else:
            change_pct = 0.0

        passed = change_pct <= -threshold_pct  # must decrease by ≥ 30%
        return {
            "passed": passed,
            "baseline": round(baseline_speed, 4),
            "intervention": round(treated_speed, 4),
            "change_pct": round(change_pct, 2),
            "expected_direction": expected_direction,
        }

    def _celegans_mean_speed(self, serotonin: float, n_episodes: int) -> float:
        from ..environments.agar_plate import AgarPlate
        from ..species.celegans.brain import CElegansNervousSystem

        env = AgarPlate()
        brain = CElegansNervousSystem()
        brain.set_param("neuromod.serotonin_tone", serotonin)
        organism = VirtualOrganism(brain, env)

        speeds = []
        for ep in range(n_episodes):
            organism.reset(seed=ep + 30)
            for t in range(200):
                motor, _ = organism.step()
                speeds.append(motor.speed)

        return float(np.mean(speeds)) if speeds else 0.0

    def _celegans_hebbian_weight(
        self,
        hebb_lr: float,
        n_steps: int,
        threshold_delta: float,
    ) -> dict:
        """Run with Hebbian plasticity enabled; check AWC→AIY weight increase."""
        from ..environments.agar_plate import AgarPlate
        from ..species.celegans.brain import CElegansNervousSystem

        env = AgarPlate()
        brain = CElegansNervousSystem()
        organism = VirtualOrganism(brain, env)

        # Baseline weight before training
        baseline_weight = float(brain._W_hebb[brain._awcl, brain._aiy])

        # Enable Hebbian learning and run with attractant present
        brain.set_param("plasticity.hebbian_lr", hebb_lr)
        organism.reset(seed=42)
        for t in range(n_steps):
            organism.step()

        final_weight = float(brain._W_hebb[brain._awcl, brain._aiy])
        delta = final_weight - baseline_weight

        passed = delta >= threshold_delta
        if abs(baseline_weight) < 1e-6:
            change_pct = 999.9 if delta > 0 else 0.0   # baseline=0: report as capped
        else:
            change_pct = round(100.0 * delta / abs(baseline_weight), 2)
        return {
            "passed": passed,
            "baseline": round(baseline_weight, 6),
            "intervention": round(final_weight, 6),
            "change_pct": change_pct,
            "expected_direction": "increase",
            "weight_delta": round(delta, 6),
        }

    # ── Danio rerio interventions ─────────────────────────────────────────────

    def _run_danio(self) -> dict:
        results = {}

        # 1. Amygdala ablation → flee_fraction drops by ≥ 50%
        try:
            results["amygdala_ablation_fear"] = self._danio_amygdala_ablation(
                expected_direction="decrease",
                threshold_pct=50.0,
                n_episodes=5,
            )
        except Exception as exc:
            results["amygdala_ablation_fear"] = _error_intervention(exc)

        # 2. EFE exploration sweep: beta_epistemic inverted-U
        try:
            results["exploration_efe_sweep"] = self._danio_efe_sweep(n_episodes=5)
        except Exception as exc:
            results["exploration_efe_sweep"] = _error_intervention(exc)

        return results

    def _danio_amygdala_ablation(
        self,
        expected_direction: str,
        threshold_pct: float,
        n_episodes: int,
    ) -> dict:
        connectome, brain, env = build("danio_rerio")
        organism = VirtualOrganism(brain, env)
        protocol = AlarmPropagationProtocol()

        runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=0)
        baseline_result = runner.run(
            n_episodes=n_episodes, label="alarm_baseline", verbose=False
        )
        baseline_flee = float(np.mean([
            ep.extras.get("scores", {}).get("flee_fraction", 0.0)
            for ep in baseline_result.episodes
        ]))

        runner2 = ExperimentRunner(organism, protocol, log_levels=[5], seed=0)
        runner2.ablate("amygdala")
        ablated_result = runner2.run(
            n_episodes=n_episodes, label="amygdala_ablated", verbose=False
        )
        ablated_flee = float(np.mean([
            ep.extras.get("scores", {}).get("flee_fraction", 0.0)
            for ep in ablated_result.episodes
        ]))

        if baseline_flee > 1e-9:
            change_pct = 100.0 * (ablated_flee - baseline_flee) / baseline_flee
        else:
            change_pct = 0.0

        passed = change_pct <= -threshold_pct
        return {
            "passed": passed,
            "baseline": round(baseline_flee, 4),
            "intervention": round(ablated_flee, 4),
            "change_pct": round(change_pct, 2),
            "expected_direction": expected_direction,
        }

    def _danio_efe_sweep(self, n_episodes: int) -> dict:
        """
        Sweep beta_epistemic over [0.0, 1.0, 3.0].
        Expected: food_eaten peaks at 1.0 (inverted-U).
        """
        values = [0.0, 1.0, 3.0]
        food_means = []

        for beta in values:
            connectome, brain, env = build("danio_rerio")
            organism = VirtualOrganism(brain, env)
            protocol = ExplorationProtocol()
            runner = ExperimentRunner(organism, protocol, log_levels=[5], seed=0)
            runner.set_param("efe.beta_epistemic", beta)
            result = runner.run(n_episodes=n_episodes, label=f"efe_{beta}", verbose=False)
            food_means.append(result.mean_food)

        # Inverted-U: value at beta=1.0 should be higher than both beta=0.0 and beta=3.0
        food_at_0, food_at_1, food_at_3 = food_means
        passed = food_at_1 >= food_at_0 and food_at_1 >= food_at_3

        return {
            "passed": passed,
            "baseline": round(food_at_1, 4),    # beta=1.0 is "optimal"
            "intervention": round(max(food_at_0, food_at_3), 4),
            "change_pct": 0.0,
            "expected_direction": "inverted_U_peak_at_1.0",
            "sweep_values": {
                "beta_0.0": round(food_at_0, 4),
                "beta_1.0": round(food_at_1, 4),
                "beta_3.0": round(food_at_3, 4),
            },
        }


    # ── Drosophila melanogaster interventions ─────────────────────────────────

    def _run_drosophila(self) -> dict:
        results = {}

        # 1. APL ablation → KC sparsity removed → kc_fr increases ≥ 50%
        try:
            results["apl_ablation_increases_kc"] = self._drosophila_apl_ablation()
        except Exception as exc:
            results["apl_ablation_increases_kc"] = _error_intervention(exc)

        # 2. DAN ablation → KC→MBON weights do not potentiate in reward zone
        try:
            results["dan_ablation_prevents_plasticity"] = self._drosophila_dan_plasticity()
        except Exception as exc:
            results["dan_ablation_prevents_plasticity"] = _error_intervention(exc)

        return results

    def _drosophila_apl_ablation(self) -> dict:
        """APL ablation removes KC sparsity; kc_fr must increase by ≥ 50%."""
        from ..species.drosophila.brain import DrosophilaBrain
        from ..environments.olfactory_arena import OlfactoryArena

        n_steps = 30
        near_cs_plus = np.array([145.0, 100.0], dtype=np.float32)

        # Baseline (APL intact)
        env = OlfactoryArena()
        brain = DrosophilaBrain()
        organism = VirtualOrganism(brain, env)
        organism.reset(seed=0)
        env._agents[0].position = near_cs_plus.copy()

        kc_rates = []
        for _ in range(n_steps):
            _, hier = organism.step()
            kc_rates.append(float(hier.L3_circuit.get("kc_fr", 0.0)))
        baseline = float(np.mean(kc_rates))

        # APL ablated
        env2 = OlfactoryArena()
        brain2 = DrosophilaBrain()
        organism2 = VirtualOrganism(brain2, env2)
        organism2.reset(seed=0)
        env2._agents[0].position = near_cs_plus.copy()
        brain2.ablate("APL")

        kc_rates2 = []
        for _ in range(n_steps):
            _, hier = organism2.step()
            kc_rates2.append(float(hier.L3_circuit.get("kc_fr", 0.0)))
        ablated = float(np.mean(kc_rates2))

        if baseline > 1e-9:
            change_pct = 100.0 * (ablated - baseline) / baseline
        else:
            change_pct = 999.0 if ablated > 1e-9 else 0.0

        return {
            "passed": change_pct >= 50.0,
            "baseline": round(baseline, 4),
            "intervention": round(ablated, 4),
            "change_pct": round(change_pct, 2),
            "expected_direction": "increase",
        }

    def _drosophila_dan_plasticity(self) -> dict:
        """
        DAN ablation prevents KC→MBON potentiation.
        Fly held in reward zone (dist < 20mm from CS+).
        Baseline: weight delta > 0 (reward drives potentiation).
        DAN ablated: weight delta < 0 (spontaneous decay only).
        Pass: baseline_delta > ablated_delta.
        """
        from ..species.drosophila.brain import DrosophilaBrain
        from ..environments.olfactory_arena import OlfactoryArena

        n_steps = 50
        reward_pos = np.array([160.0, 100.0], dtype=np.float32)  # 10mm from CS+

        # Baseline (DAN intact)
        env = OlfactoryArena()
        brain = DrosophilaBrain()
        organism = VirtualOrganism(brain, env)
        organism.reset(seed=0)
        env._agents[0].position = reward_pos.copy()

        weights = []
        for _ in range(n_steps):
            _, hier = organism.step()
            env._agents[0].position = reward_pos.copy()  # keep in reward zone
            weights.append(float(hier.L1_synaptic.get("mean_kc_mbon_weight", 0.0)))

        baseline_delta = weights[-1] - weights[0] if len(weights) >= 2 else 0.0

        # DAN ablated
        env2 = OlfactoryArena()
        brain2 = DrosophilaBrain()
        organism2 = VirtualOrganism(brain2, env2)
        organism2.reset(seed=0)
        env2._agents[0].position = reward_pos.copy()
        brain2.ablate("DAN")

        weights2 = []
        for _ in range(n_steps):
            _, hier = organism2.step()
            env2._agents[0].position = reward_pos.copy()
            weights2.append(float(hier.L1_synaptic.get("mean_kc_mbon_weight", 0.0)))

        ablated_delta = weights2[-1] - weights2[0] if len(weights2) >= 2 else 0.0

        passed = baseline_delta > ablated_delta
        if abs(ablated_delta) > 1e-9:
            change_pct = 100.0 * (baseline_delta - ablated_delta) / abs(ablated_delta)
        else:
            change_pct = 999.0 if baseline_delta > 0 else 0.0

        return {
            "passed": passed,
            "baseline": round(baseline_delta, 6),
            "intervention": round(ablated_delta, 6),
            "change_pct": round(change_pct, 2),
            "expected_direction": "baseline_higher",
        }


def _error_intervention(exc: Exception) -> dict:
    return {
        "passed": False,
        "baseline": 0.0,
        "intervention": 0.0,
        "change_pct": 0.0,
        "expected_direction": "unknown",
        "description": f"ERROR: {type(exc).__name__}: {exc}",
    }
