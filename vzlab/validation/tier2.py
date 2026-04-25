"""
Tier 2 — Atlas Correspondence.

Compares avatar population-level firing rates to synthetic reference
templates encoding expected relative activation patterns across conditions.
Spearman rank correlation is used; pass threshold is > 0.5.
"""
from __future__ import annotations

import numpy as np

from ..core.registry import build
from ..core.types import SensorySignals, ChemField, BodyState
from ..engine.organism import VirtualOrganism


# ── Reference templates ───────────────────────────────────────────────────────
# Expected relative activation per region per condition (0–1 scale).
# Encodes biology-grounded ordering: which regions should be more active.

REFERENCE_TEMPLATES: dict[str, dict[str, dict[str, float]]] = {
    "c_elegans": {
        "attractant_present": {
            "AWC": 0.8, "AIY": 0.7, "AIZ": 0.5, "AIB": 0.2, "AVA": 0.2, "AVB": 0.6,
        },
        "repellent_present": {
            "AWC": 0.2, "AIY": 0.2, "AIZ": 0.2, "AIB": 0.8, "AVA": 0.7, "AVB": 0.2,
        },
        "baseline": {
            "AWC": 0.1, "AIY": 0.1, "AIZ": 0.8, "AIB": 0.05, "AVA": 0.3, "AVB": 0.1,
        },
    },
    "danio_rerio": {
        "predator_present": {
            "sensory": 0.9, "tectum": 0.8, "amygdala": 0.9, "motor": 0.7,
        },
        "food_present": {
            "sensory": 0.7, "tectum": 0.5, "amygdala": 0.2, "motor": 0.6,
        },
        "baseline": {
            "sensory": 0.3, "tectum": 0.3, "amygdala": 0.1, "motor": 0.2,
        },
    },
    "drosophila_melanogaster": {
        # odor_present: PN gain amplified → PNs fire most; KCs sparse (APL); DANs baseline
        # Expected rank: kc < dan < pn
        "odor_present": {
            "pn": 0.6, "kc": 0.2, "dan": 0.3,
        },
        # odor_absent: PN gain zeroed → PNs silent; KCs also silent; DANs still fire at baseline
        # Expected rank: pn = kc < dan
        "odor_absent": {
            "pn": 0.0, "kc": 0.0, "dan": 0.5,
        },
    },
}


class Tier2AtlasCorrespondence:
    """Compare avatar firing rates to reference atlas templates via rank correlation."""

    def run(self, species: str, n_steps_per_condition: int = 50) -> dict:
        """
        Returns:
          {condition: {correlation: float, passed: bool, details: dict}}
        """
        if species not in REFERENCE_TEMPLATES:
            return {
                "_error": {
                    "correlation": 0.0,
                    "passed": False,
                    "details": {"error": f"No template for species '{species}'"},
                }
            }

        templates = REFERENCE_TEMPLATES[species]
        results = {}

        for condition, ref_pattern in templates.items():
            try:
                avatar_pattern = self._collect_avatar_pattern(
                    species, condition, ref_pattern, n_steps_per_condition
                )
                corr, details = self._compute_correlation(ref_pattern, avatar_pattern)
                threshold = 0.5
                results[condition] = {
                    "correlation": round(corr, 4),
                    "passed": corr > threshold,
                    "details": details,
                }
            except Exception as exc:
                results[condition] = {
                    "correlation": 0.0,
                    "passed": False,
                    "details": {"error": f"{type(exc).__name__}: {exc}"},
                }

        return results

    # ── Avatar pattern collection ─────────────────────────────────────────────

    def _collect_avatar_pattern(
        self,
        species: str,
        condition: str,
        ref_pattern: dict[str, float],
        n_steps: int,
    ) -> dict[str, float]:
        """Run avatar under specified condition and return mean relative activations."""
        connectome, brain, env = build(species)
        organism = VirtualOrganism(brain, env)

        self._setup_condition(env, brain, organism, species, condition)

        region_accum: dict[str, list[float]] = {r: [] for r in ref_pattern}

        for t in range(n_steps):
            motor, hier = organism.step()
            self._extract_region_activations(
                species, condition, hier, region_accum
            )

        # Average and normalise to [0, 1] relative scale
        raw = {r: float(np.mean(v)) if v else 0.0 for r, v in region_accum.items()}
        max_val = max(raw.values()) if raw else 1.0
        if max_val < 1e-9:
            max_val = 1.0
        normalised = {r: v / max_val for r, v in raw.items()}
        return normalised

    def _setup_condition(self, env, brain, organism, species: str, condition: str) -> None:
        """Reset and configure environment to match the condition."""
        organism.reset(seed=0)

        if species == "c_elegans":
            if condition == "attractant_present":
                env.attractant_peak = 1.0
                env.repellent_peak = 0.0
                # Move agent close to attractant
                env._agents[0].position = np.array(
                    [env.attractant_pos[0] - 10, env.attractant_pos[1]],
                    dtype=np.float32,
                )
            elif condition == "repellent_present":
                env.attractant_peak = 0.0
                env.repellent_peak = 1.0
                if env.repellent_pos is None:
                    env.repellent_pos = np.array([80.0, 50.0], dtype=np.float32)
            elif condition == "baseline":
                env.attractant_peak = 0.0
                env.repellent_peak = 0.0

        elif species == "danio_rerio":
            if condition == "predator_present":
                brain.set_param("efe.beta_flee", 2.0)
            elif condition == "food_present":
                brain.set_param("efe.beta_epistemic", 0.5)
            elif condition == "baseline":
                brain.set_param("efe.beta_flee", 1.0)
                brain.set_param("efe.beta_epistemic", 1.0)

        elif species == "drosophila_melanogaster":
            if condition == "odor_present":
                # Amplify PN gain so odor drives the circuit strongly
                brain.set_param("sensory.pn_gain", 2.0)
            elif condition == "odor_absent":
                # Silence odor input; DANs still fire from baseline dopamine tone
                brain.set_param("sensory.pn_gain", 0.0)

    def _extract_region_activations(
        self,
        species: str,
        condition: str,
        hier,
        region_accum: dict[str, list[float]],
    ) -> None:
        """Pull region-level values from HierarchicalState.L3_circuit."""
        L3 = hier.L3_circuit

        if species == "c_elegans":
            key_map = {
                "AWC": ("awcl_v", "awcr_v"),
                "AIY": ("aiy_v",),
                "AIZ": ("aiz_v",),
                "AIB": ("aib_v",),
                "AVA": ("ava_v",),
                "AVB": ("avb_v",),
            }
            for region, keys in key_map.items():
                if region in region_accum:
                    vals = [abs(float(L3.get(k, 0.0))) for k in keys]
                    region_accum[region].append(float(np.mean(vals)))

        elif species == "danio_rerio":
            # Map platform region names to L3_circuit keys from ZebrafishBrain
            key_map = {
                "sensory": "sensory_fr",
                "tectum":  "tectum_fr",
                "amygdala": "amygdala_fr",
                "motor":   "motor_fr",
            }
            for region, key in key_map.items():
                if region in region_accum:
                    region_accum[region].append(abs(float(L3.get(key, 0.0))))

        elif species == "drosophila_melanogaster":
            # Map region names to L3_circuit keys from DrosophilaBrain
            key_map = {
                "pn":  "pn_fr",
                "kc":  "kc_fr",
                "dan": "dan_activity",
            }
            for region, key in key_map.items():
                if region in region_accum:
                    region_accum[region].append(abs(float(L3.get(key, 0.0))))

    # ── Correlation computation ───────────────────────────────────────────────

    def _compute_correlation(
        self, ref: dict[str, float], avatar: dict[str, float]
    ) -> tuple[float, dict]:
        """Spearman rank correlation between ref and avatar patterns."""
        regions = sorted(ref.keys())
        ref_vec = np.array([ref[r] for r in regions])
        avatar_vec = np.array([avatar.get(r, 0.0) for r in regions])

        details = {
            "regions": regions,
            "ref_values": [round(v, 4) for v in ref_vec.tolist()],
            "avatar_values": [round(v, 4) for v in avatar_vec.tolist()],
        }

        if len(regions) < 2:
            return 0.0, {**details, "note": "not enough regions to correlate"}

        if np.std(ref_vec) < 1e-9 or np.std(avatar_vec) < 1e-9:
            # Constant vector: undefined correlation
            corr = 0.0
            details["note"] = "constant vector, correlation undefined"
        else:
            corr = _spearman_r(ref_vec, avatar_vec)
            if np.isnan(corr):
                corr = 0.0

        return corr, details


# ── Pure-numpy Spearman rank correlation ──────────────────────────────────────

def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation without scipy."""
    n = len(x)
    if n < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    d = rx - ry
    d2 = float(np.dot(d, d))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Assign average ranks to elements (handles ties)."""
    n = len(a)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # Average tied ranks
    i = 0
    while i < n:
        j = i + 1
        while j < n and a[order[j]] == a[order[i]]:
            j += 1
        if j > i + 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks
