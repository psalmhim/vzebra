"""
ZFINConnectome — zebrafish connectome adapter.
Wraps the existing zebrav2 connectome data and exposes the platform Connectome ABC.
"""
from __future__ import annotations

import os
import numpy as np
from ...core.interfaces import Connectome

# Region list from MPIN atlas (72 bilateral regions, abbreviated)
_REGIONS = [
    "retina_L", "retina_R",
    "tectum_sfgs_L", "tectum_sfgs_R", "tectum_sgc_L", "tectum_sgc_R",
    "thalamus_L", "thalamus_R",
    "habenula_L", "habenula_R",
    "pallium_s_L", "pallium_s_R", "pallium_d_L", "pallium_d_R",
    "basal_ganglia_L", "basal_ganglia_R",
    "amygdala_L", "amygdala_R",
    "cerebellum_L", "cerebellum_R",
    "raphe_dorsal", "raphe_median",
    "locus_coeruleus_L", "locus_coeruleus_R",
    "olfactory_bulb_L", "olfactory_bulb_R",
    "lateral_line_anterior_L", "lateral_line_anterior_R",
    "vestibular_nucleus_L", "vestibular_nucleus_R",
    "reticulospinal_L", "reticulospinal_R",
    "spinal_cpg_L", "spinal_cpg_R",
    "inferior_olive_L", "inferior_olive_R",
    "hypothalamus",
    "pineal",
    "IPN",
    "pretectum_L", "pretectum_R",
]

_NEUROMOD_TARGETS = {
    "DA":  ["basal_ganglia_L", "basal_ganglia_R", "pallium_d_L", "pallium_d_R",
            "tectum_sgc_L", "tectum_sgc_R", "amygdala_L", "amygdala_R"],
    "NA":  ["tectum_sfgs_L", "tectum_sfgs_R", "amygdala_L", "amygdala_R",
            "pallium_s_L", "pallium_s_R", "hypothalamus"],
    "5HT": ["habenula_L", "habenula_R", "IPN", "raphe_dorsal", "raphe_median",
            "basal_ganglia_L", "basal_ganglia_R"],
    "ACh": ["tectum_sfgs_L", "tectum_sfgs_R", "pallium_s_L", "pallium_s_R",
            "lateral_line_anterior_L", "lateral_line_anterior_R"],
}

# Default neuron counts per region (approximate from MPIN atlas)
_REGION_SIZES = {
    "retina_L": 400, "retina_R": 400,
    "tectum_sfgs_L": 400, "tectum_sfgs_R": 400,
    "tectum_sgc_L": 400, "tectum_sgc_R": 400,
    "thalamus_L": 200, "thalamus_R": 200,
    "habenula_L": 80, "habenula_R": 80,
    "pallium_s_L": 300, "pallium_s_R": 300,
    "pallium_d_L": 200, "pallium_d_R": 200,
    "basal_ganglia_L": 150, "basal_ganglia_R": 150,
    "amygdala_L": 100, "amygdala_R": 100,
    "cerebellum_L": 200, "cerebellum_R": 200,
    "raphe_dorsal": 50, "raphe_median": 50,
    "locus_coeruleus_L": 30, "locus_coeruleus_R": 30,
    "olfactory_bulb_L": 150, "olfactory_bulb_R": 150,
    "lateral_line_anterior_L": 80, "lateral_line_anterior_R": 80,
    "vestibular_nucleus_L": 60, "vestibular_nucleus_R": 60,
    "reticulospinal_L": 60, "reticulospinal_R": 60,
    "spinal_cpg_L": 16, "spinal_cpg_R": 16,
    "inferior_olive_L": 40, "inferior_olive_R": 40,
    "hypothalamus": 80,
    "pineal": 20,
    "IPN": 60,
    "pretectum_L": 80, "pretectum_R": 80,
}


class ZFINConnectome(Connectome):

    def __init__(self, weight_data: dict | None = None):
        self._weights = weight_data or {}
        self._rng = np.random.default_rng(42)

    @property
    def species(self) -> str:
        return "danio_rerio"

    @property
    def n_neurons(self) -> int:
        return sum(_REGION_SIZES.values())

    @property
    def regions(self) -> list[str]:
        return list(_REGIONS)

    def region_neurons(self, name: str) -> dict:
        n = _REGION_SIZES.get(name, 50)
        return {
            "n": n,
            "cell_types": ["RS"] * int(n * 0.75) + ["FS"] * int(n * 0.25),
            "positions": self._rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        }

    def projection(self, src: str, tgt: str) -> np.ndarray:
        key = f"{src}->{tgt}"
        if key in self._weights:
            return self._weights[key]
        n_src = _REGION_SIZES.get(src, 50)
        n_tgt = _REGION_SIZES.get(tgt, 50)
        # sparse random initialisation (connectivity ~5%)
        W = np.zeros((n_src, n_tgt), dtype=np.float32)
        mask = self._rng.random((n_src, n_tgt)) < 0.05
        W[mask] = self._rng.normal(0, 0.1, mask.sum()).astype(np.float32)
        return W

    def neuromodulatory_targets(self, modulator: str) -> list[str]:
        return _NEUROMOD_TARGETS.get(modulator.upper(), [])

    @classmethod
    def load(cls, path: str) -> "ZFINConnectome":
        """Load weight matrices from a directory of .npy files."""
        weights = {}
        if path and os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.endswith(".npy"):
                    key = fname[:-4].replace("__", "->")
                    weights[key] = np.load(os.path.join(path, fname))
        return cls(weight_data=weights)

    @classmethod
    def default(cls) -> "ZFINConnectome":
        """Default connectome with random sparse weights."""
        return cls(weight_data={})
