"""
DrosophilaConnectome — Mushroom body olfactory learning circuit.

Neuron populations (total 2145):
  PN      50   Projection Neurons — relay odor identity from antennal lobe
  KC_ab  700   Kenyon Cells alpha/beta — sparse combinatorial olfactory coding
  KC_abp 300   Kenyon Cells alpha'/beta' — intermediate-term memory locus
  KC_g  1000   Kenyon Cells gamma — short-term/aversive memory locus
  APL      1   Anterior Paired Lateral — global feedback inhibition
  DAN     60   Dopaminergic Neurons — reward/punishment signal (3 compartments × 20)
  MBON    34   Mushroom Body Output Neurons — read out KC activity

Connectivity (from connectomics data):
  PN   → KC:   random sparse; each KC receives ~6 of 50 PNs (density 12%)
  KC   → APL:  uniform positive (all KCs excite APL)
  APL  → KC:   uniform negative (APL inhibits all KCs; global WTA)
  KC   → MBON: random sparse positive (~5% of KCs per MBON); plastic
  DAN  → KC→MBON synapse: gates plasticity (not a direct connection)
  MBON → behavior: approach vs avoidance
"""
from __future__ import annotations

import numpy as np
from ...core.interfaces import Connectome

# ── Population sizes ──────────────────────────────────────────────────────────

_N_PN    =   50
_N_KC_AB =  700
_N_KC_ABP = 300
_N_KC_G  = 1000
_N_APL   =    1
_N_DAN   =   60
_N_MBON  =   34

_N_TOTAL = _N_PN + _N_KC_AB + _N_KC_ABP + _N_KC_G + _N_APL + _N_DAN + _N_MBON
# = 50 + 700 + 300 + 1000 + 1 + 60 + 34 = 2145

# Region slices (indices into the 2145-neuron vector)
_REGIONS = ["PN", "KC_ab", "KC_abp", "KC_g", "APL", "DAN", "MBON"]

_SLICES: dict[str, slice] = {}
_off = 0
for _name, _n in zip(_REGIONS, [_N_PN, _N_KC_AB, _N_KC_ABP, _N_KC_G, _N_APL, _N_DAN, _N_MBON]):
    _SLICES[_name] = slice(_off, _off + _n)
    _off += _n

# Neuron type labels
_NEURON_TYPES: list[str] = (
    [f"PN_{i}" for i in range(_N_PN)] +
    [f"KC_ab_{i}" for i in range(_N_KC_AB)] +
    [f"KC_abp_{i}" for i in range(_N_KC_ABP)] +
    [f"KC_g_{i}" for i in range(_N_KC_G)] +
    ["APL"] +
    [f"DAN_c{c}_{i}" for c in range(3) for i in range(20)] +
    [f"MBON_{i}" for i in range(_N_MBON)]
)
assert len(_NEURON_TYPES) == _N_TOTAL, f"Expected 2145 got {len(_NEURON_TYPES)}"

# Neuromodulatory targets
_NEUROMOD_TARGETS = {
    "DA":  ["KC_ab", "KC_abp", "KC_g", "MBON"],  # dopamine gates KC→MBON plasticity
    "5HT": ["DAN", "MBON"],
    "ACh": ["KC_ab", "KC_abp", "KC_g"],            # cholinergic PNs drive KCs
    "NA":  [],
}


def _build_weight_matrix(rng: np.random.Generator) -> np.ndarray:
    """Build the 2145×2145 weight matrix from known connectivity rules."""
    N = _N_TOTAL
    W = np.zeros((N, N), dtype=np.float32)

    pn    = _SLICES["PN"]
    kc_ab = _SLICES["KC_ab"]
    kc_ap = _SLICES["KC_abp"]
    kc_g  = _SLICES["KC_g"]
    apl   = _SLICES["APL"]
    mbon  = _SLICES["MBON"]

    n_kc_total = _N_KC_AB + _N_KC_ABP + _N_KC_G
    kc_start   = pn.stop   # index where KCs begin

    # ── PN → KC (all KC subtypes): each KC receives exactly 6 of 50 PNs ──────
    for kc_local in range(n_kc_total):
        kc_global = kc_start + kc_local
        chosen = rng.choice(_N_PN, size=6, replace=False)
        for pn_local in chosen:
            W[pn.start + pn_local, kc_global] = float(rng.uniform(0.5, 1.0))

    # ── KC → APL: all KCs uniformly excite APL ───────────────────────────────
    apl_idx = apl.start
    for kc_global in range(kc_start, kc_start + n_kc_total):
        W[kc_global, apl_idx] = 0.1 / n_kc_total   # small but cumulative

    # ── APL → KC: uniform inhibition onto all KCs ────────────────────────────
    for kc_global in range(kc_start, kc_start + n_kc_total):
        W[apl_idx, kc_global] = -2.0 / n_kc_total   # global inhibition

    # ── KC → MBON: random sparse positive (~5% of KCs per MBON) ─────────────
    # These are the plastic synapses (W_KC_MBON, updated during conditioning)
    mbon_start = mbon.start
    for m in range(_N_MBON):
        n_inputs = max(1, int(0.05 * n_kc_total))
        chosen_kc = rng.choice(n_kc_total, size=n_inputs, replace=False)
        for kc_local in chosen_kc:
            W[kc_start + kc_local, mbon_start + m] = float(rng.uniform(0.1, 0.5))

    # ── DAN → behavior (no direct connections; handled in brain) ────────────

    return W


class DrosophilaConnectome(Connectome):
    """Drosophila melanogaster mushroom body connectome.

    Implements the olfactory learning circuit: PN → KC → MBON
    with APL global inhibition and DAN-gated KC→MBON plasticity.
    """

    def __init__(self, weight_matrix: np.ndarray | None = None):
        if weight_matrix is not None:
            assert weight_matrix.shape == (_N_TOTAL, _N_TOTAL)
            self._W = weight_matrix.astype(np.float32)
        else:
            rng = np.random.default_rng(42)
            self._W = _build_weight_matrix(rng)

        self._neuron_types = _NEURON_TYPES
        self._slices = _SLICES

    @property
    def species(self) -> str:
        return "drosophila_melanogaster"

    @property
    def n_neurons(self) -> int:
        return _N_TOTAL

    @property
    def regions(self) -> list[str]:
        return list(_REGIONS)

    def region_neurons(self, name: str) -> dict:
        if name not in self._slices:
            return {"n": 0, "cell_types": [], "positions": np.empty((0, 3))}
        sl = self._slices[name]
        n = sl.stop - sl.start
        cell_types = self._neuron_types[sl.start:sl.stop]
        rng = np.random.default_rng(hash(name) % (2**31))
        positions = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        return {
            "n": n,
            "cell_types": cell_types,
            "positions": positions,
        }

    def projection(self, src: str, tgt: str) -> np.ndarray:
        if src not in self._slices or tgt not in self._slices:
            return np.zeros((1, 1), dtype=np.float32)
        sl_src = self._slices[src]
        sl_tgt = self._slices[tgt]
        return self._W[sl_src, sl_tgt]

    def neuromodulatory_targets(self, modulator: str) -> list[str]:
        return _NEUROMOD_TARGETS.get(modulator.upper(), [])

    @classmethod
    def load(cls, path: str) -> "DrosophilaConnectome":
        import os
        if path and os.path.isfile(path):
            ext = os.path.splitext(path)[1]
            if ext == ".npy":
                W = np.load(path)
                return cls(weight_matrix=W)
        return cls.default()

    @classmethod
    def default(cls) -> "DrosophilaConnectome":
        return cls(weight_matrix=None)
