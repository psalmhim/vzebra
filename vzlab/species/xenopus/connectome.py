"""
XenopusConnectome — Xenopus laevis tadpole spinal swimming circuit.

Based on Roberts et al. 1998, Li et al. 2006.

Populations (150 neurons total):
  RB    20   Rohon-Beard mechanosensory neurons — detect skin contact
  dIN   60   Descending interneurons — CPG rhythm generators (30 L + 30 R)
  cIN   40   Commissural inhibitory interneurons — enforce L/R anti-phase (20 L + 20 R)
  MN    30   Motor neurons — activate axial swim muscles (15 L + 15 R)

Key connectivity (Roberts et al.):
  RB  → dIN_ipsi   monosynaptic touch response
  dIN → dIN_ipsi   local recurrent excitation (sustains rhythm)
  dIN → cIN_contra commissural excitation
  cIN → dIN_contra inhibitory cross-connection (creates alternation)
  dIN → MN_ipsi    swim motor drive
"""
from __future__ import annotations

import numpy as np
from ...core.interfaces import Connectome

_N_RB    =  20
_N_DIN   =  60   # 30 left + 30 right
_N_CIN   =  40   # 20 left + 20 right
_N_MN    =  30   # 15 left + 15 right
_N_TOTAL = _N_RB + _N_DIN + _N_CIN + _N_MN  # 150

_REGIONS = ["RB", "dIN", "cIN", "MN"]
_SIZES   = [_N_RB, _N_DIN, _N_CIN, _N_MN]

_SLICES: dict[str, slice] = {}
_off = 0
for _rname, _rn in zip(_REGIONS, _SIZES):
    _SLICES[_rname] = slice(_off, _off + _rn)
    _off += _rn

_NEUROMOD_TARGETS = {
    "5HT": ["dIN", "cIN"],   # serotonin modulates CPG period
    "ACh": ["MN"],
    "DA":  ["dIN"],
    "NA":  [],
}


def _build_weight_matrix(rng: np.random.Generator) -> np.ndarray:
    W = np.zeros((_N_TOTAL, _N_TOTAL), dtype=np.float32)

    rb  = _SLICES["RB"]
    din = _SLICES["dIN"]
    cin = _SLICES["cIN"]
    mn  = _SLICES["MN"]

    half_din = _N_DIN // 2   # 30 (left) | 30 (right)
    half_cin = _N_CIN // 2   # 20 (left) | 20 (right)
    half_mn  = _N_MN  // 2   # 15 (left) | 15 (right)

    # RB → ipsilateral dIN (all RB → left dIN; simple model)
    for rb_i in range(_N_RB):
        target_din = din.start + (rb_i % half_din)
        W[rb.start + rb_i, target_din] = float(rng.uniform(0.6, 0.9))

    # dIN_L ↔ dIN_L  (local recurrent excitation, sustains rhythm)
    for i in range(half_din):
        for j in range(half_din):
            if i != j and rng.random() < 0.15:
                W[din.start + i, din.start + j] = float(rng.uniform(0.2, 0.5))

    # dIN_R ↔ dIN_R
    for i in range(half_din, _N_DIN):
        for j in range(half_din, _N_DIN):
            if i != j and rng.random() < 0.15:
                W[din.start + i, din.start + j] = float(rng.uniform(0.2, 0.5))

    # dIN_L → MN_L
    for i in range(half_din):
        for m in range(half_mn):
            if rng.random() < 0.4:
                W[din.start + i, mn.start + m] = float(rng.uniform(0.4, 0.8))

    # dIN_R → MN_R
    for i in range(half_din, _N_DIN):
        for m in range(half_mn, _N_MN):
            if rng.random() < 0.4:
                W[din.start + i, mn.start + m] = float(rng.uniform(0.4, 0.8))

    # dIN_L → cIN_R (commissural excitation drives cross-inhibition)
    for i in range(half_din):
        for c in range(half_cin, _N_CIN):
            if rng.random() < 0.3:
                W[din.start + i, cin.start + c] = float(rng.uniform(0.3, 0.6))

    # dIN_R → cIN_L
    for i in range(half_din, _N_DIN):
        for c in range(half_cin):
            if rng.random() < 0.3:
                W[din.start + i, cin.start + c] = float(rng.uniform(0.3, 0.6))

    # cIN_L → dIN_R (inhibitory — suppresses opposite side)
    for c in range(half_cin):
        for i in range(half_din, _N_DIN):
            if rng.random() < 0.4:
                W[cin.start + c, din.start + i] = float(rng.uniform(-0.5, -0.2))

    # cIN_R → dIN_L
    for c in range(half_cin, _N_CIN):
        for i in range(half_din):
            if rng.random() < 0.4:
                W[cin.start + c, din.start + i] = float(rng.uniform(-0.5, -0.2))

    return np.clip(W, -1.0, 1.0)


class XenopusConnectome(Connectome):
    """Xenopus laevis tadpole spinal CPG connectome (Roberts et al. 1998)."""

    def __init__(self, weight_matrix: np.ndarray | None = None):
        if weight_matrix is not None:
            assert weight_matrix.shape == (_N_TOTAL, _N_TOTAL)
            self._W = weight_matrix.astype(np.float32)
        else:
            self._W = _build_weight_matrix(np.random.default_rng(7))

    @property
    def species(self) -> str:
        return "xenopus_laevis"

    @property
    def n_neurons(self) -> int:
        return _N_TOTAL

    @property
    def regions(self) -> list[str]:
        return list(_REGIONS)

    def region_neurons(self, name: str) -> dict:
        if name not in _SLICES:
            return {"n": 0, "cell_types": [], "positions": np.empty((0, 3))}
        sl = _SLICES[name]
        n  = sl.stop - sl.start
        rng = np.random.default_rng(hash(name) % (2**31))
        return {
            "n": n,
            "cell_types": [name] * n,
            "positions": rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        }

    def projection(self, src: str, tgt: str) -> np.ndarray:
        if src not in _SLICES or tgt not in _SLICES:
            return np.zeros((1, 1), dtype=np.float32)
        return self._W[_SLICES[src], _SLICES[tgt]]

    def neuromodulatory_targets(self, modulator: str) -> list[str]:
        return _NEUROMOD_TARGETS.get(modulator.upper(), [])

    @classmethod
    def load(cls, path: str) -> "XenopusConnectome":
        import os
        if path and os.path.isfile(path):
            W = np.load(path)
            return cls(weight_matrix=W)
        return cls.default()

    @classmethod
    def default(cls) -> "XenopusConnectome":
        return cls(weight_matrix=None)
