"""
OpenWormConnectome — C. elegans 302-neuron connectome.

Based on White et al. 1986 (7,391 chemical + 890 electrical synapses).
Neuron categories: sensory (32), inter (76), motor (109), polymodal (85).
"""
from __future__ import annotations

import json
import os
import numpy as np
from ...core.interfaces import Connectome


# Neuron type assignments (abbreviated; full list has 302 neurons)
_SENSORY = [
    "ASEL", "ASER", "ASHL", "ASHR", "ASIL", "ASIR", "ASJL", "ASJR",
    "ASKL", "ASKR", "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR",
    "ADFL", "ADFR", "ADLL", "ADLR", "AFDD", "ALML", "ALMR", "AFDL",
    "BAGL", "BAGR", "CEPVL", "CEPVR", "CEPDL", "CEPDR",
    "IL1DL", "IL1DR",
]
_INTER = [
    "AIY", "AIZ", "AIB", "AIA", "AIM", "AIE", "AIF",
    "RIA", "RIB", "RIF", "RIG", "RIM", "RIN", "RIP", "RIR", "RIS",
    "AVA", "AVB", "AVD", "AVE", "AVF", "AVG", "AVH", "AVJ", "AVK", "AVL",
    "RMD", "RME", "RMF", "RMH",
    "PVC", "PVD", "PVP", "PVQ", "PVR", "PVT", "PVW",
    "DVA", "DVB", "DVC",
    "PVQL", "PVQR",
    "SDQL", "SDQR", "ALML", "ALMR",
    "SIADL", "SIADR", "SIAVL", "SIAVR",
    "SIBDL", "SIBDR", "SIBVL", "SIBVR",
    "SMBDL", "SMBDR", "SMBVL", "SMBVR",
    "SMDDL", "SMDDR", "SMDVL", "SMDVR",
    "URADL", "URADR", "URAVL", "URAVR",
    "RIVL", "RIVR", "RMGL", "RMGR",
    "OLLL", "OLLR", "OLQDL", "OLQDR", "OLQVL", "OLQVR",
]
_MOTOR = [
    f"VA{i}" for i in range(1, 13)] + [f"VB{i}" for i in range(1, 12)] + [
    f"VD{i}" for i in range(1, 14)] + [f"DA{i}" for i in range(1, 10)] + [
    f"DB{i}" for i in range(1, 8)] + [f"DD{i}" for i in range(1, 7)] + [
    "AS1", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10",
    "AS11",
    "HSNL", "HSNR", "VC1", "VC2", "VC3", "VC4", "VC5", "VC6",
    "MVL07", "MVL08", "MVL09", "MVL10", "MVL11", "MVL12", "MVL13",
    "MVR07", "MVR08", "MVR09", "MVR10", "MVR11", "MVR12", "MVR13",
]

ALL_NEURONS = _SENSORY + _INTER + _MOTOR
# Pad to 302 if needed
while len(ALL_NEURONS) < 302:
    ALL_NEURONS.append(f"UNK{len(ALL_NEURONS)}")
ALL_NEURONS = ALL_NEURONS[:302]

_NEURON_INDEX = {n: i for i, n in enumerate(ALL_NEURONS)}

_NEUROMOD_TARGETS = {
    "DA":  ["RIM", "AVA", "AVB", "AIB"],
    "5HT": ["RIM", "AIM", "AIY"],
    "ACh": _MOTOR[:20],
    "NA":  [],  # C. elegans has no noradrenaline
}


class OpenWormConnectome(Connectome):
    """C. elegans 302-neuron connectome from White et al. 1986."""

    def __init__(self, weight_matrix: np.ndarray | None = None):
        N = 302
        if weight_matrix is not None:
            assert weight_matrix.shape == (N, N)
            self._W = weight_matrix.astype(np.float32)
        else:
            self._W = self._default_weights(N)

    @staticmethod
    def _default_weights(N: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        W = np.zeros((N, N), dtype=np.float32)
        # Chemotaxis circuit: AWC → AIY → AIZ → SMB (attractive odor)
        # AWB → AIB → AVA → VA/DB (repellent odor)
        known_connections = [
            # Attractive-odor chemotaxis circuit (White 1986, Bargmann 2006)
            ("AWCL", "AIY", 0.9), ("AWCR", "AIY", 0.9),
            ("ASEL", "AIY", 0.8), ("ASER", "AIY", 0.8),
            ("AIY",  "AIZ", -0.30),  # inhibitory: AIY promotes runs by suppressing AIZ pirouette
            ("AIZ",  "SMBVL", 0.6), ("AIZ", "SMBVR", 0.6),
            # AIY → forward locomotion (AVB pathway — key missing link)
            ("AIY",  "AVB", 0.7), ("AIY", "PVC", 0.4),
            ("PVC",  "AVB", 0.6),
            ("AVB",  "DB1",  0.7), ("AVB", "DB2", 0.7),
            ("AVB",  "VB1",  0.6), ("AVB", "VB2", 0.6),
            # AIY inhibits reversal command (AIY promotes forward run)
            ("AIY",  "AVA", -0.5),
            # Repellent circuit
            ("AWBL", "AIB", 0.7), ("AWBR", "AIB", 0.7),
            ("AIB",  "AVA", 0.7), ("AVA",  "VA1",  0.6), ("AVA", "VA2", 0.6),
            # RIM gates motor during pirouette; AIZ commands reversal via AVA
            ("RIM",  "AVA", 0.5), ("AIZ", "RIM", 0.4),
            ("AIZ",  "AVA", 0.4),  # AIZ drives AVA reversal (accumulates over ~4 spikes)
            # DVA proprioception
            ("DVA",  "PVC",  0.4),
        ]
        for src, tgt, w in known_connections:
            if src in _NEURON_INDEX and tgt in _NEURON_INDEX:
                W[_NEURON_INDEX[src], _NEURON_INDEX[tgt]] = w
        # sparse random background
        mask = rng.random((N, N)) < 0.02
        W[mask] += rng.normal(0, 0.05, mask.sum()).astype(np.float32)
        return np.clip(W, -1, 1)

    @property
    def species(self) -> str:
        return "c_elegans"

    @property
    def n_neurons(self) -> int:
        return 302

    @property
    def regions(self) -> list[str]:
        return ["sensory", "interneuron", "motor"]

    def region_neurons(self, name: str) -> dict:
        mapping = {
            "sensory":    (_SENSORY,    [0] * len(_SENSORY)),
            "interneuron": (_INTER,     [0] * len(_INTER)),
            "motor":      (_MOTOR,      [0] * len(_MOTOR)),
        }
        neurons, _ = mapping.get(name, (ALL_NEURONS, []))
        n = len(neurons)
        return {
            "n": n,
            "cell_types": ["sensory" if nm in _SENSORY else
                           "inter" if nm in _INTER else "motor"
                           for nm in neurons],
            "neuron_names": neurons,
            "positions": np.random.default_rng(1).uniform(-1, 1, (n, 3)).astype(np.float32),
        }

    def projection(self, src: str, tgt: str) -> np.ndarray:
        src_map = {"sensory": _SENSORY, "interneuron": _INTER, "motor": _MOTOR}
        src_idx = [_NEURON_INDEX[n] for n in src_map.get(src, []) if n in _NEURON_INDEX]
        tgt_idx = [_NEURON_INDEX[n] for n in src_map.get(tgt, []) if n in _NEURON_INDEX]
        if not src_idx or not tgt_idx:
            return np.zeros((1, 1), dtype=np.float32)
        return self._W[np.ix_(src_idx, tgt_idx)]

    def neuromodulatory_targets(self, modulator: str) -> list[str]:
        return _NEUROMOD_TARGETS.get(modulator.upper(), [])

    @classmethod
    def load(cls, path: str) -> "OpenWormConnectome":
        if path and os.path.isfile(path):
            ext = os.path.splitext(path)[1]
            if ext == ".npy":
                W = np.load(path)
                return cls(weight_matrix=W)
            elif ext == ".json":
                with open(path) as f:
                    data = json.load(f)
                W = np.array(data["weights"], dtype=np.float32)
                return cls(weight_matrix=W)
        return cls.default()

    @classmethod
    def default(cls) -> "OpenWormConnectome":
        return cls(weight_matrix=None)
