"""
Species registry: maps species name → (connectome class, brain class, default environment).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces import Connectome, BrainModule, Environment


@dataclass
class SpeciesEntry:
    name: str
    common_name: str
    connectome_cls: str     # dotted import path
    brain_cls: str
    environment_cls: str
    n_neurons_approx: int
    notes: str = ""


_REGISTRY: dict[str, SpeciesEntry] = {}


def register(entry: SpeciesEntry) -> None:
    _REGISTRY[entry.name] = entry


def list_species() -> list[str]:
    return sorted(_REGISTRY.keys())


def get(name: str) -> SpeciesEntry:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown species '{name}'. Available: {list_species()}"
        )
    return _REGISTRY[name]


def build(name: str, connectome_path: str | None = None):
    """Instantiate the default (connectome, brain, environment) for a species."""
    import importlib
    entry = get(name)

    conn_mod, conn_cls = entry.connectome_cls.rsplit(".", 1)
    ConnCls = getattr(importlib.import_module(conn_mod), conn_cls)

    brain_mod, brain_cls = entry.brain_cls.rsplit(".", 1)
    BrainCls = getattr(importlib.import_module(brain_mod), brain_cls)

    env_mod, env_cls = entry.environment_cls.rsplit(".", 1)
    EnvCls = getattr(importlib.import_module(env_mod), env_cls)

    connectome = ConnCls.load(connectome_path) if connectome_path else ConnCls.default()
    environment = EnvCls()
    brain = BrainCls(connectome)
    return connectome, brain, environment


# ── Register built-in species ─────────────────────────────────────────────────

register(SpeciesEntry(
    name="danio_rerio",
    common_name="Larval zebrafish",
    connectome_cls="vzlab.species.zebrafish.connectome.ZFINConnectome",
    brain_cls="vzlab.species.zebrafish.brain.ZebrafishBrain",
    environment_cls="vzlab.environments.aquatic_arena.AquaticArena",
    n_neurons_approx=100_000,
    notes="MPIN atlas, 10.2M synapses, 48-module SNN",
))

register(SpeciesEntry(
    name="c_elegans",
    common_name="C. elegans",
    connectome_cls="vzlab.species.celegans.connectome.OpenWormConnectome",
    brain_cls="vzlab.species.celegans.brain.CElegansNervousSystem",
    environment_cls="vzlab.environments.agar_plate.AgarPlate",
    n_neurons_approx=302,
    notes="Full White et al. 1986 connectome, 7391 chemical + 890 electrical synapses",
))

register(SpeciesEntry(
    name="drosophila_melanogaster",
    common_name="Fruit fly",
    connectome_cls="vzlab.species.drosophila.connectome.DrosophilaConnectome",
    brain_cls="vzlab.species.drosophila.brain.DrosophilaBrain",
    environment_cls="vzlab.environments.olfactory_arena.OlfactoryArena",
    n_neurons_approx=2145,
    notes="Mushroom body: 2000 KCs, 34 MBONs, 60 DANs, olfactory conditioning",
))

register(SpeciesEntry(
    name="xenopus_laevis",
    common_name="African clawed frog (tadpole)",
    connectome_cls="vzlab.species.xenopus.connectome.XenopusConnectome",
    brain_cls="vzlab.species.xenopus.brain.XenopusTadpoleBrain",
    environment_cls="vzlab.environments.swim_tank.SwimTank",
    n_neurons_approx=150,
    notes="Spinal CPG: RB sensory, dIN rhythm generators, cIN cross-inhibition, MN motor (Roberts et al. 1998)",
))
