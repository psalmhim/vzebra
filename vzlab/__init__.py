"""
vzlab — Virtual Laboratory for Multi-Species Hierarchical Brain Simulation

Quick start:
    from vzlab.core.registry import build
    from vzlab.engine.organism import VirtualOrganism
    from vzlab.engine.experiment import ExperimentRunner
    from vzlab.protocols import PredatorAvoidanceProtocol

    connectome, brain, env = build("danio_rerio")
    organism = VirtualOrganism(brain, env)
    runner = ExperimentRunner(organism, PredatorAvoidanceProtocol())
    runner.ablate("amygdala")
    result = runner.run(n_episodes=20)
    print(result.summary())
"""
__version__ = "0.1.0"

from .core.interfaces import Connectome, BrainModule, Environment, BehavioralProtocol
from .core.types import (
    SensorySignals, MotorOutput, HierarchicalState, WorldState,
    ExperimentResult, LoomingDisc, OdorantPulse, AlarmSubstance, DarkFlash,
)
from .engine.organism import VirtualOrganism
from .engine.experiment import ExperimentRunner
from .core.registry import build, list_species
from .protocols import TwoSpotAssayProtocol, ThermotaxisProtocol

__all__ = [
    "Connectome", "BrainModule", "Environment", "BehavioralProtocol",
    "SensorySignals", "MotorOutput", "HierarchicalState", "WorldState",
    "ExperimentResult", "LoomingDisc", "OdorantPulse", "AlarmSubstance", "DarkFlash",
    "VirtualOrganism", "ExperimentRunner", "build", "list_species",
    "TwoSpotAssayProtocol", "ThermotaxisProtocol",
]
