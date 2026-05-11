from .predator_avoidance import PredatorAvoidanceProtocol
from .chemotaxis import ChemotaxisProtocol
from .alarm_propagation import AlarmPropagationProtocol
from .exploration import ExplorationProtocol
from .two_spot_assay import TwoSpotAssayProtocol
from .thermotaxis import ThermotaxisProtocol
from .olfactory_conditioning import OlfactoryConditioningProtocol

PROTOCOLS = {
    "predator_avoidance":       PredatorAvoidanceProtocol,
    "chemotaxis":               ChemotaxisProtocol,
    "alarm_propagation":        AlarmPropagationProtocol,
    "exploration":              ExplorationProtocol,
    "two_spot_assay":           TwoSpotAssayProtocol,
    "thermotaxis":              ThermotaxisProtocol,
    "olfactory_conditioning":   OlfactoryConditioningProtocol,
}

__all__ = list(PROTOCOLS.keys()) + ["PROTOCOLS",
    "TwoSpotAssayProtocol", "ThermotaxisProtocol",
    "OlfactoryConditioningProtocol"]
