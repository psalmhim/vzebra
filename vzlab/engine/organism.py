"""
VirtualOrganism: wires a BrainModule to an Environment.
Single agent step loop. Multi-agent: run multiple organisms in the same env.
"""
from __future__ import annotations
from ..core.interfaces import BrainModule, Environment, BehavioralProtocol
from ..core.types import MotorOutput, HierarchicalState, WorldState


class VirtualOrganism:
    """One organism = one brain + one environment reference + one agent_id."""

    def __init__(self, brain: BrainModule, env: Environment,
                 agent_id: int = 0):
        self.brain = brain
        self.env = env
        self.agent_id = agent_id
        self._t = 0

    @property
    def species(self) -> str:
        return self.brain.species

    def reset(self, seed: int | None = None) -> WorldState:
        self.brain.reset()
        self._t = 0
        return self.env.reset(n_agents=1, seed=seed)

    def step(self) -> tuple[MotorOutput, HierarchicalState]:
        signals = self.env.get_sensory_signals(self.agent_id)
        motor = self.brain.step(signals, self._t)
        motor.agent_id = self.agent_id
        self.env.step([motor])
        state = self.brain.get_hierarchical_state(self._t)
        self._t += 1
        return motor, state

    # ── Delegation shortcuts ──────────────────────────────────────────────────

    def ablate(self, region: str) -> "VirtualOrganism":
        self.brain.ablate(region)
        return self

    def restore(self, region: str) -> "VirtualOrganism":
        self.brain.restore(region)
        return self

    def set_param(self, path: str, value: float) -> "VirtualOrganism":
        self.brain.set_param(path, value)
        return self

    def inject(self, stimulus) -> "VirtualOrganism":
        self.env.inject(stimulus)
        return self
