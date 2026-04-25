"""
SwimTank — 2-D swimming tank for Xenopus laevis tadpole.

50 mm × 20 mm tank. Tadpole starts at left wall (x = 1 mm).
Wall contact (within 2 mm of any edge) triggers the tactile stimulus
that initiates the spinal CPG.

Sensory output (SensorySignals):
  body.in_contact  True when agent is within contact margin of a wall
  body.position    Current (x, y) in mm
  body.velocity    Current velocity mm/step
  body.orientation Current heading (rad)
"""
from __future__ import annotations

import math
import numpy as np

from ..core.interfaces import Environment
from ..core.types import (
    WorldState, AgentBody, SensorySignals, MotorOutput,
    BodyState, Stimulus,
)

_TANK_W        = 50.0   # mm
_TANK_H        = 20.0   # mm
_CONTACT_MARGIN =  2.0  # mm from wall = "touch"
_AGENT_SPEED   =  3.0   # mm/step  (~0.6 body-lengths/step for a 5 mm tadpole)


class SwimTank(Environment):
    """Minimal swimming arena for Xenopus laevis tadpole CPG studies."""

    species = "xenopus_laevis"

    def __init__(
        self,
        width: float = _TANK_W,
        height: float = _TANK_H,
        agent_speed: float = _AGENT_SPEED,
    ):
        self.W = width
        self.H = height
        self.agent_speed = agent_speed

        self._agents: dict[int, AgentBody] = {}
        self._stimuli: list[Stimulus] = []
        self._t = 0
        self._rng = np.random.default_rng(0)

    # ── Environment ABC ───────────────────────────────────────────────────────

    def reset(self, n_agents: int = 1, seed: int | None = None) -> WorldState:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._agents = {}
        for i in range(n_agents):
            # Start at left wall so first step registers wall contact → CPG trigger
            pos = np.array([1.0, self.H / 2.0], dtype=np.float32)
            self._agents[i] = AgentBody(
                agent_id=i,
                position=pos,
                orientation=0.0,   # pointing right (toward open water)
                velocity=np.zeros(2, dtype=np.float32),
            )
        return self.current_state

    def step(self, motor_outputs: list[MotorOutput]) -> WorldState:
        self._t += 1
        for motor in motor_outputs:
            if motor.agent_id not in self._agents:
                continue
            agent = self._agents[motor.agent_id]
            agent.orientation += float(np.clip(motor.turn, -math.pi / 4, math.pi / 4))
            speed = self.agent_speed * float(np.clip(motor.speed, 0.0, 2.0))
            agent.velocity = np.array([
                math.cos(agent.orientation) * speed,
                math.sin(agent.orientation) * speed,
            ], dtype=np.float32)
            agent.position = np.clip(
                agent.position + agent.velocity,
                [0.0, 0.0], [self.W, self.H],
            )
        return self.current_state

    def get_sensory_signals(self, agent_id: int) -> SensorySignals:
        if agent_id not in self._agents:
            return SensorySignals()
        return SensorySignals(body=self._compute_body(self._agents[agent_id]))

    def inject(self, stimulus: Stimulus) -> None:
        self._stimuli.append(stimulus)

    def clear_stimuli(self) -> None:
        self._stimuli.clear()

    def add_agent(self, agent_id: int, position: np.ndarray | None = None) -> None:
        pos = (position if position is not None
               else np.array([1.0, self.H / 2.0], dtype=np.float32))
        self._agents[agent_id] = AgentBody(
            agent_id=agent_id, position=pos.copy(),
            orientation=0.0, velocity=np.zeros(2, dtype=np.float32),
        )

    def remove_agent(self, agent_id: int) -> None:
        self._agents.pop(agent_id, None)

    @property
    def current_state(self) -> WorldState:
        return WorldState(
            t=self._t,
            agents=list(self._agents.values()),
            extras={"reward": 0.0},
        )

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_body(self, agent: AgentBody) -> BodyState:
        x, y = agent.position
        in_contact = (
            x < _CONTACT_MARGIN or x > self.W - _CONTACT_MARGIN or
            y < _CONTACT_MARGIN or y > self.H - _CONTACT_MARGIN
        )
        return BodyState(
            position=agent.position.copy(),
            velocity=agent.velocity.copy(),
            orientation=agent.orientation,
            angular_velocity=0.0,
            in_contact=in_contact,
        )
