"""
OlfactoryArena — 200×200mm arena for Drosophila olfactory conditioning.

Two odor sources:
  CS+ (rewarded odor)  at position (170, 100) — Gaussian gradient sigma=60mm
  CS- (aversive odor)  at position (30, 100)  — Gaussian gradient sigma=60mm

Agent starts at center (100, 100). Reward is delivered when within 20mm of CS+.

Sensory output:
  chem.concentrations[0] = CS+ concentration at agent position
  chem.concentrations[1] = CS- concentration at agent position

WorldState extras:
  "reward"           1.0 if within 20mm of CS+ source, else 0.0
  "in_cs_plus_zone"  bool
  "in_cs_minus_zone" bool
  "approach_index"   (start_dist - current_dist) / start_dist, measures progress to CS+
"""
from __future__ import annotations

import math
import numpy as np

from ..core.interfaces import Environment
from ..core.types import (
    WorldState, AgentBody, SensorySignals, MotorOutput,
    ChemField, BodyState, Stimulus,
)

_ARENA_W  = 200.0   # mm
_ARENA_H  = 200.0   # mm
_CS_PLUS_POS  = np.array([170.0, 100.0], dtype=np.float32)
_CS_MINUS_POS = np.array([ 30.0, 100.0], dtype=np.float32)
_CENTER       = np.array([100.0, 100.0], dtype=np.float32)
_SIGMA        = 60.0  # mm
_REWARD_ZONE  = 20.0  # mm
_CONTACT_ZONE =  5.0  # mm from wall

_START_DIST = float(np.linalg.norm(_CS_PLUS_POS - _CENTER))   # ~70mm


class OlfactoryArena(Environment):
    """Olfactory conditioning arena for Drosophila melanogaster."""

    species = "drosophila_melanogaster"

    def __init__(
        self,
        width: float = _ARENA_W,
        height: float = _ARENA_H,
        cs_plus_pos: tuple = (170.0, 100.0),
        cs_minus_pos: tuple = (30.0, 100.0),
        gradient_sigma: float = _SIGMA,
        reward_zone_radius: float = _REWARD_ZONE,
        agent_speed: float = 2.0,   # mm/step
    ):
        self.W  = width
        self.H  = height
        self.cs_plus_pos  = np.array(cs_plus_pos,  dtype=np.float32)
        self.cs_minus_pos = np.array(cs_minus_pos, dtype=np.float32)
        self.sigma        = gradient_sigma
        self.reward_zone  = reward_zone_radius
        self.agent_speed  = agent_speed

        self._agents: dict[int, AgentBody] = {}
        self._stimuli: list[Stimulus] = []
        self._t = 0
        self._rng = np.random.default_rng(0)
        self._start_dists: dict[int, float] = {}

    def reset(self, n_agents: int = 1, seed: int | None = None) -> WorldState:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._agents = {}
        self._start_dists = {}
        for i in range(n_agents):
            pos = np.array([self.W / 2.0, self.H / 2.0], dtype=np.float32)
            self._agents[i] = AgentBody(
                agent_id=i,
                position=pos.copy(),
                orientation=float(self._rng.uniform(0, 2 * math.pi)),
                velocity=np.zeros(2, dtype=np.float32),
            )
            self._start_dists[i] = float(np.linalg.norm(pos - self.cs_plus_pos))
        return self.current_state

    def step(self, motor_outputs: list[MotorOutput]) -> WorldState:
        self._t += 1

        # Activate scheduled stimuli
        for stim in list(self._stimuli):
            if hasattr(stim, 'onset_t') and self._t >= stim.onset_t + stim.duration:
                self._stimuli.remove(stim)

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
        agent = self._agents[agent_id]
        chem  = self._compute_chem(agent)
        body  = self._compute_body(agent)
        # Pass reward into extras so brain can read it
        extras = {
            "reward": self.current_state.extras.get("reward", 0.0),
        }
        return SensorySignals(chem=chem, body=body, extras=extras)

    def inject(self, stimulus: Stimulus) -> None:
        self._stimuli.append(stimulus)

    def clear_stimuli(self) -> None:
        self._stimuli.clear()

    def add_agent(self, agent_id: int, position: np.ndarray | None = None) -> None:
        pos = (position if position is not None
               else np.array([self.W / 2.0, self.H / 2.0], dtype=np.float32))
        self._agents[agent_id] = AgentBody(
            agent_id=agent_id, position=pos.copy(),
            orientation=0.0, velocity=np.zeros(2, dtype=np.float32),
        )
        self._start_dists[agent_id] = float(np.linalg.norm(pos - self.cs_plus_pos))

    def remove_agent(self, agent_id: int) -> None:
        self._agents.pop(agent_id, None)
        self._start_dists.pop(agent_id, None)

    @property
    def current_state(self) -> WorldState:
        extras: dict = {"reward": 0.0}

        if not self._agents:
            return WorldState(t=self._t, agents=[], extras=extras)

        agents = list(self._agents.values())

        # Use first agent for scalar metrics (single-agent protocol)
        agent = agents[0]
        aid   = agent.agent_id

        dist_plus  = float(np.linalg.norm(agent.position - self.cs_plus_pos))
        dist_minus = float(np.linalg.norm(agent.position - self.cs_minus_pos))

        in_plus_zone  = dist_plus  < self.reward_zone
        in_minus_zone = dist_minus < self.reward_zone

        start_dist = self._start_dists.get(aid, _START_DIST)
        approach_index = float(np.clip(
            (start_dist - dist_plus) / (start_dist + 1e-6), -1.0, 1.0
        ))

        extras["reward"]           = 1.0 if in_plus_zone else 0.0
        extras["in_cs_plus_zone"]  = in_plus_zone
        extras["in_cs_minus_zone"] = in_minus_zone
        extras["approach_index"]   = approach_index
        extras["dist_to_cs_plus"]  = dist_plus
        extras["dist_to_cs_minus"] = dist_minus

        return WorldState(t=self._t, agents=agents, extras=extras)

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_chem(self, agent: AgentBody) -> ChemField:
        sigma = self.sigma

        dist_plus  = float(np.linalg.norm(agent.position - self.cs_plus_pos))
        dist_minus = float(np.linalg.norm(agent.position - self.cs_minus_pos))

        conc_plus  = math.exp(-dist_plus**2  / (2 * sigma**2))
        conc_minus = math.exp(-dist_minus**2 / (2 * sigma**2))

        grad_plus  = (self.cs_plus_pos  - agent.position) / (dist_plus  + 1e-6)
        grad_minus = (self.cs_minus_pos - agent.position) / (dist_minus + 1e-6)

        concentrations = np.array([conc_plus, conc_minus], dtype=np.float32)
        gradient       = np.stack([grad_plus, grad_minus]).astype(np.float32)

        return ChemField(
            species=["cs_plus", "cs_minus"],
            concentrations=concentrations,
            gradient=gradient,
        )

    def _compute_body(self, agent: AgentBody) -> BodyState:
        in_contact = (
            agent.position[0] < _CONTACT_ZONE or
            agent.position[0] > self.W - _CONTACT_ZONE or
            agent.position[1] < _CONTACT_ZONE or
            agent.position[1] > self.H - _CONTACT_ZONE
        )
        return BodyState(
            position=agent.position.copy(),
            velocity=agent.velocity.copy(),
            orientation=agent.orientation,
            angular_velocity=0.0,
            in_contact=in_contact,
        )
