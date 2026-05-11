"""
AquaticArena — 2-D aquatic environment for zebrafish (and any aquatic species).

Implements the Environment ABC. Produces raw physical signals:
  - PhotonField  (360° light field, 4 wavelengths: UV/B/G/R)
  - ChemField    (food / alarm / predator odorant concentrations)
  - PressureField (flow around the agent for lateral line)
  - BodyState    (position, velocity, orientation, energy, contact)
  - SocialField  (visible conspecifics)
"""
from __future__ import annotations

import math
import numpy as np

from ..core.interfaces import Environment
from ..core.types import (
    WorldState, AgentBody, SensorySignals, MotorOutput,
    PhotonField, ChemField, PressureField, BodyState, SocialField,
    Stimulus,
)


class AquaticArena(Environment):

    species = "aquatic"

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        n_food: int = 12,
        n_predators: int = 1,
        n_photon_bins: int = 64,
        photon_wavelengths: tuple = (360.0, 450.0, 530.0, 620.0),  # UV B G R
        predator_speed: float = 2.7,
        fish_speed_base: float = 3.0,
        eat_radius: float = 35.0,
        predation_radius: float = 28.0,
    ):
        self.W = width
        self.H = height
        self.n_food = n_food
        self.n_predators = n_predators
        self.n_bins = n_photon_bins
        self.wavelengths = np.array(photon_wavelengths, dtype=np.float32)
        self.predator_speed = predator_speed
        self.fish_speed_base = fish_speed_base
        self.eat_radius = eat_radius
        self.predation_radius = predation_radius

        self._agents: dict[int, AgentBody] = {}
        self._food: np.ndarray = np.empty((0, 2))
        self._predators: np.ndarray = np.empty((0, 2))
        self._predator_orientations: np.ndarray = np.empty((0,))
        self._stimuli: list[Stimulus] = []
        self._active_stimuli: list[Stimulus] = []
        self._t = 0
        self._food_eaten: int = 0
        self._predation_events: int = 0
        self._rng = np.random.default_rng(0)

    # ── Environment ABC ───────────────────────────────────────────────────────

    def reset(self, n_agents: int = 1, seed: int | None = None) -> WorldState:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._food_eaten = 0
        self._predation_events = 0
        self._active_stimuli = []

        # place food
        self._food = self._rng.uniform(
            [50, 50], [self.W - 50, self.H - 50],
            size=(self.n_food, 2)
        ).astype(np.float32)

        # place predators
        self._predators = self._rng.uniform(
            [0, 0], [self.W, self.H],
            size=(self.n_predators, 2)
        ).astype(np.float32)
        self._predator_orientations = self._rng.uniform(
            0, 2 * math.pi, size=(self.n_predators,)
        ).astype(np.float32)

        # place agents
        self._agents = {}
        for i in range(n_agents):
            pos = self._rng.uniform([100, 100], [self.W - 100, self.H - 100], (2,)).astype(np.float32)
            self._agents[i] = AgentBody(
                agent_id=i,
                position=pos,
                orientation=float(self._rng.uniform(0, 2 * math.pi)),
                velocity=np.zeros(2, dtype=np.float32),
            )

        return self.current_state

    def step(self, motor_outputs: list[MotorOutput]) -> WorldState:
        self._t += 1

        # apply motor to agents
        for motor in motor_outputs:
            if motor.agent_id not in self._agents:
                continue
            agent = self._agents[motor.agent_id]
            agent.orientation += np.clip(motor.turn, -0.5, 0.5)
            speed = self.fish_speed_base * np.clip(motor.speed, 0.0, 2.0)
            agent.velocity = np.array([
                math.cos(agent.orientation) * speed,
                math.sin(agent.orientation) * speed,
            ], dtype=np.float32)
            agent.position += agent.velocity
            agent.position = np.clip(
                agent.position, [0, 0], [self.W, self.H]
            )
            # expose motor state to scoring protocols via agent tags
            agent.tags["turn"] = float(motor.turn)
            agent.tags["alarmed"] = bool(motor.extras.get("alarmed", False))

        # predator AI: move toward nearest agent
        for pi in range(len(self._predators)):
            if not self._agents:
                break
            nearest = min(
                self._agents.values(),
                key=lambda a: np.linalg.norm(a.position - self._predators[pi]),
            )
            diff = nearest.position - self._predators[pi]
            dist = np.linalg.norm(diff) + 1e-6
            noise = self._rng.normal(0, 15, 2).astype(np.float32)
            self._predators[pi] += (diff / dist + noise / dist) * self.predator_speed
            self._predator_orientations[pi] = math.atan2(diff[1], diff[0])

        # eating
        for aid, agent in list(self._agents.items()):
            dists = np.linalg.norm(self._food - agent.position, axis=1)
            eaten = dists < self.eat_radius
            self._food_eaten += int(eaten.sum())
            self._food = self._food[~eaten]
            # respawn food
            while len(self._food) < self.n_food:
                new = self._rng.uniform([50, 50], [self.W - 50, self.H - 50], (1, 2)).astype(np.float32)
                self._food = np.vstack([self._food, new])

        # predation
        for pi, pred_pos in enumerate(self._predators):
            for aid, agent in list(self._agents.items()):
                if np.linalg.norm(agent.position - pred_pos) < self.predation_radius:
                    self._predation_events += 1
                    agent.alive = False

        # activate scheduled stimuli
        for stim in self._stimuli:
            if stim.onset_t <= self._t < stim.onset_t + stim.duration:
                if stim not in self._active_stimuli:
                    self._active_stimuli.append(stim)
            elif stim in self._active_stimuli:
                self._active_stimuli.remove(stim)

        return self.current_state

    def get_sensory_signals(self, agent_id: int) -> SensorySignals:
        if agent_id not in self._agents:
            return SensorySignals()
        agent = self._agents[agent_id]

        return SensorySignals(
            photon=self._compute_photon(agent),
            chem=self._compute_chem(agent),
            pressure=self._compute_pressure(agent),
            body=self._compute_body(agent),
            social=self._compute_social(agent),
        )

    def inject(self, stimulus: Stimulus) -> None:
        self._stimuli.append(stimulus)

    def clear_stimuli(self) -> None:
        self._stimuli.clear()
        self._active_stimuli.clear()

    def add_agent(self, agent_id: int, position: np.ndarray | None = None) -> None:
        pos = position if position is not None else self._rng.uniform(
            [100, 100], [self.W - 100, self.H - 100], (2,)
        ).astype(np.float32)
        self._agents[agent_id] = AgentBody(
            agent_id=agent_id,
            position=pos,
            orientation=0.0,
            velocity=np.zeros(2, dtype=np.float32),
        )

    def remove_agent(self, agent_id: int) -> None:
        self._agents.pop(agent_id, None)

    @property
    def current_state(self) -> WorldState:
        # Snapshot agents with copied tags so world_log entries don't share mutable state
        agent_snapshots = [
            AgentBody(
                agent_id=a.agent_id,
                position=a.position.copy(),
                orientation=a.orientation,
                velocity=a.velocity.copy(),
                angular_velocity=a.angular_velocity,
                alive=a.alive,
                tags=dict(a.tags),
            )
            for a in self._agents.values()
        ]
        return WorldState(
            t=self._t,
            agents=agent_snapshots,
            food_positions=self._food.copy(),
            predator_positions=self._predators.copy(),
            extras={
                "food_eaten": self._food_eaten,
                "predation_events": self._predation_events,
                "reward": float(self._food_eaten) - float(self._predation_events) * 2.0,
                "active_stimuli": [s.kind for s in self._active_stimuli],
            },
        )

    # ── Sensory signal computation (raw physics) ──────────────────────────────

    def _compute_photon(self, agent: AgentBody) -> PhotonField:
        angles = np.linspace(-math.pi, math.pi, self.n_bins, endpoint=False)
        intensity = np.zeros((self.n_bins, len(self.wavelengths)), dtype=np.float32)

        # food items: emit green/yellow (~530 nm)
        for fpos in self._food:
            diff = fpos - agent.position
            dist = np.linalg.norm(diff) + 1e-6
            if dist > 300:
                continue
            ang = math.atan2(diff[1], diff[0]) - agent.orientation
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            bin_idx = int((ang + math.pi) / (2 * math.pi) * self.n_bins) % self.n_bins
            strength = max(0.0, 1.0 - dist / 300.0)
            intensity[bin_idx, 2] += strength  # green channel

        # predators: emit dark/UV silhouette
        for ppos in self._predators:
            diff = ppos - agent.position
            dist = np.linalg.norm(diff) + 1e-6
            if dist > 400:
                continue
            ang = math.atan2(diff[1], diff[0]) - agent.orientation
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            bin_idx = int((ang + math.pi) / (2 * math.pi) * self.n_bins) % self.n_bins
            strength = max(0.0, 1.0 - dist / 400.0)
            # looming stimulus check
            for stim in self._active_stimuli:
                if stim.kind == "looming":
                    strength *= (1.0 + stim.params.get("expansion_rate", 0.2) * 10)
            intensity[bin_idx, 0] += strength  # UV channel (danger signal)

        # dark flash
        for stim in self._active_stimuli:
            if stim.kind == "dark_flash":
                intensity *= 0.0

        return PhotonField(angles=angles, wavelengths=self.wavelengths,
                           intensity=np.clip(intensity, 0, 1))

    def _compute_chem(self, agent: AgentBody) -> ChemField:
        food_conc = 0.0
        alarm_conc = 0.0
        pred_conc = 0.0

        for fpos in self._food:
            dist = np.linalg.norm(fpos - agent.position)
            food_conc += max(0.0, 1.0 - dist / 200.0)

        for stim in self._active_stimuli:
            if stim.kind == "alarm":
                src = np.array(stim.params.get("source_position", [400, 300]))
                dist = np.linalg.norm(src - agent.position)
                alarm_conc += max(0.0, 1.0 - dist / 600.0)  # full-arena detection radius
            elif stim.kind == "odorant":
                if stim.params.get("chemical") == "food":
                    food_conc += stim.params.get("concentration", 1.0)

        for ppos in self._predators:
            dist = np.linalg.norm(ppos - agent.position)
            pred_conc += max(0.0, 1.0 - dist / 250.0)

        conc = np.array([food_conc, alarm_conc, pred_conc], dtype=np.float32)

        # gradient direction toward nearest food
        grad = np.zeros((3, 2), dtype=np.float32)
        if len(self._food):
            nearest_food = self._food[
                np.argmin(np.linalg.norm(self._food - agent.position, axis=1))
            ]
            diff = nearest_food - agent.position
            dist = np.linalg.norm(diff) + 1e-6
            grad[0] = diff / dist

        return ChemField(
            species=["food", "alarm", "predator"],
            concentrations=conc,
            gradient=grad,
        )

    def _compute_pressure(self, agent: AgentBody) -> PressureField:
        # 8 neuromast positions around the body
        angles_nm = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        offset = 20.0
        positions = np.stack([
            np.cos(angles_nm) * offset,
            np.sin(angles_nm) * offset,
        ], axis=1).astype(np.float32)

        # flow from nearby agents + predators
        flow = np.zeros((8, 2), dtype=np.float32)
        for ppos in self._predators:
            diff = agent.position - ppos
            dist = np.linalg.norm(diff) + 1e-6
            if dist < 150:
                flow += (diff / dist / dist * 500.0).astype(np.float32)

        return PressureField(positions=positions, flow_vectors=flow)

    def _compute_body(self, agent: AgentBody) -> BodyState:
        in_contact = (
            agent.position[0] < 10 or agent.position[0] > self.W - 10 or
            agent.position[1] < 10 or agent.position[1] > self.H - 10
        )
        return BodyState(
            position=agent.position.copy(),
            velocity=agent.velocity.copy(),
            orientation=agent.orientation,
            angular_velocity=0.0,
            in_contact=in_contact,
            energy=agent.tags.get("energy", 1.0),
        )

    def _compute_social(self, agent: AgentBody) -> SocialField:
        other_agents = [a for aid, a in self._agents.items()
                        if aid != agent.agent_id and a.alive]
        if not other_agents:
            return SocialField(
                positions=np.empty((0, 2), dtype=np.float32),
                orientations=np.empty((0,), dtype=np.float32),
                velocities=np.empty((0, 2), dtype=np.float32),
                alarm_states=np.empty((0,), dtype=bool),
            )
        positions = np.array([a.position for a in other_agents], dtype=np.float32)
        orientations = np.array([a.orientation for a in other_agents], dtype=np.float32)
        velocities = np.array([a.velocity for a in other_agents], dtype=np.float32)
        alarm_states = np.array([a.tags.get("alarmed", False) for a in other_agents])
        return SocialField(positions=positions, orientations=orientations,
                           velocities=velocities, alarm_states=alarm_states)
