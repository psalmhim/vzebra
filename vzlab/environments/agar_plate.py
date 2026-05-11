"""
AgarPlate — 2-D environment for C. elegans assays.

Modes:
  "single"      Single attractant source + optional repellent (default)
  "two_spot"    Two attractant sources at opposite quadrants (choice assay)
  "thermal"     Thermal gradient (orthogonal to chemical) for thermotaxis
  "population"  Multiple worms; collective chemotaxis index reported

Tunable construction params:
  width, height         Plate dimensions (mm, default 100×100)
  gradient              Attractant chemical name (default "diacetyl")
  gradient_sigma        Gaussian half-width of gradient (default 25 mm)
  attractant_peak       Peak concentration at source (default 1.0)
  repellent_peak        Peak repellent concentration (default 0.8)
  attractant_position   (x, y) mm of attractant source (default (80, 50))
  repellent_position    (x, y) mm of repellent source (default None)
  agent_speed           Worm crawl speed mm/step (default 2.0; 1 step ≈ 5s)
  gradient_drift_speed  Source drift mm/step for time-varying mode (default 0)
  temp_cold             Cold-spot temperature °C (default 15)
  temp_warm             Warm-spot temperature °C (default 25)
  thermal_axis          Axis of thermal gradient: "x" or "y" (default "x")
"""
from __future__ import annotations

import math
import numpy as np

from ..core.interfaces import Environment
from ..core.types import (
    WorldState, AgentBody, SensorySignals, MotorOutput,
    ChemField, BodyState, Stimulus,
)


class _ThermalBody(BodyState):
    """BodyState augmented with temperature for AFD neuron input."""
    __slots__ = BodyState.__slots__ + ("temperature",) if hasattr(BodyState, "__slots__") else ()

    def __init__(self, *args, temperature: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature


class AgarPlate(Environment):

    species = "c_elegans"

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        mode: str = "single",
        gradient: str = "diacetyl",
        gradient_sigma: float = 25.0,
        attractant_peak: float = 1.0,
        repellent_peak: float = 0.8,
        attractant_position: tuple = (80.0, 50.0),
        second_attractant_position: tuple = (20.0, 50.0),
        repellent_position: tuple | None = None,
        agent_speed: float = 2.0,   # mm/step (1 step ≈ 5s; worm ~0.3mm/s × 5s)
        gradient_drift_speed: float = 0.0,   # mm/step, for time-varying mode
        temp_cold: float = 15.0,
        temp_warm: float = 25.0,
        thermal_axis: str = "x",             # "x" or "y"
    ):
        self.W = width
        self.H = height
        self.mode = mode
        self.gradient = gradient
        self.gradient_sigma = gradient_sigma
        self.attractant_peak = attractant_peak
        self.repellent_peak  = repellent_peak
        self.attractant_pos  = np.array(attractant_position,        dtype=np.float32)
        self.second_attr_pos = np.array(second_attractant_position, dtype=np.float32)
        self.repellent_pos   = (np.array(repellent_position, dtype=np.float32)
                                if repellent_position else None)
        self.agent_speed         = agent_speed
        self.gradient_drift_speed = gradient_drift_speed
        self.temp_cold       = temp_cold
        self.temp_warm       = temp_warm
        self.thermal_axis    = thermal_axis

        self._agents: dict[int, AgentBody] = {}
        self._stimuli: list[Stimulus] = []
        self._active_stimuli: list[Stimulus] = []
        self._t = 0
        self._rng = np.random.default_rng(0)
        self._drift_offset = np.zeros(2, dtype=np.float32)

        # Track cumulative chemotaxis scores
        self._ci_history: list[float] = []
        self._second_ci_history: list[float] = []   # for two_spot mode

    def reset(self, n_agents: int = 1, seed: int | None = None) -> WorldState:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._active_stimuli = []
        self._agents = {}
        self._drift_offset[:] = 0.0
        self._ci_history.clear()
        self._second_ci_history.clear()

        for i in range(n_agents):
            # Start at plate centre
            pos = np.array([self.W / 2.0, self.H / 2.0], dtype=np.float32)
            self._agents[i] = AgentBody(
                agent_id=i,
                position=pos,
                orientation=float(self._rng.uniform(0, 2 * math.pi)),
                velocity=np.zeros(2, dtype=np.float32),
            )
        return self.current_state

    def step(self, motor_outputs: list[MotorOutput]) -> WorldState:
        self._t += 1

        # Drift attractant source for time-varying mode
        if self.gradient_drift_speed > 0:
            angle = self._t * 0.05   # slow circle
            self._drift_offset = np.array([
                math.cos(angle) * self.gradient_drift_speed * self._t,
                math.sin(angle) * self.gradient_drift_speed * self._t,
            ], dtype=np.float32)
            self._drift_offset = np.clip(
                self._drift_offset,
                -self.W * 0.3, self.W * 0.3,
            )

        # Activate scheduled stimuli
        for stim in self._stimuli:
            if hasattr(stim, "onset") and self._t >= stim.onset:
                if stim not in self._active_stimuli:
                    self._active_stimuli.append(stim)

        for motor in motor_outputs:
            if motor.agent_id not in self._agents:
                continue
            agent = self._agents[motor.agent_id]
            # C. elegans pirouettes are essentially instantaneous reorientations.
            # Allow up to π (180°) turn per step; non-pirouette turns are bounded
            # by the brain before reaching here.
            agent.orientation += float(np.clip(motor.turn, -math.pi, math.pi))
            speed = self.agent_speed * float(np.clip(motor.speed, 0.0, 2.0))
            agent.velocity = np.array([
                math.cos(agent.orientation) * speed,
                math.sin(agent.orientation) * speed,
            ], dtype=np.float32)
            agent.position = np.clip(
                agent.position + agent.velocity,
                [0.0, 0.0], [self.W, self.H],
            )

        state = self.current_state
        self._ci_history.append(state.extras.get("chemotaxis_index", 0.0))
        return state

    def get_sensory_signals(self, agent_id: int) -> SensorySignals:
        if agent_id not in self._agents:
            return SensorySignals()
        agent = self._agents[agent_id]
        body = self._compute_body(agent)

        return SensorySignals(
            chem=self._compute_chem(agent),
            body=body,
        )

    def inject(self, stimulus: Stimulus) -> None:
        self._stimuli.append(stimulus)

    def clear_stimuli(self) -> None:
        self._stimuli.clear()
        self._active_stimuli.clear()

    def add_agent(self, agent_id: int, position: np.ndarray | None = None) -> None:
        pos = (position if position is not None
               else np.array([self.W / 2.0, self.H / 2.0], dtype=np.float32))
        self._agents[agent_id] = AgentBody(
            agent_id=agent_id, position=pos,
            orientation=0.0, velocity=np.zeros(2, dtype=np.float32),
        )

    def remove_agent(self, agent_id: int) -> None:
        self._agents.pop(agent_id, None)

    @property
    def current_state(self) -> WorldState:
        extras = {"reward": 0.0}

        if not self._agents:
            extras["chemotaxis_index"] = 0.0
            return WorldState(t=self._t, agents=[], extras=extras)

        attr_pos = self.attractant_pos + self._drift_offset
        near_radius = self.W * 0.2

        dists_attr = np.array([
            np.linalg.norm(a.position - attr_pos)
            for a in self._agents.values()
        ])
        ci = float(np.mean(dists_attr < near_radius))
        extras["chemotaxis_index"] = ci
        extras["reward"] = ci

        if self.mode == "two_spot":
            dists_b = np.array([
                np.linalg.norm(a.position - self.second_attr_pos)
                for a in self._agents.values()
            ])
            ci_b = float(np.mean(dists_b < near_radius))
            extras["ci_spot_a"] = ci
            extras["ci_spot_b"] = ci_b
            extras["preference_index"] = float(ci - ci_b)  # >0 = spot A preferred

        if self.mode == "thermal" or self.gradient_drift_speed > 0:
            extras["attractant_position"] = attr_pos.tolist()

        # Mean distance to attractant (lower = better navigation)
        extras["mean_dist_to_attractant"] = float(dists_attr.mean())

        return WorldState(t=self._t, agents=list(self._agents.values()), extras=extras)

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_chem(self, agent: AgentBody) -> ChemField:
        attr_pos = self.attractant_pos + self._drift_offset
        sigma = self.gradient_sigma

        dist_attr = float(np.linalg.norm(agent.position - attr_pos))
        attr_conc = self.attractant_peak * math.exp(-dist_attr**2 / (2 * sigma**2))

        # Gradient direction (normalised vector pointing toward source)
        direction_attr = (attr_pos - agent.position) / (dist_attr + 1e-6)

        # Second attractant spot (two_spot mode)
        second_conc = 0.0
        if self.mode == "two_spot":
            dist_b   = float(np.linalg.norm(agent.position - self.second_attr_pos))
            second_conc = self.attractant_peak * math.exp(-dist_b**2 / (2 * sigma**2))
            # Mix: higher of the two drives AWC
            if second_conc > attr_conc:
                direction_attr = (self.second_attr_pos - agent.position) / (dist_b + 1e-6)

        total_attr = max(attr_conc, second_conc)

        rep_conc   = 0.0
        grad_rep   = np.zeros(2, dtype=np.float32)
        if self.repellent_pos is not None:
            dist_rep = float(np.linalg.norm(agent.position - self.repellent_pos))
            rep_sigma = sigma * 0.8
            rep_conc = self.repellent_peak * math.exp(-dist_rep**2 / (2 * rep_sigma**2))
            grad_rep = ((self.repellent_pos - agent.position)
                        / (dist_rep + 1e-6)).astype(np.float32)

        # Stimulus overrides
        for stim in self._active_stimuli:
            if getattr(stim, "kind", None) == "odorant":
                if stim.params.get("chemical") == self.gradient:
                    total_attr = min(1.0, total_attr + stim.params.get("concentration", 0.5))

        conc = np.array([total_attr, rep_conc], dtype=np.float32)
        # gradient array: shape (2, 2) — row 0 = attractant grad, row 1 = repellent grad
        grad = np.array([direction_attr, grad_rep], dtype=np.float32)

        return ChemField(
            species=[self.gradient, "repellent"],
            concentrations=conc,
            gradient=grad,
        )

    def _compute_body(self, agent: AgentBody) -> BodyState:
        in_contact = (
            agent.position[0] < 5.0 or agent.position[0] > self.W - 5.0 or
            agent.position[1] < 5.0 or agent.position[1] > self.H - 5.0
        )
        body = BodyState(
            position=agent.position.copy(),
            velocity=agent.velocity.copy(),
            orientation=agent.orientation,
            angular_velocity=0.0,
            in_contact=in_contact,
        )
        # Attach temperature for AFD neurons
        if self.mode == "thermal":
            temp = self._compute_temperature(agent)
            body.temperature = temp
        return body

    def _compute_temperature(self, agent: AgentBody) -> float:
        if self.thermal_axis == "x":
            frac = agent.position[0] / self.W
        else:
            frac = agent.position[1] / self.H
        return float(self.temp_cold + (self.temp_warm - self.temp_cold) * frac)

    # ── Analytics helpers ─────────────────────────────────────────────────────

    def mean_ci(self) -> float:
        return float(np.mean(self._ci_history)) if self._ci_history else 0.0

    def final_ci(self) -> float:
        return float(self._ci_history[-1]) if self._ci_history else 0.0
