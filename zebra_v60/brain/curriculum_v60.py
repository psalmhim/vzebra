"""
Curriculum Scheduler for Progressive Difficulty Training (Step 22).

Manages 4 progressive difficulty phases that modify environment parameters.
The scheduler is a pure observer/configurator — it reads `info` from
env.step() and writes env attributes. The brain agent is unchanged;
its existing online learning adapts naturally.

Phase progression:
  Phase 0 (Safe)    — no predator, no energy drain, abundant food
  Phase 1 (Forage)  — energy drain starts, patrolling predator
  Phase 2 (Threat)  — faster predator with stalking, less food
  Phase 3 (Wild)    — full predator AI, scarce food, high drain

Advancement criteria: survive N steps AND eat M food in current phase.
"""
import numpy as np


# Default phase configurations — applied as env attribute overrides
DEFAULT_PHASES = [
    {   # Phase 0: Safe — learn to move and eat, predator harmless
        "energy_drain_base": 0.01,
        "energy_drain_speed": 0.005,
        "energy_food_gain": 20.0,
        "n_food_init": 25,
        "pred_speed": 0.0,
        "pred_chase_radius": 0.0,
        "pred_catch_radius": 0.0,
        "_pred_allowed_states": [],
    },
    {   # Phase 1: Forage — energy management + slow patrolling predator
        "energy_drain_base": 0.02,
        "energy_drain_speed": 0.01,
        "energy_food_gain": 18.0,
        "n_food_init": 20,
        "pred_speed": 0.8,
        "pred_chase_radius": 150.0,
        "pred_catch_radius": 16.0,
        "_pred_allowed_states": ["PATROL"],
    },
    {   # Phase 2: Threat — predator stalks slowly, moderate difficulty
        "energy_drain_base": 0.03,
        "energy_drain_speed": 0.015,
        "energy_food_gain": 16.0,
        "n_food_init": 18,
        "pred_speed": 1.0,
        "pred_chase_radius": 180.0,
        "pred_catch_radius": 16.0,
        "_pred_allowed_states": ["PATROL", "STALK"],
    },
    {   # Phase 3: Wild — full predator AI, scarce food
        "energy_drain_base": 0.05,
        "energy_drain_speed": 0.03,
        "energy_food_gain": 15.0,
        "n_food_init": 12,
        "pred_speed": 1.6,
        "pred_chase_radius": 280.0,
        "pred_catch_radius": 16.0,
        "_pred_allowed_states": ["PATROL", "STALK", "HUNT", "AMBUSH"],
    },
]


class CurriculumScheduler:
    """Manages progressive difficulty phases for environment training.

    Usage:
        scheduler = CurriculumScheduler()
        scheduler.apply_phase(env, 0)  # initialize phase 0

        for t in range(T):
            action = agent.act(obs, env)
            obs, reward, term, trunc, info = env.step(action)
            phase_changed = scheduler.step(env, info)
    """

    def __init__(self,
                 advance_survival=200,
                 advance_eaten=3,
                 min_phase_steps=150,
                 phases=None):
        """
        Args:
            advance_survival: survive N steps to advance
            advance_eaten: eat M food to advance
            min_phase_steps: minimum steps before advancement check
            phases: list of phase config dicts (or DEFAULT_PHASES)
        """
        self.advance_survival = advance_survival
        self.advance_eaten = advance_eaten
        self.min_phase_steps = min_phase_steps
        self.phases = phases if phases is not None else DEFAULT_PHASES

        # Internal state
        self.current_phase = 0
        self.phase_step_count = 0
        self.phase_eaten = 0
        self.total_steps = 0
        self.phase_history = []  # list of {phase, start_step, end_step}

    def apply_phase(self, env, phase_idx):
        """Set env attributes from phase config.

        Args:
            env: ZebrafishPreyPredatorEnv instance
            phase_idx: int — phase index to apply
        """
        phase_idx = min(phase_idx, len(self.phases) - 1)
        config = self.phases[phase_idx]
        for attr, value in config.items():
            setattr(env, attr, value)

    def step(self, env, info):
        """Track metrics and maybe advance phase.

        Args:
            env: ZebrafishPreyPredatorEnv instance
            info: dict from env.step()

        Returns:
            phase_changed: bool — True if phase advanced this step
        """
        self.phase_step_count += 1
        self.total_steps += 1

        # Track food eaten
        eaten = info.get("food_eaten_this_step", 0)
        self.phase_eaten += eaten

        # Check advancement
        phase_changed = False
        if self.check_advancement():
            old_phase = self.current_phase
            self.phase_history.append({
                "phase": old_phase,
                "start_step": self.total_steps - self.phase_step_count,
                "end_step": self.total_steps,
                "eaten": self.phase_eaten,
            })
            self.current_phase = min(
                self.current_phase + 1, len(self.phases) - 1)
            self.apply_phase(env, self.current_phase)
            self.phase_step_count = 0
            self.phase_eaten = 0
            phase_changed = (self.current_phase != old_phase)

        return phase_changed

    def check_advancement(self):
        """Check if advancement criteria are met.

        Returns:
            bool — True if should advance
        """
        if self.current_phase >= len(self.phases) - 1:
            return False
        if self.phase_step_count < self.min_phase_steps:
            return False
        return (self.phase_step_count >= self.advance_survival
                and self.phase_eaten >= self.advance_eaten)

    def get_diagnostics(self):
        """Return monitoring dict."""
        return {
            "current_phase": self.current_phase,
            "phase_step_count": self.phase_step_count,
            "phase_eaten": self.phase_eaten,
            "total_steps": self.total_steps,
            "n_transitions": len(self.phase_history),
            "phase_history": list(self.phase_history),
        }

    def reset(self):
        """Reset to phase 0."""
        self.current_phase = 0
        self.phase_step_count = 0
        self.phase_eaten = 0
        self.total_steps = 0
        self.phase_history = []
