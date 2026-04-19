"""
Recording system: captures spike trains, neuromodulator dynamics,
behavioral trajectories, and decision logs during simulation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    """Single timestep of recorded data."""
    step: int
    turn: float
    speed: float
    goal: int
    x: float
    y: float
    energy: float
    da: float = 0.0
    na: float = 0.0
    ht5: float = 0.0
    ach: float = 0.0
    amygdala: float = 0.0
    free_energy: float = 0.0
    cerebellum_pe: float = 0.0
    critic_value: float = 0.0
    td_error: float = 0.0


class Recorder:
    """Records simulation data for analysis."""

    def __init__(self, channels: set[str] | None = None):
        """
        Args:
            channels: set of data channels to record. None = all.
                Options: 'behavior', 'neuromod', 'goals', 'spikes', 'energy'
        """
        self.channels = channels or {'behavior', 'neuromod', 'goals', 'energy'}
        self.steps: list[StepRecord] = []
        self.events: list[dict] = []
        self._active = False

    def start(self) -> None:
        self.steps = []
        self.events = []
        self._active = True

    def stop(self) -> None:
        self._active = False

    def record_step(self, step_data: dict, pos: tuple[float, float] = (0, 0),
                    energy: float = 0.0) -> None:
        """Record one step of simulation output."""
        if not self._active:
            return
        rec = StepRecord(
            step=len(self.steps),
            turn=step_data.get('turn', 0.0),
            speed=step_data.get('speed', 0.0),
            goal=step_data.get('goal', 2),
            x=pos[0], y=pos[1],
            energy=energy,
        )
        if 'neuromod' in self.channels:
            rec.da = step_data.get('DA', 0.0)
            rec.na = step_data.get('NA', 0.0)
            rec.ht5 = step_data.get('5HT', 0.0)
            rec.ach = step_data.get('ACh', 0.0)
            rec.amygdala = step_data.get('amygdala', 0.0)
        if 'goals' in self.channels:
            rec.free_energy = step_data.get('free_energy', 0.0)
            rec.cerebellum_pe = step_data.get('cerebellum_pe', 0.0)
            rec.critic_value = step_data.get('critic_value', 0.0)
            rec.td_error = step_data.get('critic_td_error', 0.0)
        self.steps.append(rec)

    def record_event(self, event_type: str, step: int, **kwargs: Any) -> None:
        """Record a discrete event (food eaten, caught, goal change)."""
        if not self._active:
            return
        self.events.append({'type': event_type, 'step': step, **kwargs})

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_trajectory(self) -> list[tuple[float, float]]:
        return [(s.x, s.y) for s in self.steps]

    def get_goal_counts(self) -> dict[str, int]:
        names = {0: 'FORAGE', 1: 'FLEE', 2: 'EXPLORE', 3: 'SOCIAL'}
        counts = {n: 0 for n in names.values()}
        for s in self.steps:
            name = names.get(s.goal, 'UNKNOWN')
            counts[name] = counts.get(name, 0) + 1
        return counts

    def get_neuromod_timeseries(self) -> dict[str, list[float]]:
        return {
            'DA': [s.da for s in self.steps],
            'NA': [s.na for s in self.steps],
            '5HT': [s.ht5 for s in self.steps],
            'ACh': [s.ach for s in self.steps],
        }

    def get_energy_timeseries(self) -> list[float]:
        return [s.energy for s in self.steps]

    def get_speed_timeseries(self) -> list[float]:
        return [s.speed for s in self.steps]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            'steps': [
                {
                    'step': s.step, 'turn': s.turn, 'speed': s.speed,
                    'goal': s.goal, 'x': s.x, 'y': s.y, 'energy': s.energy,
                    'da': s.da, 'na': s.na, 'ht5': s.ht5, 'ach': s.ach,
                    'amygdala': s.amygdala, 'free_energy': s.free_energy,
                    'cerebellum_pe': s.cerebellum_pe,
                    'critic_value': s.critic_value, 'td_error': s.td_error,
                }
                for s in self.steps
            ],
            'events': self.events,
            'summary': {
                'n_steps': len(self.steps),
                'goal_counts': self.get_goal_counts(),
                'n_events': len(self.events),
            },
        }

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_csv(self, path: str) -> None:
        """Export step data as CSV."""
        import csv
        if not self.steps:
            return
        fields = ['step', 'turn', 'speed', 'goal', 'x', 'y', 'energy',
                  'da', 'na', 'ht5', 'ach', 'amygdala', 'free_energy',
                  'cerebellum_pe', 'critic_value', 'td_error']
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in self.steps:
                writer.writerow({k: getattr(s, k) for k in fields})
