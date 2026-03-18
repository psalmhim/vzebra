"""
Circadian Clock — endogenous 24h oscillator (Step 40).

A free-running oscillator (period ~24h mapped to simulation steps)
that modulates all brain systems with time-of-day gating:
  - Activity level (high during "day", low during "night")
  - Foraging drive (peaks at dawn/dusk — crepuscular feeding)
  - Predator vigilance (elevated at dusk when predators are active)
  - Sleep pressure (builds during day, triggers sleep at night)
  - Metabolism (higher during active phase)

The clock can be entrained by light cues (if day/night cycle is
simulated) or runs freely at a fixed period.

Neuroscience: zebrafish have a robust circadian system driven by
clock genes (per, cry, bmal) in the pineal gland and retina
(Cahill 2002; Vatine et al. 2011).  Light directly entrains
peripheral clocks in zebrafish — unique among vertebrates.

Pure numpy.
"""
import math
import numpy as np


class CircadianClock:
    """Endogenous circadian oscillator with time-of-day modulation.

    Args:
        period: int — cycle length in simulation steps
            (default 2000 ≈ one "day" at ~4 steps/second)
        dawn_phase: float — phase of dawn [0, 1] (0.25 = 6am)
        dusk_phase: float — phase of dusk [0, 1] (0.75 = 6pm)
    """

    def __init__(self, period=2000, dawn_phase=0.25, dusk_phase=0.75):
        self.period = period
        self.dawn_phase = dawn_phase
        self.dusk_phase = dusk_phase
        self._step = 0
        self._phase = 0.0  # [0, 1)

    def step(self):
        """Advance clock by one step.

        Returns:
            modulation: dict with time-of-day factors
        """
        self._step += 1
        self._phase = (self._step % self.period) / self.period

        # Activity level: sinusoidal, peak at noon (phase 0.5)
        activity = 0.5 + 0.5 * math.cos(
            2 * math.pi * (self._phase - 0.5))

        # Is it "daytime"?
        is_day = self.dawn_phase <= self._phase < self.dusk_phase

        # Foraging drive: peaks at dawn and dusk (crepuscular)
        dawn_dist = min(abs(self._phase - self.dawn_phase),
                        1 - abs(self._phase - self.dawn_phase))
        dusk_dist = min(abs(self._phase - self.dusk_phase),
                        1 - abs(self._phase - self.dusk_phase))
        forage_drive = math.exp(-dawn_dist * 20) + math.exp(-dusk_dist * 20)
        forage_drive = min(1.0, forage_drive)

        # Vigilance: elevated at dusk (predators hunt at twilight)
        vigilance = math.exp(-dusk_dist * 15)

        # Sleep pressure: builds during day, peaks at night
        sleep_pressure = 1.0 - activity

        # Metabolic rate: higher during active phase
        metabolic_rate = 0.8 + 0.4 * activity

        return {
            "phase": self._phase,
            "activity": activity,
            "is_day": is_day,
            "forage_drive": forage_drive,
            "vigilance": vigilance,
            "sleep_pressure": sleep_pressure,
            "metabolic_rate": metabolic_rate,
        }

    def get_efe_bias(self):
        """Time-of-day EFE bias for goal selection.

        Returns:
            bias: np.array[4] — [FORAGE, FLEE, EXPLORE, SOCIAL]
        """
        mod = self.step()
        self._step -= 1  # undo the advance (peek only)

        forage = -0.1 * mod["forage_drive"]
        flee = -0.05 * mod["vigilance"]
        explore = -0.1 * mod["activity"]  # explore during day
        social = -0.05 * (1.0 - mod["activity"])  # social at rest

        return np.array([forage, flee, explore, social], dtype=np.float32)

    def reset(self):
        self._step = 0
        self._phase = 0.0

    def get_diagnostics(self):
        return {
            "phase": self._phase,
            "step": self._step,
            "period": self.period,
        }
