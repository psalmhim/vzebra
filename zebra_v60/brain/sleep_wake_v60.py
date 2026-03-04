"""
Sleep/Wake Cycle Regulator with Memory Consolidation (Step 23).

Homeostatic sleep pressure accumulates during wakefulness and triggers
consolidation bouts when threshold is reached. During sleep:
  - Near-zero locomotion (zebrafish quiescence)
  - Enhanced place cell replay (SWR-analog, 4x rate)
  - Extra VAE ELBO updates from buffer (no new sensory push)
  - Easier habit consolidation (lower threshold, reduced decay)
  - Suppressed dopaminergic arousal
  - Frozen precision updates (gamma freeze)
  - Accelerated fatigue recovery

Sleep pressure dynamics:
  Wake: pressure += rate + fatigue_gain * fatigue
        Post-stress rebound: pressure += stress_rebound
  Sleep: pressure -= 1/consolidation_steps (drains to zero)
  Emergency: pred_proximity > 0.6 → immediate wake, pressure = 0.3

Biological basis:
  - Zebrafish larvae show rest/wake bouts modulated by adenosine
  - Hippocampal sharp-wave ripples increase 4-5x during NREM sleep
  - Dopaminergic arousal pathways are suppressed during sleep
  - Sleep homeostasis follows process-S dynamics (Borbely model)
"""
import numpy as np


class SleepWakeRegulator:
    """Homeostatic sleep pressure regulator with consolidation modulation."""

    def __init__(self,
                 # Sleep pressure
                 pressure_rate=0.002,
                 pressure_fatigue_gain=0.003,
                 pressure_stress_rebound=0.01,
                 pressure_threshold=0.7,
                 # Sleep duration
                 consolidation_steps=40,
                 min_wake_steps=100,
                 # Modulation during sleep
                 sleep_speed_mult=0.05,
                 replay_mult=4,
                 vae_train_mult=3,
                 habit_threshold_mult=0.5,
                 habit_decay_sleep=0.995,
                 dopa_beta_mult=0.3,
                 precision_freeze=True,
                 fatigue_recovery_mult=3.0):
        """
        Args:
            pressure_rate: per-step pressure gain during wake
            pressure_fatigue_gain: extra pressure from fatigue
            pressure_stress_rebound: post-stress sleep need bump
            pressure_threshold: pressure level that triggers sleep
            consolidation_steps: sleep bout length in steps
            min_wake_steps: minimum steps between sleep bouts
            sleep_speed_mult: speed multiplier during sleep (near-zero)
            replay_mult: place cell replay multiplier during sleep
            vae_train_mult: VAE training iterations during sleep
            habit_threshold_mult: habit consolidation threshold during sleep
            habit_decay_sleep: habit decay rate during sleep
            dopa_beta_mult: dopamine sensitivity during sleep
            precision_freeze: whether to freeze gamma during sleep
            fatigue_recovery_mult: fatigue recovery rate multiplier
        """
        # Parameters
        self.pressure_rate = pressure_rate
        self.pressure_fatigue_gain = pressure_fatigue_gain
        self.pressure_stress_rebound = pressure_stress_rebound
        self.pressure_threshold = pressure_threshold
        self.consolidation_steps = consolidation_steps
        self.min_wake_steps = min_wake_steps
        self.sleep_speed_mult = sleep_speed_mult
        self.replay_mult = replay_mult
        self.vae_train_mult = vae_train_mult
        self.habit_threshold_mult = habit_threshold_mult
        self.habit_decay_sleep = habit_decay_sleep
        self.dopa_beta_mult = dopa_beta_mult
        self.precision_freeze = precision_freeze
        self.fatigue_recovery_mult = fatigue_recovery_mult

        # Internal state
        self.is_sleeping = False
        self.sleep_pressure = 0.0
        self.wake_step_count = 0
        self.consolidation_remaining = 0
        self.total_sleep_bouts = 0
        self._prev_stress = 0.0
        self._emergency_wake_count = 0
        self._total_sleep_steps = 0

    def step(self, fatigue, stress, pred_proximity):
        """Update sleep/wake state and return modulation parameters.

        Args:
            fatigue: float [0, 1] — current fatigue level
            stress: float [0, 1] — current stress level
            pred_proximity: float [0, 1] — 0=far, 1=very close

        Returns:
            dict with modulation params:
                is_sleeping, speed_multiplier, replay_multiplier,
                vae_train_multiplier, dopa_beta_multiplier,
                precision_freeze, fatigue_recovery_multiplier,
                habit_modulation: {threshold_mult, decay}
        """
        # Emergency wake: predator too close
        if self.is_sleeping and pred_proximity > 0.6:
            self.is_sleeping = False
            self.consolidation_remaining = 0
            self.sleep_pressure = 0.3  # partial pressure retained
            self.wake_step_count = 0
            self._emergency_wake_count += 1

        if self.is_sleeping:
            # Sleep: drain pressure
            self.sleep_pressure -= 1.0 / max(1, self.consolidation_steps)
            self.sleep_pressure = max(0.0, self.sleep_pressure)
            self.consolidation_remaining -= 1
            self._total_sleep_steps += 1

            # End of sleep bout
            if self.consolidation_remaining <= 0:
                self.is_sleeping = False
                self.wake_step_count = 0

        else:
            # Wake: accumulate pressure
            self.wake_step_count += 1
            self.sleep_pressure += self.pressure_rate
            self.sleep_pressure += self.pressure_fatigue_gain * fatigue

            # Post-stress rebound: when stress drops after being high
            if (stress < self._prev_stress
                    and self._prev_stress > 0.3):
                rebound = self.pressure_stress_rebound * (
                    self._prev_stress - stress)
                self.sleep_pressure += rebound

            self.sleep_pressure = min(1.0, self.sleep_pressure)

            # Check sleep onset
            if (self.sleep_pressure >= self.pressure_threshold
                    and self.wake_step_count >= self.min_wake_steps
                    and pred_proximity < 0.4):
                self.is_sleeping = True
                self.consolidation_remaining = self.consolidation_steps
                self.total_sleep_bouts += 1

        self._prev_stress = stress

        # Build modulation dict
        if self.is_sleeping:
            return {
                "is_sleeping": True,
                "speed_multiplier": self.sleep_speed_mult,
                "replay_multiplier": self.replay_mult,
                "vae_train_multiplier": self.vae_train_mult,
                "dopa_beta_multiplier": self.dopa_beta_mult,
                "precision_freeze": self.precision_freeze,
                "fatigue_recovery_multiplier": self.fatigue_recovery_mult,
                "habit_modulation": {
                    "threshold_mult": self.habit_threshold_mult,
                    "decay": self.habit_decay_sleep,
                },
            }
        else:
            return {
                "is_sleeping": False,
                "speed_multiplier": 1.0,
                "replay_multiplier": 1,
                "vae_train_multiplier": 1,
                "dopa_beta_multiplier": 1.0,
                "precision_freeze": False,
                "fatigue_recovery_multiplier": 1.0,
                "habit_modulation": {
                    "threshold_mult": 1.0,
                    "decay": None,  # use default
                },
            }

    def get_diagnostics(self):
        """Return monitoring dict."""
        return {
            "is_sleeping": self.is_sleeping,
            "sleep_pressure": self.sleep_pressure,
            "wake_step_count": self.wake_step_count,
            "consolidation_remaining": self.consolidation_remaining,
            "total_sleep_bouts": self.total_sleep_bouts,
            "total_sleep_steps": self._total_sleep_steps,
            "emergency_wakes": self._emergency_wake_count,
        }

    def reset(self):
        """Reset to awake state with zero pressure."""
        self.is_sleeping = False
        self.sleep_pressure = 0.0
        self.wake_step_count = 0
        self.consolidation_remaining = 0
        self.total_sleep_bouts = 0
        self._prev_stress = 0.0
        self._emergency_wake_count = 0
        self._total_sleep_steps = 0
