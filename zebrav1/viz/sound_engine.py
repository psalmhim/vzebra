"""
Sound Engine — biological sound effects for the zebrafish simulation.

Synthesizes audio in real-time using pygame.mixer:
  - Heartbeat: rhythmic thump synced to heart rate
  - Splash: burst noise on C-start escape
  - Predator rumble: low-frequency drone when predator approaches
  - Eating click: short click when food captured
  - Ambient water: gentle noise background

Requires: pygame with mixer support.
"""
import math
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class SoundEngine:
    """Real-time biological sound effects.

    Args:
        sample_rate: int — audio sample rate
        enabled: bool — master enable
    """

    def __init__(self, sample_rate=22050, enabled=True):
        self.enabled = enabled
        self.sr = sample_rate

        if not enabled or not HAS_PYGAME:
            self.enabled = False
            return

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=sample_rate, size=-16,
                              channels=1, buffer=512)

        # Pre-generate sound buffers
        self._heartbeat = self._make_heartbeat()
        self._splash = self._make_splash()
        self._click = self._make_click()
        self._rumble = self._make_rumble()

        # Channels
        self._ch_heart = pygame.mixer.Channel(0)
        self._ch_fx = pygame.mixer.Channel(1)
        self._ch_ambient = pygame.mixer.Channel(2)

        # State
        self._last_beat_step = -999
        self._beat_interval = 30  # steps between heartbeats

    def _make_heartbeat(self):
        """Low thump sound (80Hz, 0.1s)."""
        dur = 0.1
        t = np.linspace(0, dur, int(self.sr * dur))
        wave = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 20)
        wave = (wave * 16000).astype(np.int16)
        return pygame.sndarray.make_sound(wave)

    def _make_splash(self):
        """Burst noise (C-start escape)."""
        dur = 0.15
        t = np.linspace(0, dur, int(self.sr * dur))
        noise = np.random.randn(len(t)) * np.exp(-t * 15)
        wave = (noise * 8000).astype(np.int16)
        return pygame.sndarray.make_sound(wave)

    def _make_click(self):
        """Short click (food eaten)."""
        dur = 0.03
        t = np.linspace(0, dur, int(self.sr * dur))
        wave = np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 50)
        wave = (wave * 10000).astype(np.int16)
        return pygame.sndarray.make_sound(wave)

    def _make_rumble(self):
        """Low rumble (predator approaching)."""
        dur = 0.3
        t = np.linspace(0, dur, int(self.sr * dur))
        wave = np.sin(2 * np.pi * 40 * t) * 0.3
        wave += np.random.randn(len(t)) * 0.1 * np.exp(-t * 5)
        wave = (wave * 6000).astype(np.int16)
        return pygame.sndarray.make_sound(wave)

    def update(self, step, heart_rate=0.3, is_fleeing=False,
               mauthner_fired=False, food_eaten=False,
               enemy_proximity=0.0):
        """Update sounds based on current state.

        Args:
            step: int — current simulation step
            heart_rate: float [0, 1]
            is_fleeing: bool
            mauthner_fired: bool — C-start this step
            food_eaten: bool
            enemy_proximity: float [0, 1] — 0=far, 1=close
        """
        if not self.enabled:
            return

        # Heartbeat: interval shortens with HR
        self._beat_interval = max(5, int(30 * (1.0 - heart_rate * 0.8)))
        if step - self._last_beat_step >= self._beat_interval:
            vol = 0.3 + 0.7 * heart_rate
            self._heartbeat.set_volume(vol)
            self._ch_heart.play(self._heartbeat)
            self._last_beat_step = step

        # C-start splash
        if mauthner_fired:
            self._splash.set_volume(0.8)
            self._ch_fx.play(self._splash)

        # Food eaten click
        if food_eaten:
            self._click.set_volume(0.6)
            self._ch_fx.play(self._click)

        # Predator rumble (when close)
        if enemy_proximity > 0.4 and not self._ch_ambient.get_busy():
            vol = min(0.5, enemy_proximity * 0.6)
            self._rumble.set_volume(vol)
            self._ch_ambient.play(self._rumble)

    def close(self):
        if self.enabled and HAS_PYGAME:
            pygame.mixer.quit()
