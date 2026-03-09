"""
Spike sonification — all neuron groups mixed into one composite chord.

Each group contributes a frequency component whose amplitude is proportional
to its spike count. The result is a single combined tone that changes
character as different brain regions activate:

  - OT  (optic tectum):  1200 Hz  — high shimmer when visual activity is strong
  - Eye (eye control):    900 Hz  — bright overtone during saccades
  - Motor:                600 Hz  — mid punch on movement commands
  - PT  (pretectum):      400 Hz  — warm mid when processing objects
  - DA  (dopamine):       300 Hz  — resonant hum on reward signals
  - PC_int (intent):      180 Hz  — deep bass on goal commitment

Quiet brain → silence.  Active brain → rich layered chord.
"""

import numpy as np

try:
    import pygame
    import pygame.sndarray
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

# Each group: frequency, SNN output key, neuron count, min spikes to activate
GROUPS = [
    {"name": "OT",     "freq": 1200, "key": "oF",     "n": 800, "min": 40,  "norm": 400.0},
    {"name": "eye",    "freq":  900, "key": "eye",    "n": 100, "min":  3,  "norm":  50.0},
    {"name": "motor",  "freq":  600, "key": "motor",  "n": 200, "min":  5,  "norm": 100.0},
    {"name": "PT",     "freq":  400, "key": "pt",     "n": 400, "min": 15,  "norm": 200.0},
    {"name": "DA",     "freq":  300, "key": "DA",     "n":  50, "min":  2,  "norm":  25.0},
    {"name": "PC_int", "freq":  180, "key": "intent", "n":  30, "min":  1,  "norm":  15.0},
]

SPIKE_THRESHOLD = 0.15
TONE_DURATION_MS = 35          # length of each combined tone burst
COOLDOWN_FRAMES = 2            # min gap between tone bursts


class SpikeAudioEngine:
    """Real-time spike sonification — combined chord per frame.

    Usage:
        audio = SpikeAudioEngine()
        audio.update(snn_out)   # call each frame
    """

    def __init__(self, master_volume=0.5, sample_rate=22050):
        if not HAS_PYGAME:
            raise ImportError("pygame required for SpikeAudioEngine")

        self._enabled = True
        self._master_vol = np.clip(master_volume, 0.0, 1.0)
        self._sr = sample_rate

        # Initialize mixer if needed
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(sample_rate, -16, 2, 256)
            pygame.mixer.init()

        pygame.mixer.set_num_channels(4)

        # Detect stereo
        _, _, self._n_ch = pygame.mixer.get_init()

        # Pre-compute time vector for tone synthesis
        n_samples = int(self._sr * TONE_DURATION_MS / 1000.0)
        self._t = np.linspace(0, TONE_DURATION_MS / 1000.0,
                              n_samples, dtype=np.float32)
        # Exponential decay envelope (sharp attack, quick fade)
        self._env = np.exp(-12.0 * self._t)
        # Transient click at onset
        self._onset = np.zeros(n_samples, dtype=np.float32)
        if n_samples > 4:
            self._onset[:4] = np.array([0.3, 0.2, 0.1, 0.05])

        # Pre-compute sine tables per group frequency
        self._sines = {}
        for g in GROUPS:
            f = g["freq"]
            self._sines[f] = np.sin(2 * np.pi * f * self._t)
            # Add slight 2nd harmonic for richness
            self._sines[f] += 0.15 * np.sin(2 * np.pi * f * 2.0 * self._t)

        self._cooldown = 0

    def _tensor_to_np(self, t):
        if hasattr(t, 'detach'):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def update(self, snn_out):
        """Measure spike activity per group, mix into one chord, play."""
        if not self._enabled:
            return

        if self._cooldown > 0:
            self._cooldown -= 1
            return

        # Compute amplitude weight per group from spike counts
        weights = {}
        any_active = False
        for g in GROUPS:
            arr = self._tensor_to_np(
                snn_out.get(g["key"], np.zeros((1, g["n"])))).flatten()
            n_spikes = int(np.sum(np.abs(arr) > SPIKE_THRESHOLD))
            if n_spikes >= g["min"]:
                w = min(1.0, n_spikes / g["norm"])
                weights[g["freq"]] = w
                any_active = True
            else:
                weights[g["freq"]] = 0.0

        if not any_active:
            return

        # Mix: sum weighted sinusoids
        mixed = np.zeros_like(self._t)
        total_weight = 0.0
        for g in GROUPS:
            w = weights[g["freq"]]
            if w > 0:
                mixed += w * self._sines[g["freq"]]
                total_weight += w

        # Apply envelope and onset transient
        mixed = mixed / (total_weight + 1e-8)
        mixed = mixed * self._env + self._onset * min(1.0, total_weight)

        # Normalize and convert to int16
        peak = np.abs(mixed).max() + 1e-8
        mixed = mixed / peak
        amplitude = min(1.0, total_weight / 3.0) * self._master_vol
        wave_int16 = (mixed * amplitude * 28000).astype(np.int16)

        # Make stereo if needed
        if self._n_ch == 2:
            stereo = np.column_stack([wave_int16, wave_int16])
        else:
            stereo = wave_int16

        snd = pygame.sndarray.make_sound(np.ascontiguousarray(stereo))
        ch = pygame.mixer.find_channel()
        if ch is not None:
            ch.play(snd)

        self._cooldown = COOLDOWN_FRAMES

    def set_volume(self, vol):
        """Set master volume (0.0 to 1.0)."""
        self._master_vol = np.clip(vol, 0.0, 1.0)

    def set_enabled(self, enabled):
        """Toggle audio on/off."""
        self._enabled = enabled

    def close(self):
        """Clean up mixer resources."""
        pass
