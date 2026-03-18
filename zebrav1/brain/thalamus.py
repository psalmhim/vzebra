"""
Thalamic relay for v1: cross-modal salience (CMS) computation.

Computes CMS from visual and auditory (or simulated) free energy signals.
CMS modulates dopamine gain, BG exploration, and precision bias.
"""
import math


class ThalamusRelay:
    def __init__(self, alpha=0.6, smooth=0.9):
        self.alpha = alpha       # weight for temporal change vs cross-modal mismatch
        self.smooth = smooth     # exponential smoothing
        self.prev_Fv = 0.0
        self.prev_Fa = 0.0
        self.cms = 0.0

    def step(self, F_visual, F_audio):
        """
        Compute cross-modal salience from visual and audio free energies.

        Args:
            F_visual: visual free energy (scalar)
            F_audio: auditory free energy (scalar, can be simulated)

        Returns:
            cms: cross-modal salience [0, ~1]
        """
        # Temporal change in each modality
        dFv = abs(F_visual - self.prev_Fv)
        dFa = abs(F_audio - self.prev_Fa)

        # Cross-modal mismatch
        mismatch = abs(F_visual - F_audio)

        # CMS = alpha * mean_temporal_change + (1-alpha) * cross-modal mismatch
        raw_cms = self.alpha * (dFv + dFa) / 2.0 + (1.0 - self.alpha) * mismatch

        # Exponential smoothing
        self.cms = self.smooth * self.cms + (1.0 - self.smooth) * raw_cms

        # Store for next step
        self.prev_Fv = F_visual
        self.prev_Fa = F_audio

        return self.cms

    def modulate_dopamine_gain(self, base_gain=3.0):
        """CMS amplifies dopamine gain for novelty-driven exploration."""
        return base_gain * (1.0 + 0.5 * self.cms)

    def modulate_bg_exploration(self, base_noise=0.05):
        """CMS increases BG exploration noise."""
        return base_noise * (1.0 + 2.0 * self.cms)

    def modulate_precision_bias(self, base_precision):
        """CMS biases precision toward novelty detection."""
        return base_precision * (1.0 - 0.2 * min(self.cms, 1.0))

    def reset(self):
        self.prev_Fv = 0.0
        self.prev_Fa = 0.0
        self.cms = 0.0
