"""
Color Vision — cone-type spectral channels (Step 39).

Zebrafish larvae have 4 cone types: UV, blue (short), green (medium),
red (long).  Different entities have distinct spectral signatures,
enabling color-based discrimination beyond shape/size cues.

Spectral signatures per entity type:
  - Food (plankton): high UV + green (bioluminescent/reflective)
  - Enemy (predator): high red + low UV (dark body, warm color)
  - Colleague: moderate green + blue (zebrafish pigmentation)
  - Obstacle (rock): flat spectrum (grey/brown)
  - Boundary (wall): low all channels

Neuroscience: zebrafish retinal cones are UV (360nm), S/blue (415nm),
M/green (480nm), L/red (570nm).  Color opponent processing begins
in bipolar cells (Zimmermann et al. 2018).

Pure numpy — operates on existing type-channel encoding.
"""
import numpy as np


# Spectral signatures: [UV, blue, green, red] per entity type
SPECTRA = {
    "food":      np.array([0.8, 0.3, 0.7, 0.2], dtype=np.float32),
    "enemy":     np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
    "colleague": np.array([0.3, 0.6, 0.7, 0.4], dtype=np.float32),
    "obstacle":  np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32),
    "boundary":  np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32),
    "none":      np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}

# Type encoding → entity name (matching retina_sampling.py)
TYPE_MAP = {
    0.0: "none", 1.0: "food", 0.5: "enemy",
    0.25: "colleague", 0.75: "obstacle", 0.12: "boundary",
}


class ColorVisionProcessor:
    """Convert type-channel pixels to 4-channel spectral representation.

    Augments the existing monochromatic retina with color information
    that improves entity discrimination, especially in ambiguous scenes.

    Args:
        n_pixels: int — pixels per eye (default 400 type pixels)
    """

    def __init__(self, n_pixels=400):
        self.n_pixels = n_pixels

    def process(self, type_channel):
        """Convert type values to 4-channel color representation.

        Args:
            type_channel: np.array[n_pixels] — type values from retina

        Returns:
            color_channels: np.array[4] — mean [UV, blue, green, red]
            color_contrast: dict — opponent channel features
        """
        uv = np.zeros(len(type_channel), dtype=np.float32)
        blue = np.zeros_like(uv)
        green = np.zeros_like(uv)
        red = np.zeros_like(uv)

        for type_val, name in TYPE_MAP.items():
            mask = np.abs(type_channel - type_val) < 0.1
            if mask.any():
                spec = SPECTRA[name]
                uv[mask] = spec[0]
                blue[mask] = spec[1]
                green[mask] = spec[2]
                red[mask] = spec[3]

        # Mean activation per channel
        mean_color = np.array([
            float(uv.mean()), float(blue.mean()),
            float(green.mean()), float(red.mean())],
            dtype=np.float32)

        # Color opponent channels (biological: double-opponent cells)
        rg_opponent = float(red.mean() - green.mean())  # red-green
        by_opponent = float(blue.mean() - (red.mean() + green.mean()) / 2)  # blue-yellow
        uv_luminance = float(uv.mean() - (red.mean() + green.mean() + blue.mean()) / 3)

        return mean_color, {
            "rg_opponent": rg_opponent,     # high = enemy-like
            "by_opponent": by_opponent,     # high = food-like UV
            "uv_contrast": uv_luminance,   # high = food
            "mean_uv": mean_color[0],
            "mean_blue": mean_color[1],
            "mean_green": mean_color[2],
            "mean_red": mean_color[3],
        }

    def get_entity_likelihood(self, color_features):
        """Compute likelihood of each entity type from color features.

        Returns:
            likelihoods: dict — {entity_name: float [0, 1]}
        """
        # High red-green opponent → enemy
        # High UV → food
        # Balanced green-blue → colleague
        rg = color_features["rg_opponent"]
        uv = color_features["uv_contrast"]

        return {
            "food": max(0.0, min(1.0, 0.5 + 2.0 * uv)),
            "enemy": max(0.0, min(1.0, 0.5 + 2.0 * rg)),
            "colleague": max(0.0, min(1.0, 0.5 - abs(rg) - abs(uv))),
        }

    def reset(self):
        pass

    def get_diagnostics(self):
        return {}
