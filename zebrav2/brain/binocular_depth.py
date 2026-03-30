"""
Binocular depth estimation: stereo overlap for distance.

Zebrafish larvae have ~160° FoV per eye with ~40° frontal binocular
overlap. Objects in the overlap zone appear on both L and R retinas,
enabling distance estimation via disparity.

Mechanism:
  - Frontal ~40° zone: objects appear on both L and R retinal arrays
  - Disparity = position difference between L and R retinal images
  - Smaller disparity → farther object
  - Larger disparity → closer object

Output: distance estimates for food and predator in binocular zone,
confidence based on overlap quality.
"""
import math
import numpy as np


class BinocularDepth:
    def __init__(self, fov_deg=200, overlap_deg=40, retinal_width=400):
        self.fov_rad = math.radians(fov_deg)
        self.overlap_rad = math.radians(overlap_deg)
        self.retinal_width = retinal_width
        # Binocular zone in retinal coordinates
        # L eye: rightmost pixels (overlap zone is nasal/medial)
        # R eye: leftmost pixels
        self.overlap_pixels = int(retinal_width * overlap_deg / fov_deg)
        # Interocular distance (zebrafish ~0.3mm, scaled to arena)
        self.baseline = 5.0  # pixels
        # Focal length proxy
        self.focal = 200.0

        self.food_distance = 999.0
        self.enemy_distance = 999.0
        self.food_confidence = 0.0
        self.enemy_confidence = 0.0

    def estimate(self, L_type, R_type, L_intensity, R_intensity):
        """
        Estimate distances from binocular disparity.
        L_type, R_type: (400,) type channel from retinal arrays
        L_intensity, R_intensity: (400,) intensity channel
        Returns dict with distance estimates.
        """
        # Binocular overlap zone
        # L eye nasal (rightmost pixels) = L_type[400-overlap : 400]
        # R eye nasal (leftmost pixels) = R_type[0 : overlap]
        op = self.overlap_pixels
        L_overlap_type = L_type[-op:]  # nasal L
        R_overlap_type = R_type[:op]   # nasal R
        L_overlap_int = L_intensity[-op:]
        R_overlap_int = R_intensity[:op]

        # Food detection in binocular zone (type > 0.7)
        food_L = (L_overlap_type > 0.7).astype(np.float32)
        food_R = (R_overlap_type > 0.7).astype(np.float32)

        if food_L.sum() > 0 and food_R.sum() > 0:
            # Centroid of food in each eye's overlap
            cols = np.arange(op, dtype=np.float32)
            centroid_L = float(np.dot(cols, food_L) / (food_L.sum() + 1e-8))
            centroid_R = float(np.dot(cols, food_R) / (food_R.sum() + 1e-8))
            # Disparity (pixels)
            disparity = abs(centroid_L - centroid_R) + 1e-8
            # Distance = baseline * focal / disparity
            self.food_distance = min(500.0, self.baseline * self.focal / disparity)
            self.food_confidence = min(1.0, (food_L.sum() + food_R.sum()) / 10.0)
        else:
            self.food_distance = 999.0
            self.food_confidence = 0.0

        # Enemy detection in binocular zone (type ≈ 0.5)
        enemy_L = (np.abs(L_overlap_type - 0.5) < 0.1).astype(np.float32)
        enemy_R = (np.abs(R_overlap_type - 0.5) < 0.1).astype(np.float32)

        if enemy_L.sum() > 0 and enemy_R.sum() > 0:
            cols = np.arange(op, dtype=np.float32)
            centroid_L = float(np.dot(cols, enemy_L) / (enemy_L.sum() + 1e-8))
            centroid_R = float(np.dot(cols, enemy_R) / (enemy_R.sum() + 1e-8))
            disparity = abs(centroid_L - centroid_R) + 1e-8
            self.enemy_distance = min(500.0, self.baseline * self.focal / disparity)
            self.enemy_confidence = min(1.0, (enemy_L.sum() + enemy_R.sum()) / 10.0)
        else:
            self.enemy_distance = 999.0
            self.enemy_confidence = 0.0

        # Overall depth from intensity correlation in overlap
        corr = float(np.corrcoef(L_overlap_int.flatten(),
                                  R_overlap_int.flatten())[0, 1]
                     if op > 2 else 0.0)
        if np.isnan(corr):
            corr = 0.0

        return {
            'food_distance': self.food_distance,
            'food_confidence': self.food_confidence,
            'enemy_distance': self.enemy_distance,
            'enemy_confidence': self.enemy_confidence,
            'stereo_correlation': corr,
            'overlap_pixels': op,
        }

    def get_approach_gain(self):
        """Returns speed multiplier based on food distance.
        Closer food → slow down for precise approach (prey capture).
        """
        if self.food_distance < 50 and self.food_confidence > 0.3:
            return 0.6  # slow approach for capture
        elif self.food_distance < 100 and self.food_confidence > 0.2:
            return 0.8  # moderate approach
        return 1.0  # normal speed

    def reset(self):
        self.food_distance = 999.0
        self.enemy_distance = 999.0
        self.food_confidence = 0.0
        self.enemy_confidence = 0.0
