"""
Spatial neuron registry: assigns 3D atlas positions to every neuron in the v2 brain.

Each brain region's neurons get positions from the zebrafish calcium imaging atlas
(subject 12, 58K cells, micron coordinates). Connections between regions are
weighted by physical distance: closer neurons → stronger initial weights.

Usage:
    registry = SpatialRegistry(device='mps')
    registry.assign_positions(brain)  # assigns .positions to each layer
    mask = registry.distance_mask('tectum', 'thalamus', lambda_um=100.0)
"""
import os
import math
import numpy as np
import torch

ATLAS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'atlas')


class SpatialRegistry:
    def __init__(self, device='cpu'):
        self.device = device
        self._atlas_xyz = None
        self._atlas_labels = None
        self._region_positions = {}  # region_name → (N, 3) tensor in microns

        # Atlas label → brain region mapping
        # Labels from cell_label_indices_s12.npy (0-36)
        self.label_map = {
            'tectum':       [0, 30],
            'cerebellum':   [1, 36],
            'pallium':      [2],
            'thalamus':     [3, 5],
            'tegmentum':    [4, 15],
            'hypothalamus': [12, 13, 14],
            'medulla':      [23, 29, 31],
            'reticular':    [25, 26],
            'habenula':     [8],
            'subpallium':   [11],
            'preoptic':     [18],
            'olfactory':    [17, 32],
            'spinal':       [21],
            'raphe':        [33],
            'pretectum':    [27],
        }

    def _load_atlas(self):
        if self._atlas_xyz is not None:
            return
        self._atlas_xyz = np.load(
            os.path.join(ATLAS_DIR, 'cell_xyz_s12.npy')).astype(np.float32)
        self._atlas_labels = np.load(
            os.path.join(ATLAS_DIR, 'cell_label_indices_s12.npy'))

    def _get_region_stats(self, labels):
        """Get centroid and spread for a set of atlas labels."""
        self._load_atlas()
        mask = np.isin(self._atlas_labels, labels)
        pts = self._atlas_xyz[mask]
        if len(pts) == 0:
            return np.zeros(3), np.ones(3) * 20.0
        return pts.mean(axis=0), np.clip(pts.std(axis=0), 5.0, None)

    def generate_positions(self, region_name, n_neurons, offset_um=None):
        """Generate n positions as Gaussian cloud around atlas centroid.

        Returns: (n_neurons, 3) tensor in microns.
        """
        labels = self.label_map.get(region_name)
        if labels is None:
            # Fallback: place near origin with small spread
            pos = np.random.randn(n_neurons, 3).astype(np.float32) * 20.0
            return torch.tensor(pos, device=self.device)

        centroid, spread = self._get_region_stats(labels)
        # Scale spread for smaller regions
        scale = min(1.0, max(0.3, n_neurons / 300))
        pos = np.random.default_rng(hash(region_name) % 2**32).normal(
            loc=centroid,
            scale=spread * scale,
            size=(n_neurons, 3)
        ).astype(np.float32)

        if offset_um is not None:
            pos += np.array(offset_um, dtype=np.float32)

        self._region_positions[region_name] = torch.tensor(pos, device=self.device)
        return self._region_positions[region_name]

    def distance_matrix(self, region_a, region_b):
        """Compute pairwise Euclidean distance matrix (um) between two regions.

        Returns: (n_a, n_b) tensor.
        """
        pos_a = self._region_positions.get(region_a)
        pos_b = self._region_positions.get(region_b)
        if pos_a is None or pos_b is None:
            return None
        # Efficient pairwise distance: ||a-b||² = ||a||² + ||b||² - 2a·b
        a2 = (pos_a ** 2).sum(dim=1, keepdim=True)
        b2 = (pos_b ** 2).sum(dim=1, keepdim=True)
        dist_sq = a2 + b2.T - 2 * pos_a @ pos_b.T
        return torch.sqrt(torch.clamp(dist_sq, min=0))

    def distance_weight_mask(self, region_a, region_b, lambda_um=100.0):
        """Create distance-dependent weight mask: w = exp(-dist / lambda).

        Connections between nearby neurons are strong; distant ones are weak.
        lambda_um: length constant in microns (typical: 50-200um for zebrafish).
        Returns: (n_a, n_b) tensor of weights in [0, 1].
        """
        dist = self.distance_matrix(region_a, region_b)
        if dist is None:
            return None
        return torch.exp(-dist / lambda_um)

    def topographic_weight(self, src_region, tgt_region, src_axis=1, tgt_axis=1):
        """Create topographic mapping weight: preserve spatial order.

        Sorts source and target neurons along an axis, then creates a
        Gaussian connectivity pattern preserving retinotopy.
        Returns: (n_src, n_tgt) weight matrix.
        """
        pos_s = self._region_positions.get(src_region)
        pos_t = self._region_positions.get(tgt_region)
        if pos_s is None or pos_t is None:
            return None

        # Normalize positions along the specified axis to [0, 1]
        s_vals = pos_s[:, src_axis]
        t_vals = pos_t[:, tgt_axis]
        s_norm = (s_vals - s_vals.min()) / (s_vals.max() - s_vals.min() + 1e-8)
        t_norm = (t_vals - t_vals.min()) / (t_vals.max() - t_vals.min() + 1e-8)

        # Topographic Gaussian: neurons at same relative position connect strongly
        diff = s_norm.unsqueeze(1) - t_norm.unsqueeze(0)  # (n_src, n_tgt)
        sigma = 0.15  # topographic sharpness
        return torch.exp(-diff ** 2 / (2 * sigma ** 2))

    def assign_to_brain(self, brain):
        """Assign spatial positions to all regions of a ZebrafishBrainV2.

        Stores positions as .spatial_pos attribute on each module.
        """
        mappings = {
            # module_name: (region_name, n_neurons, optional_offset)
            # Retinae (most anterior, laterally separated)
            'retina_L':      ('olfactory', 1000, [0, -30, 0]),  # olfactory region = anterior
            'retina_R':      ('olfactory', 1000, [0, 30, 0]),
            'tectum.sfgs_b': ('tectum', (brain.tectum.sfgs_b_L.n_e + brain.tectum.sfgs_b_L.n_i) * 2, [0, 0, 20]),
            'tectum.sfgs_d': ('tectum', (brain.tectum.sfgs_d_L.n_e + brain.tectum.sfgs_d_L.n_i) * 2, [0, 0, 0]),
            'tectum.sgc':    ('tectum', (brain.tectum.sgc_L.n_e + brain.tectum.sgc_L.n_i) * 2, [0, 0, -15]),
            'tectum.so':     ('tectum', (brain.tectum.so_L.n_e + brain.tectum.so_L.n_i) * 2, [0, 0, -30]),
            'thalamus.TC':   ('thalamus', brain.thalamus_L.TC.n + brain.thalamus_R.TC.n, None),
            'thalamus.TRN':  ('thalamus', brain.thalamus_L.TRN.n + brain.thalamus_R.TRN.n, [0, 0, 10]),
            'pallium.pal_s': ('pallium', brain.pallium.pal_s.n_e + brain.pallium.pal_s.n_i, None),
            'pallium.pal_d': ('pallium', brain.pallium.pal_d.n_e + brain.pallium.pal_d.n_i, [0, 0, -15]),
            'amygdala':      ('subpallium', 50, [0, 0, -20]),
            'habenula':      ('habenula', 50, None),
            'cerebellum':    ('cerebellum', 270, None),
            'bg':            ('subpallium', 760, None),
            'place':         ('pallium', 128, [15, 0, 0]),
            'critic':        ('tegmentum', 68, None),
            'insula':        ('hypothalamus', 34, None),
        }

        brain._spatial_registry = self
        for key, (region, n, offset) in mappings.items():
            pos = self.generate_positions(
                region,  # use atlas region name for centroid lookup
                n, offset_um=offset)
            # Also store with full key for distance lookups
            self._region_positions[key] = pos
            # Store on the actual module
            parts = key.split('.')
            obj = brain
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is not None:
                obj.spatial_pos = pos

    def apply_distance_weights(self, weight_matrix, region_a, region_b,
                                lambda_um=100.0, strength=0.5):
        """Modulate an existing weight matrix by distance-dependent mask.

        weight_matrix: (n_a, n_b) nn.Parameter or tensor
        strength: 0 = no effect, 1 = fully distance-dependent
        """
        mask = self.distance_weight_mask(region_a, region_b, lambda_um)
        if mask is None:
            return weight_matrix
        with torch.no_grad():
            blended = weight_matrix.data * (1 - strength + strength * mask)
            weight_matrix.data.copy_(blended)
        return weight_matrix
