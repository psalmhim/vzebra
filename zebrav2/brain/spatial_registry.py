"""
Spatial neuron registry: assigns 3D atlas positions to every neuron in the v2 brain.

Each brain region's neurons get positions from the 72-region bilateral zebrafish
atlas (subject 12, 47010 cells, voxel coordinates).  CSV files:
    subject_12_CellXYZ.csv     — (x, y, z) voxel per cell
    subject_12_region_num.csv  — region number 0-71 per cell

Region numbering matches generate_brain_atlas.py:
    left hemisphere 0-35, right hemisphere 36-71 (right = left + 36)

Connections between regions are weighted by physical distance:
    closer neurons → stronger initial weights.

Usage:
    registry = SpatialRegistry(device='mps')
    registry.assign_positions(brain)  # assigns .positions to each layer
    mask = registry.distance_mask('tectum', 'thalamus', lambda_vx=100.0)
"""
import os
import ast
import csv
import math
import numpy as np
import torch

ATLAS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'atlas')


class SpatialRegistry:
    def __init__(self, device='cpu'):
        self.device = device
        self._atlas_xyz = None
        self._atlas_labels = None
        self._region_positions = {}  # region_name → (N, 3) tensor in voxels

        # 72-region bilateral atlas label map
        # Keys = abstract brain region names used by the v2 brain
        # Values = list of atlas region numbers (left/right)
        # 72-region bilateral atlas label map (synced with generate_brain_atlas.py)
        # Keys = abstract brain region names, Values = atlas region numbers
        # Left hemisphere 0-35, Right hemisphere 36-71 (right = left + 36)
        self.label_map = {
            # Core regions
            'tectum':           [29, 65],       # lTeO / rTeO
            'cerebellum':       [1, 37],        # lCb / rCb
            'pallium':          [22, 58],       # lP / rP
            'thalamus':         [30, 66],       # lTh / rTh
            'tegmentum':        [11, 47],       # lT / rT
            'hypothalamus':     [17, 53],       # lHi / rHi
            'medulla':          [0, 36],        # lMON / rMON
            'reticular':        [12, 48, 13, 49, 14, 50],  # aRF + imRF + pRF
            'habenula':         [16, 52],       # lHb / rHb
            'subpallium':       [28, 64],       # lSP / rSP
            'preoptic':         [25, 61],       # lPO / rPO
            'olfactory':        [20, 56],       # lOB / rOB
            'spinal':           [35, 71],       # lNX / rNX
            'raphe':            [10, 46],       # lRa / rRa
            'pretectum':        [26, 62],       # lPrT / rPrT
            'posterior_tuberculum': [24, 60],   # lPT / rPT (reticulospinal)
            'posterior_rf':     [14, 50],       # lpRF / rpRF (habit, NA)
            'hindbrain':        [0, 36, 12, 48, 13, 49, 14, 50],

            # Fine-grained sub-regions (for brain_v2 modules)
            'tectum_L':         [29],           # left TeO only
            'tectum_R':         [65],           # right TeO only
            'thalamus_L':       [30],           # left Th only
            'thalamus_R':       [66],           # right Th only
            'amygdala':         [28, 64],       # SP (amygdala homolog)
            'basal_ganglia':    [28, 64],       # SP (D1/D2/GPi)
            'place_cells':      [22, 58],       # Pallium (hippocampal homolog)
            'classifier':       [29, 65],       # Tectum-anchored visual classifier
            'goal_selector':    [25, 61],       # PO / preoptic
            'reticulospinal':   [24, 60],       # PT (posterior tuberculum)
            'lateral_line':     [0, 36],        # MON
            'vestibular':       [0, 36],        # MON (vestibular portion)
            'color_vision':     [29, 65],       # TeO (spectral layers)
            'saccade':          [26, 62],       # PrT (pretectum)
            'predictive':       [26, 62],       # PrT (predictive coding)
            'critic':           [11, 47],       # Tegmentum (VTA/SNc homolog)
            'habit':            [14, 50],       # pRF (habit network)
            'insula':           [17, 53],       # Hypothalamus (interoceptive cortex)
            'circadian':        [17, 53],       # Hypothalamus (SCN homolog)
            'sleep_wake':       [17, 53],       # Hypothalamus (VLPO homolog)
            'allostasis':       [17, 53],       # Hypothalamus (allostatic regulation)
            'ipn':              [11, 47],       # T (tegmentum — IPN is ventral)
            'locus_coeruleus':  [14, 50],       # lpRF / rpRF (LC in posterior hindbrain)
            'pectoral_fin':     [35, 71],       # NX (spinal motor — pectoral fin MNs)
            'cpg_L':            [35],           # NX left (spinal CPG)
            'cpg_R':            [71],           # NX right (spinal CPG)
        }

    def _load_atlas(self):
        """Load the 72-region bilateral CSV atlas (voxel coordinates)."""
        if self._atlas_xyz is not None:
            return
        self._atlas_xyz = np.loadtxt(
            os.path.join(ATLAS_DIR, 'subject_12_CellXYZ.csv'),
            delimiter=',', skiprows=1, dtype=np.float32)

        region_nums = []
        with open(os.path.join(ATLAS_DIR, 'subject_12_region_num.csv')) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                region_nums.append(ast.literal_eval(row[0])[0])
        self._atlas_labels = np.array(region_nums, dtype=np.int32)

    def _get_region_stats(self, labels):
        """Get centroid and spread for a set of atlas labels."""
        self._load_atlas()
        mask = np.isin(self._atlas_labels, labels)
        pts = self._atlas_xyz[mask]
        if len(pts) == 0:
            return np.zeros(3), np.ones(3) * 20.0
        return pts.mean(axis=0), np.clip(pts.std(axis=0), 5.0, None)

    def _atlas_jitter_sigma(self, atlas_pts: np.ndarray) -> float:
        """Estimate jitter σ from median nearest-neighbour spacing (voxels).

        The atlas only contains observed/segmented cells (~47K of ~100K real).
        Jitter fills the gaps: σ ≈ half the typical inter-cell spacing so
        interpolated positions stay within the real tissue volume.
        """
        if len(atlas_pts) < 2:
            return 5.0
        # Subsample for speed on large regions
        pts = atlas_pts if len(atlas_pts) <= 500 else atlas_pts[
            np.random.default_rng(0).choice(len(atlas_pts), 500, replace=False)]
        # Pairwise distances (upper triangle)
        diff = pts[:, None, :] - pts[None, :, :]          # (N, N, 3)
        d = np.sqrt((diff ** 2).sum(-1))                   # (N, N)
        np.fill_diagonal(d, np.inf)
        nn_dist = d.min(axis=1).mean()                     # mean nearest-neighbour
        return max(1.0, nn_dist * 0.5)                     # half-spacing as σ

    def generate_positions(self, region_name, n_neurons, offset_um=None):
        """Sample n neuron positions using the atlas as a spatial density prior.

        The atlas is incomplete (only observed/segmented cells). Positions are
        drawn by sampling atlas cells with replacement and adding Gaussian jitter
        scaled to half the local inter-cell spacing. This interpolates through
        unobserved gaps while preserving the real volumetric distribution.
        Atlas coordinates are in voxels ≈ 1 μm/voxel.

        Returns: (n_neurons, 3) tensor in atlas voxel units (~μm).
        """
        labels = self.label_map.get(region_name)
        if labels is None:
            pos = np.random.randn(n_neurons, 3).astype(np.float32) * 20.0
            return torch.tensor(pos, device=self.device)

        self._load_atlas()
        mask = np.isin(self._atlas_labels, labels)
        atlas_pts = self._atlas_xyz[mask]

        rng = np.random.default_rng(hash(region_name) % 2**32)
        if len(atlas_pts) == 0:
            centroid, spread = self._get_region_stats(labels)
            pos = rng.normal(loc=centroid, scale=spread,
                             size=(n_neurons, 3)).astype(np.float32)
        else:
            sigma = self._atlas_jitter_sigma(atlas_pts)
            idx = rng.choice(len(atlas_pts), n_neurons, replace=True)
            jitter = rng.normal(0, sigma, (n_neurons, 3)).astype(np.float32)
            pos = atlas_pts[idx] + jitter

        if offset_um is not None:
            pos += np.array(offset_um, dtype=np.float32)

        result = torch.tensor(pos, device=self.device)
        self._region_positions[region_name] = result
        return result

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
        After position assignment, re-initialises all EILayer intra-layer
        synapses with distance-dependent connectivity (λ from PNAS paper) and
        applies retinotopic weighting to retina→tectum projection weights.
        """
        from zebrav2.brain.connectome import get_region_lambda
        mappings = {
            # key: (atlas_region_name, n_neurons, offset_um)
            # -- Retinae (anterior, bilaterally separated) --
            'retina_L':              ('olfactory',    1000, [0, -40,  0]),
            'retina_R':              ('olfactory',    1000, [0,  40,  0]),
            # -- Tectum layers (bilateral, each hemisphere separately) --
            'tectum.sfgs_b_L':       ('tectum', brain.tectum.sfgs_b_L.n_e + brain.tectum.sfgs_b_L.n_i, [0, -50,  20]),
            'tectum.sfgs_b_R':       ('tectum', brain.tectum.sfgs_b_R.n_e + brain.tectum.sfgs_b_R.n_i, [0,  50,  20]),
            'tectum.sfgs_d_L':       ('tectum', brain.tectum.sfgs_d_L.n_e + brain.tectum.sfgs_d_L.n_i, [0, -50,   0]),
            'tectum.sfgs_d_R':       ('tectum', brain.tectum.sfgs_d_R.n_e + brain.tectum.sfgs_d_R.n_i, [0,  50,   0]),
            'tectum.sgc_L':          ('tectum', brain.tectum.sgc_L.n_e + brain.tectum.sgc_L.n_i,       [0, -50, -15]),
            'tectum.sgc_R':          ('tectum', brain.tectum.sgc_R.n_e + brain.tectum.sgc_R.n_i,       [0,  50, -15]),
            'tectum.so_L':           ('tectum', brain.tectum.so_L.n_e + brain.tectum.so_L.n_i,         [0, -50, -30]),
            'tectum.so_R':           ('tectum', brain.tectum.so_R.n_e + brain.tectum.so_R.n_i,         [0,  50, -30]),
            # -- Thalamus (bilateral) --
            'thalamus.tc_L':         ('thalamus', brain.thalamus_L.TC.n,  [0, -25,  0]),
            'thalamus.tc_R':         ('thalamus', brain.thalamus_R.TC.n,  [0,  25,  0]),
            'thalamus.trn_L':        ('thalamus', brain.thalamus_L.TRN.n, [0, -25, 10]),
            'thalamus.trn_R':        ('thalamus', brain.thalamus_R.TRN.n, [0,  25, 10]),
            # -- Pallium --
            'pallium.pal_s':         ('pallium', brain.pallium.pal_s.n_e + brain.pallium.pal_s.n_i, None),
            'pallium.pal_d':         ('pallium', brain.pallium.pal_d.n_e + brain.pallium.pal_d.n_i, [0, 0, -15]),
            # -- Basal ganglia --
            'bg':                    ('subpallium', 760, None),
            # -- Limbic --
            'amygdala':              ('subpallium',  50, [0, 0, -20]),
            'habenula':              ('habenula',    50, None),
            # -- Cerebellum --
            'cerebellum':            ('cerebellum', 270, None),
            # -- Hypothalamus / internal state --
            'insula':                ('hypothalamus', 34, None),
            # -- Motor --
            'critic':                ('tegmentum',   68, None),
            'place':                 ('pallium',    128, [15, 0, 0]),
            # -- Sensory --
            'lateral_line':          ('medulla',     24, None),
            'olfaction':             ('olfactory',   28, [0, 0, -10]),
            'vestibular':            ('medulla',      6, [0, 0, 10]),
            'proprioception':        ('spinal',       8, [5, 0, 0]),
            'color_vision':          ('tectum',      32, [0, 0, 30]),
            # -- Temporal / state --
            'circadian':             ('hypothalamus', 6, [-20, 0, 0]),
            'sleep_wake':            ('hypothalamus', 4, [-25, 0, 0]),
            'working_memory':        ('pallium',     40, [20, 0, 0]),
            'saccade':               ('pretectum',    6, [10, 0, 0]),
            'pretectum':             ('pretectum',   60, None),
            'ipn':                   ('tegmentum',   24, [0, 0, -10]),
            'raphe':                 ('raphe',       40, None),
            'locus_coeruleus':       ('posterior_rf', 20, [0, 0, 5]),
            'pectoral_fin':          ('spinal',       8, [10, 0, -5]),
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

        # ── Distance-dependent EILayer intra-layer connectivity ──────────────
        # Calls init_distance() on every EILayer to replace flat random
        # init_sparse with distance-weighted connectivity (λ from PNAS paper).
        ei_layers = [
            # Tectum (λ=80 μm, Mes)
            ('tectum.sfgs_b_L', 'tectum', brain.tectum.sfgs_b_L),
            ('tectum.sfgs_d_L', 'tectum', brain.tectum.sfgs_d_L),
            ('tectum.sgc_L',    'tectum', brain.tectum.sgc_L),
            ('tectum.so_L',     'tectum', brain.tectum.so_L),
            ('tectum.sfgs_b_R', 'tectum', brain.tectum.sfgs_b_R),
            ('tectum.sfgs_d_R', 'tectum', brain.tectum.sfgs_d_R),
            ('tectum.sgc_R',    'tectum', brain.tectum.sgc_R),
            ('tectum.so_R',     'tectum', brain.tectum.so_R),
            # Pallium (λ=120 μm, Tel sparse-modular)
            ('pallium.pal_s', 'pallium', brain.pallium.pal_s),
            ('pallium.pal_d', 'pallium', brain.pallium.pal_d),
        ]
        for pos_key, region_name, layer in ei_layers:
            pos_all = self._region_positions.get(pos_key)
            if pos_all is None or not hasattr(layer, 'init_distance'):
                continue
            lam = get_region_lambda(region_name)
            n_e = layer.n_e
            pos_e = pos_all[:n_e]
            pos_i = pos_all[n_e:n_e + layer.n_i]
            if len(pos_i) == 0:
                pos_i = pos_e  # fallback for tiny layers
            layer.init_distance(pos_e, pos_i, lam)

        # ── Retinotopic weighting: retina → tectum ON/OFF projections ────────
        # Topographic mapping preserves spatial order along nasal-temporal axis
        # (axis=1 = y, which separates left/right retinal fields).
        # W_on  → SFGS-b (broad); W_off → SFGS-d (deep) — different n_e, so
        # the topo mask is resized separately for each weight matrix.
        for ret_key, tec_key, W_on, W_off in [
            ('retina_R', 'tectum.sfgs_b_L', brain.tectum.W_on_L, brain.tectum.W_off_L),
            ('retina_L', 'tectum.sfgs_b_R', brain.tectum.W_on_R, brain.tectum.W_off_R),
        ]:
            topo = self.topographic_weight(ret_key, tec_key, src_axis=1, tgt_axis=1)
            if topo is None:
                continue
            # topo: (n_ret_full, n_tec_all); topo_T: (n_tec_all, n_ret_full)
            topo_T = topo.T.float()
            with torch.no_grad():
                for W in (W_on, W_off):
                    w_shape = W.weight.shape  # (n_tec_e, N_RET_PER_TYPE)
                    if topo_T.shape != w_shape:
                        t = torch.nn.functional.interpolate(
                            topo_T.unsqueeze(0).unsqueeze(0),
                            size=w_shape, mode='bilinear', align_corners=False
                        ).squeeze(0).squeeze(0)
                    else:
                        t = topo_T
                    W.weight.data.mul_(t)

    def apply_distance_weights(self, weight_matrix, region_a, region_b,
                                lambda_um=100.0, strength=0.5):
        """Modulate an existing weight matrix by distance-dependent mask.

        weight_matrix: (n_a, n_b) nn.Parameter or tensor
        strength: 0 = no effect, 1 = fully distance-dependent
        """
        mask = self.distance_weight_mask(region_a, region_b, lambda_um)
        if mask is None:
            return weight_matrix
        # Handle dimension mismatch: weight shape may differ from position count
        w_shape = weight_matrix.data.shape
        if mask.shape != w_shape:
            # Resize mask via bilinear interpolation to match weight matrix
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=w_shape, mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
        with torch.no_grad():
            blended = weight_matrix.data * (1 - strength + strength * mask)
            weight_matrix.data.copy_(blended)
        return weight_matrix
