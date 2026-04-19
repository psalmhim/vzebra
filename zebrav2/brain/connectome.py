"""
MapZebrain connectome constraints.

Uses MPIN-Atlas brain region centroids and known zebrafish connectivity
to constrain inter-region projection weights based on anatomical distance.

Connectivity rules (from zebrafish literature):
  - Retina → Tectum (contralateral, strong)
  - Tectum → Pretectum → Thalamus (ipsilateral)
  - Thalamus → Pallium (bilateral, reciprocal)
  - Pallium → Subpallium/BG (ipsilateral)
  - Tectum → Reticular formation (bilateral, for escape)
  - Habenula → Raphe (midline)
  - Raphe → widespread (5-HT)
  - PT (posterior tuberculum) → widespread (DA)

Distance-dependent connectivity: P(connection) ∝ exp(-d / λ)
"""
import os
import csv
import math
import numpy as np
import torch

ATLAS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'atlas')


def load_regions():
    """Load brain region names and abbreviations."""
    path = os.path.join(ATLAS_DIR, 'MPIN-Atlas_brain_region_Combined_brain_regions_index.csv')
    regions = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['Grey_level'])
            regions[idx] = {
                'name': row['Brain_region'].strip(),
                'abbr': row['Abbr'].strip(),
            }
    return regions


def load_centroids():
    """Load 3D centroids (72 = 36 regions × 2 hemispheres)."""
    path = os.path.join(ATLAS_DIR, 'centroids.csv')
    centroids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                x, y, z = line.split(',')
                centroids.append((float(x), float(y), float(z)))
    return np.array(centroids)


# Known anatomical projections (source_abbr → target_abbr, strength)
# Based on zebrafish connectome literature (Kunst et al. 2019, Förster et al. 2017)
ANATOMICAL_PROJECTIONS = [
    ('R',   'TeO', 1.0),   # Retina → Tectum (strongest, contralateral)
    ('TeO', 'PrT', 0.8),   # Tectum → Pretectum
    ('TeO', 'Th',  0.6),   # Tectum → Thalamus
    ('PrT', 'Th',  0.5),   # Pretectum → Thalamus
    ('Th',  'P',   0.7),   # Thalamus → Pallium
    ('P',   'Th',  0.4),   # Pallium → Thalamus (feedback)
    ('P',   'SP',  0.6),   # Pallium → Subpallium (BG)
    ('SP',  'P',   0.3),   # Subpallium → Pallium (feedback)
    ('TeO', 'aRF', 0.7),   # Tectum → Reticular formation (escape)
    ('TeO', 'imRF',0.5),   # Tectum → intermediate RF
    ('Hb',  'Ra',  0.6),   # Habenula → Raphe (5-HT)
    ('Ra',  'TeO', 0.3),   # Raphe → Tectum (5-HT modulation)
    ('Ra',  'P',   0.3),   # Raphe → Pallium
    ('PT',  'SP',  0.5),   # Post. tuberculum → Subpallium (DA)
    ('PT',  'P',   0.4),   # Post. tuberculum → Pallium (DA)
    ('P',   'Hr',  0.3),   # Pallium → Hypothalamus
    ('Cb',  'TeO', 0.3),   # Cerebellum → Tectum
    ('TeO', 'PrT', 0.8),   # Tectum → Pretectum (already listed above, but explicit)
    ('PrT', 'Th',  0.5),   # Pretectum → Thalamus (already listed above)
    ('Hb',  'IPN', 0.9),   # Habenula → IPN (strongest projection: MHb→vIPN)
    ('IPN', 'Ra',  0.5),   # IPN → Raphe (5-HT modulation)
    ('IPN', 'PT',  0.4),   # IPN → Post. tuberculum (DA feedback)
    ('Ra',  'SP',  0.4),   # Raphe → Subpallium (5-HT impulse control)
    ('pRF', 'Th',  0.5),   # LC (pRF) → Thalamus (wake/sleep gating)
    ('pRF', 'P',   0.4),   # LC (pRF) → Pallium (attention, signal-to-noise)
    ('pRF', 'SP',  0.3),   # LC (pRF) → Subpallium (arousal → amygdala)
]

# Map v2 modules to atlas regions
V2_TO_ATLAS = {
    'retina':   'R',
    'tectum':   'TeO',
    'thalamus': 'Th',
    'pallium':  'P',
    'bg':       'SP',
    'rs':       'aRF',
    'neuromod_da':  'PT',
    'neuromod_5ht': 'Ra',
    'pretectum':    'PrT',
    'ipn':          'IPN',
    'raphe':        'Ra',
    'lc':           'pRF',
}


def get_projection_strength(source_module: str, target_module: str) -> float:
    """
    Look up anatomical projection strength between two v2 modules.
    Returns 0.0 if no known projection, otherwise strength in [0, 1].
    """
    src = V2_TO_ATLAS.get(source_module, '')
    tgt = V2_TO_ATLAS.get(target_module, '')
    if not src or not tgt:
        return 0.0
    for s, t, strength in ANATOMICAL_PROJECTIONS:
        if s == src and t == tgt:
            return strength
    return 0.0


def distance_dependent_connectivity(n_pre: int, n_post: int,
                                     source_module: str, target_module: str,
                                     lambda_scale: float = 100.0,
                                     device='cpu') -> torch.Tensor:
    """
    Generate connectivity mask based on anatomical distance + projection strength.

    Returns: (n_post, n_pre) float tensor with connection probabilities.
    Multiply with random weights to get the final weight matrix.
    """
    strength = get_projection_strength(source_module, target_module)
    if strength < 0.01:
        return torch.zeros(n_post, n_pre, device=device)

    # Base probability from anatomical strength
    p_base = strength * 0.3  # max 30% connectivity for strongest projections

    # Generate mask
    mask = (torch.rand(n_post, n_pre, device=device) < p_base).float()
    return mask * strength


def init_connectome_weights(W: torch.nn.Parameter,
                            source_module: str, target_module: str,
                            g_scale: float = 1.0):
    """
    Initialize a weight matrix using connectome-constrained connectivity.
    Modifies W in-place.
    """
    n_post, n_pre = W.shape
    device = W.device
    mask = distance_dependent_connectivity(
        n_pre, n_post, source_module, target_module, device=device)
    with torch.no_grad():
        random_W = torch.randn_like(W) * g_scale * 0.1
        W.copy_(random_W * mask)


def get_connectome_summary() -> str:
    """Print summary of anatomical connectivity constraints."""
    regions = load_regions()
    lines = ["MapZebrain Connectome Constraints:"]
    lines.append(f"  {len(regions)} brain regions, {len(ANATOMICAL_PROJECTIONS)} projections")
    for src, tgt, strength in ANATOMICAL_PROJECTIONS:
        lines.append(f"  {src:5s} → {tgt:5s}  strength={strength:.1f}")
    return '\n'.join(lines)
