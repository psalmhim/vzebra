"""
MapZebrain connectome constraints — EM-grounded inter-region weights.

Primary source: Hildebrand et al. 2017 (Nature 545) ssEM dataset.
  subject_12_sc_list_pre_post.csv — 10.2M synapse pairs (47K neurons).
  Strengths derived from synapse density = count / (n_pre × n_post).
  Normalized so density=0.05 → strength=1.0 (95th-percentile ceiling).

Secondary: Kunst et al. 2019 (mapZebrain), Kölsch 2021, Carbo-Tano 2023,
  Agetsuma 2010, Dreosti 2014, Tay 2011, Del Bene 2010, Henriques 2019.

Key EM findings (correcting textbook assumptions):
  - TeO → pRF stronger than TeO → aRF (0.015 vs 0.005)
  - aRF → Th surprisingly strong (0.031) — ascending arousal path
  - Hb → IPN weak in EM specimen (0.002) — likely volume edge artifact;
    kept at 0.90 from Agetsuma 2010 tract-tracing / calcium imaging
  - MOS → TeO is the dominant tectum input (motor feedback)
  - PO → IPN is the strongest density in the dataset (0.198)

Projection tuple format: (source_abbr, target_abbr, strength, is_excitatory)
  strength in [0, 1]; is_excitatory=False means GABAergic or glycinergic.
  Atlas abbreviations match MPIN-Atlas_brain_region_Combined_brain_regions_index.csv.
"""
import os
import csv
import math
import numpy as np
import torch

ATLAS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'atlas')

# ─── EM normalization factor ─────────────────────────────────────────────────
# density 0.05 → strength 1.0  (95th-pct of modelled connections)
_EM_NORM = 0.05


def _d(density: float) -> float:
    """Convert EM synapse density to normalized [0,1] strength."""
    return min(1.0, density / _EM_NORM)


def load_regions():
    """Load brain region names and abbreviations from MPIN-Atlas CSV."""
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
    """Load 3D centroids (72 entries = 36 atlas regions × 2 hemispheres)."""
    path = os.path.join(ATLAS_DIR, 'centroids.csv')
    centroids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                x, y, z = line.split(',')
                centroids.append((float(x), float(y), float(z)))
    return np.array(centroids)


# ─── Anatomical projections ───────────────────────────────────────────────────
# Tuple: (source_abbr, target_abbr, strength [0-1], is_excitatory)
# EM densities: d(0.015)=0.30, d(0.030)=0.60, d(0.050)=1.0
# Retina→Tectum is assigned 1.0 by definition (not in EM volume but well-established).
ANATOMICAL_PROJECTIONS = [

    # ═══ VISUAL PATHWAY ══════════════════════════════════════════════════════
    # Kölsch 2021: 97% RGCs → TeO via AF10; 3% collaterals to PrT/Th
    ('R',    'TeO',  1.00, True),   # Retina → Tectum (canonical strongest, excitatory)
    ('R',    'PrT',  0.30, True),   # Retina → Pretectum (AF5-AF9 collaterals)
    ('R',    'Th',   0.10, True),   # Retina → Thalamus (AF1-AF4, circadian minority)

    # Tectum outputs — EM-derived densities:
    ('TeO',  'pRF',  0.30, True),   # TeO → posterior RF  EM d=0.015
    ('TeO',  'imRF', 0.15, True),   # TeO → intermediate RF  EM d=0.0076
    ('TeO',  'aRF',  0.11, True),   # TeO → anterior RF  EM d=0.0053
    ('TeO',  'PrT',  0.10, True),   # TeO → Pretectum  EM d=0.0050
    ('TeO',  'Ra',   0.09, True),   # TeO → Raphe  EM d=0.0044
    ('TeO',  'PT',   0.05, True),   # TeO → Post. tuberculum  EM d=0.0025
    ('TeO',  'Th',   0.04, True),   # TeO → Thalamus (weak direct)  EM d=0.00217
    ('TeO',  'Cb',   0.05, True),   # TeO → Cerebellum (mossy fibers)  EM d=0.0024
    ('TeO',  'T',    0.08, True),   # TeO → Tegmentum  EM d=0.0038
    ('TeO',  'TeO',  0.20, False),  # TeO ↔ TeO commissural (GABAergic intertectal)

    # Tectum inputs — EM-derived:
    ('aRF',  'TeO',  0.22, True),   # aRF → TeO feedback  EM d=0.011
    ('imRF', 'TeO',  0.09, True),   # imRF → TeO  EM d=0.0045
    ('T',    'TeO',  0.08, True),   # Tegmentum → TeO  EM d=0.0038
    # Pretectum — EM-derived:
    ('PrT',  'aRF',  0.54, True),   # PrT → anterior RF  EM d=0.027
    ('PrT',  'imRF', 0.55, True),   # PrT → intermediate RF  EM d=0.028
    ('PrT',  'pRF',  0.68, True),   # PrT → posterior RF  EM d=0.034
    ('PrT',  'Th',   0.23, True),   # PrT → Thalamus (OKR, PLR relay)

    # Thalamus — EM-derived:
    ('Th',   'P',    0.70, True),   # Th → Pallium (sensory relay, established)
    ('P',    'Th',   0.40, True),   # Pallium → Th feedback / TRN gate
    ('aRF',  'Th',   0.62, True),   # aRF → Th (ascending arousal)  EM d=0.031  ← surprising
    ('imRF', 'Th',   0.20, True),   # imRF → Th  EM d=0.0098
    ('Th',   'aRF',  0.20, True),   # Th → aRF  EM d=0.010

    # ═══ TOP-DOWN ATTENTION ═══════════════════════════════════════════════════
    ('P',    'TeO',  0.50, True),   # Pallium → Tectum (top-down spatial attention)

    # ═══ MOTOR PATHWAY ════════════════════════════════════════════════════════
    # Carbo-Tano 2023: MLR → V2a RSN → spinal; Hildebrand 2017: RS neuron catalog
    ('aRF',  'imRF', 0.60, True),   # aRF → imRF (descending relay)
    ('aRF',  'pRF',  0.50, True),   # aRF → pRF (brainstem/LC region)
    ('imRF', 'aRF',  0.20, False),  # imRF → aRF (glycinergic inhibitory feedback)
    ('T',    'imRF', 1.00, True),   # T → imRF  EM d=0.057 (strongest modeled)
    ('T',    'pRF',  1.00, True),   # T → pRF  EM d=0.151
    ('T',    'Ra',   1.00, True),   # T → Ra  EM d=0.051
    ('T',    'aRF',  0.27, True),   # T → aRF  EM d=0.014

    # Cerebellum outputs:
    ('Cb',   'aRF',  0.33, True),   # Cb → aRF (eurydendroid → RF)  EM d=0.0165
    ('Cb',   'imRF', 0.21, True),   # Cb → imRF  EM d=0.0103
    ('Cb',   'Th',   0.30, True),   # Cb → Th (eurydendroid → thalamus)  EM d=0.015
    ('Cb',   'PrT',  0.13, True),   # Cb → PrT  EM d=0.0066
    ('IO',   'Cb',   0.90, True),   # Inferior Olive → Cb (climbing fibers, excitatory)
    ('aRF',  'Cb',   0.30, True),   # RF → Cb (mossy fibers)

    # ═══ HABENULA-IPN PATHWAY ═════════════════════════════════════════════════
    # Agetsuma 2010, Dreosti 2014: Hb → IPN is anatomically the strongest tract
    # (EM weakness is likely volume-edge artifact; IPN has only 54 neurons in EM)
    ('Hb',   'IPN',  0.90, True),   # Hb → IPN (fasciculus retroflexus, excitatory)
    ('Hb',   'Ra',   0.78, True),   # Hb → Ra  EM d=0.039 (directly from EM)
    ('IPN',  'Ra',   0.50, True),   # IPN → Raphe (5-HT aversion/fear)
    ('IPN',  'PT',   0.40, True),   # IPN → Post. tuberculum (DA feedback)
    ('IPN',  'PO',   0.60, True),   # IPN → Preoptic area  EM d=0.061
    ('PO',   'IPN',  1.00, True),   # PO → IPN  EM d=0.198 (strongest in dataset)
    ('SP',   'IPN',  0.72, True),   # SP → IPN  EM d=0.036
    ('P',    'IPN',  0.47, True),   # P → IPN  EM d=0.023
    ('Th',   'IPN',  0.30, True),   # Th → IPN  EM d=0.015
    ('OB',   'Hb',   0.60, True),   # OB → dHb (medial glomerulus, right-biased, excitatory)
    ('P',    'Hb',   0.35, True),   # Pallium → dHb (indirect fear via vENT)
    ('Hb',   'T',    0.02, True),   # Hb → Tegmentum  EM d=0.0011

    # ═══ NEUROMODULATORY ══════════════════════════════════════════════════════
    # Tay 2011: DA from PT (posterior tuberculum), NO VTA/SNc in zebrafish
    ('Ra',   'TeO',  0.09, True),   # Raphe → Tectum (5-HT visual sensitivity)  EM d=0.0044
    ('Ra',   'P',    0.07, True),   # Raphe → Pallium (5-HT cognitive)  EM d=0.0033
    ('Ra',   'SP',   0.08, True),   # Raphe → Subpallium (5-HT impulse control)  EM d=0.0043
    ('Ra',   'PO',   0.46, True),   # Raphe → PO  EM d=0.063
    ('Ra',   'Th',   0.17, True),   # Ra → Th  EM d=0.0086
    ('Ra',   'imRF', 0.14, True),   # Ra → imRF  EM d=0.0068
    ('PT',   'SP',   0.50, True),   # PT → Subpallium (DA ascending, direct pathway)
    ('PT',   'P',    0.40, True),   # PT → Pallium (DA reward signal)
    ('PT',   'aRF',  0.43, True),   # PT → aRF  EM d=0.021
    ('PT',   'imRF', 0.51, True),   # PT → imRF  EM d=0.026
    ('PT',   'Th',   0.61, True),   # PT → Th  EM d=0.030
    ('pRF',  'Th',   0.50, True),   # LC (pRF) → Thalamus (NA: wake/sleep gating)
    ('pRF',  'P',    0.40, True),   # LC (pRF) → Pallium (NA: signal/noise)
    ('pRF',  'SP',   0.30, True),   # LC (pRF) → Subpallium (NA: arousal→amygdala)
    ('pRF',  'TeO',  0.13, True),   # LC (pRF) → Tectum (NA: stimulus selectivity)
    ('pRF',  'PrT',  0.68, True),   # pRF → PrT  EM d=0.034

    # ═══ OLFACTORY PATHWAY ════════════════════════════════════════════════════
    # Yaksi 2009, Miyasaka 2009: OB → pallium (olfactory cortex analog)
    ('OB',   'P',    0.70, True),   # OB → Pallium (olfactory cortex analog)
    ('OB',   'Hr',   0.50, True),   # OB → rostral Hypothalamus (feeding)
    ('OB',   'SP',   0.40, True),   # OB → Subpallium (olfactory valence)
    ('P',    'OB',   0.30, True),   # Pallium → OB (top-down olfactory attention)

    # ═══ LATERAL LINE / AUDITORY PATHWAY ═════════════════════════════════════
    ('MON',  'TeO',  0.17, True),   # MON → Tectum  EM d=0.0083
    ('MON',  'TS',   0.70, True),   # MON → Torus semicircularis (ascending LL)
    ('TS',   'TeO',  0.50, True),   # TS → Tectum (auditory/LL → visual integration)
    ('TS',   'aRF',  0.40, True),   # TS → aRF (auditory startle relay)

    # ═══ PALLIUM / TELENCEPHALON ══════════════════════════════════════════════
    ('P',    'SP',   0.60, True),   # Pallium → Subpallium (BG input)
    ('SP',   'P',    0.30, False),  # Subpallium → Pallium (GABAergic BG feedback)
    ('P',    'Hr',   0.30, True),   # Pallium → rostral Hypothalamus
    ('P',    'Hi',   0.30, True),   # Pallium → intermediate Hypothalamus
    ('P',    'Hc',   0.25, True),   # Pallium → caudal Hypothalamus (stress axis)

    # ═══ HYPOTHALAMUS ═════════════════════════════════════════════════════════
    ('Hi',   'pRF',  0.52, True),   # intermediate Hyp → pRF  EM d=0.026
    ('Hi',   'aRF',  0.40, True),   # intermediate Hyp → aRF (feeding motor)
    ('Hr',   'PT',   0.30, True),   # rostral Hyp → PT (DA modulation)
    ('Hc',   'pRF',  0.30, True),   # caudal Hyp → pRF (stress response)
]

# ─── Lookup helpers ───────────────────────────────────────────────────────────

# Index for O(1) lookup by (src, tgt)
_PROJ_INDEX: dict[tuple[str, str], tuple[float, bool]] = {
    (s, t): (strength, excitatory)
    for s, t, strength, excitatory in ANATOMICAL_PROJECTIONS
}


def get_projection_strength(source_module: str, target_module: str) -> float:
    """Return anatomical projection strength in [0, 1] (0 = no known projection)."""
    src = V2_TO_ATLAS.get(source_module, source_module)
    tgt = V2_TO_ATLAS.get(target_module, target_module)
    val = _PROJ_INDEX.get((src, tgt))
    return val[0] if val is not None else 0.0


def is_inhibitory(source_module: str, target_module: str) -> bool:
    """Return True if projection is GABAergic/glycinergic (inhibitory)."""
    src = V2_TO_ATLAS.get(source_module, source_module)
    tgt = V2_TO_ATLAS.get(target_module, target_module)
    val = _PROJ_INDEX.get((src, tgt))
    if val is None:
        return False
    return not val[1]


# ─── Module → Atlas abbreviation mapping ─────────────────────────────────────
V2_TO_ATLAS: dict[str, str] = {
    # Core sensory
    'retina':       'R',
    'tectum':       'TeO',
    'pretectum':    'PrT',
    'thalamus':     'Th',
    # Telencephalon
    'pallium':      'P',
    'bg':           'SP',
    'subpallium':   'SP',
    'ipn':          'IPN',
    # Brainstem / motor
    'rs':           'aRF',
    'arf':          'aRF',
    'imrf':         'imRF',
    'prf':          'pRF',
    'tegmentum':    'T',
    # Neuromodulatory
    'neuromod_da':  'PT',
    'neuromod_5ht': 'Ra',
    'raphe':        'Ra',
    'lc':           'pRF',
    # Habenula
    'habenula':     'Hb',
    'habenula_l':   'Hb',
    'habenula_r':   'Hb',
    # Cerebellum / IO
    'cerebellum':   'Cb',
    'io':           'IO',
    # Olfactory
    'olfactory':    'OB',
    'ob':           'OB',
    # Hypothalamus
    'hypothalamus': 'Hi',
    'hyp_r':        'Hr',
    'hyp_i':        'Hi',
    'hyp_c':        'Hc',
    # Lateral line / auditory
    'mon':          'MON',
    'ts':           'TS',
}


# ─── Region-specific λ for intra-region distance-dependent connectivity ──────
# PNAS empirical estimate: λ=80 μm (Fig. S2B distance-correlation curve).
# Telencephalon (clustering=0.027, modularity=0.673): sparse/modular → larger λ.
# Hindbrain (clustering=0.066, global-eff=0.122): dense/clustered → smaller λ.
REGION_LAMBDA: dict[str, float] = {
    'pallium':       120.0,   # Tel low-clustering sparse-modular
    'subpallium':    120.0,   # Tel same
    'tectum':         80.0,   # Mes — baseline
    'thalamus':       80.0,   # Di — baseline
    'pretectum':      80.0,   # Di — baseline
    'tegmentum':      80.0,   # Di/Mes — baseline
    'hypothalamus':   80.0,   # Di — baseline
    'habenula':       80.0,   # Di — baseline
    'olfactory':      80.0,   # Tel/Di — baseline
    'cerebellum':     50.0,   # Hind high-clustering dense
    'reticular':      50.0,   # Hind same
    'medulla':        50.0,   # Hind same
    'raphe':          50.0,   # Hind same
    'spinal':         50.0,   # Hind-like
}


def get_region_lambda(region_name: str) -> float:
    """Return λ (μm) for within-region distance-dependent connectivity."""
    return REGION_LAMBDA.get(region_name, 80.0)


# ─── Weight initialisation ────────────────────────────────────────────────────

def distance_dependent_connectivity(n_pre: int, n_post: int,
                                    source_module: str, target_module: str,
                                    lambda_scale: float = 100.0,
                                    device='cpu') -> torch.Tensor:
    """
    Generate (n_post, n_pre) connectivity mask scaled by anatomical strength.
    Returns float tensor of connection probabilities (0 where no projection).
    """
    strength = get_projection_strength(source_module, target_module)
    if strength < 0.01:
        return torch.zeros(n_post, n_pre, device=device)
    p_base = strength * 0.3   # max 30% connectivity for strongest projections
    mask = (torch.rand(n_post, n_pre, device=device) < p_base).float()
    return mask * strength


def init_connectome_weights(W: torch.Tensor,
                            source_module: str, target_module: str,
                            g_scale: float = 1.0):
    """
    Initialise weight matrix in-place with connectome-constrained connectivity.
    Sign is set correctly for inhibitory projections (negative weights).

    Args:
        W:             (n_post, n_pre) weight parameter tensor
        source_module: v2 module name (key in V2_TO_ATLAS) or raw atlas abbr
        target_module: v2 module name (key in V2_TO_ATLAS) or raw atlas abbr
        g_scale:       conductance scale (multiplies all weights)
    """
    n_post, n_pre = W.shape
    device = W.device
    mask = distance_dependent_connectivity(
        n_pre, n_post, source_module, target_module, device=device)
    sign = -1.0 if is_inhibitory(source_module, target_module) else 1.0
    with torch.no_grad():
        random_W = torch.randn_like(W).abs() * g_scale * 0.1  # always positive initially
        W.copy_(sign * random_W * mask)


# ─── Summary ─────────────────────────────────────────────────────────────────

def get_connectome_summary() -> str:
    """Human-readable summary of anatomical connectivity constraints."""
    regions = load_regions()
    lines = [f"MapZebrain EM-grounded Connectome ({len(ANATOMICAL_PROJECTIONS)} projections):"]
    exc = [(s, t, st) for s, t, st, e in ANATOMICAL_PROJECTIONS if e]
    inh = [(s, t, st) for s, t, st, e in ANATOMICAL_PROJECTIONS if not e]
    lines.append(f"  Excitatory: {len(exc)}  Inhibitory: {len(inh)}")
    lines.append("  Top 20 by strength:")
    top = sorted(ANATOMICAL_PROJECTIONS, key=lambda x: -x[2])[:20]
    for s, t, strength, exc_flag in top:
        pol = 'E' if exc_flag else 'I'
        lines.append(f"  [{pol}] {s:6s} → {t:6s}  {strength:.2f}")
    return '\n'.join(lines)
