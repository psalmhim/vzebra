"""
Disorder phenotype presets — one-call parameter manipulation for ZebrafishBrainV2.

Each preset models a specific genetic or pharmacological manipulation:

  wildtype      — no changes; baseline controls
  hypodopamine  — DA depletion (MPTP/reserpine)
  asd           — hyperserotonin + social withdrawal (shank3/SERT mutants)
  schizophrenia — NMDA hypofunction + DA dysregulation (DISC1/ketamine)
  anxiety       — elevated NA, low flee threshold, hypervigilant phenotype
  depression    — low DA/5-HT, elevated cortisol (chronic mild stress model)
  adhd          — low DA/ACh, high CPG noise, exploration bias
  ptsd          — high amygdala gain + fear baseline + cortisol

Usage:
    from zebrav2.brain.disorder import apply_disorder
    changes = apply_disorder(brain, 'anxiety', intensity=0.8)

All parameter writes use linear interpolation between wildtype baselines and
disorder target values, parameterised by intensity ∈ [0, 1]. This enables
dose–response curves without re-instantiating the brain.

No brain_v2 module is imported at runtime — only attribute access is used.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zebrav2.brain.brain_v2 import ZebrafishBrainV2

# ---------------------------------------------------------------------------
# Goal index constants (mirrors brain_v2.py top-level constants)
# ---------------------------------------------------------------------------
_GOAL_FORAGE  = 0
_GOAL_FLEE    = 1
_GOAL_EXPLORE = 2
_GOAL_SOCIAL  = 3

# ---------------------------------------------------------------------------
# Human-readable descriptions
# ---------------------------------------------------------------------------
DISORDER_DESCRIPTIONS: dict[str, str] = {
    'wildtype':      'Baseline — no manipulation',
    'hypodopamine':  'DA depletion — MPTP/reserpine model, impairs STDP consolidation',
    'asd':           'Hyperserotonin + social withdrawal — shank3/SERT mutant model',
    'schizophrenia': 'NMDA hypofunction + DA dysregulation — DISC1/ketamine model',
    'anxiety':       'Elevated NA + low flee threshold — hypervigilant phenotype',
    'depression':    'Low DA/5-HT + elevated cortisol — CMS (chronic mild stress) model',
    'adhd':          'Low DA/ACh + high CPG noise + exploration bias — dopamine deficit model',
    'ptsd':          'High amygdala gain + fear baseline + cortisol — trauma model',
}

# ---------------------------------------------------------------------------
# Internal disorder parameter tables
#
# Each entry is a dict of (wildtype_value, disorder_value) pairs keyed by a
# logical parameter name.  apply_disorder() reads these and writes to the brain.
# ---------------------------------------------------------------------------
_WILDTYPE_DEFAULTS: dict[str, float] = {
    'DA':               0.5,
    'NA':               0.3,
    'HT5':              0.5,
    'ACh':              0.5,
    'amy_retinal_gain': 0.08,
    'amy_fear_baseline':0.0,
    'flee_threshold':   0.25,
    'habenula_threshold':0.4,
    'cpg_noise':        0.12,
    'cortisol':         0.0,
    'oxt_baseline':     0.0,   # OxytocinSystem.oxt initial value
    'social_bias':      0.0,   # meta_goal goal_bias[SOCIAL] offset
    'w_food_cue':       1.0,
    'w_alarm':          1.0,
    'explore_bias':     0.0,   # meta_goal goal_bias[EXPLORE] offset
    # Active inference precision parameters (Adams et al. 2012)
    'gaze_precision':   1.0,   # saccade PE→gaze coupling (sensory precision)
    'pursuit_gain':     1.0,   # smooth pursuit tracking gain
    'motor_ai_blend':   0.3,   # active inference motor blend factor
    'sensory_precision':1.0,   # general sensory precision weighting
    'column_gamma':     0.0,   # TwoCompColumn precision γ (π = σ(γ) = 0.5)
}

# Disorder target values — only keys that deviate from wildtype are listed.
_DISORDER_TARGETS: dict[str, dict[str, float]] = {
    'wildtype': {},

    'hypodopamine': {
        'DA':   0.05,
    },

    'asd': {
        'HT5':              0.85,
        'social_bias':      -0.3,
        'habenula_threshold':0.2,
        'w_food_cue':       0.2,
        'w_alarm':          0.3,
        'oxt_baseline':     0.1,
    },

    'schizophrenia': {
        'DA':               0.85,
        # NMDA and classifier temperature handled separately (non-scalar)
        # Active inference: aberrant precision (Adams et al. 2012, 2013)
        'gaze_precision':   0.3,   # reduced → noisy/imprecise saccades
        'pursuit_gain':     0.4,   # reduced → smooth pursuit deficit
        'sensory_precision':0.5,   # reduced → weak sensory PE weighting
        # Two-compartment column precision (Lee, Lee & Park 2026)
        # Reduced γ → lower π = σ(γ) → weaker PE gain → aberrant salience
        'column_gamma':     -1.5,  # wildtype 0.0 → scz -1.5 (π ≈ 0.18)
    },

    'anxiety': {
        'NA':               0.75,
        'flee_threshold':   0.12,
        'amy_retinal_gain': 0.08 * 1.8,   # amy_gain = 1.8
        'habenula_threshold':0.3,
    },

    'depression': {
        'DA':               0.15,
        'HT5':              0.2,
        'habenula_threshold':0.25,
        'cortisol':         0.6,
        'column_gamma':     -0.8,  # low precision → anhedonia (π ≈ 0.31)
    },

    'adhd': {
        'DA':               0.35,
        'ACh':              0.25,
        'cpg_noise':        0.30,
        'explore_bias':     -0.3,
    },

    'ptsd': {
        'amy_retinal_gain': 0.08 * 2.0,   # amy_gain = 2.0
        'amy_fear_baseline':0.4,
        'NA':               0.8,
        'flee_threshold':   0.10,
        'cortisol':         0.6,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _apply_column_gamma(brain: ZebrafishBrainV2, gamma_val: float) -> None:
    """Set structural precision ceiling on all TwoCompColumn modules.

    Sets gamma_ceiling so that online learning cannot recover precision
    above σ(gamma_val).  This models the NMDA hypofunction in
    schizophrenia as a permanent structural deficit (Adams et al. 2013),
    not a transient initial condition.
    """
    modules_with_pc = [
        brain.retina, brain.olfaction, brain.lateral_line_mod,
        brain.proprioception, brain.vestibular, brain.color_vision,
        brain.thalamus_L, brain.thalamus_R, brain.pallium,
    ]
    for mod in modules_with_pc:
        pc = getattr(mod, 'pc', None)
        if pc is not None and hasattr(pc, 'gamma'):
            pc.gamma_ceiling = gamma_val
            pc.gamma.clamp_(max=gamma_val)
    # Active motor uses .column instead of .pc
    col = getattr(brain.active_motor, 'column', None)
    if col is not None and hasattr(col, 'gamma'):
        col.gamma_ceiling = gamma_val
        col.gamma.clamp_(max=gamma_val)


def _apply_column_gamma_floor(brain: ZebrafishBrainV2, gamma_val: float) -> None:
    """Set structural precision floor on all TwoCompColumn modules.

    Models hypervigilance: online learning cannot reduce precision below
    σ(gamma_val).  This is the mirror of _apply_column_gamma (ceiling)
    and is used for anxiety / PTSD phenotypes where sensory precision
    is tonically elevated (Paulus & Stein 2006).
    """
    modules_with_pc = [
        brain.retina, brain.olfaction, brain.lateral_line_mod,
        brain.proprioception, brain.vestibular, brain.color_vision,
        brain.thalamus_L, brain.thalamus_R, brain.pallium,
    ]
    for mod in modules_with_pc:
        pc = getattr(mod, 'pc', None)
        if pc is not None and hasattr(pc, 'gamma'):
            pc.gamma_floor = gamma_val
            pc.gamma.clamp_(min=gamma_val)
    # Active motor uses .column instead of .pc
    col = getattr(brain.active_motor, 'column', None)
    if col is not None and hasattr(col, 'gamma'):
        col.gamma_floor = gamma_val
        col.gamma.clamp_(min=gamma_val)


def apply_disorder(brain: ZebrafishBrainV2, name: str,
                   intensity: float = 1.0) -> dict:
    """
    Apply disorder phenotype to brain in-place.

    Parameters
    ----------
    brain:
        A ZebrafishBrainV2 instance (passed by reference; mutated in-place).
    name:
        One of the keys in DISORDER_DESCRIPTIONS.
    intensity:
        Interpolation factor: 0.0 = wildtype, 1.0 = full disorder.
        Values outside [0, 1] are clamped.

    Returns
    -------
    dict
        Log of every parameter change: {'param': (old_val, new_val), ...}.
    """
    intensity = max(0.0, min(1.0, float(intensity)))
    changes: dict = {}

    if name not in _DISORDER_TARGETS:
        raise ValueError(
            f"Unknown disorder '{name}'. "
            f"Available: {sorted(_DISORDER_TARGETS.keys())}"
        )

    if name == 'wildtype' or intensity == 0.0:
        return changes

    targets = _DISORDER_TARGETS[name]

    def _interp(key: str) -> tuple[float, float]:
        """Return (old_val, new_val) interpolated at intensity."""
        wt  = _WILDTYPE_DEFAULTS[key]
        tgt = targets.get(key, wt)
        return wt, wt + intensity * (tgt - wt)

    # --- Neuromodulators ---
    for nm_key, attr in [('DA', 'DA'), ('NA', 'NA'), ('HT5', 'HT5'), ('ACh', 'ACh')]:
        if nm_key in targets:
            old, new = _interp(nm_key)
            getattr(brain.neuromod, attr).fill_(new)
            changes[nm_key] = (old, new)

    # --- Amygdala ---
    if 'amy_retinal_gain' in targets:
        old, new = _interp('amy_retinal_gain')
        brain.amygdala.retinal_gain = new
        changes['amy_retinal_gain'] = (old, new)

    if 'amy_fear_baseline' in targets:
        old, new = _interp('amy_fear_baseline')
        brain.amygdala.fear_baseline = new
        changes['amy_fear_baseline'] = (old, new)

    # --- Flee threshold ---
    if 'flee_threshold' in targets:
        old, new = _interp('flee_threshold')
        brain._flee_threshold = new
        changes['flee_threshold'] = (old, new)

    # --- Habenula ---
    if 'habenula_threshold' in targets:
        old, new = _interp('habenula_threshold')
        brain.habenula.threshold = new
        changes['habenula_threshold'] = (old, new)

    # --- CPG noise ---
    if 'cpg_noise' in targets:
        old, new = _interp('cpg_noise')
        brain.cpg.noise = new
        changes['cpg_noise'] = (old, new)

    # --- HPA cortisol pre-load ---
    if 'cortisol' in targets:
        old, new = _interp('cortisol')
        if hasattr(brain, 'hpa'):
            brain.hpa.cortisol = new
        changes['cortisol'] = (old, new)

    # --- Oxytocin baseline ---
    if 'oxt_baseline' in targets:
        old, new = _interp('oxt_baseline')
        if hasattr(brain, 'oxytocin'):
            brain.oxytocin.oxt = new
        changes['oxt_baseline'] = (old, new)

    # --- Social memory weights ---
    if 'social_bias' in targets:
        old, new = _interp('social_bias')
        with __import__('torch').no_grad():
            brain.meta_goal.goal_bias[_GOAL_SOCIAL] += new - old
            brain.meta_goal.goal_bias.clamp_(-0.5, 0.5)
        changes['social_bias'] = (old, new)

    if 'explore_bias' in targets:
        old, new = _interp('explore_bias')
        with __import__('torch').no_grad():
            brain.meta_goal.goal_bias[_GOAL_EXPLORE] += new - old
            brain.meta_goal.goal_bias.clamp_(-0.5, 0.5)
        changes['explore_bias'] = (old, new)

    if 'w_food_cue' in targets:
        old, new = _interp('w_food_cue')
        brain.social_mem.w_food_cue = new
        changes['w_food_cue'] = (old, new)

    if 'w_alarm' in targets:
        old, new = _interp('w_alarm')
        brain.social_mem.w_alarm = new
        changes['w_alarm'] = (old, new)

    # --- Active inference precision (Adams et al. 2012) ---
    if 'gaze_precision' in targets:
        old, new = _interp('gaze_precision')
        brain.saccade.gaze_precision = new
        changes['gaze_precision'] = (old, new)

    if 'pursuit_gain' in targets:
        old, new = _interp('pursuit_gain')
        brain.saccade.pursuit_gain = new
        changes['pursuit_gain'] = (old, new)

    if 'sensory_precision' in targets:
        old, new = _interp('sensory_precision')
        brain.saccade.sensory_precision = new
        changes['sensory_precision'] = (old, new)

    if 'motor_ai_blend' in targets:
        old, new = _interp('motor_ai_blend')
        brain.active_motor.ai_blend = new
        changes['motor_ai_blend'] = (old, new)

    # --- TwoCompColumn precision perturbation (Lee, Lee & Park 2026) ---
    if 'column_gamma' in targets:
        old, new = _interp('column_gamma')
        _apply_column_gamma(brain, new)
        changes['column_gamma'] = (old, new)

    # --- Disorder-specific extras ---
    if name == 'hypodopamine':
        # Disable phasic DA bursts
        brain._da_phasic_steps = 0
        changes['phasic_DA_disabled'] = (True, True)

    if name == 'schizophrenia':
        # NMDA hypofunction: scale g_NMDA in thalamus and pallium
        nmda_scale = 1.0 - intensity * (1.0 - 0.3)   # 0.3 at full intensity
        for thal in [getattr(brain, 'thalamus_L', None), getattr(brain, 'thalamus_R', None)]:
            if thal is None:
                continue
            for layer_name in ['TC', 'TRN']:
                layer = getattr(thal, layer_name, None)
                if layer is not None and hasattr(layer, 'g_NMDA'):
                    old_g = layer.g_NMDA
                    layer.g_NMDA = old_g * nmda_scale
                    changes[f'thalamus.{layer_name}.g_NMDA'] = (old_g, layer.g_NMDA)
        for layer_name in ['pal_s', 'pal_d']:
            layer = getattr(brain.pallium, layer_name, None)
            if layer is not None and hasattr(layer, 'g_NMDA'):
                old_g = layer.g_NMDA
                layer.g_NMDA = old_g * nmda_scale
                changes[f'pallium.{layer_name}.g_NMDA'] = (old_g, layer.g_NMDA)
        # Classifier temperature (noisy perception)
        if hasattr(brain.classifier, 'temperature'):
            target_temp = 0.3
            old_temp = brain.classifier.temperature
            new_temp = old_temp + intensity * (target_temp - old_temp)
            brain.classifier.temperature = new_temp
            changes['classifier_temperature'] = (old_temp, new_temp)
        # Also disable phasic DA (as in hypodopamine)
        brain._da_phasic_steps = 0
        changes['phasic_DA_disabled'] = (True, True)

    if name == 'asd':
        # Uneven precision: high sensory, low integrative (Lawson et al. 2014)
        # Sensory modules: elevated γ → detail-focused perception
        sensory_mods = [brain.retina, brain.lateral_line_mod,
                        brain.color_vision, brain.proprioception]
        for mod in sensory_mods:
            pc = getattr(mod, 'pc', None)
            if pc is not None and hasattr(pc, 'gamma'):
                tgt = intensity * 1.5
                pc.gamma.fill_(tgt)
                pc.gamma_floor = tgt * 0.8  # lock floor high
        # Integrative modules: reduced γ → weak top-down integration
        integ_mods = [brain.thalamus_L, brain.thalamus_R, brain.pallium]
        for mod in integ_mods:
            pc = getattr(mod, 'pc', None)
            if pc is not None and hasattr(pc, 'gamma'):
                tgt = -intensity * 1.0
                pc.gamma.fill_(tgt)
                pc.gamma_ceiling = tgt + 0.5  # ceiling limits recovery
        changes['asd_sensory_gamma'] = (0.0, 1.5 * intensity)
        changes['asd_integrative_gamma'] = (0.0, -1.0 * intensity)

    if name == 'anxiety':
        # Hypervigilance: elevated precision floor — fish can't relax
        # (Paulus & Stein 2006; high interoceptive precision)
        _apply_column_gamma_floor(brain, 1.0 * intensity)
        changes['gamma_floor'] = (0.0, 1.0 * intensity)

    if name == 'ptsd':
        # Trauma-locked precision floor — less extreme than anxiety
        # (Linson et al. 2020; aberrant threat precision)
        _apply_column_gamma_floor(brain, 0.8 * intensity)
        changes['gamma_floor'] = (0.0, 0.8 * intensity)

    return changes


def list_disorders() -> None:
    """Print a formatted table of all available disorder presets."""
    col_w = max(len(k) for k in DISORDER_DESCRIPTIONS) + 2
    print(f"{'Disorder':<{col_w}}  Description")
    print("-" * (col_w + 2) + "  " + "-" * 55)
    for name, desc in DISORDER_DESCRIPTIONS.items():
        print(f"  {name:<{col_w - 2}}  {desc}")
