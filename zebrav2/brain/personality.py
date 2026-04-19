"""
Individual personality profiles for zebrafish.

Zebrafish show consistent individual differences in boldness, exploration,
and sociality (Teles et al. 2013, Norton et al. 2011). These map onto
neuromodulatory baselines:

  Habenula asymmetry → bold/shy (Amo et al. 2014)
  DA baseline → reward sensitivity
  5-HT baseline → patience/impulsivity
  NA baseline → arousal reactivity
  Amygdala gain → fear sensitivity
  Cortisol → stress reactivity

Usage:
  personality = get_personality('bold')
  brain.apply_personality(personality)
"""
import numpy as np


# Predefined personality types
PERSONALITIES = {
    'bold': {
        'name': 'bold',
        'DA_baseline': 0.7,       # high reward drive → aggressive foraging
        'HT5_baseline': 0.3,     # low patience → impulsive
        'NA_baseline': 0.5,       # moderate arousal
        'ACh_baseline': 0.6,      # high attention
        'amy_gain': 0.5,          # low fear sensitivity
        'flee_threshold': 0.35,   # high threshold → flees late
        'explore_bias': 0.0,
        'social_bias': 0.0,
        'habenula_threshold': 0.6,  # frustration tolerant
        'cpg_noise': 0.2,         # vigorous movement
        'description': 'Aggressive forager, low fear, approaches predator closely',
    },
    'shy': {
        'name': 'shy',
        'DA_baseline': 0.3,       # low reward drive
        'HT5_baseline': 0.7,     # high patience → cautious
        'NA_baseline': 0.6,       # high arousal → hypervigilant
        'ACh_baseline': 0.4,
        'amy_gain': 1.5,          # high fear sensitivity
        'flee_threshold': 0.15,   # low threshold → flees early
        'explore_bias': 0.1,      # less exploration
        'social_bias': -0.1,      # seeks group safety
        'habenula_threshold': 0.3,  # easily frustrated
        'cpg_noise': 0.08,        # cautious movement
        'description': 'Cautious, flees early, stays near group',
    },
    'explorer': {
        'name': 'explorer',
        'DA_baseline': 0.5,
        'HT5_baseline': 0.2,     # low patience → restless
        'NA_baseline': 0.4,
        'ACh_baseline': 0.7,      # high attention → curious
        'amy_gain': 0.7,
        'flee_threshold': 0.25,
        'explore_bias': -0.3,     # strong exploration drive
        'social_bias': 0.1,       # less social (independent)
        'habenula_threshold': 0.5,
        'cpg_noise': 0.15,
        'description': 'Curious, explores widely, independent from group',
    },
    'social': {
        'name': 'social',
        'DA_baseline': 0.4,
        'HT5_baseline': 0.6,
        'NA_baseline': 0.3,       # calm
        'ACh_baseline': 0.5,
        'amy_gain': 0.8,
        'flee_threshold': 0.25,
        'explore_bias': 0.05,
        'social_bias': -0.3,      # strong social drive
        'habenula_threshold': 0.4,
        'cpg_noise': 0.1,
        'description': 'Stays near group, follows neighbors, moderate foraging',
    },
    'default': {
        'name': 'default',
        'DA_baseline': 0.5,
        'HT5_baseline': 0.5,
        'NA_baseline': 0.3,
        'ACh_baseline': 0.5,
        'amy_gain': 1.0,
        'flee_threshold': 0.25,
        'explore_bias': 0.0,
        'social_bias': 0.0,
        'habenula_threshold': 0.4,
        'cpg_noise': 0.12,
        'description': 'Balanced, no personality bias',
    },
}


def get_personality(name='default'):
    """Get personality profile by name. Returns default if not found."""
    return PERSONALITIES.get(name, PERSONALITIES['default']).copy()


def random_personality(rng=None):
    """Generate a random personality by interpolating traits."""
    if rng is None:
        rng = np.random
    p = get_personality('default')
    # Random variation around default (±30%)
    for key in ['DA_baseline', 'HT5_baseline', 'NA_baseline', 'ACh_baseline']:
        p[key] = float(np.clip(p[key] + rng.normal(0, 0.15), 0.1, 0.9))
    p['amy_gain'] = float(np.clip(1.0 + rng.normal(0, 0.3), 0.3, 2.0))
    p['flee_threshold'] = float(np.clip(0.25 + rng.normal(0, 0.05), 0.10, 0.40))
    p['explore_bias'] = float(np.clip(rng.normal(0, 0.1), -0.3, 0.3))
    p['social_bias'] = float(np.clip(rng.normal(0, 0.1), -0.3, 0.3))
    p['habenula_threshold'] = float(np.clip(0.4 + rng.normal(0, 0.1), 0.2, 0.7))
    p['cpg_noise'] = float(np.clip(0.12 + rng.normal(0, 0.03), 0.05, 0.25))
    p['description'] = 'Random personality'
    return p


def assign_personalities(n_fish, mode='mixed'):
    """
    Assign personalities to n fish.
    mode: 'mixed' (1 of each type), 'random', 'uniform' (all default)
    """
    if mode == 'uniform':
        return [get_personality('default') for _ in range(n_fish)]
    elif mode == 'random':
        rng = np.random.RandomState(42)
        return [random_personality(rng) for _ in range(n_fish)]
    else:  # mixed
        types = ['bold', 'shy', 'explorer', 'social', 'default']
        return [get_personality(types[i % len(types)]) for i in range(n_fish)]
