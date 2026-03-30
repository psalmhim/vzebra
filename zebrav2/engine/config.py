"""
Configuration system for training environments, personalities, and objectives.

All settings are JSON-serializable for web dashboard communication.
"""
import json
import os

DEFAULT_CONFIG = {
    # Environment
    'env': {
        'arena_w': 800,
        'arena_h': 600,
        'n_food': 20,
        'max_steps': 500,
        'food_respawn': True,
        'food_respawn_min': 5,
        'predator_enabled': True,
        'predator_ai': 'intelligent',  # 'simple', 'intelligent', 'none'
        'rocks_enabled': True,
        'seed': None,  # None = random
    },
    # Fish
    'fish': {
        'n_fish': 1,
        'personality_mode': 'default',  # 'default', 'bold', 'shy', 'explorer', 'social', 'random', 'mixed'
        'energy_start': 100.0,
        'energy_drain_rate': 0.2,
        'eat_radius': 35,
    },
    # Training
    'training': {
        'n_rounds': 10,
        'save_every': 5,          # save checkpoint every N rounds
        'load_checkpoint': None,   # path to load, None = fresh start
        'objective': 'survival',   # 'survival', 'foraging', 'efe_minimize', 'balanced'
        'learning_rate_scale': 1.0,
    },
    # Objectives (weights for fitness scoring)
    'objectives': {
        'survival_weight': 1.0,
        'food_weight': 50.0,
        'efe_weight': -10.0,      # negative = minimize EFE
        'energy_weight': 0.5,
        'exploration_weight': 5.0,
    },
    # Dashboard
    'dashboard': {
        'update_interval_ms': 200,
        'show_spike_raster': True,
        'show_neuromod': True,
        'show_place_cells': True,
    },
}


class TrainingConfig:
    def __init__(self, config_dict=None):
        self.data = json.loads(json.dumps(DEFAULT_CONFIG))
        if config_dict:
            self._deep_update(self.data, config_dict)

    def _deep_update(self, base, updates):
        for k, v in updates.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v

    def get(self, dotpath, default=None):
        """Get nested value: config.get('env.n_food')"""
        keys = dotpath.split('.')
        val = self.data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, dotpath, value):
        """Set nested value: config.set('env.n_food', 20)"""
        keys = dotpath.split('.')
        d = self.data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def to_json(self):
        return json.dumps(self.data, indent=2)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def __repr__(self):
        return f"TrainingConfig({json.dumps(self.data, indent=2)[:200]}...)"


# Training repertoires: environment presets for learning survival skills
REPERTOIRES = {
    'safe_foraging': {
        'name': 'Safe Foraging',
        'description': 'Learn to find food without predator threat',
        'env': {'n_food': 25, 'max_steps': 300, 'predator_ai': 'none'},
        'training': {'n_rounds': 5},
    },
    'predator_evasion': {
        'name': 'Predator Evasion',
        'description': 'Learn to escape from aggressive predator (no food)',
        'env': {'n_food': 0, 'max_steps': 200, 'predator_ai': 'intelligent'},
        'training': {'n_rounds': 5},
    },
    'foraging_under_threat': {
        'name': 'Foraging Under Threat',
        'description': 'Balance foraging and predator avoidance',
        'env': {'n_food': 15, 'max_steps': 500, 'predator_ai': 'intelligent'},
        'training': {'n_rounds': 5},
    },
    'scarce_resources': {
        'name': 'Scarce Resources',
        'description': 'Survive with minimal food — energy management critical',
        'env': {'n_food': 3, 'max_steps': 400, 'predator_ai': 'simple'},
        'training': {'n_rounds': 5},
    },
    'exploration_challenge': {
        'name': 'Exploration Challenge',
        'description': 'Food hidden in corners — must explore to find patches',
        'env': {'n_food': 10, 'max_steps': 500, 'predator_ai': 'none'},
        'training': {'n_rounds': 5},
    },
    'full_survival': {
        'name': 'Full Survival',
        'description': 'Complete challenge — food, predator, obstacles, everything',
        'env': {'n_food': 20, 'max_steps': 1000, 'predator_ai': 'intelligent'},
        'training': {'n_rounds': 10, 'save_every': 3},
    },
    'curriculum': {
        'name': 'Curriculum (Auto)',
        'description': 'Progressive: safe → threat → scarce → full',
        'env': {'n_food': 20, 'max_steps': 300, 'predator_ai': 'none'},
        'training': {'n_rounds': 20},
        'curriculum_stages': [
            {'rounds': 5, 'env': {'predator_ai': 'none', 'n_food': 25}},
            {'rounds': 5, 'env': {'predator_ai': 'simple', 'n_food': 15}},
            {'rounds': 5, 'env': {'predator_ai': 'intelligent', 'n_food': 10}},
            {'rounds': 5, 'env': {'predator_ai': 'intelligent', 'n_food': 5}},
        ],
    },
}
