"""
Tests for Phase 3: module registry, trainer config wiring, personality names.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ═══════════════════════════════════════════════════════════════════════
# 1. Module registry
# ═══════════════════════════════════════════════════════════════════════

class TestModuleRegistry:
    def test_importable(self):
        from zebrav2.brain.module_registry import BRAIN_MODULES
        assert len(BRAIN_MODULES) > 30

    def test_required_modules(self):
        from zebrav2.brain.module_registry import get_required_modules
        required = get_required_modules()
        assert 'retina' in required
        assert 'tectum' in required
        assert 'thalamus' in required
        assert 'pallium' in required
        assert 'basal_ganglia' in required
        assert 'amygdala' in required

    def test_optional_modules(self):
        from zebrav2.brain.module_registry import get_optional_modules
        optional = get_optional_modules()
        assert 'cerebellum' in optional
        assert 'habenula' in optional
        assert 'insula' in optional
        assert 'lateral_line' in optional

    def test_enabled_modules(self):
        from zebrav2.brain.module_registry import get_enabled_modules
        ablated = {'habenula', 'insula'}
        enabled = get_enabled_modules(ablated)
        assert 'habenula' not in enabled
        assert 'insula' not in enabled
        assert 'cerebellum' in enabled
        assert 'retina' in enabled

    def test_modules_by_group(self):
        from zebrav2.brain.module_registry import get_modules_by_group
        sensory = get_modules_by_group('sensory')
        assert 'retina' in sensory
        assert 'lateral_line' in sensory

    def test_region_groups(self):
        from zebrav2.brain.module_registry import list_region_groups
        groups = list_region_groups()
        assert 'midbrain' in groups
        assert 'telencephalon' in groups
        assert 'sensory' in groups

    def test_summary(self):
        from zebrav2.brain.module_registry import summary
        s = summary()
        assert s['total_modules'] > 30
        assert s['required'] >= 8
        assert s['optional'] > 20
        assert 'region_groups' in s

    def test_all_modules_have_metadata(self):
        from zebrav2.brain.module_registry import BRAIN_MODULES
        for name, entry in BRAIN_MODULES.items():
            assert entry.name == name
            assert entry.class_name, f"{name} missing class_name"
            assert entry.module_path, f"{name} missing module_path"
            assert entry.role, f"{name} missing role"
            assert entry.region_group, f"{name} missing region_group"

    def test_ablation_config_matches_registry(self):
        """AblationConfig fields should match optional registry entries."""
        from zebrav2.config.brain_config import AblationConfig
        from zebrav2.brain.module_registry import get_optional_modules
        from dataclasses import fields

        ablation_fields = {f.name for f in fields(AblationConfig)}
        optional_names = set(get_optional_modules().keys())
        # AblationConfig may have extra fields not in registry (future-proof)
        # but all optional registry modules should be in AblationConfig
        for name in optional_names:
            assert name in ablation_fields, \
                f"Module '{name}' is optional in registry but missing from AblationConfig"


# ═══════════════════════════════════════════════════════════════════════
# 2. Trainer config wiring
# ═══════════════════════════════════════════════════════════════════════

class TestTrainerConfigWiring:
    def test_trainer_accepts_brain_config(self):
        from zebrav2.engine.trainer import TrainingEngine
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        engine = TrainingEngine(brain_config=cfg)
        assert engine.brain_config is cfg

    def test_trainer_accepts_body_config(self):
        from zebrav2.engine.trainer import TrainingEngine
        from zebrav2.config.body_config import BodyConfig
        cfg = BodyConfig()
        engine = TrainingEngine(body_config=cfg)
        assert engine.body_config is cfg

    def test_trainer_accepts_world_config(self):
        from zebrav2.engine.trainer import TrainingEngine
        from zebrav2.config.world_config import WorldConfig
        cfg = WorldConfig()
        engine = TrainingEngine(world_config=cfg)
        assert engine.world_config is cfg

    def test_trainer_passes_configs_to_brain(self):
        from zebrav2.engine.trainer import TrainingEngine
        from zebrav2.config.brain_config import BrainConfig
        from zebrav2.config.body_config import BodyConfig
        brain_cfg = BrainConfig()
        body_cfg = BodyConfig()
        engine = TrainingEngine(brain_config=brain_cfg, body_config=body_cfg)
        brain = engine._create_brain()
        assert brain.cfg is brain_cfg
        assert brain.body_cfg is body_cfg

    def test_trainer_backward_compat(self):
        """Trainer without platform configs still works."""
        from zebrav2.engine.trainer import TrainingEngine
        engine = TrainingEngine()
        assert engine.brain_config is None
        brain = engine._create_brain()
        assert brain.cfg is not None  # gets default BrainConfig


# ═══════════════════════════════════════════════════════════════════════
# 3. Personality name key
# ═══════════════════════════════════════════════════════════════════════

class TestPersonalityNames:
    def test_all_personalities_have_name(self):
        from zebrav2.brain.personality import PERSONALITIES
        for key, profile in PERSONALITIES.items():
            assert 'name' in profile, f"Personality '{key}' missing 'name' key"
            assert profile['name'] == key

    def test_get_personality_has_name(self):
        from zebrav2.brain.personality import get_personality
        for name in ['bold', 'shy', 'explorer', 'social', 'default']:
            p = get_personality(name)
            assert p['name'] == name

    def test_random_personality_has_name(self):
        from zebrav2.brain.personality import random_personality
        p = random_personality()
        assert 'name' not in p or p.get('name') is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
