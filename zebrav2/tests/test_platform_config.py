"""
Tests for the Phase 1 platform skeleton:
    - Config dataclasses (BrainConfig, BodyConfig, WorldConfig)
    - JSON round-trip serialization
    - VirtualZebrafish facade
    - CLI entry point
    - Ablation API
"""
import json
import os
import sys
import tempfile

import pytest

# ── ensure project root is importable ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ═══════════════════════════════════════════════════════════════════════
# 1. Config dataclass construction & defaults
# ═══════════════════════════════════════════════════════════════════════

class TestBrainConfig:
    def test_default_construction(self):
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        assert cfg.efe.forage_offset == 0.15
        assert cfg.efe.flee_offset == 0.35
        assert cfg.personality == 'default'
        assert cfg.fidelity == 'spiking'

    def test_efe_coefficients_match_brain_v2(self):
        """EFE defaults must match current brain_v2.py hardcoded values."""
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        assert cfg.efe.forage_food_weight == -0.8
        assert cfg.efe.flee_enemy_weight == -0.8
        assert cfg.efe.starvation_threshold == 0.75
        assert cfg.efe.critical_energy == 25.0
        assert cfg.efe.world_model_efe_scale == 0.15

    def test_stdp_pathway_defaults(self):
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        assert cfg.plasticity.tect_tc.a_plus == 0.002
        assert cfg.plasticity.tect_tc.a_minus == 0.001
        assert cfg.plasticity.pal_d.a_plus == 0.001
        assert cfg.plasticity.pal_d.consolidation_eta == 5e-5

    def test_ablation_defaults(self):
        """Habenula and insula ablated by default (matching brain_v2.py)."""
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        assert cfg.ablation.habenula is False
        assert cfg.ablation.insula is False
        assert cfg.ablation.cerebellum is True
        assert cfg.ablation.amygdala is True

    def test_get_ablated_set(self):
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        ablated = cfg.get_ablated_set()
        assert 'habenula' in ablated
        assert 'insula' in ablated
        assert 'cerebellum' not in ablated

    def test_goal_selection_defaults(self):
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        assert cfg.goal_selection.new_goal_persistence == 8
        assert cfg.goal_selection.force_explore_duration == 15
        assert cfg.goal_selection.no_food_timeout == 30

    def test_threat_defaults(self):
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        assert cfg.threat.enemy_pixel_normalization == 15.0
        assert cfg.threat.tect_threat_scale == 5.0
        assert cfg.threat.patrol_threshold_distance == 150.0


class TestBodyConfig:
    def test_default_construction(self):
        from zebrav2.config.body_config import BodyConfig
        cfg = BodyConfig()
        assert cfg.fish_speed_base == 3.0
        assert cfg.eat_radius == 35.0
        assert cfg.metabolism.energy_start == 100.0

    def test_sensory_defaults(self):
        from zebrav2.config.body_config import BodyConfig
        cfg = BodyConfig()
        assert cfg.sensory.retina.n_per_type == 400
        assert cfg.sensory.lateral_line.sn_range == 200.0
        assert cfg.sensory.olfaction.lambda_food == 70.0
        assert cfg.sensory.vestibular.n_neurons == 6
        assert cfg.sensory.proprioception.n_neurons == 8

    def test_color_vision_spectra(self):
        from zebrav2.config.body_config import BodyConfig
        cfg = BodyConfig()
        assert len(cfg.sensory.color_vision.food_spectrum) == 4
        assert cfg.sensory.color_vision.food_spectrum[2] == 0.8  # green peak

    def test_motor_defaults(self):
        from zebrav2.config.body_config import BodyConfig
        cfg = BodyConfig()
        assert cfg.motor.speed_flee == 1.5
        assert cfg.motor.speed_forage == 1.0
        assert cfg.motor.turn_max_flee == 0.45
        assert cfg.motor.wall_margin == 120.0

    def test_cpg_defaults(self):
        from zebrav2.config.body_config import BodyConfig
        cfg = BodyConfig()
        cpg = cfg.motor.cpg
        assert cpg.n_v2a == 16
        assert cpg.n_mn == 12
        assert cpg.w_v0d_cross == 0.9  # half-centre coupling
        assert cpg.noise == 0.15

    def test_metabolism_defaults(self):
        from zebrav2.config.body_config import BodyConfig
        cfg = BodyConfig()
        met = cfg.metabolism
        assert met.energy_max == 100.0
        assert met.energy_per_food == 8.0
        assert met.min_motility == 0.15
        assert met.critical_energy == 25.0


class TestWorldConfig:
    def test_default_construction(self):
        from zebrav2.config.world_config import WorldConfig
        cfg = WorldConfig()
        assert cfg.arena.width == 800
        assert cfg.arena.height == 600
        assert cfg.max_steps == 500

    def test_arena_defaults(self):
        from zebrav2.config.world_config import WorldConfig
        cfg = WorldConfig()
        assert cfg.arena.n_food == 20
        assert cfg.arena.food_respawn is True
        assert cfg.arena.rocks_enabled is True

    def test_predator_defaults(self):
        from zebrav2.config.world_config import WorldConfig
        cfg = WorldConfig()
        pred = cfg.predator
        assert pred.enabled is True
        assert pred.ai_mode == 'intelligent'
        assert pred.speed_hunt == 4.0
        assert pred.catch_radius == 20.0
        assert pred.navigation_noise == 8.0

    def test_vision_physics(self):
        from zebrav2.config.world_config import WorldConfig
        cfg = WorldConfig()
        assert cfg.fov_degrees == 200.0
        assert cfg.max_vision_distance == 500.0
        assert cfg.type_code_predator == 0.5
        assert cfg.type_code_food == 0.8


# ═══════════════════════════════════════════════════════════════════════
# 2. JSON round-trip serialization
# ═══════════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_brain_config_json_roundtrip(self):
        from zebrav2.config.brain_config import BrainConfig
        original = BrainConfig()
        original.efe.forage_offset = 0.25
        original.ablation.cerebellum = False

        restored = BrainConfig.from_json(original.to_json())
        assert restored.efe.forage_offset == 0.25
        assert restored.ablation.cerebellum is False
        assert restored.plasticity.tect_tc.a_plus == 0.002

    def test_body_config_json_roundtrip(self):
        from zebrav2.config.body_config import BodyConfig
        original = BodyConfig()
        original.motor.speed_flee = 2.0
        original.sensory.retina.n_per_type = 200

        restored = BodyConfig.from_json(original.to_json())
        assert restored.motor.speed_flee == 2.0
        assert restored.sensory.retina.n_per_type == 200

    def test_world_config_json_roundtrip(self):
        from zebrav2.config.world_config import WorldConfig
        original = WorldConfig()
        original.arena.width = 400
        original.predator.speed_hunt = 6.0

        restored = WorldConfig.from_json(original.to_json())
        assert restored.arena.width == 400
        assert restored.predator.speed_hunt == 6.0

    def test_brain_config_file_roundtrip(self):
        from zebrav2.config.brain_config import BrainConfig
        original = BrainConfig()
        original.efe.flee_offset = 0.5

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            original.save(path)
            restored = BrainConfig.load(path)
            assert restored.efe.flee_offset == 0.5
        finally:
            os.unlink(path)

    def test_body_config_file_roundtrip(self):
        from zebrav2.config.body_config import BodyConfig
        original = BodyConfig()
        original.fish_speed_base = 5.0

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            original.save(path)
            restored = BodyConfig.load(path)
            assert restored.fish_speed_base == 5.0
        finally:
            os.unlink(path)

    def test_world_config_file_roundtrip(self):
        from zebrav2.config.world_config import WorldConfig
        original = WorldConfig()
        original.max_steps = 2000

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            original.save(path)
            restored = WorldConfig.load(path)
            assert restored.max_steps == 2000
        finally:
            os.unlink(path)

    def test_all_configs_json_valid(self):
        """All configs must produce valid JSON."""
        from zebrav2.config import BrainConfig, BodyConfig, WorldConfig
        for Cls in (BrainConfig, BodyConfig, WorldConfig):
            cfg = Cls()
            j = cfg.to_json()
            parsed = json.loads(j)
            assert isinstance(parsed, dict)


# ═══════════════════════════════════════════════════════════════════════
# 3. VirtualZebrafish facade
# ═══════════════════════════════════════════════════════════════════════

class TestVirtualZebrafish:
    def test_default_construction(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        assert fish.brain_config.fidelity == 'spiking'
        assert fish.world_config.arena.width == 800

    def test_lesion_api(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.lesion('cerebellum')
        assert fish.brain_config.ablation.cerebellum is False
        assert 'cerebellum' in fish.brain_config.get_ablated_set()

    def test_enable_api(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        # habenula is ablated by default
        assert fish.brain_config.ablation.habenula is False
        fish.enable('habenula')
        assert fish.brain_config.ablation.habenula is True

    def test_lesion_invalid_region(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        with pytest.raises(ValueError, match="Unknown region"):
            fish.lesion('nonexistent_region')

    def test_set_personality(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.set_personality('bold')
        assert fish.brain_config.personality == 'bold'

    def test_set_fidelity(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.set_fidelity('rate_coded')
        assert fish.brain_config.fidelity == 'rate_coded'

    def test_set_fidelity_invalid(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        with pytest.raises(ValueError, match="Unknown fidelity"):
            fish.set_fidelity('turbo')

    def test_chaining_api(self):
        """Fluent API: methods return self for chaining."""
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = (VirtualZebrafish()
                .lesion('cerebellum')
                .lesion('amygdala')
                .set_personality('shy'))
        assert fish.brain_config.ablation.cerebellum is False
        assert fish.brain_config.ablation.amygdala is False
        assert fish.brain_config.personality == 'shy'

    def test_load_pretrained(self):
        """VirtualZebrafish.load('pretrained') finds latest checkpoint."""
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish.load('pretrained')
        # Should have found a checkpoint (round 95 exists)
        if fish._checkpoint_path:
            assert fish._checkpoint_path.endswith('.pt')

    def test_export_import_config(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.lesion('cerebellum')
        fish.brain_config.efe.forage_offset = 0.30

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            fish.export_config(path)
            fish2 = VirtualZebrafish.from_config(path)
            assert fish2.brain_config.ablation.cerebellum is False
            assert fish2.brain_config.efe.forage_offset == 0.30
        finally:
            os.unlink(path)

    def test_repr(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        r = repr(fish)
        assert 'spiking' in r
        assert '800x600' in r


# ═══════════════════════════════════════════════════════════════════════
# 4. CLI entry point
# ═══════════════════════════════════════════════════════════════════════

class TestCLI:
    def test_cli_import(self):
        from zebrav2.cli import main
        assert callable(main)

    def test_info_command(self, capsys):
        from zebrav2.cli import cmd_info
        import argparse
        args = argparse.Namespace()
        cmd_info(args)
        output = capsys.readouterr().out
        assert 'Virtual Zebrafish' in output
        assert 'v2.0.0' in output
        assert 'ablated' in output

    def test_export_command(self, capsys):
        from zebrav2.cli import cmd_export
        import argparse
        args = argparse.Namespace(output=None)
        cmd_export(args)
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert 'world' in parsed
        assert 'body' in parsed
        assert 'brain' in parsed

    def test_export_to_file(self):
        from zebrav2.cli import cmd_export
        import argparse
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            args = argparse.Namespace(output=path)
            cmd_export(args)
            with open(path) as f:
                data = json.load(f)
            assert 'brain' in data
            assert data['brain']['efe']['forage_offset'] == 0.15
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════
# 5. Integration with existing TrainingConfig
# ═══════════════════════════════════════════════════════════════════════

class TestConfigIntegration:
    def test_training_config_still_works(self):
        """Existing TrainingConfig must remain functional."""
        from zebrav2.engine.config import TrainingConfig, REPERTOIRES
        cfg = TrainingConfig()
        assert cfg.get('env.n_food') == 20
        assert cfg.get('fish.eat_radius') == 35
        assert 'full_survival' in REPERTOIRES

    def test_world_config_matches_training_config(self):
        """WorldConfig defaults match TrainingConfig defaults."""
        from zebrav2.config.world_config import WorldConfig
        from zebrav2.engine.config import TrainingConfig

        wcfg = WorldConfig()
        tcfg = TrainingConfig()

        assert wcfg.arena.width == tcfg.get('env.arena_w')
        assert wcfg.arena.height == tcfg.get('env.arena_h')
        assert wcfg.arena.n_food == tcfg.get('env.n_food')
        assert wcfg.max_steps == tcfg.get('env.max_steps')

    def test_body_config_matches_training_config(self):
        """BodyConfig defaults match TrainingConfig fish section."""
        from zebrav2.config.body_config import BodyConfig
        from zebrav2.engine.config import TrainingConfig

        bcfg = BodyConfig()
        tcfg = TrainingConfig()

        assert bcfg.metabolism.energy_start == tcfg.get('fish.energy_start')
        assert bcfg.eat_radius == tcfg.get('fish.eat_radius')


# ═══════════════════════════════════════════════════════════════════════
# 6. Parameter completeness checks
# ═══════════════════════════════════════════════════════════════════════

class TestParameterCompleteness:
    def test_brain_config_has_efe_subsystem(self):
        from zebrav2.config.brain_config import BrainConfig, EFEConfig
        cfg = BrainConfig()
        assert isinstance(cfg.efe, EFEConfig)
        # Check key coefficients exist
        assert hasattr(cfg.efe, 'forage_offset')
        assert hasattr(cfg.efe, 'flee_offset')
        assert hasattr(cfg.efe, 'explore_offset')
        assert hasattr(cfg.efe, 'social_offset')

    def test_brain_config_has_all_subsystems(self):
        from zebrav2.config.brain_config import (
            BrainConfig, EFEConfig, GoalSelectionConfig,
            NeuromodConfig, PlasticityConfig, ThreatConfig,
            NoveltyConfig, AblationConfig)
        cfg = BrainConfig()
        assert isinstance(cfg.efe, EFEConfig)
        assert isinstance(cfg.goal_selection, GoalSelectionConfig)
        assert isinstance(cfg.neuromod, NeuromodConfig)
        assert isinstance(cfg.plasticity, PlasticityConfig)
        assert isinstance(cfg.threat, ThreatConfig)
        assert isinstance(cfg.novelty, NoveltyConfig)
        assert isinstance(cfg.ablation, AblationConfig)

    def test_body_config_has_all_sensory_channels(self):
        from zebrav2.config.body_config import BodyConfig
        cfg = BodyConfig()
        assert hasattr(cfg.sensory, 'retina')
        assert hasattr(cfg.sensory, 'lateral_line')
        assert hasattr(cfg.sensory, 'olfaction')
        assert hasattr(cfg.sensory, 'vestibular')
        assert hasattr(cfg.sensory, 'proprioception')
        assert hasattr(cfg.sensory, 'color_vision')

    def test_ablation_covers_all_optional_modules(self):
        """AblationConfig must cover all optional brain modules."""
        from zebrav2.config.brain_config import AblationConfig
        from dataclasses import fields
        ablation_fields = {f.name for f in fields(AblationConfig)}
        expected = {
            'habenula', 'insula', 'cerebellum', 'amygdala', 'place_cells',
            'working_memory', 'predictive_net', 'habit_network',
            'circadian', 'sleep_wake', 'lateral_line', 'olfaction',
            'vestibular', 'proprioception', 'color_vision', 'shoaling',
            'binocular_depth', 'social_memory', 'hpa_axis', 'oxytocin',
            'meta_goal', 'vae_world_model', 'geographic_model',
        }
        assert expected.issubset(ablation_fields), \
            f"Missing: {expected - ablation_fields}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
