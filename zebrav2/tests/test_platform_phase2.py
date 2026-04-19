"""
Tests for Phase 2: config wiring, AbstractBrain, recording, perturbation, batch.
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ═══════════════════════════════════════════════════════════════════════
# 1. AbstractBrain protocol
# ═══════════════════════════════════════════════════════════════════════

class TestAbstractBrain:
    def test_protocol_importable(self):
        from zebrav2.brain.abstract_brain import AbstractBrain
        assert AbstractBrain is not None

    def test_spiking_brain_satisfies_protocol(self):
        """ZebrafishBrainV2 must satisfy AbstractBrain protocol."""
        from zebrav2.brain.abstract_brain import AbstractBrain
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        brain = ZebrafishBrainV2()
        assert hasattr(brain, 'step')
        assert hasattr(brain, 'reset')
        assert hasattr(brain, 'current_goal')
        assert hasattr(brain, 'energy')
        assert hasattr(brain, 'set_region_enabled')


# ═══════════════════════════════════════════════════════════════════════
# 2. Config wiring into brain_v2.py
# ═══════════════════════════════════════════════════════════════════════

class TestConfigWiring:
    def test_brain_accepts_config(self):
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        brain = ZebrafishBrainV2(brain_config=cfg)
        assert brain.cfg is cfg

    def test_brain_accepts_body_config(self):
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        from zebrav2.config.body_config import BodyConfig
        body = BodyConfig()
        brain = ZebrafishBrainV2(body_config=body)
        assert brain.body_cfg is body

    def test_ablation_from_config(self):
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        cfg.ablation.cerebellum = False
        brain = ZebrafishBrainV2(brain_config=cfg)
        assert 'cerebellum' in brain._ablated

    def test_backward_compat_no_config(self):
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        brain = ZebrafishBrainV2()
        assert brain.cfg is not None
        assert 'habenula' in brain._ablated

    def test_personality_from_config(self):
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        from zebrav2.config.brain_config import BrainConfig
        cfg = BrainConfig()
        cfg.personality = 'bold'
        brain = ZebrafishBrainV2(brain_config=cfg)
        assert brain.personality['name'] == 'bold'

    def test_personality_param_overrides_config(self):
        """Explicit personality parameter overrides config."""
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        from zebrav2.config.brain_config import BrainConfig
        from zebrav2.brain.personality import get_personality
        cfg = BrainConfig()
        cfg.personality = 'bold'
        brain = ZebrafishBrainV2(brain_config=cfg, personality=get_personality('shy'))
        assert brain.personality['name'] == 'shy'


# ═══════════════════════════════════════════════════════════════════════
# 3. Recording system
# ═══════════════════════════════════════════════════════════════════════

class TestRecording:
    def test_recorder_start_stop(self):
        from zebrav2.recording import Recorder
        rec = Recorder()
        rec.start()
        assert rec._active
        rec.stop()
        assert not rec._active

    def test_record_step(self):
        from zebrav2.recording import Recorder
        rec = Recorder()
        rec.start()
        rec.record_step({'turn': 0.1, 'speed': 0.5, 'goal': 0,
                         'DA': 0.8, 'NA': 0.3}, pos=(100, 200), energy=80)
        assert len(rec.steps) == 1
        assert rec.steps[0].da == 0.8
        assert rec.steps[0].x == 100

    def test_record_event(self):
        from zebrav2.recording import Recorder
        rec = Recorder()
        rec.start()
        rec.record_event('food_eaten', step=5, food_id=3)
        assert len(rec.events) == 1
        assert rec.events[0]['type'] == 'food_eaten'

    def test_inactive_recorder_ignores(self):
        from zebrav2.recording import Recorder
        rec = Recorder()
        rec.record_step({'turn': 0.1}, pos=(0, 0), energy=0)
        assert len(rec.steps) == 0

    def test_goal_counts(self):
        from zebrav2.recording import Recorder
        rec = Recorder()
        rec.start()
        for goal in [0, 0, 1, 2, 2, 2, 3]:
            rec.record_step({'turn': 0, 'speed': 0, 'goal': goal}, pos=(0, 0), energy=0)
        counts = rec.get_goal_counts()
        assert counts['FORAGE'] == 2
        assert counts['FLEE'] == 1
        assert counts['EXPLORE'] == 3
        assert counts['SOCIAL'] == 1

    def test_neuromod_timeseries(self):
        from zebrav2.recording import Recorder
        rec = Recorder()
        rec.start()
        rec.record_step({'turn': 0, 'speed': 0, 'goal': 0, 'DA': 0.5, 'NA': 0.3,
                         '5HT': 0.7, 'ACh': 0.4}, pos=(0, 0), energy=0)
        ts = rec.get_neuromod_timeseries()
        assert ts['DA'] == [0.5]
        assert ts['5HT'] == [0.7]

    def test_to_dict(self):
        from zebrav2.recording import Recorder
        rec = Recorder()
        rec.start()
        rec.record_step({'turn': 0.1, 'speed': 0.5, 'goal': 2}, pos=(50, 60), energy=90)
        d = rec.to_dict()
        assert 'steps' in d
        assert 'events' in d
        assert 'summary' in d
        assert d['summary']['n_steps'] == 1

    def test_save_json(self):
        from zebrav2.recording import Recorder
        rec = Recorder()
        rec.start()
        rec.record_step({'turn': 0, 'speed': 1, 'goal': 0}, pos=(10, 20), energy=100)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            rec.save(path)
            with open(path) as f:
                data = json.load(f)
            assert data['summary']['n_steps'] == 1
        finally:
            os.unlink(path)

    def test_save_csv(self):
        from zebrav2.recording import Recorder
        rec = Recorder()
        rec.start()
        rec.record_step({'turn': 0, 'speed': 1, 'goal': 0}, pos=(10, 20), energy=100)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            rec.to_csv(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # header + 1 row
            assert 'turn' in lines[0]
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════
# 4. Perturbation API
# ═══════════════════════════════════════════════════════════════════════

class TestPerturbation:
    def test_lesion(self):
        from zebrav2.perturbation import PerturbationManager
        pm = PerturbationManager()
        pm.lesion('habenula')
        assert 'habenula' in pm.lesions

    def test_inject_from_library(self):
        from zebrav2.perturbation import PerturbationManager
        pm = PerturbationManager()
        pm.inject('haloperidol')
        assert len(pm.drugs) == 1
        assert pm.drugs[0].name == 'Haloperidol'

    def test_inject_unknown_drug(self):
        from zebrav2.perturbation import PerturbationManager
        pm = PerturbationManager()
        with pytest.raises(ValueError, match="Unknown drug"):
            pm.inject('magic_potion')

    def test_inject_custom_dose(self):
        from zebrav2.perturbation import PerturbationManager
        pm = PerturbationManager()
        pm.inject('haloperidol', dose=0.3)
        assert pm.drugs[0].dose == 0.3

    def test_da_antagonist_multiplier(self):
        from zebrav2.perturbation import PerturbationManager
        pm = PerturbationManager()
        pm.inject('haloperidol')  # DA antagonist, dose=0.7
        mults = pm.get_neuromod_multipliers()
        assert mults['DA'] == pytest.approx(0.3, abs=0.01)
        assert mults['NA'] == 1.0  # unaffected

    def test_da_agonist_multiplier(self):
        from zebrav2.perturbation import PerturbationManager
        pm = PerturbationManager()
        pm.inject('amphetamine')  # DA agonist, dose=0.8
        mults = pm.get_neuromod_multipliers()
        assert mults['DA'] == pytest.approx(1.8, abs=0.01)

    def test_stimulate(self):
        from zebrav2.perturbation import PerturbationManager
        pm = PerturbationManager()
        pm.stimulate('amygdala', intensity=0.8, duration=20)
        assert len(pm.stimulations) == 1
        active = pm.get_active_stimulations()
        assert len(active) == 1
        # After duration expires
        for _ in range(25):
            pm.step()
        active = pm.get_active_stimulations()
        assert len(active) == 0

    def test_drug_onset_delay(self):
        from zebrav2.perturbation import DrugEffect
        drug = DrugEffect(name='Delayed', target_neuromod='DA',
                          effect='antagonist', dose=0.5, onset_steps=10)
        assert drug.compute_multiplier(step=5) == 1.0  # before onset
        assert drug.compute_multiplier(step=15) == 0.5  # after onset

    def test_summary(self):
        from zebrav2.perturbation import PerturbationManager
        pm = PerturbationManager()
        pm.lesion('cerebellum')
        pm.inject('fluoxetine')
        s = pm.summary()
        assert 'cerebellum' in s['lesions']
        assert len(s['drugs']) == 1

    def test_virtual_fish_inject(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.inject('haloperidol').inject('fluoxetine')
        assert len(fish.perturbations.drugs) == 2

    def test_virtual_fish_stimulate(self):
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.stimulate('amygdala', duration=10)
        assert len(fish.perturbations.stimulations) == 1


# ═══════════════════════════════════════════════════════════════════════
# 5. Batch experiment runner
# ═══════════════════════════════════════════════════════════════════════

class TestBatch:
    def test_parameter_sweep_combinations(self):
        from zebrav2.batch import BatchRunner, ParameterSweep
        sweep1 = ParameterSweep('brain.efe.forage_offset', [0.1, 0.2])
        sweep2 = ParameterSweep('brain.efe.flee_offset', [0.3, 0.4])
        runner = BatchRunner(sweeps=[sweep1, sweep2], n_seeds=2)
        combos = runner._get_combinations()
        assert len(combos) == 4  # 2 x 2
        assert {'brain.efe.forage_offset': 0.1, 'brain.efe.flee_offset': 0.3} in combos

    def test_make_configs(self):
        from zebrav2.batch import BatchRunner, ParameterSweep
        runner = BatchRunner()
        brain, body, world = runner._make_configs(
            {'brain.efe.forage_offset': 0.30})
        assert brain.efe.forage_offset == 0.30

    def test_make_configs_body(self):
        from zebrav2.batch import BatchRunner
        runner = BatchRunner()
        brain, body, world = runner._make_configs(
            {'body.motor.speed_flee': 2.5})
        assert body.motor.speed_flee == 2.5

    def test_make_configs_world(self):
        from zebrav2.batch import BatchRunner
        runner = BatchRunner()
        brain, body, world = runner._make_configs(
            {'world.arena.width': 400})
        assert world.arena.width == 400

    def test_lesion_in_batch(self):
        from zebrav2.batch import BatchRunner
        runner = BatchRunner(lesions=['cerebellum', 'amygdala'])
        brain, _, _ = runner._make_configs({})
        assert brain.ablation.cerebellum is False
        assert brain.ablation.amygdala is False

    def test_no_sweep_single_combo(self):
        from zebrav2.batch import BatchRunner
        runner = BatchRunner()
        combos = runner._get_combinations()
        assert len(combos) == 1
        assert combos[0] == {}


# ═══════════════════════════════════════════════════════════════════════
# 6. Drug library completeness
# ═══════════════════════════════════════════════════════════════════════

class TestDrugLibrary:
    def test_all_drugs_have_valid_targets(self):
        from zebrav2.perturbation import DRUG_LIBRARY
        valid_targets = {'DA', 'NA', '5HT', 'ACh'}
        for name, drug in DRUG_LIBRARY.items():
            assert drug.target_neuromod in valid_targets, f"{name} has invalid target"

    def test_all_drugs_have_valid_effects(self):
        from zebrav2.perturbation import DRUG_LIBRARY
        valid_effects = {'agonist', 'antagonist', 'reuptake_inhibitor'}
        for name, drug in DRUG_LIBRARY.items():
            assert drug.effect in valid_effects, f"{name} has invalid effect"

    def test_drug_doses_in_range(self):
        from zebrav2.perturbation import DRUG_LIBRARY
        for name, drug in DRUG_LIBRARY.items():
            assert 0.0 <= drug.dose <= 1.0, f"{name} dose out of range"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
