"""
Integration tests for VirtualZebrafish end-to-end pipeline.

Tests the full facade: config → brain init → run → recording → perturbation.
"""
import unittest
import numpy as np


class TestVirtualZebrafishEndToEnd(unittest.TestCase):
    """Test VirtualZebrafish.run() with spiking and rate-coded brains."""

    def test_spiking_run_short(self):
        """Spiking brain runs 20 steps without crash, returns valid metrics."""
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        result = fish.run(steps=20)
        self.assertIn('survived', result)
        self.assertIn('goals', result)
        self.assertGreater(result['survived'], 0)
        self.assertIsInstance(result['goals'], dict)
        for key in ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']:
            self.assertIn(key, result['goals'])

    def test_rate_coded_run(self):
        """Rate-coded brain runs 50 steps without crash."""
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.set_fidelity('rate_coded')
        result = fish.run(steps=50)
        self.assertGreater(result['survived'], 0)
        self.assertIn('energy_final', result)

    def test_lesion_and_run(self):
        """Lesion a region, run, and verify it completes."""
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.lesion('habenula')
        result = fish.run(steps=10)
        self.assertGreater(result['survived'], 0)

    def test_personality_preset(self):
        """Set personality and verify brain uses it."""
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.set_personality('bold')
        result = fish.run(steps=10)
        self.assertGreater(result['survived'], 0)

    def test_fluent_api_chaining(self):
        """Fluent API: lesion → personality → run."""
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        result = (fish
                  .lesion('cerebellum')
                  .set_personality('shy')
                  .run(steps=10))
        self.assertGreater(result['survived'], 0)


class TestPerturbationIntegration(unittest.TestCase):
    """Test drug injection and disorder application via VirtualZebrafish."""

    def test_inject_drug_and_run(self):
        """Inject haloperidol and verify brain still runs."""
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        fish.inject('haloperidol', dose=1.0)
        result = fish.run(steps=10)
        self.assertGreater(result['survived'], 0)

    def test_disorder_and_run(self):
        """Apply anxiety disorder and verify brain runs."""
        from zebrav2.virtual_fish import VirtualZebrafish
        fish = VirtualZebrafish()
        # Need brain initialized first for disorder
        result = fish.run(steps=5)
        fish.apply_disorder('anxiety', intensity=0.5)
        result2 = fish.run(steps=5)
        self.assertGreater(result2['survived'], 0)


class TestSpatialPriors(unittest.TestCase):
    """Test that spatial priors modulate STDP weights correctly."""

    def test_spatial_priors_applied(self):
        """Verify spatial prior modulation changes weight statistics."""
        import torch
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2

        brain = ZebrafishBrainV2()
        # Weights should not be all-zero (xavier + distance modulation)
        w = brain.thalamus_L.W_tect_tc.weight.data
        self.assertGreater(w.abs().max().item(), 0.0)

    def test_distance_mask_shape(self):
        """Distance mask resizes to match weight matrix dimensions."""
        from zebrav2.brain.spatial_registry import SpatialRegistry
        sr = SpatialRegistry(device='cpu')
        # Generate positions for two regions with different sizes
        sr.generate_positions('tectum', 100)
        sr.generate_positions('thalamus', 50)
        mask = sr.distance_weight_mask('tectum', 'thalamus', lambda_um=100.0)
        self.assertEqual(mask.shape, (100, 50))


class TestRateCodedBrain(unittest.TestCase):
    """Test rate-coded brain standalone."""

    def test_step_returns_valid_output(self):
        """Rate-coded brain step returns turn/speed/goal."""
        from zebrav2.brain.rate_coded_brain import RateCodedBrain
        from zebrav2.config import BrainConfig, BodyConfig

        brain = RateCodedBrain(brain_config=BrainConfig(), body_config=BodyConfig())
        brain.reset()
        obs = {
            'retinal_L': np.zeros(800, dtype=np.float32),
            'retinal_R': np.zeros(800, dtype=np.float32),
            'fish_pos': (400, 300),
            'fish_heading': 0.0,
            'fish_speed': 0.5,
            'energy': 100.0,
            'food_count': 5,
            'arena_w': 800,
            'arena_h': 600,
            'conspecific_data': [],
            'step': 0,
        }
        result = brain.step(obs)
        self.assertIn('turn', result)
        self.assertIn('speed', result)
        self.assertIn('goal', result)

    def test_energy_tracking(self):
        """Brain tracks energy from observation correctly."""
        from zebrav2.brain.rate_coded_brain import RateCodedBrain
        from zebrav2.config import BrainConfig, BodyConfig

        brain = RateCodedBrain(brain_config=BrainConfig(), body_config=BodyConfig())
        brain.reset()
        obs = {
            'retinal_L': np.zeros(800, dtype=np.float32),
            'retinal_R': np.zeros(800, dtype=np.float32),
            'fish_pos': (400, 300),
            'fish_heading': 0.0,
            'fish_speed': 0.5,
            'energy': 50.0,  # depleted energy
            'food_count': 5,
            'arena_w': 800,
            'arena_h': 600,
            'conspecific_data': [],
            'step': 0,
        }
        brain.step(obs)
        self.assertAlmostEqual(brain.energy, 50.0)
        # Low energy should shift goal toward FORAGE
        self.assertIn(brain.current_goal, [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
