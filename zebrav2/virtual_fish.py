"""
VirtualZebrafish: high-level facade for the downloadable virtual zebrafish.

Provides the researcher-facing API:
    fish = VirtualZebrafish.load("pretrained")
    fish.lesion("habenula")
    results = fish.run(steps=1000)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from zebrav2.config import WorldConfig, BodyConfig, BrainConfig
from zebrav2.recording import Recorder
from zebrav2.perturbation import PerturbationManager, DrugEffect, DRUG_LIBRARY


class VirtualZebrafish:
    """A configurable virtual zebrafish with full brain, body, and world."""

    def __init__(
        self,
        world: WorldConfig | None = None,
        body: BodyConfig | None = None,
        brain: BrainConfig | None = None,
    ):
        self.world_config = world or WorldConfig()
        self.body_config = body or BodyConfig()
        self.brain_config = brain or BrainConfig()

        self._brain = None     # lazy: ZebrafishBrainV2 or RateCodedBrain
        self._env = None       # lazy: MultiAgentEnvV2
        self._checkpoint_path: str | None = None
        self.recorder = Recorder()
        self.perturbations = PerturbationManager()

    # ------------------------------------------------------------------
    # Configuration API
    # ------------------------------------------------------------------

    def lesion(self, region: str) -> VirtualZebrafish:
        """Disable a brain region (ablation study)."""
        if hasattr(self.brain_config.ablation, region):
            setattr(self.brain_config.ablation, region, False)
        else:
            raise ValueError(
                f"Unknown region '{region}'. Available: "
                f"{list(asdict(self.brain_config.ablation).keys())}")
        self._brain = None  # force rebuild
        return self

    def enable(self, region: str) -> VirtualZebrafish:
        """Re-enable a previously lesioned brain region."""
        if hasattr(self.brain_config.ablation, region):
            setattr(self.brain_config.ablation, region, True)
        else:
            raise ValueError(f"Unknown region '{region}'.")
        self._brain = None
        return self

    def set_personality(self, preset: str) -> VirtualZebrafish:
        """Set personality preset: 'default', 'bold', 'shy', 'explorer', 'social'."""
        self.brain_config.personality = preset
        self._brain = None
        return self

    def set_fidelity(self, level: str) -> VirtualZebrafish:
        """Set brain fidelity: 'spiking', 'rate_coded', 'minimal'."""
        if level not in ('spiking', 'rate_coded', 'minimal'):
            raise ValueError(f"Unknown fidelity '{level}'.")
        self.brain_config.fidelity = level
        self._brain = None
        return self

    # ------------------------------------------------------------------
    # Perturbation API
    # ------------------------------------------------------------------

    def inject(self, drug: str | DrugEffect, dose: float | None = None) -> VirtualZebrafish:
        """Administer a drug. Use name from library or custom DrugEffect.

        Available drugs: haloperidol, fluoxetine, scopolamine,
                        amphetamine, propranolol, buspirone
        """
        self.perturbations.inject(drug, dose)
        return self

    def stimulate(self, region: str, pattern: str = 'pulse',
                  intensity: float = 1.0, duration: int = 10) -> VirtualZebrafish:
        """Optogenetic-style stimulation of a brain region."""
        self.perturbations.stimulate(region, pattern, intensity, duration)
        return self

    def apply_disorder(self, name: str, intensity: float = 1.0) -> VirtualZebrafish:
        """Apply a disorder phenotype to the brain.

        Available: wildtype, hypodopamine, asd, schizophrenia,
                  anxiety, depression, adhd, ptsd
        """
        from zebrav2.brain.disorder import apply_disorder
        brain = self._get_brain()
        if brain is not None:
            apply_disorder(brain, name, intensity)
        return self

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str) -> VirtualZebrafish:
        """Load pre-trained weights from a checkpoint file."""
        self._checkpoint_path = path
        self._brain = None  # force rebuild with new weights
        return self

    @classmethod
    def load(cls, name_or_path: str, **overrides: Any) -> VirtualZebrafish:
        """Load a pre-configured virtual zebrafish.

        Args:
            name_or_path: 'pretrained' for latest checkpoint, or path to .pt file
            **overrides: override any config parameter
        """
        fish = cls()

        if name_or_path == 'pretrained':
            ckpt_dir = os.path.join(
                os.path.dirname(__file__), 'checkpoints')
            if os.path.isdir(ckpt_dir):
                pts = sorted(f for f in os.listdir(ckpt_dir)
                             if f.endswith('.pt'))
                if pts:
                    fish._checkpoint_path = os.path.join(ckpt_dir, pts[-1])
        elif os.path.exists(name_or_path):
            fish._checkpoint_path = name_or_path

        return fish

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self, steps: int | None = None) -> dict[str, Any]:
        """Run an episode and return behavioral metrics.

        Returns:
            dict with keys: survived (int), food_eaten (int),
            energy_final (float), goals (dict of goal counts),
            trajectory (list of (x, y) if recording enabled)
        """
        n_steps = steps or self.world_config.max_steps
        brain = self._get_brain()

        survived = 0
        food_eaten = 0
        goal_counts = {'FORAGE': 0, 'FLEE': 0, 'EXPLORE': 0, 'SOCIAL': 0}
        trajectory: list[tuple[float, float]] = []

        # Use rate-coded pipeline for fast execution
        if self.brain_config.fidelity == 'rate_coded':
            return self._run_rate_coded(n_steps)

        # Spiking pipeline (full brain_v2)
        from zebrav2.brain.brain_v2 import GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, GOAL_SOCIAL
        goal_names = {GOAL_FORAGE: 'FORAGE', GOAL_FLEE: 'FLEE',
                      GOAL_EXPLORE: 'EXPLORE', GOAL_SOCIAL: 'SOCIAL'}

        brain.reset()

        for step in range(n_steps):
            # Minimal observation (no rendering)
            obs = self._make_observation(step)
            result = brain.step(obs)

            survived += 1
            goal_name = goal_names.get(brain.current_goal, 'EXPLORE')
            goal_counts[goal_name] += 1

            if hasattr(brain, 'energy'):
                if brain.energy <= 0:
                    break

        return {
            'survived': survived,
            'food_eaten': food_eaten,
            'energy_final': float(getattr(brain, 'energy', 0)),
            'goals': goal_counts,
        }

    def _run_rate_coded(self, n_steps: int) -> dict[str, Any]:
        """Run using the rate-coded brain pipeline."""
        from zebrav2.brain.rate_coded_brain import GOAL_NAMES
        brain = self._get_brain()
        brain.reset()

        survived = 0
        goal_counts = {'FORAGE': 0, 'FLEE': 0, 'EXPLORE': 0, 'SOCIAL': 0}

        for step in range(n_steps):
            obs = self._make_observation(step)
            result = brain.step(obs)
            survived += 1
            goal_name = GOAL_NAMES[brain.current_goal]
            goal_counts[goal_name] += 1
            if brain.energy <= 0:
                break

        return {
            'survived': survived,
            'food_eaten': 0,
            'energy_final': float(brain.energy),
            'goals': goal_counts,
        }

    def _make_observation(self, step: int) -> dict:
        """Create a minimal observation dict for brain.step()."""
        import numpy as np
        w = self.world_config.arena.width
        h = self.world_config.arena.height
        return {
            'retinal_L': np.zeros(800, dtype=np.float32),
            'retinal_R': np.zeros(800, dtype=np.float32),
            'fish_pos': (w / 2, h / 2),
            'fish_heading': 0.0,
            'fish_speed': 0.5,
            'energy': self.body_config.metabolism.energy_start,
            'food_count': self.world_config.arena.n_food,
            'arena_w': w,
            'arena_h': h,
            'conspecific_data': [],
            'step': step,
        }

    # ------------------------------------------------------------------
    # Config export / import
    # ------------------------------------------------------------------

    def export_config(self, path: str) -> None:
        """Export full configuration (world + body + brain) to JSON."""
        cfg = {
            'world': self.world_config.to_dict(),
            'body': self.body_config.to_dict(),
            'brain': self.brain_config.to_dict(),
        }
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_config(cls, path: str) -> VirtualZebrafish:
        """Load a VirtualZebrafish from a JSON config file."""
        with open(path) as f:
            cfg = json.load(f)
        return cls(
            world=WorldConfig.from_dict(cfg.get('world', {})),
            body=BodyConfig.from_dict(cfg.get('body', {})),
            brain=BrainConfig.from_dict(cfg.get('brain', {})),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_brain(self):
        """Lazily initialize the brain module."""
        if self._brain is not None:
            return self._brain

        if self.brain_config.fidelity == 'spiking':
            from zebrav2.brain.brain_v2 import ZebrafishBrainV2
            from zebrav2.brain.personality import get_personality

            personality = get_personality(self.brain_config.personality)
            self._brain = ZebrafishBrainV2(personality=personality)

            # Apply ablation config
            self._brain._ablated = self.brain_config.get_ablated_set()

            # Attach perturbation manager (drugs/stimulation affect neuromod)
            if self.perturbations.drugs or self.perturbations.stimulations:
                self._brain.set_perturbations(self.perturbations)

            # Load checkpoint if specified
            if self._checkpoint_path:
                from zebrav2.engine.checkpoint import CheckpointManager
                ckpt = CheckpointManager()
                ckpt.load(self._brain, self._checkpoint_path)

        elif self.brain_config.fidelity == 'rate_coded':
            from zebrav2.brain.rate_coded_brain import RateCodedBrain
            self._brain = RateCodedBrain(
                brain_config=self.brain_config,
                body_config=self.body_config)

        return self._brain

    def __repr__(self) -> str:
        ablated = self.brain_config.get_ablated_set()
        return (f"VirtualZebrafish(fidelity={self.brain_config.fidelity!r}, "
                f"personality={self.brain_config.personality!r}, "
                f"ablated={ablated}, "
                f"arena={self.world_config.arena.width}x"
                f"{self.world_config.arena.height})")
