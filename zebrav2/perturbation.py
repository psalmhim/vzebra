"""
Perturbation API: lesion, drug injection, optogenetic stimulation.

Provides experiment-level manipulations of the virtual zebrafish brain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DrugEffect:
    """Pharmacological perturbation parameters."""
    name: str
    target_neuromod: str     # 'DA', 'NA', '5HT', 'ACh'
    effect: str              # 'agonist', 'antagonist', 'reuptake_inhibitor'
    dose: float              # 0.0-1.0 (normalized)
    onset_steps: int = 0     # delay before effect starts
    duration_steps: int = -1 # -1 = permanent

    def compute_multiplier(self, step: int) -> float:
        """Compute neuromodulator multiplier at given step."""
        if step < self.onset_steps:
            return 1.0
        if self.duration_steps > 0 and step > self.onset_steps + self.duration_steps:
            return 1.0
        if self.effect == 'agonist':
            return 1.0 + self.dose
        elif self.effect == 'antagonist':
            return max(0.0, 1.0 - self.dose)
        elif self.effect == 'reuptake_inhibitor':
            return 1.0 + 0.5 * self.dose  # partial boost
        return 1.0


# Pre-defined drug profiles (common in zebrafish pharmacology)
DRUG_LIBRARY = {
    'haloperidol': DrugEffect(
        name='Haloperidol', target_neuromod='DA',
        effect='antagonist', dose=0.7),
    'fluoxetine': DrugEffect(
        name='Fluoxetine (SSRI)', target_neuromod='5HT',
        effect='reuptake_inhibitor', dose=0.5),
    'scopolamine': DrugEffect(
        name='Scopolamine', target_neuromod='ACh',
        effect='antagonist', dose=0.6),
    'amphetamine': DrugEffect(
        name='Amphetamine', target_neuromod='DA',
        effect='agonist', dose=0.8),
    'propranolol': DrugEffect(
        name='Propranolol', target_neuromod='NA',
        effect='antagonist', dose=0.5),
    'buspirone': DrugEffect(
        name='Buspirone (anxiolytic)', target_neuromod='5HT',
        effect='agonist', dose=0.4),
}


class PerturbationManager:
    """Manages all active perturbations on a virtual zebrafish."""

    def __init__(self):
        self.lesions: set[str] = set()
        self.drugs: list[DrugEffect] = []
        self.stimulations: list[dict] = []
        self._step = 0

    def lesion(self, region: str) -> None:
        """Permanently disable a brain region."""
        self.lesions.add(region)

    def inject(self, drug: str | DrugEffect, dose: float | None = None) -> None:
        """Administer a drug. Use name from DRUG_LIBRARY or custom DrugEffect."""
        if isinstance(drug, str):
            if drug not in DRUG_LIBRARY:
                raise ValueError(
                    f"Unknown drug '{drug}'. Available: {list(DRUG_LIBRARY.keys())}")
            d = DrugEffect(**{**DRUG_LIBRARY[drug].__dict__})
            if dose is not None:
                d.dose = dose
            self.drugs.append(d)
        else:
            self.drugs.append(drug)

    def stimulate(self, region: str, pattern: str = 'pulse',
                  intensity: float = 1.0, duration: int = 10) -> None:
        """Optogenetic-style stimulation of a brain region."""
        self.stimulations.append({
            'region': region, 'pattern': pattern,
            'intensity': intensity, 'duration': duration,
            'start_step': self._step,
        })

    def get_neuromod_multipliers(self) -> dict[str, float]:
        """Get combined neuromodulator multipliers from all active drugs."""
        mults = {'DA': 1.0, 'NA': 1.0, '5HT': 1.0, 'ACh': 1.0}
        for drug in self.drugs:
            if drug.target_neuromod in mults:
                mults[drug.target_neuromod] *= drug.compute_multiplier(self._step)
        return mults

    def get_active_stimulations(self) -> list[dict]:
        """Get stimulations active at current step."""
        active = []
        for stim in self.stimulations:
            elapsed = self._step - stim['start_step']
            if 0 <= elapsed < stim['duration']:
                active.append(stim)
        return active

    def step(self) -> None:
        """Advance perturbation clock by one step."""
        self._step += 1

    def reset(self) -> None:
        self._step = 0

    def summary(self) -> dict:
        return {
            'lesions': list(self.lesions),
            'drugs': [{'name': d.name, 'target': d.target_neuromod,
                       'dose': d.dose} for d in self.drugs],
            'stimulations': len(self.stimulations),
        }
