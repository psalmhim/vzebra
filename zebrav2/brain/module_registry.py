"""
Module registry: declarative catalog of all brain modules with metadata.

Each module entry tracks:
  - region name (matches AblationConfig field names)
  - nn.Module class reference (lazy)
  - biological role
  - whether it's optional (can be ablated)
  - dependencies on other modules
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModuleEntry:
    """Metadata for a single brain module."""
    name: str
    class_name: str
    module_path: str        # e.g. 'zebrav2.brain.cerebellum'
    role: str               # brief biological description
    optional: bool = True   # can be disabled without crashing
    depends_on: list[str] = field(default_factory=list)
    region_group: str = ''  # anatomical grouping


# Full registry of all 34 spiking + 13 pure-Python modules
BRAIN_MODULES: dict[str, ModuleEntry] = {
    # === Core pipeline (required) ===
    'retina': ModuleEntry(
        'retina', 'RetinaV2', 'zebrav2.brain.retina',
        'Photoreceptor array → 800-dim bilateral retinal input',
        optional=False, region_group='sensory'),
    'tectum': ModuleEntry(
        'tectum', 'Tectum', 'zebrav2.brain.tectum',
        'Optic tectum: 4 bilateral layers (sfgs_b, sfgs_d, sgc, so)',
        optional=False, region_group='midbrain'),
    'thalamus': ModuleEntry(
        'thalamus', 'Thalamus', 'zebrav2.brain.thalamus',
        'Thalamic relay: TC + TRN gating',
        optional=False, region_group='diencephalon'),
    'pallium': ModuleEntry(
        'pallium', 'Pallium', 'zebrav2.brain.pallium',
        'Cortical homolog: pal_s (superficial) + pal_d (deep)',
        optional=False, region_group='telencephalon'),
    'basal_ganglia': ModuleEntry(
        'basal_ganglia', 'BasalGanglia', 'zebrav2.brain.basal_ganglia',
        'Action selection: D1/D2 pathways, motor gating',
        optional=False, region_group='telencephalon'),
    'neuromod': ModuleEntry(
        'neuromod', 'NeuromodSystem', 'zebrav2.brain.neuromod',
        'DA/NA/5HT/ACh modulation system',
        optional=False, region_group='brainstem'),
    'amygdala': ModuleEntry(
        'amygdala', 'SpikingAmygdalaV2', 'zebrav2.brain.amygdala',
        'Fear conditioning: BLA → CeA → threat output',
        optional=False, region_group='telencephalon'),
    'reticulospinal': ModuleEntry(
        'reticulospinal', 'ReticulospinalSystem', 'zebrav2.brain.reticulospinal',
        'Descending motor command relay',
        optional=False, region_group='brainstem'),

    # === Optional modules (can be ablated) ===
    'cerebellum': ModuleEntry(
        'cerebellum', 'SpikingCerebellum', 'zebrav2.brain.cerebellum',
        'Prediction error, motor learning',
        optional=True, region_group='hindbrain'),
    'habenula': ModuleEntry(
        'habenula', 'SpikingHabenula', 'zebrav2.brain.habenula',
        'Frustration/disappointment → DA suppression',
        optional=True, region_group='diencephalon'),
    'predictive_net': ModuleEntry(
        'predictive_net', 'SpikingPredictiveNet', 'zebrav2.brain.predictive_net',
        'Predictive coding: surprise signals',
        optional=True, region_group='telencephalon'),
    'rl_critic': ModuleEntry(
        'rl_critic', 'SpikingCritic', 'zebrav2.brain.rl_critic',
        'TD learning critic: value estimation',
        optional=True, region_group='telencephalon'),
    'habit_network': ModuleEntry(
        'habit_network', 'SpikingHabitNet', 'zebrav2.brain.habit_network',
        'Habit formation via stimulus-response learning',
        optional=True, region_group='telencephalon'),
    'insula': ModuleEntry(
        'insula', 'SpikingInsularCortex', 'zebrav2.brain.interoception',
        'Interoception: hunger, stress, heart rate, valence',
        optional=True, region_group='telencephalon'),
    'lateral_line': ModuleEntry(
        'lateral_line', 'SpikingLateralLine', 'zebrav2.brain.lateral_line',
        'Hydrodynamic flow sensing',
        optional=True, region_group='sensory'),
    'olfaction': ModuleEntry(
        'olfaction', 'SpikingOlfaction', 'zebrav2.brain.olfaction',
        'Chemical sensing: food odor + alarm substance',
        optional=True, region_group='sensory'),
    'working_memory': ModuleEntry(
        'working_memory', 'SpikingWorkingMemory', 'zebrav2.brain.working_memory',
        'Short-term storage for goal maintenance',
        optional=True, region_group='telencephalon'),
    'vestibular': ModuleEntry(
        'vestibular', 'SpikingVestibular', 'zebrav2.brain.vestibular',
        'Angular velocity and tilt sensing',
        optional=True, region_group='sensory'),
    'proprioception': ModuleEntry(
        'proprioception', 'SpikingProprioception', 'zebrav2.brain.proprioception',
        'Body state and collision sensing',
        optional=True, region_group='sensory'),
    'color_vision': ModuleEntry(
        'color_vision', 'SpikingColorVision', 'zebrav2.brain.color_vision',
        'UV/blue/green/red spectral analysis',
        optional=True, region_group='sensory'),
    'circadian': ModuleEntry(
        'circadian', 'SpikingCircadian', 'zebrav2.brain.circadian',
        'Day/night cycle modulation',
        optional=True, region_group='hypothalamus'),
    'sleep_wake': ModuleEntry(
        'sleep_wake', 'SpikingSleepWake', 'zebrav2.brain.sleep_wake',
        'Sleep/wake state regulation',
        optional=True, region_group='hypothalamus'),
    'saccade': ModuleEntry(
        'saccade', 'SpikingSaccade', 'zebrav2.brain.saccade',
        'Eye movement: saccade and gaze control',
        optional=True, region_group='midbrain'),
    'geographic_model': ModuleEntry(
        'geographic_model', 'GeographicModel', 'zebrav2.brain.geographic_model',
        'Spatial exploration tracking',
        optional=True, region_group='telencephalon'),
    'binocular_depth': ModuleEntry(
        'binocular_depth', 'BinocularDepth', 'zebrav2.brain.binocular_depth',
        'Binocular disparity → depth estimation',
        optional=True, region_group='midbrain'),
    'shoaling': ModuleEntry(
        'shoaling', 'ShoalingModule', 'zebrav2.brain.shoaling',
        'Social cohesion: group following behavior',
        optional=True, region_group='telencephalon'),
    'prey_capture': ModuleEntry(
        'prey_capture', 'PreyCaptureKinematics', 'zebrav2.brain.prey_capture',
        'J-turn / S-start prey capture sequences',
        optional=True, region_group='midbrain'),
    'vae_world_model': ModuleEntry(
        'vae_world_model', 'VAEWorldModelV2', 'zebrav2.brain.vae_world_model',
        'Variational world model: memory + novelty',
        optional=True, region_group='telencephalon'),
    'place_cells': ModuleEntry(
        'place_cells', 'ThetaPlaceCells', 'zebrav2.brain.place_cells',
        'Spatial navigation: place cell grid',
        optional=True, region_group='telencephalon'),
    'classifier': ModuleEntry(
        'classifier', 'ClassifierV2', 'zebrav2.brain.classifier',
        'Visual object classification',
        optional=True, region_group='telencephalon'),
    'predator_model': ModuleEntry(
        'predator_model', 'PredatorModel', 'zebrav2.brain.predator_model',
        'Predator behavior prediction',
        optional=True, region_group='telencephalon'),
    'internal_model': ModuleEntry(
        'internal_model', 'InternalWorldModel', 'zebrav2.brain.internal_model',
        'Internal state prediction',
        optional=True, region_group='telencephalon'),
    'spinal_cpg': ModuleEntry(
        'spinal_cpg', 'SpinalCPG', 'zebrav2.brain.spinal_cpg',
        'Central pattern generator: rhythmic locomotion',
        optional=True, region_group='spinal'),
    'goal_selector': ModuleEntry(
        'goal_selector', 'SpikingGoalSelector', 'zebrav2.brain.goal_selector',
        'Winner-take-all goal competition',
        optional=True, region_group='telencephalon'),
    'allostasis': ModuleEntry(
        'allostasis', 'AllostaticRegulator', 'zebrav2.brain.allostasis',
        'Energy homeostasis regulation',
        optional=True, region_group='hypothalamus'),
    'meta_goal': ModuleEntry(
        'meta_goal', 'MetaGoalWeights', 'zebrav2.brain.meta_goal',
        'Meta-learning: episodic goal weight adjustment',
        optional=True, region_group='telencephalon'),
    'social_memory': ModuleEntry(
        'social_memory', 'SocialMemory', 'zebrav2.brain.social_memory',
        'Individual recognition + social learning',
        optional=True, region_group='telencephalon'),
    'hpa_axis': ModuleEntry(
        'hpa_axis', 'HPAAxis', 'zebrav2.brain.hpa_axis',
        'Stress response: cortisol release',
        optional=True, region_group='hypothalamus'),
    'oxytocin': ModuleEntry(
        'oxytocin', 'OxytocinSystem', 'zebrav2.brain.oxytocin',
        'Social bonding + prosocial motivation',
        optional=True, region_group='hypothalamus'),
}


def get_enabled_modules(ablated: set[str]) -> dict[str, ModuleEntry]:
    """Return modules that are NOT ablated (i.e., enabled)."""
    return {name: entry for name, entry in BRAIN_MODULES.items()
            if name not in ablated}


def get_optional_modules() -> dict[str, ModuleEntry]:
    """Return all optional (ablatable) modules."""
    return {name: entry for name, entry in BRAIN_MODULES.items()
            if entry.optional}


def get_required_modules() -> dict[str, ModuleEntry]:
    """Return all required (non-ablatable) modules."""
    return {name: entry for name, entry in BRAIN_MODULES.items()
            if not entry.optional}


def get_modules_by_group(group: str) -> dict[str, ModuleEntry]:
    """Return modules in an anatomical group."""
    return {name: entry for name, entry in BRAIN_MODULES.items()
            if entry.region_group == group}


def list_region_groups() -> list[str]:
    """Return all unique anatomical groups."""
    return sorted({e.region_group for e in BRAIN_MODULES.values()})


def summary() -> dict[str, Any]:
    """Summary statistics for the registry."""
    groups = {}
    for entry in BRAIN_MODULES.values():
        g = entry.region_group or 'ungrouped'
        groups.setdefault(g, []).append(entry.name)
    return {
        'total_modules': len(BRAIN_MODULES),
        'required': len(get_required_modules()),
        'optional': len(get_optional_modules()),
        'region_groups': groups,
    }
