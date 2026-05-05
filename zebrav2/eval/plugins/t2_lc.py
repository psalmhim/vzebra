"""
Tier-2 plugin: Spiking Locus Coeruleus NA validation.

Verifies tonic NA baseline, phasic burst on amygdala threat,
circadian gating, and phasic flag.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.locus_coeruleus import SpikingLocusCoeruleus


@register
class LCPlugin(EvalPlugin):
    tier = 2
    name = 't2_lc'
    tags = ['locus_coeruleus', 'noradrenaline', 'neuromod', 'arousal']
    requires = []
    purpose = (
        'Verify spiking LC encodes NA: tonic baseline stable, '
        'phasic burst on amygdala threat, night circadian reduces NA.'
    )
    method = (
        'Warm up 30 steps → baseline. '
        'Continue 10 steps with amygdala_alpha=0.8 → na > baseline, phasic=True. '
        'Night: warm up 30 steps with circadian=0.1 → na_night < day baseline.'
    )
    reference = (
        'Aston-Jones & Cohen (2005) Annu Rev Neurosci — LC-NE function; '
        'Singh et al. (2015) Curr Biol — noradrenergic modulation in zebrafish'
    )

    def run(self, ctx: EvalContext) -> EvalResult:
        lc = SpikingLocusCoeruleus(device='cpu')

        # Warm up to EMA steady-state (day default circadian=0.7)
        for _ in range(30):
            lc.forward()
        baseline = lc.forward()['na_level']

        # Phasic burst: 3 steps of high amygdala threat (EMA tau=0.85 dampens single step)
        phasic_raised = False
        for _ in range(3):
            out_threat = lc.forward(amygdala_alpha=0.8)
            phasic_raised = phasic_raised or out_threat['phasic']
        na_threat = out_threat['na_level']

        # Night: separate warm-up with low circadian
        lc.reset()
        for _ in range(30):
            lc.forward(circadian=0.1)
        na_night = lc.forward(circadian=0.1)['na_level']

        metrics = {
            'baseline_na':       baseline,
            'na_threat':         na_threat,
            'na_night':          na_night,
            'threat_vs_base':    na_threat - baseline,
            'night_suppression': baseline - na_night,
            'phasic_on_threat':  float(phasic_raised),
        }
        pass_criteria = {
            'baseline_na':       ('>', 0.05),
            'threat_vs_base':    ('>', 0.02),
            'night_suppression': ('>', 0.0),
            'phasic_on_threat':  ('==', 1.0),
        }
        return self._result(metrics, pass_criteria, artifacts={'baseline': baseline})
