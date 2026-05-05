"""
Tier-2 plugin: Spiking Dorsal Raphe 5-HT validation.

Verifies tonic 5-HT baseline, LHb suppression, flee suppression,
and IPN-driven excitation.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.raphe import SpikingRaphe


@register
class RaphePlugin(EvalPlugin):
    tier = 2
    name = 't2_raphe'
    tags = ['raphe', 'serotonin', 'neuromod']
    requires = []
    purpose = (
        'Verify spiking dorsal raphe encodes 5-HT: tonic baseline stable, '
        'suppressed by LHb disappointment, suppressed during flee, '
        'elevated by IPN excitation.'
    )
    method = (
        'Warm up 30 steps → baseline. '
        'Continue 30 steps with LHb=0.6 → ht5 drops below baseline. '
        'Separate warm-up, then 30 steps flee_active=True → ht5 drops. '
        'Separate warm-up, then 30 steps ipn_raphe_drive=1.0 → ht5 rises.'
    )
    reference = (
        'Lillesaar (2011) J Chem Neuroanat — zebrafish serotonergic system; '
        'Yokogawa et al. (2012) J Neurosci — dorsal raphe sensory gating'
    )

    def run(self, ctx: EvalContext) -> EvalResult:
        raphe = SpikingRaphe(device='cpu')

        # Warm up to EMA steady-state
        for _ in range(30):
            raphe.forward()
        baseline = raphe.forward()['ht5_level']

        # LHb suppression: continue from warm state (EMA already at baseline)
        for _ in range(30):
            raphe.forward(lhb_rate=0.6)
        ht5_lhb = raphe.forward(lhb_rate=0.6)['ht5_level']

        # Flee suppression: fresh warm-up then flee
        raphe.reset()
        for _ in range(30):
            raphe.forward()
        for _ in range(30):
            raphe.forward(flee_active=True)
        ht5_flee = raphe.forward(flee_active=True)['ht5_level']

        # IPN excitation: fresh warm-up then IPN
        raphe.reset()
        for _ in range(30):
            raphe.forward()
        for _ in range(30):
            raphe.forward(ipn_raphe_drive=1.0)
        ht5_ipn = raphe.forward(ipn_raphe_drive=1.0)['ht5_level']

        metrics = {
            'baseline_ht5':     baseline,
            'ht5_lhb':          ht5_lhb,
            'ht5_flee':         ht5_flee,
            'ht5_ipn':          ht5_ipn,
            'lhb_suppression':  baseline - ht5_lhb,
            'flee_suppression': baseline - ht5_flee,
            'ipn_excitation':   ht5_ipn - baseline,
        }
        pass_criteria = {
            'baseline_ht5':     ('>', 0.05),
            'lhb_suppression':  ('>', 0.0),
            'flee_suppression': ('>', 0.0),
            'ipn_excitation':   ('>', 0.0),
        }
        return self._result(metrics, pass_criteria, artifacts={'baseline': baseline})
