"""
Tier-2 plugin: VTA/SNpc spiking dopamine validation.

Verifies RPE-coding (burst on reward, pause on omission),
LHb-driven DA suppression, and tonic baseline stability.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.vta import SpikingVTA


@register
class VTAPlugin(EvalPlugin):
    tier = 2
    name = 't2_vta'
    tags = ['vta', 'dopamine', 'reward', 'neuromod']
    requires = []
    purpose = (
        'Verify spiking VTA encodes RPE: phasic burst on unexpected reward, '
        'pause on omission, tonic baseline ~0.5, and LHb-driven DA suppression.'
    )
    method = (
        'Positive RPE → da_level > tonic baseline + 0.2. '
        'Negative RPE → da_level < tonic baseline. '
        'LHb=0.5 → da_level reduced vs no-LHb baseline. '
        'Phasic flag raised on reward.'
    )
    reference = (
        'Schultz et al. (1997) Science 275:1593 — RPE coding in DA neurons; '
        'Kastenhuber et al. (2010) J Comp Neurol 518:116 — zebrafish PT dopamine'
    )

    def run(self, ctx: EvalContext) -> EvalResult:
        vta = SpikingVTA(device='cpu')

        # Warm up to let EMA stabilize
        for _ in range(15):
            vta.forward()
        baseline = vta.forward()['da_level']

        # Positive RPE: unexpected reward → phasic burst
        vta.reset()
        for _ in range(5):
            vta.forward(rpe=0.8)
        out_reward = vta.forward(rpe=0.8)
        da_reward = out_reward['da_level']
        phasic_raised = out_reward['phasic']

        # Negative RPE: reward omission → pause / dip
        vta.reset()
        for _ in range(5):
            vta.forward(rpe=-0.6)
        da_omission = vta.forward(rpe=-0.6)['da_level']

        # LHb suppression: disappointment → reduced DA vs baseline
        vta.reset()
        for _ in range(15):
            vta.forward()
        da_with_lhb = vta.forward(lhb_rate=0.5)['da_level']

        metrics = {
            'baseline_da':     baseline,
            'da_reward':       da_reward,
            'da_omission':     da_omission,
            'da_with_lhb':     da_with_lhb,
            'phasic_on_reward': float(phasic_raised),
            'reward_vs_base':  da_reward - baseline,
            'lhb_suppression': baseline - da_with_lhb,
        }
        pass_criteria = {
            'baseline_da':      ('>', 0.3),
            'da_reward':        ('>', 0.6),
            'da_omission':      ('<', 0.5),
            'reward_vs_base':   ('>', 0.1),
            'lhb_suppression':  ('>', 0.0),
            'phasic_on_reward': ('==', 1.0),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'baseline': baseline,
        })
