"""
Tier-2 plugin: Habenula negative RPE and L-R asymmetry validation.

Verifies negative RPE drives LHb, L-R asymmetry in dHb,
and helplessness accumulation after repeated failures.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.habenula import SpikingHabenula


@register
class HabenulaPlugin(EvalPlugin):
    tier = 2
    name = 't2_habenula'
    tags = ['habenula', 'reward', 'limbic']
    requires = []
    purpose = (
        'Verify negative RPE drives LHb, L-R asymmetry in dHb, '
        'and helplessness accumulates after repeated failures.'
    )
    method = (
        'reward < expected → LHb fires. reward > expected → LHb quiescent. '
        'high aversion → rMHb > lMHb. '
        '5 repeated failures → helplessness accumulates.'
    )
    reference = (
        'Amo 2014 — Left-right asymmetry of the habenular circuit; Matsumoto 2007'
    )

    def run(self, ctx: EvalContext) -> EvalResult:
        hab = SpikingHabenula(device='cpu')

        # Negative RPE: reward=-0.3 vs expected=0.1 → LHb should fire
        hab.reset()
        out_neg = hab(reward=-0.3, expected_reward=0.1,
                      aversion=0.0, current_goal=0, DA=0.5)
        lhb_neg_rpe = out_neg['lhb_rate']

        # Positive RPE: reward=0.3 vs expected=0.0 → LHb should be low
        hab.reset()
        out_pos = hab(reward=0.3, expected_reward=0.0,
                      aversion=0.0, current_goal=0, DA=0.5)
        lhb_pos_rpe = out_pos['lhb_rate']

        # L-R asymmetry: high aversion → rMHb > lMHb
        hab.reset()
        out_aversion = hab(reward=0.0, expected_reward=0.0,
                           aversion=0.8, current_goal=0, DA=0.5)
        rmhb_rate = out_aversion['rmhb_rate']
        lmhb_rate = out_aversion['lmhb_rate']
        rmhb_vs_lmhb = rmhb_rate / (lmhb_rate + 1e-6)

        # Helplessness: 5 repeated failures (negative RPE calls)
        hab.reset()
        for i in range(5):
            hab(reward=-0.5, expected_reward=0.2,
                aversion=0.0, current_goal=0, DA=0.5)
        helplessness = float(hab.helplessness)

        metrics = {
            'lhb_neg_rpe':    lhb_neg_rpe,
            'lhb_pos_rpe':    lhb_pos_rpe,
            'rmhb_vs_lmhb':   rmhb_vs_lmhb,
            'helplessness':    helplessness,
        }
        pass_criteria = {
            'lhb_neg_rpe':  ('>', 0.0),
            'lhb_pos_rpe':  ('<', 0.3),
            'rmhb_vs_lmhb': ('>', 1.1),
            'helplessness':  ('>', 0.01),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'rmhb_rate': rmhb_rate,
            'lmhb_rate': lmhb_rate,
        })
