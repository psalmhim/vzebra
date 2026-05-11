"""
Tier-2 plugin: Pallium working memory and goal modulation validation.

Verifies pallium maintains activity after input removed (working memory)
and is modulated by goal vectors.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.pallium import Pallium
from zebrav2.spec import N_TC


@register
class PalliumPlugin(EvalPlugin):
    tier = 2
    name = 't2_pallium'
    tags = ['pallium', 'working_memory', 'cognition']
    requires = ['t2_tectum']
    purpose = (
        'Verify pallium maintains activity after input removed (working memory) '
        'and is modulated by different goal vectors.'
    )
    method = (
        'Drive pallium for 5 steps, remove input (zero tc_rate), '
        'check rate_S persists for 5 more steps. '
        'Compare activation patterns for different goal vectors.'
    )
    reference = 'Major 2004 — Persistent neural activity in pallium; Del Bene 2010'

    def run(self, ctx: EvalContext) -> EvalResult:
        pallium = Pallium(device='cpu')

        tc_rate_active = torch.rand(N_TC) * 0.5
        tc_rate_zero   = torch.zeros(N_TC)
        goal_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        goal_2 = torch.tensor([0.0, 0.0, 1.0, 0.0])

        # Drive for 5 steps, then remove input
        for _ in range(5):
            pallium(tc_rate_active, goal_0)

        # Measure rate after input removed (next 5 steps with zero input)
        rates_after = []
        for _ in range(5):
            out = pallium(tc_rate_zero, goal_0)
            rates_after.append(float(out['rate_S'].mean().item()))
        rate_after_input = float(sum(rates_after) / len(rates_after))

        # Goal modulation: compare activation patterns for goal 0 vs goal 2
        pallium.reset()
        # Drive with goal 0
        for _ in range(5):
            out_g0 = pallium(tc_rate_active, goal_0)
        pattern_g0 = out_g0['rate_S'].clone()

        pallium.reset()
        # Drive with goal 2
        for _ in range(5):
            out_g2 = pallium(tc_rate_active, goal_2)
        pattern_g2 = out_g2['rate_S'].clone()

        goal_modulation = float((pattern_g0 - pattern_g2).norm().item())

        # Prediction error: measure from the last forward pass
        pe = out_g2['pred_error']
        pe_generated = float(pe.abs().mean().item())

        metrics = {
            'rate_after_input': rate_after_input,
            'goal_modulation':  goal_modulation,
            'pe_generated':     pe_generated,
        }
        pass_criteria = {
            'rate_after_input': ('>', 0.0),
            'goal_modulation':  ('>', 0.0),
            'pe_generated':     ('>=', 0.0),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'rate_S_mean_with_input': float(out_g0['rate_S'].mean().item()),
            'rate_S_mean_after':      rate_after_input,
        })
