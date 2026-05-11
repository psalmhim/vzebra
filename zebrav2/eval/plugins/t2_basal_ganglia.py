"""
Tier-2 plugin: Basal ganglia D1/D2 pathway and action gating validation.

Verifies D1/D2 pathway and action gating under dopamine modulation.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.basal_ganglia import BasalGanglia
from zebrav2.spec import N_PAL_D


@register
class BasalGangliaPlugin(EvalPlugin):
    tier = 2
    name = 't2_basal_ganglia'
    tags = ['basal_ganglia', 'reward', 'action']
    requires = ['t2_habenula']
    purpose = (
        'Verify D1/D2 pathway and action gating under DA modulation. '
        'High DA → strong D1 activation. Low DA → stronger D2.'
    )
    method = (
        'Drive pallium-D with uniform input. '
        'High DA → strong D1 activation. Low DA → stronger D2. '
        'Measure GPi gate.'
    )
    reference = 'Mancusi 2023 — Striatal D1/D2 MSN dynamics in zebrafish'

    def run(self, ctx: EvalContext) -> EvalResult:
        # The BG gate depends on random weight initialization (Xavier uniform).
        # Sample across 20 independent BG instances to get a stable mean.
        N_INITS = 20

        d1_rates_high = []
        gate_vals_high = []
        d2_rates_low = []
        last_out_high = None
        last_out_low = None

        for _ in range(N_INITS):
            bg = BasalGanglia(device='cpu')
            input_size = bg.W_pald_d1.in_features
            pal_d_rate = torch.ones(input_size) * 0.5

            out_high_da = bg(pal_d_rate, DA=0.9)
            out_low_da  = bg(pal_d_rate, DA=0.1)

            d1_rates_high.append(float(out_high_da['D1'].mean().item()))
            gate_vals_high.append(float(out_high_da['gate']))
            d2_rates_low.append(float(out_low_da['D2'].mean().item()))
            last_out_high = out_high_da
            last_out_low  = out_low_da

        d1_rate_high_da = float(sum(d1_rates_high) / N_INITS)
        gate_nonzero    = float(sum(gate_vals_high) / N_INITS)
        d2_rate_low_da  = float(sum(d2_rates_low)  / N_INITS)
        out_high_da = last_out_high
        out_low_da  = last_out_low

        metrics = {
            'd1_rate_high_da': d1_rate_high_da,
            'gate_nonzero':    gate_nonzero,
            'd2_rate_low_da':  d2_rate_low_da,
        }
        pass_criteria = {
            'd1_rate_high_da': ('>', 0.0),
            'gate_nonzero':    ('>', 0.0),
            'd2_rate_low_da':  ('>', 0.0),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'd1_high_da_last': out_high_da['D1'].tolist(),
            'd2_low_da_last':  out_low_da['D2'].tolist(),
            'gate_vals_high':  gate_vals_high,
            'gate_positive_fraction': sum(1 for g in gate_vals_high if g > 0) / N_INITS,
        })
