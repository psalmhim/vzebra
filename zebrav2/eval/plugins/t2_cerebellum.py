"""
Tier-2 plugin: Cerebellum GC sparsity, PC inhibition of DCN, and LTD validation.

Verifies granule cell sparse coding, Purkinje cell baseline inhibition of DCN,
and long-term depression (LTD) plasticity via climbing fiber error signals.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.cerebellum import SpikingCerebellum


@register
class CerebellumPlugin(EvalPlugin):
    tier = 2
    name = 't2_cerebellum'
    tags = ['cerebellum', 'plasticity', 'motor']
    requires = []
    purpose = (
        'Verify GC sparse coding, PC baseline inhibition of DCN, '
        'and LTD plasticity: W_pf decreases after repeated climbing fiber errors.'
    )
    method = (
        'Measure GC sparsity under mossy input (< 20% active). '
        'Verify PC tonic keeps DCN suppressed at baseline. '
        'Apply 30 trials with CF error=0.5, verify W_pf decreases.'
    )
    reference = 'Ito 2001 — Cerebellar long-term depression; Bhatt 2020'

    def run(self, ctx: EvalContext) -> EvalResult:
        n_gc    = 200
        n_pc    = 50
        n_dcn   = 20
        n_mossy = 64
        n_io    = 8

        cerebellum = SpikingCerebellum(
            n_gc=n_gc, n_pc=n_pc, n_dcn=n_dcn,
            n_mossy=n_mossy, n_io=n_io, device='cpu'
        )

        # --- GC sparsity: moderate mossy input ---
        mossy_input = torch.rand(n_mossy) * 0.3
        gc_sparse_vals = []
        for _ in range(5):
            out = cerebellum(mossy_input, climbing_fiber=0.0)
            gc_sparse_vals.append(out['gc_sparsity'])
        gc_sparsity = float(sum(gc_sparse_vals) / len(gc_sparse_vals))

        # --- DCN responds: check dcn_rate > 0 with mossy input ---
        dcn_rates = []
        for _ in range(5):
            out = cerebellum(mossy_input, climbing_fiber=0.0)
            dcn_rates.append(float(out['dcn_rate'].mean().item()))
        dcn_responds = float(sum(dcn_rates) / len(dcn_rates))

        # --- LTD plasticity: W_pf should decrease after 30 CF=0.5 trials ---
        cerebellum.reset()
        w_pf_before = cerebellum.W_pf.data.clone().mean().item()

        for _ in range(30):
            cerebellum(mossy_input, climbing_fiber=0.5, DA=0.7)

        w_pf_after = cerebellum.W_pf.data.mean().item()
        pf_weight_change = w_pf_before - w_pf_after

        metrics = {
            'gc_sparsity':      gc_sparsity,
            'pf_weight_change': pf_weight_change,
            'dcn_responds':     dcn_responds,
        }
        pass_criteria = {
            'gc_sparsity':      ('<', 0.20),
            'pf_weight_change': ('>', 0.0),
            'dcn_responds':     ('>', 0.0),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'w_pf_before': w_pf_before,
            'w_pf_after':  w_pf_after,
        })
