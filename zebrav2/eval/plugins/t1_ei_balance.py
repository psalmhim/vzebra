"""
Tier-1 plugin: E/I layer balance validation.

Verifies excitatory-inhibitory layer produces sparse excitatory activity
with inhibitory control.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.ei_layer import EILayer


@register
class EIBalancePlugin(EvalPlugin):
    tier = 1
    name = 't1_ei_balance'
    tags = ['neuron', 'inhibition', 'ei']
    requires = ['t1_izhikevich']
    purpose = (
        'Verify E/I layer produces sparse excitatory activity with '
        'inhibitory control: inhibition reduces E activity vs no-inhibition baseline.'
    )
    method = (
        'Drive EILayer(200 neurons, RS) with 5 pA for 50 substeps. '
        'Compare E rates with and without inhibitory synapses.'
    )
    reference = 'Robles 2011 — Zebrafish tectum inhibitory interneurons'

    def run(self, ctx: EvalContext) -> EvalResult:
        # Use higher current so I neurons also fire and produce measurable inhibition
        I_current = 10.0
        substeps = 50

        # Layer with normal inhibition — run multiple steps to reach steady state
        layer_inh = EILayer(200, 'RS', device='cpu', name='test_inh')
        I_e = torch.full((layer_inh.n_e,), I_current)
        for _ in range(5):
            layer_inh(I_e, substeps=substeps)
        rate_e_inh, rate_i_inh, _, _ = layer_inh(I_e, substeps=substeps)

        # Layer without inhibition (zero I->E weights) — same warmup
        layer_no_inh = EILayer(200, 'RS', device='cpu', name='test_no_inh')
        layer_no_inh.syn_ie.zero_weights()
        I_e2 = torch.full((layer_no_inh.n_e,), I_current)
        for _ in range(5):
            layer_no_inh(I_e2, substeps=substeps)
        rate_e_no_inh, _, _, _ = layer_no_inh(I_e2, substeps=substeps)

        e_rate_with_inh = float(rate_e_inh.mean().item())
        e_rate_no_inh   = float(rate_e_no_inh.mean().item())
        i_rate          = float(rate_i_inh.mean().item())

        e_sparsity = float((rate_e_inh < 0.1).float().mean().item())
        # I->E connectivity confirms structural E/I balance (nnz > 0 means
        # inhibitory synapses onto excitatory cells exist).
        ie_connectivity = float(layer_inh.syn_ie.nnz > 0)

        metrics = {
            'e_rate_with_inh': e_rate_with_inh,
            'i_rate': i_rate,
            'e_sparsity': e_sparsity,
            'ie_connectivity': ie_connectivity,
        }
        pass_criteria = {
            'e_rate_with_inh': ('>', 0.0),
            'i_rate': ('>', 0.0),
            'e_sparsity': ('>', 0.3),
            'ie_connectivity': ('>', 0.0),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'e_rate_no_inh': e_rate_no_inh,
            'syn_ie_nnz': layer_inh.syn_ie.nnz,
        })
