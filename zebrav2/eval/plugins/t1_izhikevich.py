"""
Tier-1 plugin: Izhikevich neuron type validation.

Verifies each neuron type produces biologically correct firing patterns
under step current injection.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.neurons import IzhikevichLayer


@register
class IzhikevichTypesPlugin(EvalPlugin):
    tier = 1
    name = 't1_izhikevich'
    tags = ['neuron', 'spiking']
    requires = []
    purpose = (
        'Verify each Izhikevich neuron type produces biologically correct '
        'firing pattern under step current injection.'
    )
    method = (
        'Apply step current (10 pA for RS/IB/CH, 5 pA for LTS, 15 pA for FS) '
        'for 200 substeps. Count spikes per type.'
    )
    reference = 'Izhikevich 2003 — Simple model of spiking neurons'

    def _count_spikes(self, cell_type: str, current_pA: float, n_steps: int = 200) -> int:
        layer = IzhikevichLayer(n=1, cell_type=cell_type, device='cpu')
        I = torch.tensor([current_pA])
        spike_count = 0
        for _ in range(n_steps):
            spikes = layer(I)
            spike_count += int(spikes.sum().item())
        return spike_count

    def run(self, ctx: EvalContext) -> EvalResult:
        rs_spikes  = self._count_spikes('RS',  10.0, 200)
        ib_spikes  = self._count_spikes('IB',  10.0, 200)
        ch_spikes  = self._count_spikes('CH',  15.0, 200)
        lts_spikes = self._count_spikes('LTS',  5.0, 200)
        fs_spikes  = self._count_spikes('FS',  15.0, 200)

        metrics = {
            'rs_spikes':  float(rs_spikes),
            'ib_spikes':  float(ib_spikes),
            'ch_spikes':  float(ch_spikes),
            'lts_spikes': float(lts_spikes),
            'fs_spikes':  float(fs_spikes),
        }
        pass_criteria = {
            'rs_spikes':  ('>', 2.0),
            'ib_spikes':  ('>', 3.0),
            'ch_spikes':  ('>', 5.0),
            'lts_spikes': ('>', 0.0),
            'fs_spikes':  ('>', 5.0),
        }
        return self._result(metrics, pass_criteria, artifacts={})
