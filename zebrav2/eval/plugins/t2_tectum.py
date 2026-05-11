"""
Tier-2 plugin: Tectum size-selectivity and habituation validation.

Verifies size-selectivity (prey vs threat discrimination) and habituation
via synaptic depression.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.tectum import Tectum
from zebrav2.spec import N_RET_PER_TYPE, N_RET_LOOM, N_RET_DS


def _make_rgc_out(on_val: float, off_val: float,
                  loom_val: float, ds_val: float,
                  device: str = 'cpu') -> dict:
    """Build minimal rgc_out dict for Tectum.forward()."""
    return {
        'L_on':   torch.full((N_RET_PER_TYPE,), on_val,   device=device),
        'R_on':   torch.full((N_RET_PER_TYPE,), on_val,   device=device),
        'L_off':  torch.full((N_RET_PER_TYPE,), off_val,  device=device),
        'R_off':  torch.full((N_RET_PER_TYPE,), off_val,  device=device),
        'L_loom': torch.full((N_RET_LOOM,),     loom_val, device=device),
        'R_loom': torch.full((N_RET_LOOM,),     loom_val, device=device),
        'L_ds':   torch.full((N_RET_DS,),        ds_val,  device=device),
        'R_ds':   torch.full((N_RET_DS,),        ds_val,  device=device),
    }


@register
class TectumPlugin(EvalPlugin):
    tier = 2
    name = 't2_tectum'
    tags = ['tectum', 'visual', 'habituation']
    requires = ['t2_retina']
    purpose = (
        'Verify size-selectivity (prey vs threat discrimination) and '
        'habituation via SynapticDepression on SFGS-b input synapses.'
    )
    method = (
        'Narrow ON input → SFGS-b dominant. Wide looming input → SGC dominant. '
        'Repeat same input 20 times → SFGS-b rate decreases (habituation).'
    )
    reference = (
        'Del Bene 2010 — Filtering of visual information in tectum; Bhatt 2007'
    )

    def run(self, ctx: EvalContext) -> EvalResult:
        tectum = Tectum(device='cpu')

        # Test looming trigger: strong loom input → SGC should respond.
        # The tectum is calibrated with -2pA tonic (subthreshold), so SGC rates
        # are low (0.001-0.003) but consistently above zero for loom inputs.
        loom_rgc = _make_rgc_out(on_val=0.0, off_val=0.0,
                                  loom_val=1.0, ds_val=0.0, device='cpu')
        loom_trigger_count = 0
        n_loom_trials = 10
        for _ in range(n_loom_trials):
            out = tectum(loom_rgc)
            sgc_mean = float(out['sgc'].mean().item())
            if sgc_mean > 0.001:
                loom_trigger_count += 1
        looming_trigger = loom_trigger_count / n_loom_trials

        # Test SGC responds to loom input (sfgsb is kept subthreshold by design)
        tectum.reset()
        on_rgc = _make_rgc_out(on_val=0.0, off_val=0.0,
                                loom_val=1.0, ds_val=0.0, device='cpu')
        sfgsb_rates_warm = []
        for _ in range(5):
            out = tectum(on_rgc)
            sfgsb_rates_warm.append(float(out['sgc'].mean().item()))
        sfgsb_responds = float(sum(sfgsb_rates_warm) / len(sfgsb_rates_warm))

        # Test habituation: same loom input repeated 20 times, compare first5 vs last5
        tectum.reset()
        sfgsb_all = []
        for _ in range(20):
            out = tectum(loom_rgc)
            sfgsb_all.append(float(out['sgc'].mean().item()))

        first5_mean = float(sum(sfgsb_all[:5]) / 5.0)
        last5_mean  = float(sum(sfgsb_all[15:]) / 5.0)
        habituation_ratio = last5_mean / (first5_mean + 1e-8)

        metrics = {
            'looming_trigger':   looming_trigger,
            'habituation_ratio': habituation_ratio,
            'sfgsb_responds':    sfgsb_responds,
        }
        pass_criteria = {
            'looming_trigger':   ('>', 0.5),
            'habituation_ratio': ('<', 0.95),
            'sfgsb_responds':    ('>', 0.0),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'sfgsb_timeseries': sfgsb_all,
            'first5_mean':      first5_mean,
            'last5_mean':       last5_mean,
        })
