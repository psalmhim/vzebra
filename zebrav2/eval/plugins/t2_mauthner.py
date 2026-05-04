"""
Tier-2 plugin: Mauthner cell circuit validation.

Verifies directional selectivity of the M-cell C-start reflex,
CoLo crossed inhibition for unidirectional escape, silence under
zero input, and bilateral acoustic/lateral-line co-activation.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.mauthner import SpikingMauthner


@register
class MauthnerEval(EvalPlugin):
    tier = 2
    name = 't2_mauthner'
    tags = ['mauthner', 'escape', 'hindbrain']
    requires = []
    purpose = (
        'Verify Mauthner cell directional selectivity (crossed tectobulbar), '
        'CoLo glycinergic inhibition ensuring unidirectional C-start, '
        'silence under zero input, and bilateral acoustic activation.'
    )
    method = (
        'Silence: zero sgc + zero ll_pressure → neither M-cell fires. '
        'DSI: 5 trials strong right sgc → left M-cell fires, not right '
        '(crossed tectobulbar: fraction > 0.5). '
        'CoLo: after M_L fires on right threat, colo_L_active=True. '
        'Acoustic bilateral: ll_pressure=0.7 + zero sgc → both M-cells fire.'
    )
    reference = 'Korn & Faber (2005) Neuroscience; Bhatt et al. (2007) J Neurosci'

    def run(self, ctx: EvalContext) -> EvalResult:
        zero_sgc = torch.zeros(100)
        right_sgc = torch.ones(100) * 0.5

        # --- Test 1: No-stimulus silence ---
        mauthner = SpikingMauthner(device='cpu')
        mauthner.reset()
        out_silence = mauthner(zero_sgc, zero_sgc, ll_pressure=0.0)
        silence_ok = (not out_silence['m_L_spike']) and (not out_silence['m_R_spike'])

        # --- Test 2: Directional selectivity ---
        # Strong right sgc → left M-cell should fire (crossed tectobulbar).
        # Right threat (sgc_R) drives M_L; pass if M_L fires and M_R does not.
        correct_trials = 0
        for _ in range(5):
            mauthner = SpikingMauthner(device='cpu')
            mauthner.reset()
            out = mauthner(zero_sgc, right_sgc, ll_pressure=0.0)
            if out['m_L_spike'] and not out['m_R_spike']:
                correct_trials += 1
        dsi = correct_trials / 5.0

        # --- Test 3: CoLo crossed inhibition ---
        # After M_L fires on a right-sgc trial, colo_L_active must be True.
        colo_ok = False
        for _ in range(5):
            mauthner = SpikingMauthner(device='cpu')
            mauthner.reset()
            out = mauthner(zero_sgc, right_sgc, ll_pressure=0.0)
            if out['m_L_spike'] and out['colo_L_active']:
                colo_ok = True
                break

        # --- Test 4: Acoustic bilateral ---
        # Lateral-line pressure alone (no visual threat) → both M-cells fire.
        mauthner = SpikingMauthner(device='cpu')
        mauthner.reset()
        out_acoustic = mauthner(zero_sgc, zero_sgc, ll_pressure=0.7)
        acoustic_bilateral = out_acoustic['m_L_spike'] and out_acoustic['m_R_spike']

        metrics = {
            'silence_ok':        float(silence_ok),
            'dsi':               dsi,
            'colo_ok':           float(colo_ok),
            'acoustic_bilateral': float(acoustic_bilateral),
        }
        pass_criteria = {
            'silence_ok':        ('==', True),
            'dsi':               ('>', 0.5),
            'colo_ok':           ('==', True),
            'acoustic_bilateral': ('==', True),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'silence_out':    out_silence,
            'acoustic_out':   out_acoustic,
            'dsi_trials':     correct_trials,
        })
