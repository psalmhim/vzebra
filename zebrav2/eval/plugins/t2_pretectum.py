"""
Tier-2 plugin: Pretectum OKR validation.

Verifies optokinetic response: rightward DS input → positive okr_velocity;
leftward → negative.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.pretectum import SpikingPretectum


@register
class PretectumOKRPlugin(EvalPlugin):
    tier = 2
    name = 't2_pretectum'
    tags = ['pretectum', 'okr', 'visual', 'motor']
    requires = ['t2_retina']
    purpose = (
        'Verify OKR: rightward DS input → positive okr_velocity; '
        'leftward DS input → negative okr_velocity.'
    )
    method = (
        'Feed ds_L=0.02, ds_R=0.0 → OKR < 0 (leftward). '
        'Feed ds_L=0.0, ds_R=0.02 → OKR > 0 (rightward). '
        'Run 30 steps each condition. Measure DSI of pretectum population.'
    )
    reference = (
        'Kubo 2014 — Optic flow-responsive area driving horizontal eye movements'
    )

    def _run_condition(self, pretectum: SpikingPretectum,
                       ds_L: float, ds_R: float, n_steps: int = 30) -> tuple:
        """Run n_steps, return (mean_okr, mean_dsi)."""
        pretectum.reset()
        okr_vals = []
        dsi_vals = []
        for _ in range(n_steps):
            out = pretectum(ds_L=ds_L, ds_R=ds_R, eye_velocity=0.0)
            okr_vals.append(out['okr_velocity'])
            dsi_vals.append(out['dsi'])
        mean_okr = sum(okr_vals) / len(okr_vals)
        mean_dsi = sum(dsi_vals) / len(dsi_vals)
        return mean_okr, mean_dsi

    def run(self, ctx: EvalContext) -> EvalResult:
        pretectum = SpikingPretectum(device='cpu')

        # Right-eye stim (ds_R=0.02): L_pretectum driven → positive OKR
        okr_right, dsi_right = self._run_condition(pretectum, ds_L=0.0, ds_R=0.02)

        # Left-eye stim (ds_L=0.02): R_pretectum driven → negative OKR
        okr_left, dsi_left = self._run_condition(pretectum, ds_L=0.02, ds_R=0.0)

        # Static (no motion)
        okr_static, _ = self._run_condition(pretectum, ds_L=0.0, ds_R=0.0)

        # Use right-stim DSI as the main DSI metric
        dsi = dsi_right

        metrics = {
            'okr_right':  okr_right,
            'okr_left':   okr_left,
            'okr_static': okr_static,
            'dsi':        dsi,
        }
        pass_criteria = {
            'okr_right': ('>', 0.0),
            'okr_left':  ('<', 0.0),
            'dsi':       ('>', 0.05),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'dsi_left': dsi_left,
        })
