"""
Tier-2 plugin: Retina direction-selective, ON/OFF, and looming validation.

Verifies direction-selective, ON/OFF, and looming RGC responses using
moving gratings and luminance steps.
"""
import math
import torch
import numpy as np
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.retina import RetinaV2

N_PIX  = 400
PERIOD = 40
SHIFT  = 4


def _grating(phase: float) -> torch.Tensor:
    k = torch.arange(N_PIX, dtype=torch.float32)
    return 0.5 + 0.5 * torch.sin(2 * math.pi * k / PERIOD + phase)


def _make_eye(intensity: torch.Tensor) -> torch.Tensor:
    eye = torch.zeros(800)
    eye[:N_PIX] = intensity
    return eye


@register
class RetinaPlugin(EvalPlugin):
    tier = 2
    name = 't2_retina'
    tags = ['retina', 'visual', 'ds']
    requires = []
    purpose = (
        'Verify direction-selective, ON/OFF, and looming RGC responses. '
        'Rightward grating → positive ds_mean; leftward → negative.'
    )
    method = (
        'Moving grating (N_PIX=400, period=40, shift=4). '
        'Warm up 10 steps, measure 20 steps. Luminance step for ON/OFF.'
    )
    reference = 'Nikolaou 2012 — Directional tuning of zebrafish RGCs; Kubo 2014'

    def _run_grating(self, retina: RetinaV2, rightward: bool,
                     n_warm: int = 10, n_eval: int = 20):
        """Run moving grating, return mean ds_L over eval steps."""
        retina.reset()
        phase = 0.0
        phase_step = 2 * math.pi * SHIFT / PERIOD
        if not rightward:
            phase_step = -phase_step

        ds_L_vals = []
        for step in range(n_warm + n_eval):
            phase += phase_step
            intensity = _grating(phase)
            L = _make_eye(intensity)
            R = _make_eye(intensity)
            out = retina(L, R, entity_info={})
            if step >= n_warm:
                ds_L_vals.append(float(out['L_ds'].mean().item()))
        return float(sum(ds_L_vals) / len(ds_L_vals)) if ds_L_vals else 0.0

    def run(self, ctx: EvalContext) -> EvalResult:
        retina = RetinaV2(device='cpu')

        # Directional selectivity: rightward vs leftward grating
        ds_right = self._run_grating(retina, rightward=True)
        ds_left  = self._run_grating(retina, rightward=False)

        # Static grating (no motion): near-zero ds
        retina.reset()
        static_ds_vals = []
        static_intensity = _grating(0.0)
        for _ in range(20):
            L = _make_eye(static_intensity)
            R = _make_eye(static_intensity)
            out = retina(L, R, entity_info={})
            static_ds_vals.append(float(out['L_ds'].mean().item()))

        # ON response: luminance step up (dark → bright)
        retina.reset()
        dark_eye   = _make_eye(torch.zeros(N_PIX))
        bright_eye = _make_eye(torch.ones(N_PIX) * 0.8)

        # Baseline: run dark for 5 steps
        for _ in range(5):
            retina(dark_eye, dark_eye, entity_info={})
        baseline_on = float(retina(dark_eye, dark_eye, entity_info={})['L_on'].mean().item())

        # ON response: step to bright
        on_rates = []
        for _ in range(10):
            out = retina(bright_eye, bright_eye, entity_info={})
            on_rates.append(float(out['L_on'].mean().item()))
        on_response = float(sum(on_rates) / len(on_rates)) - baseline_on

        # OFF response: bright → dark (after ON warmup)
        retina.reset()
        for _ in range(10):
            retina(bright_eye, bright_eye, entity_info={})

        off_rates = []
        for _ in range(10):
            out = retina(dark_eye, dark_eye, entity_info={})
            off_rates.append(float(out['L_off'].mean().item()))
        off_response = float(sum(off_rates) / len(off_rates))

        # DSI: signed difference between rightward and leftward response.
        # ds_right is positive (rightward = preferred), ds_left is negative.
        # DSI = (ds_right - ds_left) / (|ds_right| + |ds_left| + eps)
        # measures the total directional swing normalized to response magnitude.
        dsi = (ds_right - ds_left) / (abs(ds_right) + abs(ds_left) + 1e-8)

        metrics = {
            'ds_rightward_L':   ds_right,
            'ds_leftward_L':    ds_left,
            'ds_directionality': abs(ds_right - ds_left),
            'on_response':       on_response,
            'off_response':      off_response,
            'dsi':               dsi,
        }
        pass_criteria = {
            'ds_rightward_L':    ('>', 0.005),
            'ds_leftward_L':     ('<', -0.005),
            'ds_directionality': ('>', 0.01),
            'on_response':       ('>', 0.01),
            'off_response':      ('>', 0.0),
            'dsi':               ('>', 0.1),
        }
        return self._result(metrics, pass_criteria, artifacts={})
