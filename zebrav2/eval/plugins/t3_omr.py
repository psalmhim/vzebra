"""
Tier 3: End-to-end OMR circuit — retina → pretectum → directional turn.

Reference: Naumann 2016 — Functional circuit models of the zebrafish optomotor response
"""
from __future__ import annotations
import math
import numpy as np
import torch

from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register

DEVICE = 'cpu'
N_PIX  = 400
PERIOD = 40
SHIFT  = 4
N_WARM = 10
N_EVAL = 20


def _grating(phase: float) -> np.ndarray:
    k = np.arange(N_PIX, dtype=np.float32)
    return (0.5 + 0.5 * np.sin(2 * math.pi * k / PERIOD + phase)).astype(np.float32)


def _make_eye(intensity: np.ndarray) -> np.ndarray:
    eye = np.zeros(800, dtype=np.float32)
    eye[:N_PIX] = intensity
    return eye


def _run_condition(retina, pretectum, stim_left: bool, stim_right: bool):
    retina.reset()
    pretectum.reset()
    phase = 0.0
    phase_step = 2 * math.pi * SHIFT / PERIOD
    okr_log = []
    for step in range(N_WARM + N_EVAL):
        phase += phase_step
        L_int = _grating(phase) if stim_left  else np.zeros(N_PIX, np.float32)
        R_int = _grating(phase) if stim_right else np.zeros(N_PIX, np.float32)
        L = torch.tensor(_make_eye(L_int), device=DEVICE)
        R = torch.tensor(_make_eye(R_int), device=DEVICE)
        rgc  = retina(L, R, entity_info={})
        ds_L = float(rgc['L_ds'].mean())
        ds_R = float(rgc['R_ds'].mean())
        pret = pretectum(ds_L, ds_R, eye_velocity=0.0)
        if step >= N_WARM:
            okr_log.append(pret['okr_velocity'])
    return float(np.mean(okr_log))


@register
class T3OMR(EvalPlugin):
    tier = 3
    name = 't3_omr'
    tags = ['omr', 'visual', 'motor', 'circuit']
    requires = ['t2_retina', 't2_pretectum']
    purpose = 'End-to-end OMR circuit: retina → pretectum → directional turn.'
    method = (
        'Moving sinusoidal grating (period=40px, shift=4px/step). '
        'Right-eye only → OKR > 0. Left-eye only → OKR < 0. '
        'Bilateral static → |OKR| < 0.15. '
        '10 warm-up + 20 eval steps.'
    )
    reference = 'Naumann 2016 — Functional circuit models of the zebrafish optomotor response'

    def run(self, ctx: EvalContext) -> EvalResult:
        from zebrav2.brain.retina import RetinaV2
        from zebrav2.brain.pretectum import SpikingPretectum

        # --- Right-eye only ---
        ret_r = RetinaV2(device=DEVICE)
        pre_r = SpikingPretectum(device=DEVICE)
        okr_right = _run_condition(ret_r, pre_r, stim_left=False, stim_right=True)

        # --- Left-eye only ---
        ret_l = RetinaV2(device=DEVICE)
        pre_l = SpikingPretectum(device=DEVICE)
        okr_left = _run_condition(ret_l, pre_l, stim_left=True, stim_right=False)

        # --- Bilateral static ---
        ret_s = RetinaV2(device=DEVICE)
        pre_s = SpikingPretectum(device=DEVICE)
        okr_static_vals = []
        for step in range(N_WARM + N_EVAL):
            I = torch.tensor(_make_eye(_grating(0.0)), device=DEVICE)
            rgc = ret_s(I.clone(), I.clone(), entity_info={})
            pret = pre_s(float(rgc['L_ds'].mean()), float(rgc['R_ds'].mean()), eye_velocity=0.0)
            if step >= N_WARM:
                okr_static_vals.append(pret['okr_velocity'])
        okr_static = float(np.mean(okr_static_vals))
        okr_static_abs = abs(okr_static)

        directional_accuracy = (
            1.0 if (okr_right > 0.0 and okr_left < 0.0 and okr_right > okr_left)
            else 0.0
        )

        metrics = {
            'okr_right_eye':       okr_right,
            'okr_left_eye':        okr_left,
            'okr_static_abs':      okr_static_abs,
            'directional_accuracy': directional_accuracy,
        }
        pass_criteria = {
            'okr_right_eye':       ('>', 0.0),
            'okr_left_eye':        ('<', 0.0),
            'okr_static_abs':      ('<', 0.15),
            'directional_accuracy': ('>', 0.9),
        }
        return self._result(metrics, pass_criteria)
