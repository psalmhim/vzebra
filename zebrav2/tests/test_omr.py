"""
OMR (optomotor response) benchmark for ZebrafishBrainV2.

The optomotor response is a whole-field visual motion reflex: the fish
turns/swims in the same direction as perceived environmental motion.

Circuit under test:
  Retina DS cells → Pretectum (contralateral) → okr_velocity → turn

Test protocol mirrors Naumann et al. 2016 / Kubo et al. 2014:
  - Moving sinusoidal grating (spatial period 40px, temporal phase shift 4px/step)
  - Three conditions: right-eye-only, left-eye-only, bilateral
  - Directional prediction: monocular → turn toward stimulated side
  - DS cells use spatial-shift Reichardt correlator (Hassenstein & Reichardt 1956)

Anatomy:
  Left retina  → Right pretectum  (contralateral optic chiasm crossing)
  Right retina → Left pretectum
  Left pretectum more active → positive OKR → rightward turn
"""

import math
import numpy as np
import torch
from zebrav2.brain.retina import RetinaV2
from zebrav2.brain.pretectum import SpikingPretectum

DEVICE = 'cpu'
N_PIX  = 400          # intensity pixels per eye
PERIOD = 40           # grating spatial period (pixels)
SHIFT  = 4            # pixels per step (temporal frequency)
N_WARM = 10           # warm-up steps (fill retina temporal buffers)
N_EVAL = 30           # evaluation steps


# ── Stimulus generator ────────────────────────────────────────────────────────

def grating(phase: float) -> np.ndarray:
    """Sinusoidal intensity grating for one eye at a given phase (radians)."""
    k = np.arange(N_PIX, dtype=np.float32)
    return (0.5 + 0.5 * np.sin(2 * math.pi * k / PERIOD + phase)).astype(np.float32)


def make_eye(intensity: np.ndarray) -> np.ndarray:
    """Pack intensity into 800-element eye array ([:400]=intensity, [400:]=type=0)."""
    eye = np.zeros(800, dtype=np.float32)
    eye[:N_PIX] = intensity
    return eye


def run_condition(retina: RetinaV2, pretectum: SpikingPretectum,
                  stim_left: bool, stim_right: bool,
                  n_warm: int = N_WARM, n_eval: int = N_EVAL):
    """
    Run N steps with a moving grating (rightward motion: phase increases).
    Returns mean ds_L, ds_R, okr_velocity over evaluation steps.
    """
    retina.reset() if hasattr(retina, 'reset') else None
    pretectum.reset()

    phase = 0.0
    phase_step = 2 * math.pi * SHIFT / PERIOD   # radians per step

    ds_L_log, ds_R_log, okr_log = [], [], []

    for step in range(n_warm + n_eval):
        phase += phase_step
        L_int = grating(phase) if stim_left  else np.zeros(N_PIX, np.float32)
        R_int = grating(phase) if stim_right else np.zeros(N_PIX, np.float32)

        L = torch.tensor(make_eye(L_int), device=DEVICE)
        R = torch.tensor(make_eye(R_int), device=DEVICE)

        rgc = retina(L, R, entity_info={})
        ds_L = float(rgc['L_ds'].mean())
        ds_R = float(rgc['R_ds'].mean())
        pret = pretectum(ds_L, ds_R, eye_velocity=0.0)
        okr  = pret['okr_velocity']

        if step >= n_warm:
            ds_L_log.append(ds_L)
            ds_R_log.append(ds_R)
            okr_log.append(okr)

    return np.mean(ds_L_log), np.mean(ds_R_log), np.mean(okr_log)


# ── Checks ────────────────────────────────────────────────────────────────────

def check(label: str, condition: bool, detail: str = ''):
    status = 'PASS' if condition else 'FAIL'
    msg = f'  {status}  {label}'
    if detail:
        msg += f'  ({detail})'
    print(msg)
    return condition


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print('=' * 60)
    print('OMR Benchmark: Optomotor Response')
    print('=' * 60)
    print(f'Grating: period={PERIOD}px, velocity={SHIFT}px/step (rightward)')
    print()

    retina    = RetinaV2(device=DEVICE)
    pretectum = SpikingPretectum(device=DEVICE)

    passed = 0
    total  = 0

    # ── DS directionality unit test ───────────────────────────────────────────
    print('--- Test 1: DS cell directionality (retina only) ---')
    retina_test = RetinaV2(device=DEVICE)

    # Right-moving grating (phase increases) → ds_net > 0
    phase = 0.0
    ds_right_vals = []
    for _ in range(N_WARM + N_EVAL):
        phase += 2 * math.pi * SHIFT / PERIOD
        L = torch.tensor(make_eye(grating(phase)), device=DEVICE)
        R = torch.tensor(make_eye(grating(phase)), device=DEVICE)
        out = retina_test(L, R, entity_info={})
        ds_right_vals.append((float(out['L_ds'].mean()), float(out['R_ds'].mean())))
    ds_r_L = np.mean([x[0] for x in ds_right_vals[N_WARM:]])
    ds_r_R = np.mean([x[1] for x in ds_right_vals[N_WARM:]])

    # Left-moving grating (phase decreases) → ds_net < 0
    retina_test2 = RetinaV2(device=DEVICE)
    phase = 0.0
    ds_left_vals = []
    for _ in range(N_WARM + N_EVAL):
        phase -= 2 * math.pi * SHIFT / PERIOD
        L = torch.tensor(make_eye(grating(phase)), device=DEVICE)
        R = torch.tensor(make_eye(grating(phase)), device=DEVICE)
        out = retina_test2(L, R, entity_info={})
        ds_left_vals.append((float(out['L_ds'].mean()), float(out['R_ds'].mean())))
    ds_l_L = np.mean([x[0] for x in ds_left_vals[N_WARM:]])
    ds_l_R = np.mean([x[1] for x in ds_left_vals[N_WARM:]])

    total += 1; passed += check(
        'Rightward grating → positive mean DS (left eye)',
        ds_r_L > 0,
        f'ds_L={ds_r_L:.4f}')
    total += 1; passed += check(
        'Rightward grating → positive mean DS (right eye)',
        ds_r_R > 0,
        f'ds_R={ds_r_R:.4f}')
    total += 1; passed += check(
        'Leftward grating → negative mean DS (left eye)',
        ds_l_L < 0,
        f'ds_L={ds_l_L:.4f}')
    total += 1; passed += check(
        'Leftward grating → negative mean DS (right eye)',
        ds_l_R < 0,
        f'ds_R={ds_l_R:.4f}')
    total += 1; passed += check(
        'Rightward > leftward DS (left eye)',
        ds_r_L > ds_l_L,
        f'{ds_r_L:.4f} > {ds_l_L:.4f}')

    # ── OMR condition 1: right eye only (rightward grating) ──────────────────
    print('\n--- Test 2: Right-eye monocular OMR (expect rightward turn) ---')
    retina_r  = RetinaV2(device=DEVICE)
    pretect_r = SpikingPretectum(device=DEVICE)
    ds_L, ds_R, okr = run_condition(retina_r, pretect_r,
                                     stim_left=False, stim_right=True)
    total += 1; passed += check(
        'Right-eye stim: ds_R > 0',
        ds_R > 0, f'ds_R={ds_R:.4f}')
    total += 1; passed += check(
        'Right-eye stim: ds_L ≈ 0 (unstimulated)',
        abs(ds_L) < 0.05, f'ds_L={ds_L:.4f}')
    total += 1; passed += check(
        'Right-eye stim: OKR > 0 (rightward turn)',
        okr > 0, f'okr={okr:.4f}')

    # ── OMR condition 2: left eye only (rightward grating) ───────────────────
    print('\n--- Test 3: Left-eye monocular OMR (expect leftward turn) ---')
    retina_l  = RetinaV2(device=DEVICE)
    pretect_l = SpikingPretectum(device=DEVICE)
    ds_L2, ds_R2, okr2 = run_condition(retina_l, pretect_l,
                                        stim_left=True, stim_right=False)
    total += 1; passed += check(
        'Left-eye stim: ds_L > 0',
        ds_L2 > 0, f'ds_L={ds_L2:.4f}')
    total += 1; passed += check(
        'Left-eye stim: ds_R ≈ 0 (unstimulated)',
        abs(ds_R2) < 0.05, f'ds_R={ds_R2:.4f}')
    total += 1; passed += check(
        'Left-eye stim: OKR < 0 (leftward turn)',
        okr2 < 0, f'okr={okr2:.4f}')

    # ── OMR condition 3: bilateral static (no motion) ────────────────────────
    print('\n--- Test 4: Static bilateral (no motion, expect OKR ≈ 0) ---')
    retina_s  = RetinaV2(device=DEVICE)
    pretect_s = SpikingPretectum(device=DEVICE)

    # Static grating: run with fixed phase
    okr_static_vals = []
    for _ in range(N_WARM + N_EVAL):
        I = torch.tensor(make_eye(grating(0.0)), device=DEVICE)
        out = retina_s(I.clone(), I.clone(), entity_info={})
        p = pretect_s(float(out['L_ds'].mean()), float(out['R_ds'].mean()), eye_velocity=0.0)
        if _ >= N_WARM:
            okr_static_vals.append(p['okr_velocity'])
    okr_static = float(np.mean(okr_static_vals))

    total += 1; passed += check(
        'Static bilateral: |OKR| < 0.3 (near-zero turn)',
        abs(okr_static) < 0.3, f'okr={okr_static:.4f}')

    # ── OMR directionality: right turn > left turn ───────────────────────────
    print('\n--- Test 5: Directional consistency ---')
    total += 1; passed += check(
        'Right-eye OKR > Left-eye OKR (correct ipsiversive direction)',
        okr > okr2, f'{okr:.4f} > {okr2:.4f}')
    total += 1; passed += check(
        'Right-eye OKR > static OKR',
        okr > okr_static, f'{okr:.4f} > {okr_static:.4f}')
    total += 1; passed += check(
        'Left-eye OKR < static OKR',
        okr2 < okr_static, f'{okr2:.4f} < {okr_static:.4f}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(f'Result: {passed}/{total} passed')
    return passed == total


if __name__ == '__main__':
    ok = run()
    raise SystemExit(0 if ok else 1)
