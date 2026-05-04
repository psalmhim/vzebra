"""
Tests for SpikingMauthner module.

Run from project root:
    .venv/bin/python zebrav2/tests/test_mauthner.py
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from zebrav2.brain.mauthner import SpikingMauthner


def make_sgc(strength: float, n: int = 100) -> torch.Tensor:
    return torch.ones(n) * strength


def fresh() -> SpikingMauthner:
    m = SpikingMauthner(device='cpu')
    m.reset()
    return m


# ---------------------------------------------------------------------------
# Test 1: No spontaneous firing
# ---------------------------------------------------------------------------
def test_no_spontaneous_firing():
    m = fresh()
    sgc_zero = make_sgc(0.0)
    for _ in range(10):
        out = m(sgc_zero, sgc_zero, ll_pressure=0.0)
    assert not out['m_L_spike'], "M_L should not fire with zero input"
    assert not out['m_R_spike'], "M_R should not fire with zero input"
    print("PASS  test_no_spontaneous_firing")


# ---------------------------------------------------------------------------
# Test 2: Laterality — right-field threat (sgc_R) drives LEFT M-cell
# ---------------------------------------------------------------------------
def test_right_threat_fires_left_mcell():
    m = fresh()
    sgc_zero = make_sgc(0.0)
    sgc_strong = make_sgc(1.0)  # strong right-tectum activity

    out = m(sgc_zero, sgc_strong, ll_pressure=0.0)
    assert out['m_L_spike'], "Strong sgc_R should fire LEFT M-cell (contralateral)"
    assert not out['m_R_spike'], "Strong sgc_R alone should NOT fire right M-cell"
    print("PASS  test_right_threat_fires_left_mcell")


# ---------------------------------------------------------------------------
# Test 3: Laterality — left-field threat (sgc_L) drives RIGHT M-cell
# ---------------------------------------------------------------------------
def test_left_threat_fires_right_mcell():
    m = fresh()
    sgc_zero = make_sgc(0.0)
    sgc_strong = make_sgc(1.0)  # strong left-tectum activity

    out = m(sgc_strong, sgc_zero, ll_pressure=0.0)
    assert out['m_R_spike'], "Strong sgc_L should fire RIGHT M-cell (contralateral)"
    assert not out['m_L_spike'], "Strong sgc_L alone should NOT fire left M-cell"
    print("PASS  test_left_threat_fires_right_mcell")


# ---------------------------------------------------------------------------
# Test 4: Acoustic (ll_pressure) fires bilaterally
# ---------------------------------------------------------------------------
def test_acoustic_fires_bilaterally():
    m = fresh()
    sgc_zero = make_sgc(0.0)

    out = m(sgc_zero, sgc_zero, ll_pressure=1.0)
    assert out['m_L_spike'], "Strong ll_pressure should fire LEFT M-cell"
    assert out['m_R_spike'], "Strong ll_pressure should fire RIGHT M-cell"
    print("PASS  test_acoustic_fires_bilaterally")


# ---------------------------------------------------------------------------
# Test 5: CoLo crossed inhibition
#   - After M_L fires, colo_L_active=True
#   - After M_R fires, colo_R_active=True
# ---------------------------------------------------------------------------
def test_colo_crossed_inhibition():
    # Trigger M_L (via sgc_R) and check colo_L_active
    m = fresh()
    sgc_zero = make_sgc(0.0)
    sgc_strong = make_sgc(1.0)

    out = m(sgc_zero, sgc_strong, ll_pressure=0.0)
    assert out['m_L_spike'], "Prerequisite: M_L must fire"
    assert out['colo_L_active'], "colo_L_active must be True when M_L fires"

    # Trigger M_R (via sgc_L) and check colo_R_active
    m = fresh()
    out = m(sgc_strong, sgc_zero, ll_pressure=0.0)
    assert out['m_R_spike'], "Prerequisite: M_R must fire"
    assert out['colo_R_active'], "colo_R_active must be True when M_R fires"

    print("PASS  test_colo_crossed_inhibition")


# ---------------------------------------------------------------------------
# Test 6: Gap junction accumulation
#   Repeated sub-threshold sgc_R (0.03) should eventually lower threshold
#   enough to fire L M-cell within 20 steps via club-ending facilitation.
#   sgc_R=0.03 → I_vis_L = 3.6 pA, insufficient alone on step 1.
# ---------------------------------------------------------------------------
def test_gap_junction_accumulation():
    sgc_sub = make_sgc(0.03)
    sgc_zero = make_sgc(0.0)

    # Verify it does NOT fire on the very first step
    m = fresh()
    out_first = m(sgc_zero, sgc_sub, ll_pressure=0.0)
    assert not out_first['m_L_spike'], (
        "Sub-threshold sgc_R (0.03) should not fire M_L on first step"
    )

    # Reset and run up to 20 steps; gap accumulation should eventually fire M_L
    m = fresh()
    fired = False
    for _ in range(20):
        out = m(sgc_zero, sgc_sub, ll_pressure=0.0)
        if out['m_L_spike']:
            fired = True
            break

    assert fired, (
        f"Gap junction accumulation should trigger M_L within 20 sub-threshold steps "
        f"(sgc_R=0.03); gap_L reached {float(m.gap_L):.4f} without firing"
    )
    print("PASS  test_gap_junction_accumulation")


# ---------------------------------------------------------------------------
# Test 7: Rate tracking — after firing, rate > 0
# ---------------------------------------------------------------------------
def test_rate_tracking_after_firing():
    # Left M-cell rate after sgc_R fires it
    m = fresh()
    sgc_zero = make_sgc(0.0)
    sgc_strong = make_sgc(1.0)

    out = m(sgc_zero, sgc_strong, ll_pressure=0.0)
    assert out['m_L_spike'], "Prerequisite: M_L must have fired"
    assert out['m_L_rate'] > 0, (
        f"m_L_rate should be > 0 after M_L fired, got {out['m_L_rate']}"
    )

    # Right M-cell rate after sgc_L fires it
    m = fresh()
    out = m(sgc_strong, sgc_zero, ll_pressure=0.0)
    assert out['m_R_spike'], "Prerequisite: M_R must have fired"
    assert out['m_R_rate'] > 0, (
        f"m_R_rate should be > 0 after M_R fired, got {out['m_R_rate']}"
    )

    print("PASS  test_rate_tracking_after_firing")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    tests = [
        test_no_spontaneous_firing,
        test_right_threat_fires_left_mcell,
        test_left_threat_fires_right_mcell,
        test_acoustic_fires_bilaterally,
        test_colo_crossed_inhibition,
        test_gap_junction_accumulation,
        test_rate_tracking_after_firing,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
