"""
Action-perception cycle tests — active inference motor, spontaneity, FE tracking.

Tests:
  1. Iterative convergence (multi-pass FE reduction)
  2. Adaptive blend (precision-dependent, schizophrenia reduces it)
  3. Spontaneity triggers (baseline, habenula, boredom)
  4. FE gradient tracking (dF/dt computation)
  5. F-based goal modulation (code path doesn't crash)
  6. Multi-pass count (default n_inference_passes == 3)
  7. Return dict keys (ai_convergence, fe_gradient, spontaneity)

Run:
  .venv/bin/python -m zebrav2.tests.test_action_perception_cycle
"""
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.active_motor import ActiveInferenceMotor
from zebrav2.spec import DEVICE

results = {}
pass_count = 0
fail_count = 0


def _record(key, passed, desc):
    global pass_count, fail_count
    status = 'PASS' if passed else 'FAIL'
    results[key] = (status, desc)
    tag = '[OK]  ' if passed else '[FAIL]'
    print(f"  {tag} {key}: {desc}")
    if passed:
        pass_count += 1
    else:
        fail_count += 1


def _make_env():
    """Minimal mock env for brain.step()."""
    class E:
        brain_L = np.zeros(800, dtype=np.float32)
        brain_R = np.zeros(800, dtype=np.float32)
        _enemy_pixels_total = 0
        fish_x = 400; fish_y = 300; fish_heading = 0.0; fish_energy = 80.0
        pred_x = -9999; pred_y = -9999
        foods = [[420, 300, 's']]
        rock_formations = []; all_fish = []
        gaze_offset = 0.0; arena_w = 800; arena_h = 600
    env = E()
    env.brain_R[100:120] = 0.8
    env.brain_R[500:520] = 0.8
    return env


# ---------------------------------------------------------------------------
# Test 1: Iterative convergence
# ---------------------------------------------------------------------------

def test_iterative_convergence():
    """ActiveInferenceMotor.step() _fe_per_pass shows convergence."""
    print("\n=== Test 1: Iterative Convergence ===")

    motor = ActiveInferenceMotor(device=DEVICE)

    # Run several steps with varying inputs
    converged_count = 0
    n_trials = 5
    for i in range(n_trials):
        motor.step(
            goal=0, food_bearing=0.3 * (i + 1) / n_trials,
            enemy_bearing=0.0, wall_proximity=0.0,
            food_visible=True, enemy_visible=False,
            gaze_target=0.0, explore_phase=0.0,
            DA=0.5, NA=0.3, HT5=0.5, ACh=0.5,
            actual_speed=1.0, heading_delta=0.1,
            tail_L=0.2, tail_R=0.2,
            gaze_offset=0.0, collision=False, turn_rate=0.0,
        )

        fe_passes = motor._fe_per_pass
        if len(fe_passes) >= 3:
            # Check FE decreased or PE absolute value decreased across passes
            if fe_passes[0] >= fe_passes[2]:
                converged_count += 1

    ok = converged_count > 0
    _record(
        'CONVERGENCE_iterative', ok,
        f"converged in {converged_count}/{n_trials} trials "
        f"(pass1 FE >= pass3 FE)"
    )
    return {'converged_count': converged_count, 'n_trials': n_trials, 'ok': ok}


# ---------------------------------------------------------------------------
# Test 2: Adaptive blend
# ---------------------------------------------------------------------------

def test_adaptive_blend():
    """ai_blend is NOT fixed 0.3; schizophrenia (low precision) lowers it."""
    print("\n=== Test 2: Adaptive Blend ===")

    obs = np.zeros(10)

    # --- Wildtype brain ---
    brain_wt = ZebrafishBrainV2(device=DEVICE)
    brain_wt.reset()
    env_wt = _make_env()

    out_wt = None
    for _ in range(3):
        out_wt = brain_wt.step(obs, env_wt)

    wt_blend = out_wt['ai_blend']
    not_fixed = abs(wt_blend - 0.3) > 1e-4
    _record(
        'BLEND_not_fixed', not_fixed,
        f"wildtype ai_blend={wt_blend:.4f} (expected != 0.3)"
    )

    # --- Schizophrenia brain (low precision → lower blend) ---
    from zebrav2.brain.disorder import apply_disorder

    brain_scz = ZebrafishBrainV2(device=DEVICE)
    brain_scz.reset()
    apply_disorder(brain_scz, 'schizophrenia', intensity=1.0)

    env_scz = _make_env()
    out_scz = None
    for _ in range(3):
        out_scz = brain_scz.step(obs, env_scz)

    scz_blend = out_scz['ai_blend']
    scz_lower = scz_blend < wt_blend
    _record(
        'BLEND_scz_lower', scz_lower,
        f"schiz ai_blend={scz_blend:.4f} < wildtype={wt_blend:.4f}"
    )

    return {
        'wt_blend': wt_blend,
        'scz_blend': scz_blend,
        'not_fixed': not_fixed,
        'scz_lower': scz_lower,
    }


# ---------------------------------------------------------------------------
# Test 3: Spontaneity triggers
# ---------------------------------------------------------------------------

def test_spontaneity_triggers():
    """_compute_spontaneity() returns > 0.05 baseline; habenula and boredom increase it."""
    print("\n=== Test 3: Spontaneity Triggers ===")

    obs = np.zeros(10)

    # --- Baseline spontaneity ---
    brain = ZebrafishBrainV2(device=DEVICE)
    brain.reset()
    env = _make_env()

    brain.step(obs, env)
    s_baseline = brain._spontaneity
    baseline_ok = s_baseline >= 0.05
    _record(
        'SPONT_baseline', baseline_ok,
        f"baseline spontaneity={s_baseline:.4f} (>= 0.05 neural noise)"
    )

    # --- High habenula helplessness → higher spontaneity ---
    brain2 = ZebrafishBrainV2(device=DEVICE)
    brain2.reset()
    env2 = _make_env()

    # Warm up one step first
    brain2.step(obs, env2)
    s_before_hab = brain2._spontaneity

    # Inject high helplessness
    brain2.habenula.helplessness = 0.8
    brain2.step(obs, env2)
    s_with_hab = brain2._spontaneity

    hab_ok = s_with_hab > s_before_hab
    _record(
        'SPONT_habenula', hab_ok,
        f"hab helplessness=0.8 → spontaneity={s_with_hab:.4f} > "
        f"before={s_before_hab:.4f}"
    )

    # --- Boredom: flat FE gradient + low novelty → higher spontaneity ---
    brain3 = ZebrafishBrainV2(device=DEVICE)
    brain3.reset()
    env3 = _make_env()

    brain3.step(obs, env3)
    s_before_bore = brain3._spontaneity

    # Force boredom conditions
    brain3._fe_gradient = 0.0
    brain3._novelty_ema = 0.1
    brain3.step(obs, env3)
    s_bored = brain3._spontaneity

    boredom_ok = s_bored > s_before_bore
    _record(
        'SPONT_boredom', boredom_ok,
        f"boredom (fe_grad~0, novelty=0.1) → spontaneity={s_bored:.4f} > "
        f"before={s_before_bore:.4f}"
    )

    return {
        's_baseline': s_baseline,
        's_with_hab': s_with_hab,
        's_bored': s_bored,
        'baseline_ok': baseline_ok,
        'hab_ok': hab_ok,
        'boredom_ok': boredom_ok,
    }


# ---------------------------------------------------------------------------
# Test 4: FE gradient tracking
# ---------------------------------------------------------------------------

def test_fe_gradient_tracking():
    """After 5+ steps, _fe_gradient is non-zero and _fe_history has entries."""
    print("\n=== Test 4: FE Gradient Tracking ===")

    obs = np.zeros(10)
    brain = ZebrafishBrainV2(device=DEVICE)
    brain.reset()
    env = _make_env()

    fe_gradients = []
    for i in range(6):
        brain.step(obs, env)
        fe_gradients.append(brain._fe_gradient)

    # After step 2 (index 1), _fe_gradient should be non-zero
    # (needs at least 2 entries in _fe_history)
    nonzero_after_2 = any(abs(g) > 1e-8 for g in fe_gradients[2:])
    has_history = len(brain._fe_history) >= 2

    grad_ok = nonzero_after_2 and has_history
    _record(
        'FE_GRADIENT_tracked', grad_ok,
        f"nonzero after step 2={nonzero_after_2}, "
        f"history len={len(brain._fe_history)}, "
        f"gradients={[round(g, 4) for g in fe_gradients]}"
    )

    return {
        'fe_gradients': fe_gradients,
        'history_len': len(brain._fe_history),
        'ok': grad_ok,
    }


# ---------------------------------------------------------------------------
# Test 5: F-based goal modulation (code path test)
# ---------------------------------------------------------------------------

def test_fe_goal_modulation():
    """Set _fe_gradient high positive, verify EFE code path doesn't crash."""
    print("\n=== Test 5: F-Based Goal Modulation ===")

    obs = np.zeros(10)
    brain = ZebrafishBrainV2(device=DEVICE)
    brain.reset()
    env = _make_env()

    # Run a few steps to initialize state
    for _ in range(3):
        brain.step(obs, env)

    # Force high positive FE gradient (= current goal failing)
    brain._fe_gradient = 0.5
    brain._step_count = 20  # past the threshold check (> 10)

    # This should not crash and should produce a valid output
    crashed = False
    try:
        out = brain.step(obs, env)
        has_turn = 'turn' in out
        has_speed = 'speed' in out
    except Exception as e:
        crashed = True
        has_turn = False
        has_speed = False
        print(f"    EXCEPTION: {e}")

    ok = (not crashed) and has_turn and has_speed
    _record(
        'FE_GOAL_modulation', ok,
        f"fe_gradient=0.5 step survived without crash={not crashed}, "
        f"has_turn={has_turn}, has_speed={has_speed}"
    )

    return {'crashed': crashed, 'ok': ok}


# ---------------------------------------------------------------------------
# Test 6: Multi-pass count
# ---------------------------------------------------------------------------

def test_multi_pass_count():
    """ActiveInferenceMotor.n_inference_passes == 3 by default."""
    print("\n=== Test 6: Multi-Pass Count ===")

    motor = ActiveInferenceMotor(device=DEVICE)
    n = motor.n_inference_passes
    ok = n == 3
    _record(
        'MULTI_PASS_count', ok,
        f"n_inference_passes={n} (expected 3)"
    )
    return {'n_inference_passes': n, 'ok': ok}


# ---------------------------------------------------------------------------
# Test 7: Return dict keys
# ---------------------------------------------------------------------------

def test_return_dict_keys():
    """brain.step() returns 'ai_convergence', 'fe_gradient', 'spontaneity'."""
    print("\n=== Test 7: Return Dict Keys ===")

    obs = np.zeros(10)
    brain = ZebrafishBrainV2(device=DEVICE)
    brain.reset()
    env = _make_env()

    out = brain.step(obs, env)

    required_keys = ['ai_convergence', 'fe_gradient', 'spontaneity']
    missing = [k for k in required_keys if k not in out]

    ok = len(missing) == 0
    _record(
        'RETURN_KEYS', ok,
        f"required keys present={ok}, missing={missing}"
    )
    return {'missing': missing, 'ok': ok}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 60)
    print("Action-Perception Cycle — Test Suite")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    test_iterative_convergence()
    test_adaptive_blend()
    test_spontaneity_triggers()
    test_fe_gradient_tracking()
    test_fe_goal_modulation()
    test_multi_pass_count()
    test_return_dict_keys()

    print("\n" + "=" * 60)
    print(f"Results: {pass_count} PASS / {fail_count} FAIL")
    print("=" * 60)
    for key, (status, desc) in results.items():
        tag = '[OK]  ' if status == 'PASS' else '[FAIL]'
        print(f"  {tag} {key}: {desc}")

    return pass_count, fail_count


if __name__ == '__main__':
    p, f = run_all()
    sys.exit(1 if f > 0 else 0)
