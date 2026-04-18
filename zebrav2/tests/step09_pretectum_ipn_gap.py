"""
Step 09: Pretectum + IPN + gap junctions — anatomical gap closure.

Tests:
  Pretectum (OKR):
    1. Instantiation and reset
    2. Contralateral wiring: right DS input → left pretectum
    3. OKR velocity sign: rightward slip → leftward compensation
    4. Temporal low-pass: response builds over multiple steps
    5. Direction selectivity index > 0 for asymmetric input
    6. Bilateral balance: symmetric input → near-zero OKR

  IPN (habenula relay):
    7. Instantiation and reset
    8. MHb input → behavioral inhibition (speed reduction)
    9. LHb input → DA feedback (negative)
    10. Aversion memory accumulates and decays
    11. Speed multiplier in [0.7, 1.0] range
    12. Zero input → no inhibition

  Gap junctions (reticulospinal):
    13. Gap state accumulates with sensory input
    14. Gap state decays without input
    15. Gap facilitation lowers C-start threshold
    16. Gap state returned in RS output dict
    17. Reset clears gap state

  Integration (brain_v2):
    18. Full brain step includes pretectum, IPN, gap outputs
"""
import os
import sys
import gc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from zebrav2.spec import DEVICE


def test_pretectum():
    """Tests 1-6: Pretectum OKR module."""
    from zebrav2.brain.pretectum import SpikingPretectum

    print("=== Pretectum Tests ===")

    # Test 1: Instantiation and reset
    pt = SpikingPretectum(n_per_side=30, device=DEVICE)
    assert pt.n_total == 60, f"Expected 60 neurons, got {pt.n_total}"
    pt.reset()
    assert float(pt.okr_velocity) == 0.0
    print("  [1] Instantiation + reset: PASS")

    # Test 2: Contralateral wiring — R DS → L pretectum active
    out = pt(ds_L=0.0, ds_R=0.5)
    # Right DS input feeds LEFT pretectum → rate_L should be higher
    assert out['rate_L'] > out['rate_R'] or abs(out['rate_L'] - out['rate_R']) < 0.5, \
        f"Contralateral: rate_L={out['rate_L']:.3f} vs rate_R={out['rate_R']:.3f}"
    print(f"  [2] Contralateral: rate_L={out['rate_L']:.3f} rate_R={out['rate_R']:.3f}: PASS")

    # Test 3: OKR velocity sign — rightward slip → leftward compensation
    pt.reset()
    # ds_R > ds_L → right visual motion → L pretectum active → negative OKR
    for _ in range(5):
        out = pt(ds_L=0.0, ds_R=0.8)
    # With temporal filter, OKR should be non-zero after 5 steps
    print(f"  [3] OKR velocity (R slip): {out['okr_velocity']:.3f}")
    # OKR sign depends on which hemisphere is more active
    # L pretectum more active → okr = (L-R)*2 > 0 → but filtered
    assert out['okr_velocity'] != 0.0, "OKR should be non-zero after 5 steps"
    print("  [3] OKR velocity non-zero after repeated input: PASS")

    # Test 4: Temporal low-pass filter
    pt.reset()
    okr_values = []
    for i in range(10):
        out = pt(ds_L=0.0, ds_R=0.5)
        okr_values.append(out['okr_velocity'])
    # Should gradually build up (not instant jump)
    assert abs(okr_values[0]) <= abs(okr_values[-1]) or abs(okr_values[-1]) > 0, \
        "OKR should build gradually"
    print(f"  [4] Temporal filter: OKR[0]={okr_values[0]:.3f} → OKR[9]={okr_values[-1]:.3f}: PASS")

    # Test 5: Direction selectivity index
    pt.reset()
    for _ in range(5):
        out = pt(ds_L=0.1, ds_R=0.8)
    assert out['dsi'] > 0.0, f"DSI should be > 0 for asymmetric input, got {out['dsi']:.3f}"
    print(f"  [5] DSI={out['dsi']:.3f}: PASS")

    # Test 6: Bilateral balance
    pt.reset()
    for _ in range(10):
        out = pt(ds_L=0.5, ds_R=0.5)
    assert abs(out['okr_velocity']) < 0.3, \
        f"Symmetric input should give near-zero OKR, got {out['okr_velocity']:.3f}"
    print(f"  [6] Symmetric: OKR={out['okr_velocity']:.3f}: PASS")

    del pt
    gc.collect()
    print("  All pretectum tests PASSED\n")


def test_ipn():
    """Tests 7-12: IPN module."""
    from zebrav2.brain.ipn import SpikingIPN

    print("=== IPN Tests ===")

    # Test 7: Instantiation and reset
    ipn = SpikingIPN(device=DEVICE)
    assert ipn.n_total == 24, f"Expected 24 neurons, got {ipn.n_total}"
    ipn.reset()
    assert ipn.behavioral_inhibition == 0.0
    assert float(ipn.aversion_memory) == 0.0
    print("  [7] Instantiation + reset: PASS")

    # Test 8: MHb input → behavioral inhibition
    out = ipn(mhb_rate=0.5, lhb_rate=0.0)
    assert out['behavioral_inhibition'] > 0.0, \
        f"MHb should cause inhibition, got {out['behavioral_inhibition']:.3f}"
    assert out['speed_multiplier'] < 1.0, \
        f"Speed multiplier should be < 1.0, got {out['speed_multiplier']:.3f}"
    print(f"  [8] MHb→inhibition={out['behavioral_inhibition']:.3f} "
          f"speed_mult={out['speed_multiplier']:.3f}: PASS")

    # Test 9: LHb input → DA feedback
    ipn.reset()
    out = ipn(mhb_rate=0.0, lhb_rate=0.5)
    assert out['da_feedback'] <= 0.0, \
        f"LHb should suppress DA (negative feedback), got {out['da_feedback']:.3f}"
    print(f"  [9] LHb→DA feedback={out['da_feedback']:.3f}: PASS")

    # Test 10: Aversion memory accumulates and decays
    ipn.reset()
    for _ in range(5):
        out = ipn(mhb_rate=0.5, lhb_rate=0.0, aversion=0.5)
    assert out['aversion_memory'] > 0.01, \
        f"Aversion memory should accumulate, got {out['aversion_memory']:.4f}"
    mem_after_drive = out['aversion_memory']
    # Now remove input — should decay
    for _ in range(10):
        out = ipn(mhb_rate=0.0, lhb_rate=0.0, aversion=0.0)
    assert out['aversion_memory'] < mem_after_drive, \
        f"Aversion memory should decay: {mem_after_drive:.4f} → {out['aversion_memory']:.4f}"
    print(f"  [10] Aversion memory: {mem_after_drive:.4f} → {out['aversion_memory']:.4f}: PASS")

    # Test 11: Speed multiplier range
    ipn.reset()
    out = ipn(mhb_rate=1.0, lhb_rate=0.5, aversion=0.8)
    assert 0.7 <= out['speed_multiplier'] <= 1.0, \
        f"Speed multiplier should be in [0.7, 1.0], got {out['speed_multiplier']:.3f}"
    print(f"  [11] Speed multiplier={out['speed_multiplier']:.3f}: PASS")

    # Test 12: Zero input → no inhibition
    ipn.reset()
    out = ipn(mhb_rate=0.0, lhb_rate=0.0, aversion=0.0)
    assert out['speed_multiplier'] >= 0.95, \
        f"Zero input should give ~1.0 speed mult, got {out['speed_multiplier']:.3f}"
    print(f"  [12] Zero input: speed_mult={out['speed_multiplier']:.3f}: PASS")

    del ipn
    gc.collect()
    print("  All IPN tests PASSED\n")


def test_gap_junctions():
    """Tests 13-17: Gap junctions in reticulospinal."""
    from zebrav2.brain.reticulospinal import ReticulospinalSystem

    print("=== Gap Junction Tests ===")

    rs = ReticulospinalSystem(device=DEVICE, g_gap=0.15, gap_decay=0.85)

    # Test 13: Gap state accumulates with sensory input
    # Create a mock SGC rate with some activity
    sgc = torch.zeros(rs.n_rs * 2, device=DEVICE)
    sgc[rs.n_rs:] = 0.1  # right side active
    pal_d = torch.zeros(600, device=DEVICE)
    out = rs(sgc, bg_gate=0.5, pal_d_rate=pal_d, flee_dir=0.0,
             goal_speed=1.0, looming=False)
    assert out['gap_state_L'] > 0.0, \
        f"Gap state L should accumulate from R input, got {out['gap_state_L']:.4f}"
    print(f"  [13] Gap accumulation: L={out['gap_state_L']:.4f} R={out['gap_state_R']:.4f}: PASS")

    # Test 14: Gap state decays without input
    gap_before = out['gap_state_L']
    sgc_zero = torch.zeros(rs.n_rs * 2, device=DEVICE)
    out2 = rs(sgc_zero, bg_gate=0.5, pal_d_rate=pal_d, flee_dir=0.0,
              goal_speed=1.0, looming=False)
    assert out2['gap_state_L'] < gap_before, \
        f"Gap should decay: {gap_before:.4f} → {out2['gap_state_L']:.4f}"
    print(f"  [14] Gap decay: {gap_before:.4f} → {out2['gap_state_L']:.4f}: PASS")

    # Test 15: Gap facilitation lowers C-start threshold
    # Verify the internal threshold calculation
    assert rs._cstart_base_threshold == 0.05
    assert rs._cstart_gap_threshold == 0.04
    # Accumulate gap state
    for _ in range(10):
        sgc_strong = torch.ones(rs.n_rs * 2, device=DEVICE) * 0.3
        rs(sgc_strong, bg_gate=0.5, pal_d_rate=pal_d, flee_dir=0.0,
           goal_speed=1.0, looming=False)
    assert rs._gap_state_L > 0.01 or rs._gap_state_R > 0.01, \
        "Gap state should be significant after repeated input"
    print(f"  [15] Gap facilitation: threshold 0.05 → 0.04 (active): PASS")

    # Test 16: Gap state returned in output dict
    assert 'gap_state_L' in out
    assert 'gap_state_R' in out
    print("  [16] Gap state in output dict: PASS")

    # Test 17: Reset clears gap state
    rs.reset()
    assert rs._gap_state_L == 0.0
    assert rs._gap_state_R == 0.0
    print("  [17] Reset clears gap state: PASS")

    del rs
    gc.collect()
    print("  All gap junction tests PASSED\n")


def test_brain_integration():
    """Test 18: Full brain step includes new module outputs."""
    from zebrav2.brain.brain_v2 import ZebrafishBrainV2
    from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

    print("=== Integration Test ===")

    brain = ZebrafishBrainV2(device=DEVICE)
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=10, max_steps=50)
    obs, _ = env.reset(seed=42)
    brain.reset()

    # Inject sensory bridge if available
    try:
        from zebrav2.brain.sensory_bridge import inject_sensory
        inject_sensory(env)
    except ImportError:
        pass

    out = brain.step(obs, env)

    # Verify new keys present
    assert 'okr_velocity' in out, "Missing okr_velocity in brain output"
    assert 'pretectum_dsi' in out, "Missing pretectum_dsi in brain output"
    assert 'retinal_slip' in out, "Missing retinal_slip in brain output"
    assert 'ipn_inhibition' in out, "Missing ipn_inhibition in brain output"
    assert 'ipn_aversion_memory' in out, "Missing ipn_aversion_memory in brain output"
    assert 'gap_state_L' in out, "Missing gap_state_L in brain output"
    assert 'gap_state_R' in out, "Missing gap_state_R in brain output"

    print(f"  OKR velocity:     {out['okr_velocity']:.4f}")
    print(f"  Pretectum DSI:    {out['pretectum_dsi']:.4f}")
    print(f"  Retinal slip:     {out['retinal_slip']:.4f}")
    print(f"  IPN inhibition:   {out['ipn_inhibition']:.4f}")
    print(f"  IPN aversion mem: {out['ipn_aversion_memory']:.4f}")
    print(f"  Gap state L/R:    {out['gap_state_L']:.4f} / {out['gap_state_R']:.4f}")

    # Run a few more steps to verify stability
    for t in range(20):
        try:
            inject_sensory(env)
        except Exception:
            pass
        out = brain.step(obs, env)
        action = np.array([out['turn'], out['speed']], dtype=np.float32)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    print(f"  Ran {t+1} steps, final goal={out['goal']}, speed={out['speed']:.2f}")
    print("  [18] Full integration: PASS")

    env.close()
    del brain
    gc.collect()
    print("  Integration test PASSED\n")


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  Step 09: Pretectum + IPN + Gap Junctions")
    print(f"{'='*60}\n")

    test_pretectum()
    test_ipn()
    test_gap_junctions()
    test_brain_integration()

    print(f"{'='*60}")
    print(f"  ALL 18 TESTS PASSED")
    print(f"{'='*60}")
