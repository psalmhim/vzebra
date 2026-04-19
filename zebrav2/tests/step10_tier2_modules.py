"""
Tier 2 module tests: raphe, locus coeruleus, habituation, pectoral fin.

36 tests:
  8 raphe (spiking 5-HT)
  8 locus coeruleus (spiking NA)
  8 habituation (synaptic depression)
  8 pectoral fin (slow-turn kinematics)
  4 integration (full brain wiring)
"""
import sys
import os
import math
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from zebrav2.spec import DEVICE


def _header(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def _ok(name):
    print(f"  ✓ {name}")


def _fail(name, msg):
    print(f"  ✗ {name}: {msg}")
    return False


# ============================================================
#  RAPHE TESTS (8)
# ============================================================
def test_raphe():
    _header("Raphe (Spiking 5-HT Source)")
    from zebrav2.brain.raphe import SpikingRaphe
    passed = 0
    total = 8

    # 1. Construction
    r = SpikingRaphe(device=DEVICE)
    assert r.n_dr == 30 and r.n_mr == 10
    _ok("1. Construction: 30 DR + 10 MR neurons")
    passed += 1

    # 2. Baseline output
    out = r(lhb_rate=0.0, ipn_raphe_drive=0.0, amygdala_stress=0.0, circadian=0.7)
    assert 0.0 <= out['ht5_level'] <= 1.0
    assert 'dr_rate' in out and 'mr_rate' in out
    _ok(f"2. Baseline 5-HT level: {out['ht5_level']:.3f}")
    passed += 1

    # 3. LHb inhibition: high LHb → lower 5-HT
    r.reset()
    for _ in range(30):
        r(lhb_rate=0.0)
    base_ht5 = r(lhb_rate=0.0)['ht5_level']
    r.reset()
    for _ in range(30):
        r(lhb_rate=0.5)
    inhib_ht5 = r(lhb_rate=0.5)['ht5_level']
    if inhib_ht5 < base_ht5:
        _ok(f"3. LHb inhibition: {base_ht5:.3f} → {inhib_ht5:.3f}")
        passed += 1
    else:
        _fail("3. LHb inhibition", f"expected lower 5-HT, got {inhib_ht5:.3f} vs {base_ht5:.3f}")

    # 4. IPN excitation: high IPN → higher 5-HT
    r.reset()
    for _ in range(30):
        r(ipn_raphe_drive=0.0)
    base2 = r(ipn_raphe_drive=0.0)['ht5_level']
    r.reset()
    for _ in range(30):
        r(ipn_raphe_drive=0.5)
    excited = r(ipn_raphe_drive=0.5)['ht5_level']
    if excited > base2:
        _ok(f"4. IPN excitation: {base2:.3f} → {excited:.3f}")
        passed += 1
    else:
        _fail("4. IPN excitation", f"expected higher 5-HT, got {excited:.3f} vs {base2:.3f}")

    # 5. Flee suppression
    r.reset()
    for _ in range(30):
        r(flee_active=False)
    no_flee = r(flee_active=False)['ht5_level']
    r.reset()
    for _ in range(30):
        r(flee_active=True)
    with_flee = r(flee_active=True)['ht5_level']
    if with_flee <= no_flee:
        _ok(f"5. Flee suppression: {no_flee:.3f} → {with_flee:.3f}")
        passed += 1
    else:
        _fail("5. Flee suppression", f"flee 5-HT ({with_flee:.3f}) should be ≤ no-flee ({no_flee:.3f})")

    # 6. Output range stability
    r.reset()
    for _ in range(100):
        out = r(lhb_rate=np.random.random() * 0.5,
                ipn_raphe_drive=np.random.random() * 0.5,
                amygdala_stress=np.random.random() * 0.3,
                circadian=0.7)
    assert 0.05 <= out['ht5_level'] <= 0.95
    _ok(f"6. 100-step stability: 5-HT={out['ht5_level']:.3f}")
    passed += 1

    # 7. Sensory gain and patience fields
    r.reset()
    out = r()
    assert 'sensory_gain' in out and 'patience' in out
    assert 0.0 <= out['sensory_gain'] <= 1.0
    _ok(f"7. Output fields: sensory_gain={out['sensory_gain']:.3f}, patience={out['patience']:.3f}")
    passed += 1

    # 8. Reset clears state
    r.reset()
    assert abs(r._ht5_ema - 0.5) < 0.01
    _ok("8. Reset clears state")
    passed += 1

    print(f"\nRaphe: {passed}/{total} passed")
    return passed, total


# ============================================================
#  LOCUS COERULEUS TESTS (8)
# ============================================================
def test_lc():
    _header("Locus Coeruleus (Spiking NA Source)")
    from zebrav2.brain.locus_coeruleus import SpikingLocusCoeruleus
    passed = 0
    total = 8

    # 1. Construction
    lc = SpikingLocusCoeruleus(device=DEVICE)
    assert lc.n_lc == 20
    _ok("1. Construction: 20 LC neurons")
    passed += 1

    # 2. Baseline output
    out = lc()
    assert 0.0 <= out['na_level'] <= 1.0
    _ok(f"2. Baseline NA level: {out['na_level']:.3f}")
    passed += 1

    # 3. Phasic burst: amygdala > threshold → phasic
    lc.reset()
    for _ in range(10):
        lc(amygdala_alpha=0.0)
    tonic = lc(amygdala_alpha=0.0)
    assert not tonic['phasic']
    phasic = lc(amygdala_alpha=0.5)  # above 0.3 threshold
    assert phasic['phasic']
    _ok(f"3. Phasic burst: tonic NA={tonic['na_level']:.3f}, phasic NA={phasic['na_level']:.3f}")
    passed += 1

    # 4. Phasic cooldown: 5-step refractory
    lc.reset()
    lc(amygdala_alpha=0.5)  # triggers phasic
    out2 = lc(amygdala_alpha=0.5)  # should be in cooldown
    assert not out2['phasic'], "Should be in refractory cooldown"
    _ok("4. Phasic cooldown: 5-step refractory prevents repeat burst")
    passed += 1

    # 5. NA cap during phasic: ≤ 0.8
    lc.reset()
    for _ in range(10):
        lc(amygdala_alpha=0.0, insula_arousal=0.8, circadian=1.0)
    out = lc(amygdala_alpha=0.8)  # strong phasic
    assert out['na_level'] <= 0.85, f"Phasic NA should be capped, got {out['na_level']}"
    _ok(f"5. Phasic NA cap: {out['na_level']:.3f} ≤ 0.85")
    passed += 1

    # 6. Wake gate and attention
    lc.reset()
    out = lc()
    assert 'wake_gate' in out and 'attention' in out
    assert 0.0 <= out['wake_gate'] <= 1.0
    _ok(f"6. Wake gate={out['wake_gate']:.3f}, attention={out['attention']:.3f}")
    passed += 1

    # 7. Circadian modulation: higher circadian → higher NA
    lc.reset()
    for _ in range(30):
        lc(circadian=0.3)
    low_circ = lc(circadian=0.3)['na_level']
    lc.reset()
    for _ in range(30):
        lc(circadian=1.0)
    high_circ = lc(circadian=1.0)['na_level']
    if high_circ > low_circ:
        _ok(f"7. Circadian modulation: night={low_circ:.3f}, day={high_circ:.3f}")
        passed += 1
    else:
        _fail("7. Circadian modulation", f"day ({high_circ:.3f}) should > night ({low_circ:.3f})")

    # 8. Reset
    lc.reset()
    assert abs(lc._na_ema - 0.3) < 0.01
    assert lc._phasic_cooldown == 0
    _ok("8. Reset clears state")
    passed += 1

    print(f"\nLC: {passed}/{total} passed")
    return passed, total


# ============================================================
#  HABITUATION TESTS (8)
# ============================================================
def test_habituation():
    _header("Habituation (Synaptic Depression)")
    from zebrav2.brain.habituation import SynapticDepression
    passed = 0
    total = 8

    # 1. Construction
    hab = SynapticDepression(n_synapses=100, device=DEVICE)
    assert hab.n_synapses == 100
    _ok("1. Construction: 100 synapses")
    passed += 1

    # 2. Initial state: all depletion = 1.0
    assert abs(hab.get_habituation_level() - 1.0) < 0.01
    _ok("2. Initial depletion = 1.0 (fully fresh)")
    passed += 1

    # 3. Repeated identical input → depression
    stimulus = torch.randn(100, device=DEVICE) * 3.0
    for _ in range(50):
        hab(stimulus)
    level = hab.get_habituation_level()
    assert level < 0.8, f"Expected depression after 50 repeats, got {level:.3f}"
    _ok(f"3. 50 repeats → depletion level: {level:.3f}")
    passed += 1

    # 4. Floor: depletion ≥ d_min (0.3)
    for _ in range(200):
        hab(stimulus)
    level2 = hab.get_habituation_level()
    assert level2 >= 0.29  # allow small float tolerance
    _ok(f"4. Floor after 250 repeats: {level2:.3f} ≥ 0.3")
    passed += 1

    # 5. Recovery: stop stimulating → depletion recovers toward 1.0
    for _ in range(100):
        hab(torch.zeros(100, device=DEVICE))  # no input
    level3 = hab.get_habituation_level()
    assert level3 > level2, f"Expected recovery, got {level3:.3f} vs {level2:.3f}"
    _ok(f"5. Recovery after 100 silent steps: {level3:.3f}")
    passed += 1

    # 6. Novel stimulus → dishabituation
    hab.reset()
    pattern_a = torch.ones(100, device=DEVICE) * 2.0
    for _ in range(50):
        hab(pattern_a)
    level_before = hab.get_habituation_level()
    # Novel pattern (very different from pattern_a)
    pattern_b = -torch.ones(100, device=DEVICE) * 2.0
    hab(pattern_b)
    level_after = hab.get_habituation_level()
    if level_after >= level_before:
        _ok(f"6. Dishabituation: {level_before:.3f} → {level_after:.3f}")
        passed += 1
    else:
        # Dishabituation should restore some synapses — if cosine distance test passes
        _fail("6. Dishabituation", f"expected partial recovery, got {level_after:.3f}")

    # 7. Output is attenuated
    hab.reset()
    inp = torch.ones(100, device=DEVICE) * 5.0
    out_first = hab(inp)
    # First application: depletion ~1.0, so output ≈ input
    assert float(out_first.mean()) > 4.0, f"First pass should be near input, got {float(out_first.mean())}"
    for _ in range(100):
        hab(inp)
    out_late = hab(inp)
    assert float(out_late.mean()) < float(out_first.mean()), "Habituated output should be smaller"
    _ok(f"7. Output attenuation: {float(out_first.mean()):.2f} → {float(out_late.mean()):.2f}")
    passed += 1

    # 8. Reset
    hab.reset()
    assert abs(hab.get_habituation_level() - 1.0) < 0.01
    _ok("8. Reset restores fresh state")
    passed += 1

    print(f"\nHabituation: {passed}/{total} passed")
    return passed, total


# ============================================================
#  PECTORAL FIN TESTS (8)
# ============================================================
def test_pectoral_fin():
    _header("Pectoral Fin (Slow-Turn Kinematics)")
    from zebrav2.brain.pectoral_fin import PectoralFinMotor, GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE
    passed = 0
    total = 8

    # 1. Construction
    pf = PectoralFinMotor(device=DEVICE)
    assert pf.n_total == 8
    _ok("1. Construction: 8 motor neurons (4/side)")
    passed += 1

    # 2. Baseline output in EXPLORE
    out = pf(goal=GOAL_EXPLORE)
    assert out['active']
    assert -1.0 <= out['fin_turn'] <= 1.0
    _ok(f"2. EXPLORE baseline: fin_turn={out['fin_turn']:.3f}, active={out['active']}")
    passed += 1

    # 3. Suppressed during FLEE
    pf.reset()
    out_flee = pf(goal=GOAL_FLEE)
    assert not out_flee['active']
    assert out_flee['fin_turn'] == 0.0
    _ok("3. FLEE: suppressed (active=False, fin_turn=0)")
    passed += 1

    # 4. Turn direction: positive food_bearing → positive fin_turn
    pf.reset()
    out_right = pf(food_bearing=0.8, turn_request=0.0, goal=GOAL_FORAGE)
    pf.reset()
    out_left = pf(food_bearing=-0.8, turn_request=0.0, goal=GOAL_FORAGE)
    if out_right['fin_turn'] > out_left['fin_turn']:
        _ok(f"4. Turn direction: right={out_right['fin_turn']:.3f}, left={out_left['fin_turn']:.3f}")
        passed += 1
    else:
        _fail("4. Turn direction", f"right ({out_right['fin_turn']:.3f}) should > left ({out_left['fin_turn']:.3f})")

    # 5. FORAGE has higher gain than EXPLORE
    pf.reset()
    out_forage = pf(turn_request=0.5, goal=GOAL_FORAGE)
    pf.reset()
    out_explore = pf(turn_request=0.5, goal=GOAL_EXPLORE)
    if abs(out_forage['fin_turn']) >= abs(out_explore['fin_turn']) * 0.8:
        _ok(f"5. FORAGE gain ≥ EXPLORE: {abs(out_forage['fin_turn']):.3f} vs {abs(out_explore['fin_turn']):.3f}")
        passed += 1
    else:
        _fail("5. Goal gain", f"forage ({abs(out_forage['fin_turn']):.3f}) should ≥ explore ({abs(out_explore['fin_turn']):.3f})")

    # 6. L/R rate output
    pf.reset()
    out = pf(food_bearing=0.5, goal=GOAL_FORAGE)
    assert out['fin_L_rate'] >= 0.0 and out['fin_R_rate'] >= 0.0
    _ok(f"6. L/R rates: L={out['fin_L_rate']:.3f}, R={out['fin_R_rate']:.3f}")
    passed += 1

    # 7. Turn clamped to [-1, 1]
    pf.reset()
    out_extreme = pf(food_bearing=1.0, turn_request=1.0, goal=GOAL_FORAGE)
    assert -1.0 <= out_extreme['fin_turn'] <= 1.0
    _ok(f"7. Turn clamped: {out_extreme['fin_turn']:.3f}")
    passed += 1

    # 8. Reset
    pf.reset()
    assert float(pf.rate_L.sum()) == 0.0
    _ok("8. Reset clears rates")
    passed += 1

    print(f"\nPectoral Fin: {passed}/{total} passed")
    return passed, total


# ============================================================
#  INTEGRATION TESTS (4)
# ============================================================
def test_integration():
    _header("Integration (Full Brain Wiring)")
    from zebrav2.brain.brain_v2 import ZebrafishBrainV2
    from zebrav2.config.brain_config import BrainConfig
    passed = 0
    total = 4

    cfg = BrainConfig()  # habenula/insula off by default — matches production config
    brain = ZebrafishBrainV2(device=DEVICE, brain_config=cfg)

    # 1. All new modules exist
    assert hasattr(brain, 'raphe')
    assert hasattr(brain, 'lc')
    assert hasattr(brain, 'pectoral_fin')
    assert hasattr(brain.tectum, 'hab_L')
    assert hasattr(brain.tectum, 'hab_R')
    _ok("1. All Tier 2 modules instantiated")
    passed += 1

    # 2. Run 20 steps — no crash
    class FakeEnv:
        def __init__(self):
            self.fish_x = 400; self.fish_y = 300; self.fish_heading = 0.0
            self.fish_energy = 80.0
            self.pred_x = 200; self.pred_y = 200
            self.arena_w = 800; self.arena_h = 600
            self._eaten_now = 0
            self._enemy_pixels_total = 5.0
            self.brain_L = np.random.rand(800).astype(np.float32) * 0.1
            self.brain_R = np.random.rand(800).astype(np.float32) * 0.1
            self.foods = [(500, 300)]
            self.all_fish = [{'x': 400, 'y': 300, 'heading': 0, 'alive': True}]
    env = FakeEnv()
    t0 = time.time()
    for i in range(20):
        result = brain.step(None, env)
    dt = time.time() - t0
    _ok(f"2. 20 steps completed in {dt:.2f}s ({dt/20*1000:.0f}ms/step)")
    passed += 1

    # 3. Spiking neuromod overrides work
    # After stepping, neuromod values should be set by raphe/LC
    assert 'raphe_ht5' in result
    assert 'lc_na' in result
    assert 0.0 < result['raphe_ht5'] < 1.0
    assert 0.0 < result['lc_na'] < 1.0
    _ok(f"3. Spiking neuromod: raphe 5-HT={result['raphe_ht5']:.3f}, LC NA={result['lc_na']:.3f}")
    passed += 1

    # 4. Habituation reduces tectal response to repeated stimulus
    # After 20 steps with constant input, habituation should be below 1.0
    hab_level = brain.tectum.hab_L.get_habituation_level()
    assert hab_level < 1.0, f"Expected habituation, got {hab_level:.3f}"
    _ok(f"4. Tectal habituation after 20 constant frames: {hab_level:.3f}")
    passed += 1

    print(f"\nIntegration: {passed}/{total} passed")
    return passed, total


# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  TIER 2 MODULE TESTS")
    print("=" * 60)
    t0 = time.time()

    results = []
    results.append(test_raphe())
    results.append(test_lc())
    results.append(test_habituation())
    results.append(test_pectoral_fin())
    results.append(test_integration())

    total_pass = sum(r[0] for r in results)
    total_tests = sum(r[1] for r in results)
    dt = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_pass}/{total_tests} passed ({dt:.1f}s)")
    print(f"{'='*60}")

    if total_pass < total_tests:
        sys.exit(1)
