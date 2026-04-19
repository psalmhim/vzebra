"""
Full evaluation of BinocularDepth module.

Tests cover:
  1. Construction & geometry (4 tests)
  2. Food distance estimation (4 tests)
  3. Enemy distance estimation (4 tests)
  4. Disparity–distance relationship (3 tests)
  5. Confidence calibration (3 tests)
  6. Stereo correlation (3 tests)
  7. Approach gain for prey capture (3 tests)
  8. Edge cases & robustness (4 tests)
  9. Integration with brain pipeline (3 tests)
Total: 31 tests
"""
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from zebrav2.brain.binocular_depth import BinocularDepth

_pass = 0
_fail = 0


def _ok(msg):
    global _pass
    _pass += 1
    print(f"  \u2713 {msg}")


def _nok(msg):
    global _fail
    _fail += 1
    print(f"  \u2717 {msg}")


def _check(cond, msg):
    if cond:
        _ok(msg)
    else:
        _nok(msg)


def _make_retinal(overlap_pixels, food_positions=None, enemy_positions=None,
                  intensity=0.8):
    """Create synthetic L/R retinal arrays with objects in binocular zone.

    food_positions: list of (L_col, R_col) tuples in overlap coords
    enemy_positions: list of (L_col, R_col) tuples in overlap coords
    """
    L_type = np.zeros(400, dtype=np.float32)
    R_type = np.zeros(400, dtype=np.float32)
    L_int = np.zeros(400, dtype=np.float32)
    R_int = np.zeros(400, dtype=np.float32)
    op = overlap_pixels

    if food_positions:
        for (lc, rc) in food_positions:
            # L eye nasal = L_type[400-op:]  → index 400-op+lc
            li = 400 - op + lc
            if 0 <= li < 400:
                L_type[li] = 0.9  # food type > 0.7
                L_int[li] = intensity
            # R eye nasal = R_type[:op]  → index rc
            if 0 <= rc < 400:
                R_type[rc] = 0.9
                R_int[rc] = intensity

    if enemy_positions:
        for (lc, rc) in enemy_positions:
            li = 400 - op + lc
            if 0 <= li < 400:
                L_type[li] = 0.5  # enemy type
                L_int[li] = intensity
            if 0 <= rc < 400:
                R_type[rc] = 0.5
                R_int[rc] = intensity

    return L_type, R_type, L_int, R_int


# ============================================================
#  1. CONSTRUCTION & GEOMETRY (4 tests)
# ============================================================
def test_construction():
    print("\n" + "=" * 60)
    print("  1. Construction & Geometry")
    print("=" * 60)

    bd = BinocularDepth()

    # 1.1 Default parameters
    _check(bd.retinal_width == 400,
           f"1.1 Retinal width={bd.retinal_width}")

    # 1.2 Overlap pixels = 400 * 40/200 = 80
    _check(bd.overlap_pixels == 80,
           f"1.2 Overlap pixels={bd.overlap_pixels} (40° of 200° FoV)")

    # 1.3 Baseline (interocular distance)
    _check(bd.baseline == 5.0,
           f"1.3 Baseline={bd.baseline}px")

    # 1.4 Custom FoV
    bd2 = BinocularDepth(fov_deg=180, overlap_deg=60, retinal_width=360)
    expected_op = int(360 * 60 / 180)
    _check(bd2.overlap_pixels == expected_op,
           f"1.4 Custom: overlap={bd2.overlap_pixels} (expected {expected_op})")


# ============================================================
#  2. FOOD DISTANCE ESTIMATION (4 tests)
# ============================================================
def test_food_distance():
    print("\n" + "=" * 60)
    print("  2. Food Distance Estimation")
    print("=" * 60)

    bd = BinocularDepth()
    op = bd.overlap_pixels  # 80

    # 2.1 Food in binocular zone → finite distance
    L_t, R_t, L_i, R_i = _make_retinal(op,
        food_positions=[(40, 40)])  # same position → small disparity → far
    out = bd.estimate(L_t, R_t, L_i, R_i)
    _check(out['food_distance'] < 999.0,
           f"2.1 Food detected: distance={out['food_distance']:.1f}")

    # 2.2 Food with large disparity → close
    bd.reset()
    L_t2, R_t2, L_i2, R_i2 = _make_retinal(op,
        food_positions=[(20, 60)])  # 40px disparity → close
    out2 = bd.estimate(L_t2, R_t2, L_i2, R_i2)
    _check(out2['food_distance'] < out['food_distance'],
           f"2.2 Large disparity: {out2['food_distance']:.1f} < {out['food_distance']:.1f}")

    # 2.3 No food in binocular zone → 999
    bd.reset()
    out3 = bd.estimate(np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32))
    _check(out3['food_distance'] == 999.0,
           f"2.3 No food: distance={out3['food_distance']}")

    # 2.4 Food only in L eye (no binocular match) → 999
    bd.reset()
    L_t4 = np.zeros(400, dtype=np.float32)
    L_t4[350] = 0.9  # in overlap zone of L eye
    out4 = bd.estimate(L_t4, np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32))
    _check(out4['food_distance'] == 999.0,
           f"2.4 Monocular food (L only): distance={out4['food_distance']}")


# ============================================================
#  3. ENEMY DISTANCE ESTIMATION (4 tests)
# ============================================================
def test_enemy_distance():
    print("\n" + "=" * 60)
    print("  3. Enemy Distance Estimation")
    print("=" * 60)

    bd = BinocularDepth()
    op = bd.overlap_pixels

    # 3.1 Enemy in binocular zone → finite distance
    L_t, R_t, L_i, R_i = _make_retinal(op,
        enemy_positions=[(40, 40)])
    out = bd.estimate(L_t, R_t, L_i, R_i)
    _check(out['enemy_distance'] < 999.0,
           f"3.1 Enemy detected: distance={out['enemy_distance']:.1f}")

    # 3.2 Large disparity → closer enemy
    bd.reset()
    L_t2, R_t2, L_i2, R_i2 = _make_retinal(op,
        enemy_positions=[(15, 65)])  # 50px disparity
    out2 = bd.estimate(L_t2, R_t2, L_i2, R_i2)
    _check(out2['enemy_distance'] < out['enemy_distance'],
           f"3.2 Close enemy: {out2['enemy_distance']:.1f} < far={out['enemy_distance']:.1f}")

    # 3.3 No enemy → 999
    bd.reset()
    out3 = bd.estimate(np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32))
    _check(out3['enemy_distance'] == 999.0,
           f"3.3 No enemy: distance={out3['enemy_distance']}")

    # 3.4 Enemy monocular → 999
    bd.reset()
    L_t4 = np.zeros(400, dtype=np.float32)
    L_t4[350] = 0.5  # enemy in L overlap only
    out4 = bd.estimate(L_t4, np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32))
    _check(out4['enemy_distance'] == 999.0,
           f"3.4 Monocular enemy: distance={out4['enemy_distance']}")


# ============================================================
#  4. DISPARITY–DISTANCE RELATIONSHIP (3 tests)
# ============================================================
def test_disparity_distance():
    print("\n" + "=" * 60)
    print("  4. Disparity–Distance Relationship")
    print("=" * 60)

    bd = BinocularDepth()
    op = bd.overlap_pixels

    # 4.1 Distance = baseline * focal / disparity → inversely proportional
    distances = []
    disparities = [5, 10, 20, 40]
    for disp in disparities:
        bd.reset()
        center = op // 2
        half = disp // 2
        L_t, R_t, L_i, R_i = _make_retinal(op,
            food_positions=[(center - half, center + half)])
        out = bd.estimate(L_t, R_t, L_i, R_i)
        distances.append(out['food_distance'])

    monotonic = all(distances[i] >= distances[i+1] for i in range(len(distances)-1))
    _check(monotonic,
           f"4.1 Monotonic: d={[f'{d:.0f}' for d in distances]} "
           f"for disparities={disparities}")

    # 4.2 Distance formula: baseline(5) * focal(200) / disparity
    bd.reset()
    # Single pixel food at positions that give known disparity
    L_t, R_t, L_i, R_i = _make_retinal(op,
        food_positions=[(30, 50)])  # disparity = |30-50| = 20
    out = bd.estimate(L_t, R_t, L_i, R_i)
    expected = 5.0 * 200.0 / 20.0  # = 50
    _check(abs(out['food_distance'] - expected) < 1.0,
           f"4.2 Formula check: {out['food_distance']:.1f} ≈ {expected:.1f}")

    # 4.3 Very small disparity → capped at 500
    bd.reset()
    # Same position → disparity ≈ epsilon → distance → 500 (cap)
    L_t, R_t, L_i, R_i = _make_retinal(op,
        food_positions=[(40, 40)])
    out = bd.estimate(L_t, R_t, L_i, R_i)
    _check(out['food_distance'] <= 500.0,
           f"4.3 Max distance cap: {out['food_distance']:.1f} ≤ 500")


# ============================================================
#  5. CONFIDENCE CALIBRATION (3 tests)
# ============================================================
def test_confidence():
    print("\n" + "=" * 60)
    print("  5. Confidence Calibration")
    print("=" * 60)

    bd = BinocularDepth()
    op = bd.overlap_pixels

    # 5.1 More food pixels → higher confidence
    bd.reset()
    L_t1 = np.zeros(400, dtype=np.float32)
    R_t1 = np.zeros(400, dtype=np.float32)
    # 2 pixels each
    L_t1[398] = 0.9; L_t1[399] = 0.9
    R_t1[0] = 0.9; R_t1[1] = 0.9
    out1 = bd.estimate(L_t1, R_t1, np.ones(400, dtype=np.float32) * 0.5,
                       np.ones(400, dtype=np.float32) * 0.5)
    conf_small = out1['food_confidence']

    bd.reset()
    L_t2 = np.zeros(400, dtype=np.float32)
    R_t2 = np.zeros(400, dtype=np.float32)
    # 8 pixels each
    for i in range(8):
        L_t2[392 + i] = 0.9
        R_t2[i] = 0.9
    out2 = bd.estimate(L_t2, R_t2, np.ones(400, dtype=np.float32) * 0.5,
                       np.ones(400, dtype=np.float32) * 0.5)
    conf_large = out2['food_confidence']
    _check(conf_large > conf_small,
           f"5.1 More pixels: conf={conf_large:.3f} > {conf_small:.3f}")

    # 5.2 Confidence capped at 1.0
    bd.reset()
    L_t3 = np.zeros(400, dtype=np.float32)
    R_t3 = np.zeros(400, dtype=np.float32)
    L_t3[320:400] = 0.9  # 80 food pixels
    R_t3[0:80] = 0.9     # 80 food pixels
    out3 = bd.estimate(L_t3, R_t3, np.ones(400, dtype=np.float32),
                       np.ones(400, dtype=np.float32))
    _check(out3['food_confidence'] <= 1.0,
           f"5.2 Confidence capped: {out3['food_confidence']:.3f} ≤ 1.0")

    # 5.3 No detection → zero confidence
    bd.reset()
    out4 = bd.estimate(np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32))
    _check(out4['food_confidence'] == 0.0 and out4['enemy_confidence'] == 0.0,
           f"5.3 No detection: food_conf={out4['food_confidence']}, "
           f"enemy_conf={out4['enemy_confidence']}")


# ============================================================
#  6. STEREO CORRELATION (3 tests)
# ============================================================
def test_stereo_correlation():
    print("\n" + "=" * 60)
    print("  6. Stereo Correlation")
    print("=" * 60)

    bd = BinocularDepth()
    op = bd.overlap_pixels

    # 6.1 Identical L/R intensity → high correlation
    intensity = np.random.RandomState(42).rand(400).astype(np.float32) * 0.5
    # Make overlap zones identical
    L_int = intensity.copy()
    R_int = np.zeros(400, dtype=np.float32)
    R_int[:op] = L_int[-op:]  # copy L overlap to R overlap
    out = bd.estimate(np.zeros(400, dtype=np.float32),
                      np.zeros(400, dtype=np.float32),
                      L_int, R_int)
    _check(out['stereo_correlation'] > 0.5,
           f"6.1 Matched intensity: corr={out['stereo_correlation']:.3f}")

    # 6.2 Random L/R → low/no correlation
    bd.reset()
    rng = np.random.RandomState(99)
    L_rand = rng.rand(400).astype(np.float32)
    R_rand = rng.rand(400).astype(np.float32)
    out2 = bd.estimate(np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       L_rand, R_rand)
    _check(abs(out2['stereo_correlation']) < 0.5,
           f"6.2 Random intensity: corr={out2['stereo_correlation']:.3f}")

    # 6.3 Zero intensity → correlation = 0 (no signal)
    bd.reset()
    out3 = bd.estimate(np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32),
                       np.zeros(400, dtype=np.float32))
    _check(out3['stereo_correlation'] == 0.0,
           f"6.3 Zero signal: corr={out3['stereo_correlation']}")


# ============================================================
#  7. APPROACH GAIN FOR PREY CAPTURE (3 tests)
# ============================================================
def test_approach_gain():
    print("\n" + "=" * 60)
    print("  7. Approach Gain (Prey Capture)")
    print("=" * 60)

    bd = BinocularDepth()
    op = bd.overlap_pixels

    # 7.1 Far food → gain = 1.0
    bd.food_distance = 200.0
    bd.food_confidence = 0.5
    _check(bd.get_approach_gain() == 1.0,
           f"7.1 Far food (200px): gain={bd.get_approach_gain()}")

    # 7.2 Medium distance → gain = 0.8
    bd.food_distance = 80.0
    bd.food_confidence = 0.5
    _check(bd.get_approach_gain() == 0.8,
           f"7.2 Medium food (80px): gain={bd.get_approach_gain()}")

    # 7.3 Close food → gain = 0.6 (slow for precise capture)
    bd.food_distance = 30.0
    bd.food_confidence = 0.5
    _check(bd.get_approach_gain() == 0.6,
           f"7.3 Close food (30px): gain={bd.get_approach_gain()}")


# ============================================================
#  8. EDGE CASES & ROBUSTNESS (4 tests)
# ============================================================
def test_edge_cases():
    print("\n" + "=" * 60)
    print("  8. Edge Cases & Robustness")
    print("=" * 60)

    bd = BinocularDepth()
    op = bd.overlap_pixels

    # 8.1 Both food and enemy simultaneously
    L_t, R_t, L_i, R_i = _make_retinal(op,
        food_positions=[(30, 50)],
        enemy_positions=[(60, 20)])
    out = bd.estimate(L_t, R_t, L_i, R_i)
    _check(out['food_distance'] < 999.0 and out['enemy_distance'] < 999.0,
           f"8.1 Both detected: food={out['food_distance']:.0f}, "
           f"enemy={out['enemy_distance']:.0f}")

    # 8.2 Type value exactly at threshold boundary
    bd.reset()
    L_t2 = np.zeros(400, dtype=np.float32)
    R_t2 = np.zeros(400, dtype=np.float32)
    L_t2[370] = 0.7  # exactly at food threshold (> 0.7 needed)
    R_t2[10] = 0.7
    out2 = bd.estimate(L_t2, R_t2, np.ones(400, dtype=np.float32) * 0.5,
                       np.ones(400, dtype=np.float32) * 0.5)
    _check(out2['food_distance'] == 999.0,
           f"8.2 At threshold (0.7 not > 0.7): distance={out2['food_distance']}")

    # 8.3 All NaN intensities → no crash
    bd.reset()
    try:
        L_nan = np.full(400, np.nan, dtype=np.float32)
        out3 = bd.estimate(np.zeros(400, dtype=np.float32),
                           np.zeros(400, dtype=np.float32),
                           L_nan, L_nan)
        ok = math.isfinite(out3['stereo_correlation']) or out3['stereo_correlation'] == 0.0
        _check(ok, f"8.3 NaN intensities: no crash, corr={out3['stereo_correlation']}")
    except Exception as e:
        _nok(f"8.3 NaN intensities crashed: {e}")

    # 8.4 Reset clears distances
    bd.food_distance = 50.0
    bd.enemy_distance = 100.0
    bd.food_confidence = 0.8
    bd.enemy_confidence = 0.7
    bd.reset()
    _check(bd.food_distance == 999.0 and bd.enemy_distance == 999.0
           and bd.food_confidence == 0.0 and bd.enemy_confidence == 0.0,
           "8.4 Reset restores defaults")


# ============================================================
#  9. INTEGRATION WITH BRAIN PIPELINE (3 tests)
# ============================================================
def test_integration():
    print("\n" + "=" * 60)
    print("  9. Integration with Brain Pipeline")
    print("=" * 60)

    import torch
    from zebrav2.spec import DEVICE
    from zebrav2.brain.brain_v2 import ZebrafishBrainV2
    from zebrav2.config.brain_config import BrainConfig

    cfg = BrainConfig()
    brain = ZebrafishBrainV2(device=DEVICE, brain_config=cfg)
    brain.reset()

    # Minimal env
    class FakeEnv:
        def __init__(self):
            self.fish_x = 400.0
            self.fish_y = 300.0
            self.fish_heading = 0.0
            self.fish_energy = 80.0
            self.brain_L = np.zeros(800, dtype=np.float32)
            self.brain_R = np.zeros(800, dtype=np.float32)
            self.pred_x = 500.0
            self.pred_y = 300.0
            self.foods = [(350.0, 300.0)]
            self.all_fish = [{'x': 400, 'y': 300, 'heading': 0, 'alive': True}]
            self._eaten_now = 0
            self._enemy_pixels_total = 0.0
            self.arena_w = 800
            self.arena_h = 600

    env = FakeEnv()

    # 9.1 Brain step includes binocular module
    result = brain.step(None, env)
    _check(brain.binocular is not None,
           "9.1 BinocularDepth module exists in brain")

    # 9.2 Binocular state updated after step
    # With food at 350 and fish at 400, food is 50px ahead — may or may not
    # be in binocular zone depending on retinal projection
    _check(isinstance(brain.binocular.food_distance, float),
           f"9.2 Food distance updated: {brain.binocular.food_distance:.1f}")

    # 9.3 Approach gain affects speed
    brain.binocular.food_distance = 30.0
    brain.binocular.food_confidence = 0.8
    gain = brain.binocular.get_approach_gain()
    _check(gain == 0.6,
           f"9.3 Close food approach gain={gain} (slows for capture)")


# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  BINOCULAR DEPTH — FULL EVALUATION (31 tests)")
    print("=" * 60)

    test_construction()
    test_food_distance()
    test_enemy_distance()
    test_disparity_distance()
    test_confidence()
    test_stereo_correlation()
    test_approach_gain()
    test_edge_cases()
    test_integration()

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {_pass}/{_pass + _fail} passed, {_fail} failed")
    print(f"{'=' * 60}")

    if _fail > 0:
        sys.exit(1)
