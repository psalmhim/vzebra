"""
Depth tests for 3 undertested sensory modules:
  1. SpikingColorVision  (8 tests)
  2. SpikingVestibular    (8 tests)
  3. SpikingProprioception (8 tests)
Total: 24 tests
"""
import sys
import os
import math
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from zebrav2.spec import DEVICE
from zebrav2.brain.color_vision import SpikingColorVision
from zebrav2.brain.vestibular import SpikingVestibular
from zebrav2.brain.proprioception import SpikingProprioception

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


# ============================================================
#  1. COLOR VISION (8 tests)
# ============================================================
def test_color_vision():
    print("\n" + "=" * 60)
    print("  1. SpikingColorVision")
    print("=" * 60)

    cv = SpikingColorVision(device=DEVICE)

    # 1.1 Construction: 4 populations × 8 neurons = 32 total
    total_n = cv.uv_pop.n + cv.blue_pop.n + cv.green_pop.n + cv.red_pop.n
    _check(total_n == 32, f"1.1 4 populations × 8 = {total_n} neurons")

    # 1.2 Food stimulus: high green, moderate red (signature [0.1, 0.2, 0.8, 0.4])
    type_L = torch.zeros(400, device=DEVICE)
    type_R = torch.zeros(400, device=DEVICE)
    type_L[100:150] = 0.9  # food type pixels
    out = cv.forward(type_L, type_R)
    _check(out['green'] > out['uv'] and out['green'] > out['blue'],
           f"1.2 Food: green={out['green']:.3f} > uv={out['uv']:.3f}, blue={out['blue']:.3f}")

    # 1.3 Enemy stimulus: high red (signature [0.05, 0.1, 0.3, 0.7])
    cv.reset()
    type_L2 = torch.zeros(400, device=DEVICE)
    type_L2[100:150] = 0.5  # enemy type pixels
    out2 = cv.forward(type_L2, torch.zeros(400, device=DEVICE))
    _check(out2['red'] > out2['uv'],
           f"1.3 Enemy: red={out2['red']:.3f} > uv={out2['uv']:.3f}")

    # 1.4 Color opponent: R-G for food should be negative (green > red)
    cv.reset()
    out_food = cv.forward(type_L, torch.zeros(400, device=DEVICE))
    _check(out_food['rg_opponent'] < 0,
           f"1.4 Food R-G opponent={out_food['rg_opponent']:.3f} < 0 (green dominant)")

    # 1.5 B-Y opponent: should be negative for food (green+red > uv+blue)
    _check(out_food['by_opponent'] < 0,
           f"1.5 Food B-Y opponent={out_food['by_opponent']:.3f} < 0")

    # 1.6 Spectral PE: non-zero after stimulus change
    cv.reset()
    # First call: establishes baseline
    cv.forward(torch.zeros(400, device=DEVICE), torch.zeros(400, device=DEVICE))
    # Second call: food stimulus → spectral change → PE
    out3 = cv.forward(type_L, torch.zeros(400, device=DEVICE))
    pe_sum = sum(abs(p) for p in out3['spectral_pe'])
    _check(pe_sum > 0.01,
           f"1.6 Spectral PE after change: {pe_sum:.4f} > 0.01")

    # 1.7 Spectral salience: precision-weighted PE magnitude
    _check(out3['spectral_salience'] > 0,
           f"1.7 Spectral salience={out3['spectral_salience']:.4f} > 0")

    # 1.8 Reset clears rates and PE
    cv.reset()
    _check(float(cv.uv_rate.sum()) == 0 and cv.free_energy == 0.0,
           "1.8 Reset clears rates and free energy")


# ============================================================
#  2. VESTIBULAR (8 tests)
# ============================================================
def test_vestibular():
    print("\n" + "=" * 60)
    print("  2. SpikingVestibular")
    print("=" * 60)

    vest = SpikingVestibular(device=DEVICE)

    # 2.1 Construction: 6 neurons (pitch, roll, yaw × 2)
    _check(vest.n == 6, f"2.1 {vest.n} neurons")

    # 2.2 Right turn: positive angular velocity → yaw-right neuron fires
    out = vest.forward(heading=0.0, speed=1.0, turn_rate=0.5)
    _check(out['angular_velocity'] == 0.5,
           f"2.2 Angular velocity={out['angular_velocity']:.2f}")

    # 2.3 Tilt proportional to turn × speed
    expected_tilt = min(1.0, abs(0.5) * 1.0 * 0.5)
    _check(abs(out['tilt'] - expected_tilt) < 0.01,
           f"2.3 Tilt={out['tilt']:.3f} ≈ {expected_tilt:.3f}")

    # 2.4 High speed + sharp turn → higher tilt
    vest.reset()
    out_fast = vest.forward(heading=0.0, speed=2.0, turn_rate=1.0)
    _check(out_fast['tilt'] > out['tilt'],
           f"2.4 Fast tilt={out_fast['tilt']:.3f} > slow tilt={out['tilt']:.3f}")

    # 2.5 Prediction error: mismatch between expected and actual turn
    vest.reset()
    out_pe = vest.forward(heading=0.0, speed=1.0, turn_rate=0.8,
                          predicted_turn=0.0, predicted_speed=1.0)
    _check(abs(out_pe['prediction_error']) > 0.01,
           f"2.5 PE with turn mismatch: {out_pe['prediction_error']:.4f}")

    # 2.6 PE remains bounded over repeated steps (no divergence)
    vest.reset()
    pes = []
    for _ in range(20):
        out_rep = vest.forward(heading=0.0, speed=1.0, turn_rate=0.5,
                               predicted_turn=0.5, predicted_speed=1.0)
        pes.append(abs(out_rep['prediction_error']))
    _check(max(pes) < 5.0 and all(math.isfinite(p) for p in pes),
           f"2.6 PE bounded: range [{min(pes):.4f}, {max(pes):.4f}]")

    # 2.7 Postural correction: opposes PE
    vest.reset()
    out_post = vest.forward(heading=0.0, speed=1.0, turn_rate=0.8,
                            predicted_turn=0.0, predicted_speed=1.0)
    _check('postural_correction' in out_post,
           f"2.7 Postural correction={out_post['postural_correction']:.4f}")

    # 2.8 Reset clears state
    vest.reset()
    _check(vest.angular_velocity == 0.0 and vest.tilt == 0.0 and vest.free_energy == 0.0,
           "2.8 Reset clears angular_velocity, tilt, free_energy")


# ============================================================
#  3. PROPRIOCEPTION (8 tests)
# ============================================================
def test_proprioception():
    print("\n" + "=" * 60)
    print("  3. SpikingProprioception")
    print("=" * 60)

    prop = SpikingProprioception(device=DEVICE)

    # 3.1 Construction: 8 neurons (speed×2, heading×2, wall×2, collision×2)
    _check(prop.n == 8, f"3.1 {prop.n} neurons")

    # 3.2 Speed encoding: movement → actual_speed > 0
    out = prop.forward(fish_x=410.0, fish_y=300.0, speed=1.0, heading=0.0)
    _check(out['actual_speed'] > 0,
           f"3.2 Actual speed={out['actual_speed']:.2f} > 0 (moved 10px)")

    # 3.3 Wall proximity: near edge → high value
    prop.reset()
    # First call to set prev position
    prop.forward(fish_x=30.0, fish_y=300.0, speed=1.0, heading=0.0)
    out_wall = prop.forward(fish_x=25.0, fish_y=300.0, speed=1.0, heading=0.0)
    _check(out_wall['wall_proximity'] > 0.3,
           f"3.3 Wall proximity at x=25: {out_wall['wall_proximity']:.3f} > 0.3")

    # 3.4 No wall proximity at center
    prop.reset()
    prop.forward(fish_x=400.0, fish_y=300.0, speed=1.0, heading=0.0)
    out_center = prop.forward(fish_x=400.0, fish_y=300.0, speed=0.0, heading=0.0)
    _check(out_center['wall_proximity'] == 0.0,
           f"3.4 Center wall proximity={out_center['wall_proximity']:.3f}")

    # 3.5 Collision: speed commanded but no movement
    prop.reset()
    prop.forward(fish_x=400.0, fish_y=300.0, speed=1.0, heading=0.0)
    out_coll = prop.forward(fish_x=400.0, fish_y=300.0, speed=1.5, heading=0.0)
    _check(out_coll['collision'] is True,
           f"3.5 Collision detected: speed=1.5 but no movement")

    # 3.6 No collision when moving normally
    prop.reset()
    prop.forward(fish_x=400.0, fish_y=300.0, speed=1.0, heading=0.0)
    out_nc = prop.forward(fish_x=410.0, fish_y=300.0, speed=1.0, heading=0.0)
    _check(out_nc['collision'] is False,
           f"3.6 No collision: moved 10px at speed=1.0")

    # 3.7 Speed PE: prediction mismatch drives PE
    prop.reset()
    prop.set_predictions(predicted_speed=0.5, predicted_heading_delta=0.0)
    prop.forward(fish_x=400.0, fish_y=300.0, speed=1.0, heading=0.0)
    out_pe = prop.forward(fish_x=420.0, fish_y=300.0, speed=1.0, heading=0.0)
    _check(abs(out_pe['speed_pe']) > 0.01,
           f"3.7 Speed PE with mismatch: {out_pe['speed_pe']:.4f}")

    # 3.8 Reset clears all state
    prop.reset()
    _check(prop.collision is False and prop.wall_proximity == 0.0
           and prop.free_energy == 0.0 and prop.speed_pe == 0.0,
           "3.8 Reset clears collision, wall, FE, PE")


# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  SENSORY DEPTH TESTS (3 modules × 8 tests)")
    print("=" * 60)

    test_color_vision()
    test_vestibular()
    test_proprioception()

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {_pass}/{_pass + _fail} passed, {_fail} failed")
    print(f"{'=' * 60}")

    if _fail > 0:
        sys.exit(1)
