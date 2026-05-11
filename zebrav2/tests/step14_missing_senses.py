"""
Depth tests for 4 missing sensory modalities:
  1. SpikingNociception  (8 tests)  — pain/noxious stimuli
  2. SpikingAuditory     (8 tests)  — inner ear / acoustic
  3. SpikingGustatory    (8 tests)  — taste/chemosensory
  4. SpikingTactile      (8 tests)  — mechanoreceptor touch
Total: 32 tests
"""
import sys
import os
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from zebrav2.spec import DEVICE
from zebrav2.brain.nociception import SpikingNociception
from zebrav2.brain.auditory import SpikingAuditory
from zebrav2.brain.gustatory import SpikingGustatory
from zebrav2.brain.tactile import SpikingTactile

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
#  1. NOCICEPTION (8 tests)
# ============================================================
def test_nociception():
    print("\n" + "=" * 60)
    print("  1. SpikingNociception")
    print("=" * 60)

    noci = SpikingNociception(device=DEVICE)

    # 1.1 Construction: 6 neurons, 3 FEP channels
    _check(noci.n == 6, f"1.1 {noci.n} neurons, 3 FEP channels")

    # 1.2 Collision → mechanical pain > 0
    out = noci.forward(collision=True)
    _check(out['mechanical_pain'] > 0.5,
           f"1.2 Collision: mechanical_pain={out['mechanical_pain']:.3f} > 0.5")

    # 1.3 Extreme heat → thermal pain
    noci.reset()
    out_heat = noci.forward(temperature=38.0)
    _check(out_heat['thermal_pain'] > 0.3,
           f"1.3 Heat (38°C): thermal_pain={out_heat['thermal_pain']:.3f} > 0.3")

    # 1.4 Comfort zone → no thermal pain
    noci.reset()
    out_comfort = noci.forward(temperature=26.0)
    _check(out_comfort['thermal_pain'] == 0.0,
           f"1.4 Comfort (26°C): thermal_pain={out_comfort['thermal_pain']:.3f} = 0")

    # 1.5 Chemical irritant → chemical pain
    noci.reset()
    out_chem = noci.forward(chemical_irritant=0.8)
    _check(out_chem['chemical_pain'] > 0.5,
           f"1.5 Chemical: chemical_pain={out_chem['chemical_pain']:.3f} > 0.5")

    # 1.6 Predator bite (close range) → mechanical pain
    noci.reset()
    out_bite = noci.forward(predator_distance=15.0)
    _check(out_bite['mechanical_pain'] > 0.3,
           f"1.6 Predator bite (15px): mechanical={out_bite['mechanical_pain']:.3f} > 0.3")

    # 1.7 Withdrawal reflex: strong pain → reflex active
    noci.reset()
    out_strong = noci.forward(collision=True, temperature=35.0, chemical_irritant=0.5)
    _check(out_strong['withdrawal_reflex'] > 0,
           f"1.7 Strong pain: withdrawal_reflex={out_strong['withdrawal_reflex']:.3f} > 0")

    # 1.8 Reset clears all state
    noci.reset()
    _check(noci.pain_level == 0.0 and noci.free_energy == 0.0
           and noci.sensitisation == 1.0,
           "1.8 Reset clears pain, FE, sensitisation")


# ============================================================
#  2. AUDITORY (8 tests)
# ============================================================
def test_auditory():
    print("\n" + "=" * 60)
    print("  2. SpikingAuditory")
    print("=" * 60)

    aud = SpikingAuditory(device=DEVICE)

    # 2.1 Construction: 6 neurons
    _check(aud.n == 6, f"2.1 {aud.n} neurons")

    # 2.2 Near predator → low frequency activation
    out = aud.forward(predator_distance=50.0)
    _check(out['low_freq'] > 0.3,
           f"2.2 Predator (50px): low_freq={out['low_freq']:.3f} > 0.3")

    # 2.3 Far predator → no low frequency
    aud.reset()
    out_far = aud.forward(predator_distance=500.0)
    _check(out_far['low_freq'] == 0.0,
           f"2.3 Far predator (500px): low_freq={out_far['low_freq']:.3f} = 0")

    # 2.4 Near conspecific → mid frequency
    aud.reset()
    out_con = aud.forward(conspecific_distances=[30.0, 100.0])
    _check(out_con['mid_freq'] > 0.3,
           f"2.4 Conspecific (30px): mid_freq={out_con['mid_freq']:.3f} > 0.3")

    # 2.5 Startle trigger: sudden loud onset
    aud.reset()
    out_startle = aud.forward(sudden_onset=0.8)
    _check(out_startle['startle_trigger'] is True,
           f"2.5 Startle: sudden_onset=0.8 → trigger={out_startle['startle_trigger']}")

    # 2.6 No startle during refractory period
    out_refrac = aud.forward(sudden_onset=0.9)
    _check(out_refrac['startle_trigger'] is False,
           f"2.6 Refractory: startle={out_refrac['startle_trigger']} (suppressed)")

    # 2.7 Acoustic salience: spectral change detection
    aud.reset()
    aud.forward(predator_distance=999.0)  # baseline
    out_change = aud.forward(predator_distance=80.0)  # sudden predator approach
    _check(out_change['acoustic_salience'] > 0.01,
           f"2.7 Salience after change: {out_change['acoustic_salience']:.4f} > 0.01")

    # 2.8 Reset clears state
    aud.reset()
    _check(aud.low_freq == 0.0 and aud.startle_trigger is False
           and aud.free_energy == 0.0,
           "2.8 Reset clears freq, startle, FE")


# ============================================================
#  3. GUSTATORY (8 tests)
# ============================================================
def test_gustatory():
    print("\n" + "=" * 60)
    print("  3. SpikingGustatory")
    print("=" * 60)

    gust = SpikingGustatory(device=DEVICE)

    # 3.1 Construction: 6 neurons
    _check(gust.n == 6, f"3.1 {gust.n} neurons")

    # 3.2 Eating good food → positive palatability
    out = gust.forward(eating=True, food_quality=0.8)
    _check(out['palatability'] > 0.3,
           f"3.2 Good food: palatability={out['palatability']:.3f} > 0.3")

    # 3.3 Eating good food → amino acid > 0
    _check(out['amino_acid'] > 0.5,
           f"3.3 Good food: amino_acid={out['amino_acid']:.3f} > 0.5")

    # 3.4 Eating toxic food → spit reflex
    gust.reset()
    out_toxic = gust.forward(eating=True, food_quality=0.2, toxin_level=0.8)
    _check(out_toxic['spit_reflex'] is True,
           f"3.4 Toxic food: spit_reflex={out_toxic['spit_reflex']}")

    # 3.5 Not eating → no taste activation
    gust.reset()
    out_idle = gust.forward(eating=False, food_distance=200.0)
    _check(out_idle['amino_acid'] == 0.0 and out_idle['umami'] == 0.0,
           f"3.5 No eating: amino={out_idle['amino_acid']:.3f}, umami={out_idle['umami']:.3f}")

    # 3.6 Close to food → mild gustatory detection
    gust.reset()
    out_close = gust.forward(eating=False, food_quality=0.9, food_distance=15.0)
    _check(out_close['amino_acid'] > 0.05,
           f"3.6 Near food (15px): amino_acid={out_close['amino_acid']:.3f} > 0.05")

    # 3.7 Bitter > appetitive → negative palatability
    gust.reset()
    out_bad = gust.forward(eating=True, food_quality=0.1, toxin_level=0.9)
    _check(out_bad['palatability'] < 0,
           f"3.7 Bad taste: palatability={out_bad['palatability']:.3f} < 0")

    # 3.8 Reset clears state
    gust.reset()
    _check(gust.amino_acid == 0.0 and gust.bitter == 0.0
           and gust.free_energy == 0.0 and gust.spit_reflex is False,
           "3.8 Reset clears amino, bitter, FE, spit")


# ============================================================
#  4. TACTILE (8 tests)
# ============================================================
def test_tactile():
    print("\n" + "=" * 60)
    print("  4. SpikingTactile")
    print("=" * 60)

    tact = SpikingTactile(device=DEVICE)

    # 4.1 Construction: 6 neurons
    _check(tact.n == 6, f"4.1 {tact.n} neurons")

    # 4.2 Head collision → head touch + startle
    out = tact.forward(collision=True, heading_to_wall=0.8)
    _check(out['head_touch'] > 0.5 and out['contact_location'] == 'head',
           f"4.2 Head collision: touch={out['head_touch']:.3f}, loc={out['contact_location']}")

    # 4.3 Head touch → startle reflex
    _check(out['startle_reflex'] is True,
           f"4.3 Head touch startle: {out['startle_reflex']}")

    # 4.4 Conspecific nearby → trunk touch (shoaling)
    tact.reset()
    out_shoal = tact.forward(conspecific_distance=10.0)
    _check(out_shoal['trunk_touch'] > 0.2,
           f"4.4 Shoaling touch (10px): trunk={out_shoal['trunk_touch']:.3f} > 0.2")

    # 4.5 No contact → no touch
    tact.reset()
    out_none = tact.forward(conspecific_distance=500.0, wall_proximity=0.0)
    _check(out_none['touch_intensity'] == 0.0 and out_none['contact_location'] == 'none',
           f"4.5 No contact: intensity={out_none['touch_intensity']:.3f}, loc={out_none['contact_location']}")

    # 4.6 Predator close → head + trunk touch
    tact.reset()
    out_pred = tact.forward(predator_distance=10.0, heading_to_wall=0.0)
    _check(out_pred['head_touch'] > 0.3 and out_pred['trunk_touch'] > 0.3,
           f"4.6 Predator (10px): head={out_pred['head_touch']:.3f}, trunk={out_pred['trunk_touch']:.3f}")

    # 4.7 Wall lateral brush → trunk touch (not head)
    tact.reset()
    out_brush = tact.forward(wall_proximity=0.9, heading_to_wall=0.0)
    _check(out_brush['trunk_touch'] > 0.1,
           f"4.7 Wall brush: trunk={out_brush['trunk_touch']:.3f} > 0.1")

    # 4.8 Reset clears state
    tact.reset()
    _check(tact.head_touch == 0.0 and tact.trunk_touch == 0.0
           and tact.contact_location == 'none' and tact.free_energy == 0.0,
           "4.8 Reset clears touch, location, FE")


# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  MISSING SENSES TESTS (4 modules × 8 tests)")
    print("=" * 60)

    test_nociception()
    test_auditory()
    test_gustatory()
    test_tactile()

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {_pass}/{_pass + _fail} passed, {_fail} failed")
    print(f"{'=' * 60}")

    if _fail > 0:
        sys.exit(1)
