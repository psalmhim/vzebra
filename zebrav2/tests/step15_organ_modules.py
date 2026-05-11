"""
Step 15: Hypothalamus, Pineal gland, Inferior olive — organ module tests.

26 tests covering construction, physiological dynamics, edge cases, and reset.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from zebrav2.brain.hypothalamus import SpikingHypothalamus
from zebrav2.brain.pineal import SpikingPineal
from zebrav2.brain.inferior_olive import SpikingInferiorOlive

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

# ===== HYPOTHALAMUS (10 tests) =====
print("\n=== Hypothalamus ===")
hypo = SpikingHypothalamus()

# 1. Construction
out = hypo(energy=50.0, hunger=0.5, stress=0.3, fatigue=0.2)
check("hypo: construction returns dict", isinstance(out, dict) and 'feeding_drive' in out)

# 2. High hunger → feeding drive
out_hungry = hypo(energy=20.0, hunger=0.8, stress=0.0, fatigue=0.0, food_odor=0.5)
check("hypo: high hunger → feeding drive > 0.3", out_hungry['feeding_drive'] > 0.3)

# 3. High energy → satiety
out_full = hypo(energy=90.0, hunger=0.0, stress=0.0)
check("hypo: high energy → satiety > 0.5", out_full['satiety'] > 0.5)

# 4. Amygdala fear → CRH release
out_fear = hypo(energy=50.0, hunger=0.0, stress=0.5, amygdala_alpha=0.8, cortisol=0.1)
check("hypo: amygdala fear → CRH > 0.3", out_fear['crh_release'] > 0.3)

# 5. Cortisol negative feedback suppresses CRH
out_neg = hypo(energy=50.0, hunger=0.0, stress=0.5, amygdala_alpha=0.8, cortisol=0.9)
check("hypo: cortisol feedback → CRH < fear CRH",
      out_neg['crh_release'] < out_fear['crh_release'])

# 6. Melatonin suppresses orexin (no feeding arousal at night)
hypo.reset()
out_day = hypo(energy=30.0, hunger=0.7, melatonin=0.0, food_odor=0.3)
hypo.reset()
out_night = hypo(energy=30.0, hunger=0.7, melatonin=0.9, food_odor=0.3)
check("hypo: melatonin suppresses orexin", out_night['orexin'] < out_day['orexin'])

# 7. Circadian gate: day vs night
hypo.reset()
out_day_g = hypo(energy=50.0, hunger=0.0, melatonin=0.0)
hypo.reset()
out_night_g = hypo(energy=50.0, hunger=0.0, melatonin=0.9)
check("hypo: day gate > night gate", out_day_g['circadian_gate'] > out_night_g['circadian_gate'])

# 8. EFE bias method
bias = hypo.get_efe_bias()
check("hypo: get_efe_bias returns dict with forage/flee",
      'forage' in bias and 'flee' in bias)

# 9. FEP prediction error computed
check("hypo: prediction error ≥ 0", out['prediction_error'] >= 0)

# 10. Reset
hypo.reset()
check("hypo: reset clears state",
      hypo.feeding_drive == 0.0 and hypo.crh_release == 0.0 and hypo.orexin == 0.0)

# ===== PINEAL (8 tests) =====
print("\n=== Pineal ===")
pin = SpikingPineal()

# 1. Construction
out = pin(light_level=0.7)
check("pin: construction returns dict", isinstance(out, dict) and 'melatonin_raw' in out)

# 2. Light suppresses melatonin
pin.reset()
pin.melatonin_raw = 0.8  # start with high melatonin
out_light = pin(light_level=0.9)
check("pin: light suppresses melatonin", out_light['melatonin_raw'] < 0.8)

# 3. Darkness allows melatonin synthesis
pin.reset()
for _ in range(50):
    out_dark = pin(light_level=0.1)
check("pin: darkness → melatonin increases", out_dark['melatonin_raw'] > 0.1)

# 4. Light detection tracks input
out_bright = pin(light_level=0.9)
check("pin: light_detected tracks input", out_bright['light_detected'] > 0.5)

# 5. Retinal luminance modulates detection
pin.reset()
out_ret = pin(light_level=0.3, retinal_luminance=0.9)
check("pin: retinal luminance contributes",
      out_ret['light_detected'] > 0.3)

# 6. Smoothed melatonin lags raw
pin.reset()
pin.melatonin_raw = 0.0
pin._mel_ema = 0.0
for _ in range(5):
    out_sm = pin(light_level=0.05)
check("pin: smoothed ≤ raw (EMA lag)",
      out_sm['melatonin_smoothed'] <= out_sm['melatonin_raw'] + 0.01)

# 7. FEP prediction error computed
check("pin: prediction error ≥ 0", out_sm['prediction_error'] >= 0)

# 8. Reset
pin.reset()
check("pin: reset clears state",
      pin.melatonin_raw == 0.0 and pin.light_detected == 0.0)

# ===== INFERIOR OLIVE (8 tests) =====
print("\n=== Inferior Olive ===")
io = SpikingInferiorOlive()

# 1. Construction
out = io(sensory_surprise=0.5, motor_mismatch=0.3)
check("io: construction returns dict", isinstance(out, dict) and 'climbing_fiber' in out)

# 2. High error → complex spike
io.reset()
_ = io(sensory_surprise=0.0)  # set baseline
out_err = io(sensory_surprise=0.8, motor_mismatch=0.5)
check("io: high error → complex spike", out_err['complex_spike'] is True)

# 3. No error → no complex spike
io.reset()
_ = io(sensory_surprise=0.0, motor_mismatch=0.0)
out_calm = io(sensory_surprise=0.0, motor_mismatch=0.0)
check("io: no error → no complex spike", out_calm['complex_spike'] is False)

# 4. Climbing fiber output bounded [0, 1]
check("io: climbing fiber in [0,1]",
      0 <= out_err['climbing_fiber'] <= 1.0)

# 5. Cerebellar PE feedback modulates output
io.reset()
_ = io(sensory_surprise=0.0)
out_cb = io(sensory_surprise=0.3, motor_mismatch=0.2, cerebellar_pe=0.5)
io.reset()
_ = io(sensory_surprise=0.0)
out_no_cb = io(sensory_surprise=0.3, motor_mismatch=0.2, cerebellar_pe=0.0)
check("io: cerebellar PE modulates climbing fiber",
      out_cb['climbing_fiber'] >= out_no_cb['climbing_fiber'])

# 6. Sensory error tracks surprise changes
io.reset()
_ = io(sensory_surprise=0.0)
out_s = io(sensory_surprise=0.9)
check("io: sensory error > 0 on surprise", out_s['sensory_error'] > 0.2)

# 7. FEP prediction error computed
check("io: prediction error ≥ 0", out_err['prediction_error'] >= 0)

# 8. Reset
io.reset()
check("io: reset clears state",
      io.sensory_error == 0.0 and io.climbing_fiber == 0.0 and io.complex_spike is False)

# ===== SUMMARY =====
print(f"\n{'='*50}")
print(f"Step 15 Organ Modules: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {FAIL}")
    sys.exit(1)
