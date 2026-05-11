"""
Step 17: Visceral organs — vagus nerve, pituitary, area postrema, NTS,
lateral line efferent.

35 tests covering construction, physiological dynamics, cross-module
interactions, edge cases, and reset.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from zebrav2.brain.vagus_nerve import SpikingVagusNerve
from zebrav2.brain.pituitary import SpikingPituitary
from zebrav2.brain.area_postrema import SpikingAreaPostrema
from zebrav2.brain.nts import SpikingNTS
from zebrav2.brain.lateral_line_efferent import SpikingLateralLineEfferent

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

# ===== VAGUS NERVE (7 tests) =====
print("\n=== Vagus Nerve ===")
vn = SpikingVagusNerve()

# 1. Construction
out = vn(heart_rate=0.6, gut_state=0.4, stress=0.3)
check("vn: construction returns dict", isinstance(out, dict) and 'vagal_tone' in out)

# 2. Low stress → high vagal tone (parasympathetic dominance)
vn.reset()
out_calm = vn(heart_rate=0.5, gut_state=0.5, stress=0.1)
check("vn: low stress → vagal tone > 0.3", out_calm['vagal_tone'] > 0.3)

# 3. High stress → low vagal tone (sympathetic override)
vn.reset()
out_stress = vn(heart_rate=0.8, gut_state=0.3, stress=0.9)
check("vn: high stress → vagal tone < calm",
      out_stress['vagal_tone'] < out_calm['vagal_tone'])

# 4. Gut signal relays gut_state
vn.reset()
out_gut = vn(heart_rate=0.5, gut_state=0.8, stress=0.2)
check("vn: gut_state=0.8 → gut_signal > 0.3", out_gut['gut_signal'] > 0.3)

# 5. Cardiac output bounded
check("vn: cardiac_output in [0,1]",
      0 <= out_stress['cardiac_output'] <= 1.0)

# 6. FEP prediction error ≥ 0
check("vn: prediction error ≥ 0", out['prediction_error'] >= 0)

# 7. Reset
vn.reset()
check("vn: reset clears gut_signal",
      vn.gut_signal == 0.0)


# ===== PITUITARY (7 tests) =====
print("\n=== Pituitary ===")
pit = SpikingPituitary()

# 8. Construction
out = pit(crh_release=0.3, stress=0.2, energy=60.0)
check("pit: construction returns dict", isinstance(out, dict) and 'acth' in out)

# 9. High CRH → high ACTH (HPA axis relay)
pit.reset()
out_crh = pit(crh_release=0.8, stress=0.5, energy=50.0)
check("pit: high CRH → ACTH > 0.3", out_crh['acth'] > 0.3)

# 10. ACTH drives cortisol_drive
check("pit: cortisol_drive > 0", out_crh['cortisol_drive'] > 0)

# 11. Dopamine inhibits MSH
pit.reset()
out_no_da = pit(crh_release=0.3, stress=0.2, energy=50.0, dopamine=0.1)
pit.reset()
out_da = pit(crh_release=0.3, stress=0.2, energy=50.0, dopamine=0.9)
check("pit: dopamine inhibits MSH", out_da['msh'] <= out_no_da['msh'])

# 12. Vasotocin output bounded
check("pit: vasotocin in [0,1]",
      0 <= out_crh.get('vasotocin', 0) <= 1.0)

# 13. FEP prediction error ≥ 0
check("pit: prediction error ≥ 0", out['prediction_error'] >= 0)

# 14. Reset
pit.reset()
check("pit: reset clears state",
      pit.acth == 0.0 and pit.cortisol_drive == 0.0)


# ===== AREA POSTREMA (7 tests) =====
print("\n=== Area Postrema ===")
ap = SpikingAreaPostrema()

# 15. Construction
out = ap(blood_toxin=0.2, blood_glucose=0.5)
check("ap: construction returns dict", isinstance(out, dict) and 'nausea_signal' in out)

# 16. High toxin → strong nausea (accumulates over a few steps)
ap.reset()
for _ in range(5):
    out_tox = ap(blood_toxin=0.9, blood_glucose=0.5)
check("ap: high toxin → nausea > 0.3", out_tox['nausea_signal'] > 0.3)

# 17. Toxin detected flag
check("ap: toxin detected = True", out_tox['toxin_detected'] is True)

# 18. No toxin → no nausea/detection
ap.reset()
out_clean = ap(blood_toxin=0.0, blood_glucose=0.5)
check("ap: no toxin → low nausea", out_clean['nausea_signal'] < 0.3)

# 19. Glucose status reflects blood glucose
ap.reset()
out_hypo = ap(blood_toxin=0.0, blood_glucose=0.1)
ap.reset()
out_hyper = ap(blood_toxin=0.0, blood_glucose=0.9)
check("ap: low glucose < high glucose status",
      out_hypo['glucose_status'] < out_hyper['glucose_status'])

# 20. FEP prediction error ≥ 0
check("ap: prediction error ≥ 0", out['prediction_error'] >= 0)

# 21. Reset
ap.reset()
check("ap: reset clears state",
      ap.nausea_signal == 0.0 and ap.toxin_detected is False)


# ===== NTS (7 tests) =====
print("\n=== Nucleus Tractus Solitarius ===")
nts = SpikingNTS()

# 22. Construction
out = nts(taste_input=0.5, vagal_afferent=0.3)
check("nts: construction returns dict", isinstance(out, dict) and 'taste_relay' in out)

# 23. Taste relay tracks input
nts.reset()
out_taste = nts(taste_input=0.8, vagal_afferent=0.2)
check("nts: taste input → taste_relay > 0.2", out_taste['taste_relay'] > 0.2)

# 24. Vagal afferent → visceral relay
nts.reset()
out_visc = nts(taste_input=0.1, vagal_afferent=0.8)
check("nts: vagal afferent → visceral_relay > 0.2", out_visc['visceral_relay'] > 0.2)

# 25. Satiety increases with repeated taste
nts.reset()
for _ in range(20):
    out_sat = nts(taste_input=0.7, vagal_afferent=0.3)
check("nts: repeated taste → satiety > 0.1", out_sat['satiety_signal'] > 0.1)

# 26. Cardio output bounded
check("nts: cardio_output in [0,1]",
      0 <= out_taste.get('cardio_output', 0) <= 1.0)

# 27. FEP prediction error ≥ 0
check("nts: prediction error ≥ 0", out['prediction_error'] >= 0)

# 28. Reset
nts.reset()
check("nts: reset clears state",
      nts.taste_relay == 0.0 and nts.satiety_signal == 0.0)


# ===== LATERAL LINE EFFERENT (7 tests) =====
print("\n=== Lateral Line Efferent ===")
lle = SpikingLateralLineEfferent()

# 29. Construction
out = lle(motor_command=0.3, lateral_line_input=0.5)
check("lle: construction returns dict", isinstance(out, dict) and 'suppression_gain' in out)

# 30. Swimming → high suppression
lle.reset()
out_swim = lle(motor_command=0.9, lateral_line_input=0.5, swim_speed=0.8)
check("lle: swimming → suppression > 0.3", out_swim['suppression_gain'] > 0.3)

# 31. Stationary → low suppression
lle.reset()
out_still = lle(motor_command=0.0, lateral_line_input=0.5, swim_speed=0.0)
check("lle: stationary → suppression < swimming",
      out_still['suppression_gain'] < out_swim['suppression_gain'])

# 32. External flow preserved when stationary
check("lle: external_flow > 0 when stationary",
      out_still['external_flow'] > 0)

# 33. External flow attenuated during swimming
check("lle: external_flow reduced during swimming",
      out_swim['external_flow'] <= out_still['external_flow'])

# 34. FEP prediction error ≥ 0
check("lle: prediction error ≥ 0", out['prediction_error'] >= 0)

# 35. Reset
lle.reset()
check("lle: reset clears state",
      lle.suppression_gain == 0.0 and lle.external_flow == 0.0)


# ===== SUMMARY =====
print(f"\n{'='*50}")
print(f"Step 17 Visceral Organs: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {FAIL}")
    sys.exit(1)
