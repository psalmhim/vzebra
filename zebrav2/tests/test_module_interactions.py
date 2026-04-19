"""
Cross-module integration tests for zebrav2 brain simulation.

Tests interactions between HPA, amygdala, oxytocin, social memory,
meta-goal REINFORCE, and disorder presets.
"""
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

results = []


def run_test(name, fn):
    try:
        status, details = fn()
        results.append((name, status, details))
        print(f"  [{'PASS' if status else 'FAIL'}] {name}: {details}")
    except Exception as e:
        results.append((name, False, f"EXCEPTION: {e}"))
        print(f"  [FAIL] {name}: EXCEPTION: {e}")


# =========================================================================
# 1. HPA x Amygdala cascade
# =========================================================================
def test_hpa_amygdala_cascade():
    from zebrav2.brain.hpa_axis import HPAAxis
    hpa = HPAAxis()

    # Feed high amygdala_alpha (0.8) for 200 steps
    for _ in range(200):
        hpa.update(amygdala_alpha=0.8)

    cortisol_high = hpa.cortisol
    sensitization = hpa.amygdala_sensitization()

    checks = []
    checks.append(cortisol_high > 0.3)
    checks.append(sensitization > 1.0)

    # Feed low amygdala_alpha (0.1) for 800 steps -> cortisol should decay
    # stress_load must first drain below 0.1 (~100 steps to halve from 1.0,
    # needs ~400+ steps), then cortisol decays with tau ~300.
    cortisol_before_decay = hpa.cortisol
    for _ in range(800):
        hpa.update(amygdala_alpha=0.1)

    cortisol_after_decay = hpa.cortisol
    checks.append(cortisol_after_decay < cortisol_before_decay)

    details = (
        f"cortisol_high={cortisol_high:.4f} (>0.3), "
        f"sensitization={sensitization:.4f} (>1.0), "
        f"decay={cortisol_before_decay:.4f}->{cortisol_after_decay:.4f}"
    )
    return all(checks), details


# =========================================================================
# 2. HPA x DA suppression
# =========================================================================
def test_hpa_da_suppression():
    from zebrav2.brain.hpa_axis import HPAAxis
    hpa = HPAAxis()
    hpa.cortisol = 0.8

    da_factor = hpa.da_suppression()
    ok = da_factor < 0.7
    details = f"da_suppression={da_factor:.4f} (<0.7)"
    return ok, details


# =========================================================================
# 3. HPA x Place cells
# =========================================================================
def test_hpa_place_cells():
    from zebrav2.brain.hpa_axis import HPAAxis
    hpa = HPAAxis()
    hpa.cortisol = 0.8

    lr_factor = hpa.place_lr_factor()
    ok = lr_factor < 0.6
    details = f"place_lr_factor={lr_factor:.4f} (<0.6)"
    return ok, details


# =========================================================================
# 4. Oxytocin x Social buffering
# =========================================================================
def test_oxytocin_social_buffering():
    from zebrav2.brain.oxytocin import OxytocinSystem
    oxt_sys = OxytocinSystem()

    for _ in range(50):
        oxt_sys.update(n_nearby_fish=3, n_crowded_foragers=0)

    oxt_val = oxt_sys.oxt
    fear_ext = oxt_sys.fear_extinction_factor()
    trust = oxt_sys.social_trust_boost()
    efe_bias = oxt_sys.social_efe_bias()

    checks = [
        oxt_val > 0.3,
        fear_ext > 0.1,
        trust > 0.05,
        efe_bias < -0.01,
    ]
    details = (
        f"oxt={oxt_val:.4f} (>0.3), "
        f"fear_ext={fear_ext:.4f} (>0.1), "
        f"trust={trust:.4f} (>0.05), "
        f"efe_bias={efe_bias:.4f} (<-0.01)"
    )
    return all(checks), details


# =========================================================================
# 5. Oxytocin x AVP competition
# =========================================================================
def test_oxytocin_avp_competition():
    from zebrav2.brain.oxytocin import OxytocinSystem
    oxt_sys = OxytocinSystem()

    for _ in range(50):
        oxt_sys.update(n_nearby_fish=0, n_crowded_foragers=5)

    avp_val = oxt_sys.avp
    comp = oxt_sys.competition_drive()

    checks = [avp_val > 0.2, comp > 0.03]
    details = f"avp={avp_val:.4f} (>0.2), competition_drive={comp:.4f} (>0.03)"
    return all(checks), details


# =========================================================================
# 6. MetaGoal REINFORCE
# =========================================================================
def test_meta_goal_reinforce():
    import torch
    import numpy as np
    from zebrav2.brain.meta_goal import MetaGoalWeights

    mg = MetaGoalWeights(device='cpu')

    fitnesses = [100, 200, 300, 400, 500]
    for episode_idx, fitness in enumerate(fitnesses):
        # Simulate 20 steps per episode
        for step in range(20):
            # Build goal probabilities WITH autograd through goal_bias
            raw_logits = torch.tensor([1.0, 0.5, 0.3, 0.2]) + mg.goal_bias
            goal_probs = torch.softmax(raw_logits, dim=0)
            chosen = torch.multinomial(goal_probs, 1).item()
            mod_contribs = np.random.rand(8).astype(np.float32) * 0.1
            mg.record_step(chosen, goal_probs, mod_contribs)

        mg.episode_update(fitness)

    # Check bounds
    gb = mg.goal_bias.data.cpu().numpy()
    mw = mg.mod_w.data.cpu().numpy()

    gb_ok = all(-0.5 <= v <= 0.5 for v in gb)
    mw_ok = all(0.1 <= v <= 3.0 for v in mw)
    ema_ok = mg._fitness_ema is not None

    checks = [gb_ok, mw_ok, ema_ok]
    details = (
        f"goal_bias={gb.tolist()}, bounded={gb_ok}; "
        f"mod_w range=[{mw.min():.3f}, {mw.max():.3f}], bounded={mw_ok}; "
        f"fitness_ema={mg._fitness_ema:.1f}"
    )
    return all(checks), details


# =========================================================================
# 7. SocialMemory alarm precision
# =========================================================================
def test_social_memory_alarm_precision():
    from zebrav2.brain.social_memory import SocialMemory

    sm = SocialMemory()

    # Phase 1: 20 alarm cycles, 18/20 predator was close (high precision)
    # Need enough cycles for precision-EMA (alpha=0.1) to converge and
    # pull w_alarm above 1.5.
    for cycle in range(20):
        sm.reset()
        pred_close = cycle < 18  # first 18 are true alarms

        # Trigger alarm via get_social_alarm
        sm._step = 0
        sm.update_states(50.0, 50.0, [
            {'x': 60.0, 'y': 60.0, 'heading': 0.0, 'speed': 2.0},
        ])
        sm.get_social_alarm(0.5)  # triggers alarm tracking

        # Run through the horizon window
        for t in range(25):
            sm._step += 1
            sm.update_alarm_outcome(pred_was_close=pred_close)

    w_alarm_after_true = sm.w_alarm

    # Phase 2: 10 false alarm cycles
    for cycle in range(10):
        sm.reset()
        sm._step = 0
        sm.update_states(50.0, 50.0, [
            {'x': 60.0, 'y': 60.0, 'heading': 0.0, 'speed': 2.0},
        ])
        sm.get_social_alarm(0.5)

        for t in range(25):
            sm._step += 1
            sm.update_alarm_outcome(pred_was_close=False)

    w_alarm_after_false = sm.w_alarm

    checks = [
        w_alarm_after_true > 1.5,
        w_alarm_after_false < w_alarm_after_true,
    ]
    details = (
        f"w_alarm after true alarms={w_alarm_after_true:.4f} (>1.5), "
        f"after false alarms={w_alarm_after_false:.4f} (decreased)"
    )
    return all(checks), details


# =========================================================================
# 8. SocialMemory food cue
# =========================================================================
def test_social_memory_food_cue():
    from zebrav2.brain.social_memory import SocialMemory

    sm = SocialMemory()
    # Place conspecific nearby, low speed (eating)
    sm.update_states(50.0, 50.0, [
        {'x': 55.0, 'y': 55.0, 'heading': 0.0, 'speed': 0.1},  # eating
    ])

    # Fish heading toward conspecific
    heading = math.atan2(55.0 - 50.0, 55.0 - 50.0)
    efe = sm.get_food_cue_efe(50.0, 50.0, heading)

    ok = efe < 0.0
    details = f"food_cue_efe={efe:.6f} (<0)"
    return ok, details


# =========================================================================
# 9. Disorder presets (mocked brain)
# =========================================================================
def test_disorder_presets():
    import torch
    from zebrav2.brain.disorder import apply_disorder, DISORDER_DESCRIPTIONS
    from zebrav2.brain.meta_goal import MetaGoalWeights
    from zebrav2.brain.social_memory import SocialMemory
    from zebrav2.brain.hpa_axis import HPAAxis
    from zebrav2.brain.oxytocin import OxytocinSystem

    class MockNeuromod:
        def __init__(self):
            self.DA = torch.tensor([0.5])
            self.NA = torch.tensor([0.3])
            self.HT5 = torch.tensor([0.5])
            self.ACh = torch.tensor([0.5])

    class MockAmygdala:
        def __init__(self):
            self.retinal_gain = 0.08
            self.fear_baseline = 0.0

    class MockHabenula:
        def __init__(self):
            self.threshold = 0.4

    class MockCPG:
        def __init__(self):
            self.noise = 0.12

    class MockLayer:
        def __init__(self):
            self.g_NMDA = 1.0

    class MockThalamus:
        def __init__(self):
            self.TC = MockLayer()
            self.TRN = MockLayer()

    class MockPallium:
        def __init__(self):
            self.pal_s = MockLayer()
            self.pal_d = MockLayer()

    class MockClassifier:
        def __init__(self):
            self.temperature = 0.1

    class MockBrain:
        def __init__(self):
            self.neuromod = MockNeuromod()
            self.amygdala = MockAmygdala()
            self.habenula = MockHabenula()
            self.cpg = MockCPG()
            self.thalamus = MockThalamus()
            self.pallium = MockPallium()
            self.classifier = MockClassifier()
            self.meta_goal = MetaGoalWeights(device='cpu')
            self.social_mem = SocialMemory()
            self.hpa = HPAAxis()
            self.oxytocin = OxytocinSystem()
            self._flee_threshold = 0.25
            self._da_phasic_steps = 10

    disorder_names = [
        'wildtype', 'hypodopamine', 'anxiety', 'depression',
        'ptsd', 'adhd', 'asd', 'schizophrenia',
    ]

    failed = []
    details_parts = []

    for name in disorder_names:
        brain = MockBrain()
        try:
            changes = apply_disorder(brain, name, intensity=0.8)
            if name == 'wildtype':
                # No changes expected
                ok = len(changes) == 0
            else:
                # At least one parameter should have changed
                ok = len(changes) > 0
            if not ok:
                failed.append(name)
            details_parts.append(f"{name}: {len(changes)} changes")
        except Exception as e:
            failed.append(name)
            details_parts.append(f"{name}: ERROR {e}")

    all_ok = len(failed) == 0
    details = "; ".join(details_parts)
    if failed:
        details = f"FAILED: {failed}. " + details
    return all_ok, details


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    print("\n=== Cross-Module Integration Tests ===\n")

    run_test("1. HPA x Amygdala cascade", test_hpa_amygdala_cascade)
    run_test("2. HPA x DA suppression", test_hpa_da_suppression)
    run_test("3. HPA x Place cells", test_hpa_place_cells)
    run_test("4. Oxytocin x Social buffering", test_oxytocin_social_buffering)
    run_test("5. Oxytocin x AVP competition", test_oxytocin_avp_competition)
    run_test("6. MetaGoal REINFORCE", test_meta_goal_reinforce)
    run_test("7. SocialMemory alarm precision", test_social_memory_alarm_precision)
    run_test("8. SocialMemory food cue", test_social_memory_food_cue)
    run_test("9. Disorder presets", test_disorder_presets)

    print("\n--- Summary ---")
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    print(f"  {passed}/{total} passed")

    if passed < total:
        print("\n  Failed tests:")
        for name, status, details in results:
            if not status:
                print(f"    {name}: {details}")
        sys.exit(1)
    else:
        print("  All tests passed.")
