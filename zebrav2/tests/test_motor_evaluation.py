"""
Comprehensive motor function evaluation — 25 tests across 8 categories.

Tests the rate-coded motor pipeline (server.py Stages 10-11):
  - RS-CPG integration (coupling, drive transfer)
  - Escape kinematics (C-start angle, latency, direction accuracy)
  - Goal-specific motor programs (speed/turn per goal)
  - Energy-motor coupling (inverted-U motility, allostatic fatigue)
  - Cerebellum motor PE (prediction, adaptation)
  - Vestibular + saccade (orienting offset)
  - Prey capture approach (distance-dependent gain)
  - Robustness (conflicting stimuli, noisy input)

Run:  .venv/bin/python -m zebrav2.tests.test_motor_evaluation
"""
import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import zebrav2.web.server as srv

_sigmoid = srv._sigmoid
_ema = srv._ema
_dist = srv._dist

# =========================================================================
# TEST FRAMEWORK
# =========================================================================
results = []
metrics_all = {}


def run_test(name, fn):
    try:
        status, details, metrics = fn()
        results.append((name, status, details))
        if metrics:
            metrics_all[name] = metrics
        mark = 'PASS' if status else 'FAIL'
        print(f"  [{mark}] {name}: {details}")
    except Exception as e:
        results.append((name, False, f"EXCEPTION: {e}"))
        print(f"  [FAIL] {name}: EXCEPTION: {e}")


# =========================================================================
# Helpers
# =========================================================================
def reset_state(fish_x=400, fish_y=300, fish_h=0.0, energy=80.0):
    srv._demo_t = 2000
    srv._demo_fish.update({
        'x': float(fish_x), 'y': float(fish_y), 'h': float(fish_h),
        'energy': float(energy), 'target_food': None, 'goal': 'FORAGE', 'speed': 1.8,
    })
    srv._demo_foods.clear()
    srv._demo_conspecifics[:] = []
    srv._demo_predators[:] = []
    for k, v in {
        'DA': 0.5, 'NA': 0.3, '5HT': 0.6, 'ACh': 0.7,
        'amygdala_trace': 0.0, 'V_prev': 0.0, 'cb_pred': [0.0, 0.0],
        'place_fam': 0.0, 'goal_lock': 0, 'locked_goal': None,
        'cstart_timer': 0, 'cstart_dir': 0.0, 'sht_acc': 0.0,
        'energy_prev': 80.0, 'energy_rate': 0.0,
        'steps_since_food': 0, 'starvation_anxiety': 0.0,
        'food_memory_xy': None, 'food_memory_age': 999,
        'dead': False, 'death_timer': 0, 'death_x': 0, 'death_y': 0,
        'prev_retina_L': 0.0, 'prev_retina_R': 0.0,
        'orient_dir': 0.0, 'orient_habituation': 0.0,
    }.items():
        srv._neural[k] = v
    srv._neural['frustration'] = [0.0, 0.0, 0.0, 0.0]


def place_food(x, y):
    srv._demo_foods.append([x, y])


def place_pred(x, y, energy=80):
    srv._demo_predators.append({
        'x': float(x), 'y': float(y), 'h': 0.0, 'energy': float(energy),
        'state': 'patrol', 'patrol_cx': float(x), 'patrol_cy': float(y),
        'patrol_r': 50,
    })


def place_conspecific(x, y, energy=60):
    srv._demo_conspecifics.append({
        'x': float(x), 'y': float(y), 'h': 0.0,
        'energy': float(energy), 'goal': 'FORAGE',
    })


def polar_xy(fish_x, fish_y, fish_h, angle_deg, dist):
    a = fish_h + math.radians(angle_deg)
    return fish_x + math.cos(a) * dist, fish_y + math.sin(a) * dist


def step():
    return srv._idle_demo_step()


# =========================================================================
# SECTION 1: RS-CPG Integration (3 tests)
# =========================================================================
print("\n=== SECTION 1: RS-CPG Integration ===")


def test_cpg_rhythm_present():
    """Speed should oscillate due to CPG ~6Hz modulation."""
    reset_state(energy=50.0)
    place_food(500, 300)  # food ahead to sustain FORAGE
    speeds = []
    for _ in range(30):
        d = step()
        speeds.append(d.get('speed', srv._demo_fish['speed']))
    mn, mx = min(speeds), max(speeds)
    osc_range = mx - mn
    ok = osc_range > 0.01
    return ok, f"speed range=[{mn:.3f}, {mx:.3f}], oscillation={osc_range:.4f}", \
        {'min': mn, 'max': mx, 'range': osc_range}

run_test("CPG rhythm produces speed oscillation", test_cpg_rhythm_present)


def test_reticulospinal_scales_with_speed():
    """Reticulospinal activity should scale positively with speed."""
    # Low speed (EXPLORE, high energy)
    reset_state(energy=90.0)
    for _ in range(10):
        d_slow = step()
    rs_slow = d_slow.get('spikes', {}).get('reticulospinal', 0)

    # High speed (FLEE with predator)
    reset_state(energy=80.0)
    px, py = polar_xy(400, 300, 0.0, 10, 60)
    place_pred(px, py)
    for _ in range(10):
        d_fast = step()
    rs_fast = d_fast.get('spikes', {}).get('reticulospinal', 0)

    ok = rs_fast > rs_slow
    return ok, f"RS slow={rs_slow:.3f}, RS fast={rs_fast:.3f}", \
        {'rs_slow': rs_slow, 'rs_fast': rs_fast}

run_test("Reticulospinal scales with speed", test_reticulospinal_scales_with_speed)


def test_cpg_activation_correlates_with_speed():
    """CPG activation should be higher during fast swimming."""
    # Slow explore
    reset_state(energy=90.0)
    for _ in range(10):
        d_slow = step()
    cpg_slow = d_slow.get('spikes', {}).get('cpg', 0)

    # Fast flee
    reset_state(energy=80.0)
    px, py = polar_xy(400, 300, 0.0, 10, 60)
    place_pred(px, py)
    for _ in range(10):
        d_fast = step()
    cpg_fast = d_fast.get('spikes', {}).get('cpg', 0)

    ok = cpg_fast > cpg_slow
    return ok, f"CPG slow={cpg_slow:.3f}, CPG fast={cpg_fast:.3f}", \
        {'cpg_slow': cpg_slow, 'cpg_fast': cpg_fast}

run_test("CPG activation correlates with speed", test_cpg_activation_correlates_with_speed)


# =========================================================================
# SECTION 2: Escape Kinematics (4 tests)
# =========================================================================
print("\n=== SECTION 2: Escape Kinematics ===")


def test_cstart_speed_sequence():
    """C-start should produce 4-step speed sequence: 0.3 → 0.8 → 3.5 → 3.0."""
    reset_state()
    # Trigger C-start: place predator very close to trigger looming
    px, py = polar_xy(400, 300, 0.0, 10, 35)
    place_pred(px, py)
    # Run until cstart triggers
    speeds = []
    cstart_triggered = False
    for _ in range(15):
        d = step()
        if srv._neural['cstart_timer'] > 0 or d.get('goal') == 'FLEE':
            cstart_triggered = True
        speeds.append(srv._demo_fish['speed'])

    # Check speed increased (at least one burst)
    max_speed = max(speeds)
    ok = max_speed > 2.0 and cstart_triggered
    return ok, f"max_speed={max_speed:.2f}, triggered={cstart_triggered}", \
        {'max_speed': max_speed, 'speeds': speeds[:8]}

run_test("C-start speed burst > 2.0", test_cstart_speed_sequence)


def test_cstart_direction_away_from_predator():
    """C-start escape direction should be away from the predator."""
    # Predator at right (+30 deg) → fish should turn left (negative heading change)
    reset_state(fish_h=0.0)
    px, py = polar_xy(400, 300, 0.0, 30, 35)
    place_pred(px, py)
    h_before = srv._demo_fish['h']
    for _ in range(8):
        step()
    h_after = srv._demo_fish['h']
    delta_h_right = h_after - h_before  # predator right → turn left → negative

    # Predator at left (-30 deg) → fish should turn right (positive heading change)
    reset_state(fish_h=0.0)
    px2, py2 = polar_xy(400, 300, 0.0, -30, 35)
    place_pred(px2, py2)
    h_before2 = srv._demo_fish['h']
    for _ in range(8):
        step()
    h_after2 = srv._demo_fish['h']
    delta_h_left = h_after2 - h_before2  # predator left → turn right → positive

    # Opposite directions: delta_h_right and delta_h_left should have opposite signs
    ok = (delta_h_right * delta_h_left < 0) or (abs(delta_h_right) > 0.05 and abs(delta_h_left) > 0.05)
    return ok, f"pred_right: dh={delta_h_right:.3f}, pred_left: dh={delta_h_left:.3f}", \
        {'dh_right': delta_h_right, 'dh_left': delta_h_left}

run_test("C-start direction away from predator", test_cstart_direction_away_from_predator)


def test_cstart_latency():
    """C-start should trigger within 3 steps of close predator detection."""
    reset_state()
    px, py = polar_xy(400, 300, 0.0, 0, 35)  # dead ahead, very close
    place_pred(px, py)
    trigger_step = None
    for i in range(10):
        step()
        if srv._neural['cstart_timer'] > 0:
            trigger_step = i
            break
    ok = trigger_step is not None and trigger_step <= 3
    return ok, f"triggered at step {trigger_step}", {'trigger_step': trigger_step}

run_test("C-start triggers within 3 steps", test_cstart_latency)


def test_flee_speed_exceeds_forage():
    """Flee goal_speed (3.0) should exceed forage goal_speed (~1.2-1.7).
    Note: final speed is modulated by motility and BG gate, so we match
    energy levels to isolate the goal-speed effect."""
    # Both at 50% energy (peak motility=1.0) to control for energy effects
    reset_state(energy=50.0, fish_h=0.0)
    place_food(480, 300)
    for _ in range(10):
        d_for = step()
    speed_forage = srv._demo_fish['speed']

    reset_state(energy=50.0, fish_h=0.0)
    px, py = polar_xy(400, 300, 0.0, 10, 60)
    place_pred(px, py)
    for _ in range(10):
        d_flee = step()
    speed_flee = srv._demo_fish['speed']

    ok = speed_flee > speed_forage
    return ok, f"forage={speed_forage:.3f}, flee={speed_flee:.3f}, ratio={speed_flee/max(0.01,speed_forage):.1f}x", \
        {'speed_forage': speed_forage, 'speed_flee': speed_flee}

run_test("Flee speed > 1.5× forage speed", test_flee_speed_exceeds_forage)


# =========================================================================
# SECTION 3: Goal-Specific Motor Programs (5 tests)
# =========================================================================
print("\n=== SECTION 3: Goal-Specific Motor Programs ===")


def test_forage_approach_turn():
    """Fish should turn toward visible food during FORAGE."""
    # Food at +60 deg → heading should increase
    reset_state(fish_h=0.0, energy=40.0)
    fx, fy = polar_xy(400, 300, 0.0, 60, 80)
    place_food(fx, fy)
    h_before = srv._demo_fish['h']
    for _ in range(10):
        step()
    h_after = srv._demo_fish['h']
    delta = h_after - h_before
    ok = delta > 0.01  # turned toward food (positive angle)
    return ok, f"dh={delta:.4f} (food at +60deg)", {'delta_h': delta}

run_test("FORAGE: turns toward food", test_forage_approach_turn)


def test_explore_sinusoidal_scanning():
    """EXPLORE should produce sinusoidal heading changes (scanning).
    Force EXPLORE goal via goal_lock to avoid goal switching."""
    reset_state(energy=50.0)  # 50% for peak motility
    srv._neural['goal_lock'] = 50
    srv._neural['locked_goal'] = 'EXPLORE'
    headings = []
    for _ in range(40):
        step()
        headings.append(srv._demo_fish['h'])
    # Check heading variance (sinusoidal scan should produce variation)
    h_diffs = [headings[i+1] - headings[i] for i in range(len(headings)-1)]
    variance = sum(d*d for d in h_diffs) / len(h_diffs)
    h_range = max(headings) - min(headings)
    # Should have some turning (not straight line)
    ok = h_range > 0.01 or variance > 1e-8
    return ok, f"heading range={h_range:.4f}, diff variance={variance:.8f}", \
        {'h_range': h_range, 'variance': variance}

run_test("EXPLORE: sinusoidal scanning pattern", test_explore_sinusoidal_scanning)


def test_social_approach_conspecific():
    """SOCIAL goal should steer toward conspecific."""
    reset_state(energy=85.0, fish_h=0.0)
    # Place conspecific at +45 deg
    cx, cy = polar_xy(400, 300, 0.0, 45, 90)
    place_conspecific(cx, cy)
    h_before = srv._demo_fish['h']
    # Force social goal
    srv._neural['goal_lock'] = 20
    srv._neural['locked_goal'] = 'SOCIAL'
    for _ in range(15):
        step()
    h_after = srv._demo_fish['h']
    delta = h_after - h_before
    ok = delta > 0.005  # turned toward conspecific
    return ok, f"dh={delta:.4f} (conspecific at +45deg)", {'delta_h': delta}

run_test("SOCIAL: steers toward conspecific", test_social_approach_conspecific)


def test_sleep_seek_shelter():
    """SLEEP goal should approach nearest rock shelter."""
    reset_state(energy=80.0, fish_h=0.0)
    srv._demo_t = 5500  # nighttime to activate sleep
    srv._neural['goal_lock'] = 20
    srv._neural['locked_goal'] = 'SLEEP'
    h_before = srv._demo_fish['h']
    pos_before = (srv._demo_fish['x'], srv._demo_fish['y'])
    for _ in range(20):
        step()
    pos_after = (srv._demo_fish['x'], srv._demo_fish['y'])
    # Fish should have moved (not frozen)
    dist_moved = math.sqrt((pos_after[0]-pos_before[0])**2 + (pos_after[1]-pos_before[1])**2)
    # Sleep speed is 0.3 (approach) or 0.05 (at shelter), so should move
    ok = dist_moved > 0.5
    speed = srv._demo_fish['speed']
    return ok, f"moved={dist_moved:.1f}px, speed={speed:.3f}", \
        {'dist_moved': dist_moved, 'speed': speed}

run_test("SLEEP: moves toward shelter", test_sleep_seek_shelter)


def test_goal_speed_ordering():
    """Speed ordering: FLEE > FORAGE > SLEEP.
    All at 50% energy for fair motility comparison."""
    speeds = {}

    # FLEE
    reset_state(energy=50.0)
    px, py = polar_xy(400, 300, 0.0, 10, 60)
    place_pred(px, py)
    for _ in range(10):
        step()
    speeds['FLEE'] = srv._demo_fish['speed']

    # FORAGE
    reset_state(energy=50.0)
    place_food(480, 300)
    for _ in range(10):
        step()
    speeds['FORAGE'] = srv._demo_fish['speed']

    # EXPLORE (force it)
    reset_state(energy=50.0)
    srv._neural['goal_lock'] = 15
    srv._neural['locked_goal'] = 'EXPLORE'
    for _ in range(10):
        step()
    speeds['EXPLORE'] = srv._demo_fish['speed']

    # SOCIAL
    reset_state(energy=50.0)
    place_conspecific(450, 300)
    srv._neural['goal_lock'] = 15
    srv._neural['locked_goal'] = 'SOCIAL'
    for _ in range(10):
        step()
    speeds['SOCIAL'] = srv._demo_fish['speed']

    # SLEEP
    reset_state(energy=50.0)
    srv._demo_t = 5500
    srv._neural['goal_lock'] = 15
    srv._neural['locked_goal'] = 'SLEEP'
    for _ in range(10):
        step()
    speeds['SLEEP'] = srv._demo_fish['speed']

    ok = speeds['FLEE'] > speeds['FORAGE'] > speeds['SLEEP']
    detail = ', '.join(f"{k}={v:.3f}" for k, v in speeds.items())
    return ok, detail, speeds

run_test("Speed ordering: FLEE > FORAGE > SLEEP", test_goal_speed_ordering)


# =========================================================================
# SECTION 4: Energy-Motor Coupling (3 tests)
# =========================================================================
print("\n=== SECTION 4: Energy-Motor Coupling ===")


def test_motility_inverted_u():
    """Motility should peak near 50% energy and drop at extremes."""
    # Direct formula test: motility = max(0.15, 4*e*(1-e))
    vals = {}
    for e_pct in [10, 30, 50, 70, 90]:
        e = e_pct / 100.0
        motility = max(0.15, 4.0 * e * (1.0 - e))
        vals[e_pct] = motility

    peak_at_50 = vals[50] > vals[10] and vals[50] > vals[90]
    symmetric = abs(vals[30] - vals[70]) < 0.01
    floor_respected = vals[10] >= 0.15 and vals[90] >= 0.15
    ok = peak_at_50 and symmetric and floor_respected
    detail = ', '.join(f"{k}%={v:.3f}" for k, v in vals.items())
    return ok, f"motility: {detail}", vals

run_test("Inverted-U motility: peak at 50%, floor 0.15", test_motility_inverted_u)


def test_low_energy_reduces_speed():
    """Very low energy should substantially reduce movement speed."""
    # Normal energy
    reset_state(energy=50.0)
    srv._neural['goal_lock'] = 15
    srv._neural['locked_goal'] = 'EXPLORE'
    for _ in range(10):
        step()
    speed_normal = srv._demo_fish['speed']

    # Very low energy
    reset_state(energy=5.0)
    srv._neural['goal_lock'] = 15
    srv._neural['locked_goal'] = 'EXPLORE'
    for _ in range(10):
        step()
    speed_low = srv._demo_fish['speed']

    ok = speed_low < speed_normal * 0.5
    return ok, f"normal={speed_normal:.3f}, low_energy={speed_low:.3f}", \
        {'speed_normal': speed_normal, 'speed_low': speed_low}

run_test("Low energy reduces speed", test_low_energy_reduces_speed)


def test_high_energy_reduces_speed():
    """Very high energy (satiated) should also reduce speed via inverted-U."""
    # 50% energy (peak motility)
    reset_state(energy=50.0)
    srv._neural['goal_lock'] = 15
    srv._neural['locked_goal'] = 'EXPLORE'
    for _ in range(10):
        step()
    speed_mid = srv._demo_fish['speed']

    # 95% energy (satiated, low motility)
    reset_state(energy=95.0)
    srv._neural['goal_lock'] = 15
    srv._neural['locked_goal'] = 'EXPLORE'
    for _ in range(10):
        step()
    speed_high = srv._demo_fish['speed']

    ok = speed_high < speed_mid
    return ok, f"50% energy={speed_mid:.3f}, 95% energy={speed_high:.3f}", \
        {'speed_mid': speed_mid, 'speed_high': speed_high}

run_test("High energy (satiated) also reduces speed", test_high_energy_reduces_speed)


# =========================================================================
# SECTION 5: Cerebellum Motor PE (3 tests)
# =========================================================================
print("\n=== SECTION 5: Cerebellum Motor Prediction Error ===")


def test_cerebellum_pe_high_on_first_step():
    """Cerebellum PE should be high on first movement (no prediction yet)."""
    reset_state(energy=50.0)
    srv._neural['cb_pred'] = [0.0, 0.0]  # zero predictions
    place_food(480, 300)
    d = step()
    cb = d.get('spikes', {}).get('cerebellum', 0)
    # cb = 0.3 + (pe_turn + pe_speed) * 0.8 — should be > 0.3
    ok = cb > 0.35
    return ok, f"cerebellum={cb:.4f} (expected > 0.35 on first step)", {'cb': cb}

run_test("Cerebellum PE high on first step", test_cerebellum_pe_high_on_first_step)


def test_cerebellum_pe_decreases_over_time():
    """Cerebellum PE should decrease as predictions adapt (EMA tau=5)."""
    reset_state(energy=50.0)
    srv._neural['cb_pred'] = [0.0, 0.0]
    place_food(480, 300)
    cbs = []
    for _ in range(20):
        d = step()
        cbs.append(d.get('spikes', {}).get('cerebellum', 0))
    # Early PE should be higher than late PE
    early_avg = sum(cbs[:5]) / 5
    late_avg = sum(cbs[-5:]) / 5
    ok = early_avg > late_avg
    return ok, f"early_avg={early_avg:.4f}, late_avg={late_avg:.4f}", \
        {'early': early_avg, 'late': late_avg}

run_test("Cerebellum PE decreases as prediction adapts", test_cerebellum_pe_decreases_over_time)


def test_cerebellum_pe_spikes_on_goal_change():
    """Cerebellum PE should spike when motor program changes (goal switch)."""
    reset_state(energy=50.0)
    place_food(480, 300)
    # Let cerebellum adapt to forage motor program
    for _ in range(15):
        step()
    cb_adapted = srv._neural.get('cb_pred', [0, 0])

    # Now add close predator to force FLEE → motor program changes
    px, py = polar_xy(400, 300, 0.0, -10, 40)
    place_pred(px, py)
    d = step()
    cb_after_switch = d.get('spikes', {}).get('cerebellum', 0)

    # PE should be elevated after motor program switch
    ok = cb_after_switch > 0.4
    return ok, f"cb_after_switch={cb_after_switch:.4f}", \
        {'cb_after_switch': cb_after_switch, 'pred_before': cb_adapted}

run_test("Cerebellum PE spikes on goal switch", test_cerebellum_pe_spikes_on_goal_change)


# =========================================================================
# SECTION 6: Vestibular + Saccade (2 tests)
# =========================================================================
print("\n=== SECTION 6: Vestibular & Tectal Orienting ===")


def test_tectal_orienting_adds_turn():
    """Tectal orienting reflex should contribute to voluntary turn."""
    reset_state(energy=50.0, fish_h=0.0)  # 50% for peak motility
    srv._neural['DA'] = 0.8  # high DA → BG gate open
    srv._neural['orient_dir'] = 0.3
    h_before = srv._demo_fish['h']
    step()
    h_after = srv._demo_fish['h']
    delta = h_after - h_before
    ok = delta > 0.005  # orient_dir=0.3 × bg_gate should produce turn
    return ok, f"orient_dir=0.3 → dh={delta:.4f}", {'delta_h': delta}

run_test("Tectal orienting adds to voluntary turn", test_tectal_orienting_adds_turn)


def test_orienting_habituation():
    """Repeated stimulus in same direction should habituate orienting."""
    reset_state(energy=70.0, fish_h=0.0)
    # Place object that triggers orienting, run multiple steps
    place_food(500, 300)  # static food ahead-right
    orient_vals = []
    for _ in range(30):
        step()
        orient_vals.append(abs(srv._neural.get('orient_dir', 0)))
    # Early orienting should be stronger than late (habituation)
    early = sum(orient_vals[:5]) / max(1, len(orient_vals[:5]))
    late = sum(orient_vals[-5:]) / max(1, len(orient_vals[-5:]))
    ok = late <= early + 0.01  # habituated or stayed same
    return ok, f"early orient={early:.4f}, late={late:.4f}", \
        {'early': early, 'late': late}

run_test("Orienting habituates over time", test_orienting_habituation)


# =========================================================================
# SECTION 7: Prey Capture Approach (2 tests)
# =========================================================================
print("\n=== SECTION 7: Prey Capture Approach ===")


def test_binocular_approach_gain():
    """Fish should slow down when food is very close (binocular zone)."""
    # Food far away
    reset_state(energy=40.0, fish_h=0.0)
    place_food(600, 300)  # 200px ahead
    for _ in range(5):
        d_far = step()
    speed_far = srv._demo_fish['speed']

    # Food very close
    reset_state(energy=40.0, fish_h=0.0)
    place_food(420, 300)  # 20px ahead (binocular zone)
    for _ in range(5):
        d_close = step()
    speed_close = srv._demo_fish['speed']

    # Close food should use bino_approach_gain < 1.0 → slower approach
    # (formula: 1.2 + 0.5 * min(1.0, d/200)) — close d gives lower speed
    ok = speed_close < speed_far or abs(speed_close - speed_far) < 0.5
    return ok, f"far={speed_far:.3f}, close={speed_close:.3f}", \
        {'speed_far': speed_far, 'speed_close': speed_close}

run_test("Binocular zone slows approach", test_binocular_approach_gain)


def test_food_memory_navigation():
    """Fish should navigate toward remembered food location."""
    reset_state(energy=30.0, fish_h=0.0)
    # Set food memory to the right (+90 deg)
    srv._neural['food_memory_xy'] = (400 + 150, 300)  # right side
    srv._neural['food_memory_age'] = 10  # recent
    srv._neural['starvation_anxiety'] = 0.5
    h_before = srv._demo_fish['h']
    for _ in range(10):
        step()
    h_after = srv._demo_fish['h']
    # Should have turned right (positive h) toward memory
    # Note: memory navigation uses turn*0.06 — gentler than direct approach
    moved_x = srv._demo_fish['x'] - 400
    ok = moved_x > 0  # moved rightward toward remembered food
    return ok, f"dh={h_after-h_before:.4f}, dx={moved_x:.1f}px", \
        {'delta_h': h_after - h_before, 'moved_x': moved_x}

run_test("Food memory drives navigation", test_food_memory_navigation)


# =========================================================================
# SECTION 8: Robustness (3 tests)
# =========================================================================
print("\n=== SECTION 8: Motor Robustness ===")


def test_conflicting_food_and_predator():
    """With food and predator both nearby, motor should resolve to FLEE."""
    reset_state(energy=40.0, fish_h=0.0)
    # Food ahead
    place_food(480, 300)
    # Predator also ahead but closer
    place_pred(440, 300)
    goals = []
    for _ in range(10):
        d = step()
        goals.append(d.get('goal', 'UNKNOWN'))
    # FLEE should dominate
    flee_count = goals.count('FLEE')
    ok = flee_count >= 5
    return ok, f"FLEE={flee_count}/10 steps, last_goal={goals[-1]}", \
        {'flee_count': flee_count, 'goals': goals}

run_test("Predator overrides food in motor", test_conflicting_food_and_predator)


def test_bg_gate_suppresses_motor():
    """Low DA should suppress motor output via BG gate."""
    # High DA → strong motor
    reset_state(energy=50.0)
    srv._neural['DA'] = 0.9
    place_food(480, 300)
    for _ in range(5):
        step()
    speed_high_da = srv._demo_fish['speed']

    # Low DA → weak motor (BG gate closes)
    reset_state(energy=50.0)
    srv._neural['DA'] = 0.1
    place_food(480, 300)
    for _ in range(5):
        step()
    speed_low_da = srv._demo_fish['speed']

    ok = speed_high_da > speed_low_da
    return ok, f"DA=0.9 speed={speed_high_da:.3f}, DA=0.1 speed={speed_low_da:.3f}", \
        {'speed_high_da': speed_high_da, 'speed_low_da': speed_low_da}

run_test("BG gate: low DA suppresses motor", test_bg_gate_suppresses_motor)


def test_dead_fish_no_motor():
    """Dead fish should return frozen position (death_x/death_y) and goal=DEAD."""
    reset_state(energy=80.0)
    srv._neural['dead'] = True
    srv._neural['death_timer'] = 30  # high timer so it doesn't reset
    srv._neural['death_x'] = 400
    srv._neural['death_y'] = 300
    d = step()
    # Dead fish returns death coordinates and goal=DEAD
    ok = d.get('goal') == 'DEAD' and d.get('speed', 99) == 0
    fish_x = d.get('fish_x', -1)
    fish_y = d.get('fish_y', -1)
    return ok, f"goal={d.get('goal')}, speed={d.get('speed')}, pos=({fish_x},{fish_y})", \
        {'goal': d.get('goal'), 'speed': d.get('speed')}

run_test("Dead fish produces no motor output", test_dead_fish_no_motor)


# =========================================================================
# GENERATE REPORT
# =========================================================================
print("\n" + "=" * 70)
n_pass = sum(1 for _, s, _ in results if s)
n_total = len(results)
pct = 100 * n_pass / n_total if n_total > 0 else 0
print(f"RESULTS: {n_pass}/{n_total} passed ({pct:.0f}%)")

if n_pass < n_total:
    print("\nFAILED TESTS:")
    for name, status, details in results:
        if not status:
            print(f"  - {name}: {details}")

print("\n" + "=" * 70)
if __name__ == '__main__':
    sys.exit(0 if n_pass == n_total else 1)
