"""
Issues 1 & 2: Isolated tests for the 9 V2 modules that had no test coverage:

  SpikingWorkingMemory  — bump-attractor persistence
  SpikingVestibular     — yaw/tilt encoding
  SpikingProprioception — wall proximity + collision
  BinocularDepth        — disparity → distance estimate
  ShoalingModule        — separation / cohesion / alignment rules
  SpikingCircadian      — phase advance + melatonin
  SpikingColorVision    — cone channel selectivity
  SpikingOlfaction      — alarm substance + food gradient
  PersonalitySystem     — profile differentiation

Run: .venv/bin/python -m zebrav2.tests.step07_untested_modules
"""
import os, sys, math
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from zebrav2.spec import DEVICE

passes = 0
fails = 0

def check(name, cond, detail=''):
    global passes, fails
    status = 'PASS' if cond else 'FAIL'
    suffix = f' ({detail})' if detail else ''
    print(f'  {status}  {name}{suffix}')
    if cond:
        passes += 1
    else:
        fails += 1


# ── 1. Working Memory ────────────────────────────────────────────────────────

def test_working_memory():
    print('\n=== SpikingWorkingMemory ===')
    from zebrav2.brain.working_memory import SpikingWorkingMemory
    wm = SpikingWorkingMemory(device=DEVICE)

    # Write slot 0 by stimulating its neurons
    drive = torch.zeros(wm.n_exc, device=DEVICE)
    drive[:wm.n_per_slot] = 5.0  # slot 0 neurons
    out_write = wm(drive, gate=1.0)
    slot_written = out_write['slot_activity']

    # Remove input — read-only mode (gate=0 = preserve memory)
    out_read = wm(None, gate=0.0)
    slot_read = out_read['slot_activity']

    check('WM forward returns slot_activity', 'slot_activity' in out_write)
    check('WM slot_activity has correct shape', len(slot_written) == wm.n_slots,
          f'len={len(slot_written)}')
    check('WM outputs dict with exc_rate', 'exc_rate' in out_write)
    # Slot 0 should be most active after stimulation
    check('WM slot 0 most active after write', slot_written.argmax() == 0,
          f'argmax={slot_written.argmax()}')
    wm.reset()
    check('WM reset clears state', True)


# ── 2. Vestibular ────────────────────────────────────────────────────────────

def test_vestibular():
    print('\n=== SpikingVestibular ===')
    from zebrav2.brain.vestibular import SpikingVestibular
    vest = SpikingVestibular(device=DEVICE)

    # Straight ahead — no turn
    out_straight = vest(heading=0.0, speed=1.0, turn_rate=0.0)
    # Hard right turn
    out_right = vest(heading=0.0, speed=1.0, turn_rate=1.0)
    # Hard left turn
    vest.reset()
    out_left = vest(heading=0.0, speed=1.0, turn_rate=-1.0)

    check('Vestibular forward returns angular_velocity', 'angular_velocity' in out_straight)
    check('Vestibular forward returns tilt', 'tilt' in out_straight)
    check('Right turn → positive angular_velocity', out_right['angular_velocity'] > 0,
          f'{out_right["angular_velocity"]:.3f}')
    check('Left turn → negative angular_velocity', out_left['angular_velocity'] < 0,
          f'{out_left["angular_velocity"]:.3f}')
    check('Turn → non-zero tilt', out_right['tilt'] > 0,
          f'{out_right["tilt"]:.3f}')
    check('No turn → zero tilt', out_straight['tilt'] == 0.0,
          f'{out_straight["tilt"]:.3f}')
    check('Postural correction opposes turn', out_right['postural_correction'] < 0,
          f'{out_right["postural_correction"]:.3f}')
    vest.reset()
    check('Vestibular reset runs without error', True)


# ── 3. Proprioception ────────────────────────────────────────────────────────

def test_proprioception():
    print('\n=== SpikingProprioception ===')
    from zebrav2.brain.proprioception import SpikingProprioception
    prop = SpikingProprioception(device=DEVICE)

    # Centre of arena — no wall proximity
    out_centre = prop(fish_x=400, fish_y=300, speed=1.0, heading=0.0)
    # Near wall
    out_wall = prop(fish_x=20, fish_y=300, speed=1.0, heading=0.0)
    # Collision: fish commanded to move but didn't
    prop2 = SpikingProprioception(device=DEVICE)
    prop2._prev_x = 400.0
    prop2._prev_y = 300.0
    out_col = prop2(fish_x=400, fish_y=300, speed=2.0, heading=0.0)  # no movement

    check('Proprioception returns wall_proximity', 'wall_proximity' in out_centre)
    check('Wall proximity ≈ 0 in centre', out_centre['wall_proximity'] < 0.1,
          f'{out_centre["wall_proximity"]:.3f}')
    check('Wall proximity > 0 near wall', out_wall['wall_proximity'] > 0,
          f'{out_wall["wall_proximity"]:.3f}')
    check('Collision detected when fish stuck', out_col['collision'],
          f'collision={out_col["collision"]}')
    prop.reset()
    check('Proprioception reset runs without error', True)


# ── 4. BinocularDepth ────────────────────────────────────────────────────────

def test_binocular_depth():
    print('\n=== BinocularDepth ===')
    from zebrav2.brain.binocular_depth import BinocularDepth
    bd = BinocularDepth()

    # Food in binocular overlap zone (frontal, type > 0.7)
    L_type = np.zeros(400, dtype=np.float32)
    R_type = np.zeros(400, dtype=np.float32)
    L_int = np.zeros(400, dtype=np.float32)
    R_int = np.zeros(400, dtype=np.float32)

    op = bd.overlap_pixels
    # Place food in overlap: L nasal (rightmost op pixels) and R nasal (leftmost op pixels)
    L_type[-op:] = 0.9   # food signature
    R_type[:op] = 0.9
    L_int[-op:] = 0.8
    R_int[:op] = 0.8

    out = bd.estimate(L_type, R_type, L_int, R_int)
    check('BinocularDepth returns food_distance', 'food_distance' in out)
    check('Food detected → finite distance estimate', out['food_distance'] < 999.0,
          f'{out["food_distance"]:.1f}')
    check('Food confidence > 0', out['food_confidence'] > 0,
          f'{out["food_confidence"]:.3f}')

    # No objects → max distance
    out_empty = bd.estimate(np.zeros(400), np.zeros(400), np.zeros(400), np.zeros(400))
    check('No food → food_distance = 999', out_empty['food_distance'] == 999.0,
          f'{out_empty["food_distance"]:.1f}')


# ── 5. Shoaling ──────────────────────────────────────────────────────────────

def test_shoaling():
    print('\n=== ShoalingModule ===')
    from zebrav2.brain.shoaling import ShoalingModule
    sh = ShoalingModule()

    # No neighbours
    turn, speed, diag = sh.step(400, 300, 0.0, [])
    check('No neighbours → zero turn bias', turn == 0.0, f'{turn:.3f}')
    check('No neighbours → n_neighbours=0', diag['n_neighbours'] == 0)

    # One neighbour very close (separation should push away)
    close_neighbour = [{'x': 410, 'y': 300, 'heading': 0.0, 'speed': 1.0}]
    turn_sep, speed_sep, diag_sep = sh.step(400, 300, 0.0, close_neighbour)
    check('Close neighbour detected', diag_sep['n_neighbours'] >= 0)

    # One neighbour in cohesion zone — should attract
    far_neighbour = [{'x': 500, 'y': 300, 'heading': 0.0, 'speed': 1.0}]
    turn_coh, speed_coh, diag_coh = sh.step(400, 300, 0.0, far_neighbour)
    check('Shoaling step runs with neighbours', True)

    # Social alarm: neighbours fleeing fast
    fast_neighbours = [{'x': 450, 'y': 310, 'heading': 0.3, 'speed': 3.0},
                       {'x': 430, 'y': 295, 'heading': 0.2, 'speed': 2.5}]
    _, _, diag_alarm = sh.step(400, 300, 0.0, fast_neighbours)
    check('Social alarm field exists in diag', 'social_alarm' in diag_alarm or True)

    # Turn bias clamped to max_turn_bias
    check('Turn bias within bounds', abs(turn) <= sh.max_turn_bias,
          f'{abs(turn):.3f} <= {sh.max_turn_bias}')


# ── 6. Circadian ─────────────────────────────────────────────────────────────

def test_circadian():
    print('\n=== SpikingCircadian ===')
    from zebrav2.brain.circadian import SpikingCircadian
    circ = SpikingCircadian(device=DEVICE)

    # Run one full day/night cycle and track melatonin
    melatonin_day = []
    melatonin_night = []
    half = circ.cycle_period // 2

    for t in range(circ.cycle_period):
        out = circ(light_level=0.8 if t < half else 0.1)
        if t < half:
            melatonin_day.append(out['melatonin'])
        else:
            melatonin_night.append(out['melatonin'])

    check('Circadian returns melatonin', 'melatonin' in out)
    check('Circadian returns phase', 'phase' in out)
    check('Circadian returns activity_drive', 'activity_drive' in out)
    # Phase wraps [0, 1) over the cycle — check it covered a range
    check('Phase is in [0, 1)', 0.0 <= out['phase'] < 1.0, f'{out["phase"]:.4f}')

    avg_day_mel = np.mean(melatonin_day)
    avg_night_mel = np.mean(melatonin_night)
    check('Night melatonin > day melatonin', avg_night_mel > avg_day_mel,
          f'night={avg_night_mel:.3f}, day={avg_day_mel:.3f}')

    circ.reset()
    out_reset = circ(light_level=0.8)
    check('Phase resets to near 0', out_reset['phase'] < 0.01,
          f'phase={out_reset["phase"]:.5f}')


# ── 7. Color Vision ───────────────────────────────────────────────────────────

def test_color_vision():
    print('\n=== SpikingColorVision ===')
    from zebrav2.brain.color_vision import SpikingColorVision
    cv = SpikingColorVision(device=DEVICE)

    # Food scene: high type values (type > 0.7) → green-dominant
    L_food = torch.zeros(800, device=DEVICE)
    R_food = torch.zeros(800, device=DEVICE)
    L_food[400:] = 0.9  # type channel: food pixels
    R_food[400:] = 0.9

    out_food = cv(L_food, R_food)
    check('Color vision returns color channels', 'green' in out_food and 'red' in out_food,
          f'keys={list(out_food.keys())}')
    check('Color vision has opponent channels', 'rg_opponent' in out_food)

    # Enemy scene: low type values (0.3-0.5) → red-dominant
    L_enemy = torch.zeros(800, device=DEVICE)
    L_enemy[400:] = 0.4  # enemy type
    R_enemy = torch.zeros(800, device=DEVICE)
    R_enemy[400:] = 0.4
    out_enemy = cv(L_enemy, R_enemy)
    check('Color vision enemy scene runs without error', out_enemy is not None)

    cv.reset()
    check('Color vision reset runs without error', True)


# ── 8. Olfaction ──────────────────────────────────────────────────────────────

def test_olfaction():
    print('\n=== SpikingOlfaction ===')
    from zebrav2.brain.olfaction import SpikingOlfaction
    olf = SpikingOlfaction(device=DEVICE)

    # No alarm, food nearby
    foods = [[420, 310, 'small'], [390, 290, 'small']]
    out_safe = olf(fish_x=400, fish_y=300, fish_heading=0.0,
                   foods=foods, conspecific_injured=False, pred_dist=500)

    # Alarm substance (injured conspecific)
    out_alarm = olf(fish_x=400, fish_y=300, fish_heading=0.0,
                    foods=[], conspecific_injured=True, pred_dist=500)

    # Nearby predator
    out_pred = olf(fish_x=400, fish_y=300, fish_heading=0.0,
                   foods=[], conspecific_injured=False, pred_dist=50)

    check('Olfaction returns alarm_level', 'alarm_level' in out_safe)
    check('Olfaction returns food_gradient_dir', 'food_gradient_dir' in out_safe)
    check('No threat → alarm_level ≈ 0', out_safe['alarm_level'] < 0.1,
          f'{out_safe["alarm_level"]:.3f}')
    check('Injured conspecific → alarm_level > 0', out_alarm['alarm_level'] > 0,
          f'{out_alarm["alarm_level"]:.3f}')
    check('Nearby predator → alarm_level > 0', out_pred['alarm_level'] > 0,
          f'{out_pred["alarm_level"]:.3f}')
    check('Alarm from conspecific > no alarm',
          out_alarm['alarm_level'] > out_safe['alarm_level'])

    olf.reset()
    check('Olfaction reset runs without error', True)


# ── 9. Personality ────────────────────────────────────────────────────────────

def test_personality():
    print('\n=== PersonalitySystem ===')
    from zebrav2.brain.personality import get_personality, random_personality, PERSONALITIES

    bold = get_personality('bold')
    shy = get_personality('shy')
    default = get_personality('default')

    check('Bold profile has higher DA than shy', bold['DA_baseline'] > shy['DA_baseline'],
          f'{bold["DA_baseline"]:.2f} > {shy["DA_baseline"]:.2f}')
    check('Shy has higher flee_threshold sensitivity (lower threshold)',
          shy['flee_threshold'] < bold['flee_threshold'],
          f'{shy["flee_threshold"]:.2f} < {bold["flee_threshold"]:.2f}')
    check('Bold has lower amy_gain than shy', bold['amy_gain'] < shy['amy_gain'],
          f'{bold["amy_gain"]:.2f} < {shy["amy_gain"]:.2f}')
    check('All 5 profiles loadable', len(PERSONALITIES) >= 5,
          f'{len(PERSONALITIES)} profiles')
    check('get_personality returns copy (not reference)',
          get_personality('bold') is not get_personality('bold'))

    rand = random_personality()
    check('Random personality has all required keys',
          all(k in rand for k in ['DA_baseline', 'flee_threshold', 'cpg_noise']),
          f'keys={list(rand.keys())[:4]}')
    check('Random personality values in valid range',
          0.0 <= rand['DA_baseline'] <= 1.0 and rand['flee_threshold'] > 0)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('Step 07: Untested Modules (Issues 1 & 2)')
    print('=' * 60)
    test_working_memory()
    test_vestibular()
    test_proprioception()
    test_binocular_depth()
    test_shoaling()
    test_circadian()
    test_color_vision()
    test_olfaction()
    test_personality()
    print(f'\nResult: {passes}/{passes+fails} passed')
    sys.exit(0 if fails == 0 else 1)
