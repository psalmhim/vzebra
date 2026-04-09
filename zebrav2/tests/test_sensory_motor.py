"""
Stimulus-response tests for 5 biologically rewritten sensory/motor modules.
  1. Retina v2: DoG RGCs, angular looming, UV prey, Reichardt DS
  2. Lateral line: superficial/canal dual-channel, multi-source
  3. Spinal CPG: 96-neuron LIF circuit, bout-glide, Renshaw
  4. Reticulospinal: C-start/T-start, CoLo crossed inhibition, voluntary turn
  5. Olfaction: Fick diffusion, bilateral gradient, receptor adaptation

Run:
  .venv/bin/python -m zebrav2.tests.test_sensory_motor
"""
import os
import sys
import math
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.retina import RetinaV2
from zebrav2.brain.lateral_line import SpikingLateralLine
from zebrav2.brain.spinal_cpg import SpinalCPG
from zebrav2.brain.reticulospinal import ReticulospinalSystem
from zebrav2.brain.olfaction import SpikingOlfaction

PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'v2_sensory_motor')
os.makedirs(PLOT_DIR, exist_ok=True)

results = {}  # criterion_key -> ('PASS'|'FAIL', value_str, threshold_str)


# ---------------------------------------------------------------------------
# Test 1: RetinaV2
# ---------------------------------------------------------------------------

def test_retina():
    """Test 1: RetinaV2 — DoG RGCs, looming, UV prey, Reichardt DS."""
    print("\n=== Test 1: RetinaV2 ===")

    retina = RetinaV2(device=DEVICE)

    # ---- 1a. ON cells: bright food pixel ----
    retina.reset()
    L = torch.zeros(800, device=DEVICE)
    R = torch.zeros(800, device=DEVICE)
    # Set a band of high-intensity food pixels in positions 50..80
    L[50:80] = 0.9   # bright centre → ON cells should respond
    out = retina(L, R, {'enemy': 0.0})
    on_mean = float(out['L_on'].mean())
    on_ok = on_mean > 0.0
    results['RET_on_cells'] = (
        'PASS' if on_ok else 'FAIL',
        f'{on_mean:.4f}', '> 0')
    print(f"  RET_on_cells: {'PASS' if on_ok else 'FAIL'} (L_on.mean={on_mean:.4f}, > 0)")

    # ---- 1b. OFF cells: bright → dark transition ----
    retina.reset()
    L2 = torch.zeros(800, device=DEVICE)
    R2 = torch.zeros(800, device=DEVICE)
    # First frame: bright
    L2[100:140] = 0.9
    retina(L2, R2, {'enemy': 0.0})
    # Second frame: dark (luminance drop → OFF transient)
    L2_dark = torch.zeros(800, device=DEVICE)
    out2 = retina(L2_dark, R2, {'enemy': 0.0})
    off_mean = float(out2['L_off'].mean())
    off_ok = off_mean > 0.0
    results['RET_off_cells'] = (
        'PASS' if off_ok else 'FAIL',
        f'{off_mean:.4f}', '> 0 after bright→dark')
    print(f"  RET_off_cells: {'PASS' if off_ok else 'FAIL'} (L_off.mean={off_mean:.4f}, > 0)")

    # ---- 1c. Looming trigger=True: enemy_frac increases 0.1 → 0.4 ----
    retina.reset()
    L3 = torch.zeros(800, device=DEVICE)
    R3 = torch.zeros(800, device=DEVICE)
    retina(L3, R3, {'enemy': 0.1})   # frame 1: small enemy
    out3 = retina(L3, R3, {'enemy': 0.4})  # frame 2: large enemy (expansion)
    loom_trigger_on = out3['loom_trigger']
    results['RET_loom_trigger_on'] = (
        'PASS' if loom_trigger_on else 'FAIL',
        str(loom_trigger_on), 'True when expanding 0.1→0.4')
    print(f"  RET_loom_trigger_on: {'PASS' if loom_trigger_on else 'FAIL'} (trigger={loom_trigger_on})")

    # ---- 1d. Looming trigger=False: enemy_frac constant ----
    retina.reset()
    L4 = torch.zeros(800, device=DEVICE)
    R4 = torch.zeros(800, device=DEVICE)
    retina(L4, R4, {'enemy': 0.4})
    out4 = retina(L4, R4, {'enemy': 0.4})  # no expansion
    loom_trigger_off = out4['loom_trigger']
    results['RET_loom_trigger_off'] = (
        'PASS' if not loom_trigger_off else 'FAIL',
        str(loom_trigger_off), 'False when constant')
    print(f"  RET_loom_trigger_off: {'PASS' if not loom_trigger_off else 'FAIL'} (trigger={loom_trigger_off})")

    # ---- 1e. UV prey > 0.4: low intensity + high type at same positions ----
    retina.reset()
    L5 = torch.zeros(800, device=DEVICE)
    R5 = torch.zeros(800, device=DEVICE)
    # intensity < 0.3 AND type > 0.7 at same pixels → UV prey signal
    L5[100:106] = 0.15   # low intensity (food-dark, UV-absorbing)
    L5[500:506] = 0.95   # type channel > 0.7 at same pixel positions (500 = type channel for idx 100)
    out5 = retina(L5, R5, {'enemy': 0.0})
    uv_prey = out5['uv_prey_L']
    uv_ok = uv_prey > 0.4
    results['RET_uv_prey'] = (
        'PASS' if uv_ok else 'FAIL',
        f'{uv_prey:.4f}', '> 0.4')
    print(f"  RET_uv_prey: {'PASS' if uv_ok else 'FAIL'} (uv_prey_L={uv_prey:.4f}, > 0.4)")

    # ---- 1f. DS cells: non-zero after second frame (delay buffer filled) ----
    retina.reset()
    L6 = torch.zeros(800, device=DEVICE)
    R6 = torch.zeros(800, device=DEVICE)
    L6[50:200] = 0.5
    retina(L6, R6, {'enemy': 0.0})  # first frame: fills delay buffer
    L6_shifted = torch.zeros(800, device=DEVICE)
    L6_shifted[60:210] = 0.5        # shifted pattern (motion)
    out6 = retina(L6_shifted, R6, {'enemy': 0.0})
    ds_mean = float(out6['ds_fused'].mean())
    ds_ok = ds_mean > 0.0
    results['RET_ds_cells'] = (
        'PASS' if ds_ok else 'FAIL',
        f'{ds_mean:.4f}', '> 0 after second frame')
    print(f"  RET_ds_cells: {'PASS' if ds_ok else 'FAIL'} (ds_fused.mean={ds_mean:.4f}, > 0)")

    # ---- Figure: time-series for a 10-step sweep ----
    retina.reset()
    on_rates, off_rates, loom_vals, ds_vals, uv_prey_vals = [], [], [], [], []
    loom_triggers = []

    for t in range(10):
        L_t = torch.zeros(800, device=DEVICE)
        R_t = torch.zeros(800, device=DEVICE)
        # Introduce food pixels after step 2
        if t >= 2:
            L_t[50:70] = 0.9
            L_t[450:456] = 0.15   # intensity
            L_t[850:856] = 0.95   # type
        # Growing enemy from step 5
        enemy_frac = 0.0 if t < 5 else (t - 4) * 0.08
        out_t = retina(L_t, R_t, {'enemy': enemy_frac})
        on_rates.append(float(out_t['L_on'].mean()))
        off_rates.append(float(out_t['L_off'].mean()))
        loom_vals.append(float(out_t['loom_fused'].mean()))
        ds_vals.append(float(out_t['ds_fused'].mean()))
        uv_prey_vals.append(out_t['uv_prey_L'])
        loom_triggers.append(1.0 if out_t['loom_trigger'] else 0.0)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Test 1: RetinaV2 Stimulus-Response', fontsize=14, fontweight='bold')
    steps = list(range(10))

    axes[0, 0].plot(steps, on_rates, 'g-o', linewidth=1.5, markersize=4)
    axes[0, 0].set_title('ON Cell Rate (L eye mean)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].axvline(2, color='gray', linestyle='--', label='food on')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(steps, off_rates, 'r-o', linewidth=1.5, markersize=4)
    axes[0, 1].set_title('OFF Cell Rate (L eye mean)')
    axes[0, 1].set_xlabel('Step')

    axes[1, 0].plot(steps, loom_vals, 'b-o', linewidth=1.5, markersize=4, label='loom_fused')
    axes[1, 0].scatter([s for s, v in zip(steps, loom_triggers) if v > 0],
                       [loom_vals[s] for s, v in zip(steps, loom_triggers) if v > 0],
                       color='red', s=80, zorder=5, label='loom_trigger')
    axes[1, 0].set_title('Looming Rate + Trigger Events')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(steps, ds_vals, 'm-o', linewidth=1.5, markersize=4)
    axes[1, 1].set_title('DS Cell Rate (fused mean)')
    axes[1, 1].set_xlabel('Step')

    axes[2, 0].plot(steps, uv_prey_vals, 'c-o', linewidth=1.5, markersize=4)
    axes[2, 0].axhline(0.4, color='r', linestyle='--', label='threshold 0.4')
    axes[2, 0].set_title('UV Prey Signal (L eye)')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].bar(['ON\nmean', 'OFF\nmean', 'Loom\nmean', 'DS\nmean', 'UV\nprey'],
                   [np.mean(on_rates), np.mean(off_rates), np.mean(loom_vals),
                    np.mean(ds_vals), np.mean(uv_prey_vals)],
                   color=['green', 'red', 'blue', 'purple', 'cyan'])
    axes[2, 1].set_title('Mean Output Summary')
    axes[2, 1].set_ylabel('Rate / Scalar')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test1_retina.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR}/test1_retina.png")
    return all(results[k][0] == 'PASS' for k in results if k.startswith('RET_'))


# ---------------------------------------------------------------------------
# Test 2: SpikingLateralLine
# ---------------------------------------------------------------------------

def test_lateral_line():
    """Test 2: SpikingLateralLine — superficial/canal dual-channel, multi-source."""
    print("\n=== Test 2: SpikingLateralLine ===")

    ll = SpikingLateralLine(device=DEVICE)

    # ---- 2a. Predator proximity > 0.3 at dist=50px ----
    ll.reset()
    out_pred = ll(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                  pred_x=450.0, pred_y=300.0,
                  pred_vx=8.0, pred_vy=0.0,
                  foods=[], conspecifics=[])
    prox_ok = out_pred['proximity'] > 0.3
    results['LL_proximity'] = (
        'PASS' if prox_ok else 'FAIL',
        f'{out_pred["proximity"]:.4f}', '> 0.3')
    print(f"  LL_proximity: {'PASS' if prox_ok else 'FAIL'} (proximity={out_pred['proximity']:.4f}, > 0.3)")

    # ---- 2b. High-freq threat > 0 with predator acceleration ----
    ll.reset()
    # Step 1: predator at dist 50, vel (8,0)
    ll(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
       pred_x=450.0, pred_y=300.0, pred_vx=8.0, pred_vy=0.0,
       foods=[], conspecifics=[])
    # Step 2: predator suddenly accelerates to (20,0)
    out_hft = ll(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                 pred_x=450.0, pred_y=300.0, pred_vx=20.0, pred_vy=0.0,
                 foods=[], conspecifics=[])
    hft_ok = out_hft['high_freq_threat'] > 0.0
    results['LL_high_freq_threat'] = (
        'PASS' if hft_ok else 'FAIL',
        f'{out_hft["high_freq_threat"]:.4f}', '> 0')
    print(f"  LL_high_freq_threat: {'PASS' if hft_ok else 'FAIL'} (hft={out_hft['high_freq_threat']:.4f}, > 0)")

    # ---- 2c. Prey detected=True: food 15px ahead of fish ----
    ll.reset()
    # Fish at (400,300) heading=0 (→ east), food at (415,300): 15px ahead, within 100px range
    out_prey = ll(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                  pred_x=-999.0, pred_y=-999.0, pred_vx=0.0, pred_vy=0.0,
                  foods=[(415.0, 300.0)], conspecifics=[])
    prey_detected_ok = out_prey['prey_detected']
    results['LL_prey_detected'] = (
        'PASS' if prey_detected_ok else 'FAIL',
        str(prey_detected_ok), 'True with food at 15px')
    print(f"  LL_prey_detected: {'PASS' if prey_detected_ok else 'FAIL'} (prey_detected={prey_detected_ok})")

    # ---- 2d. low_freq_prey > 0.005 ----
    lfp_ok = out_prey['low_freq_prey'] > 0.005
    results['LL_low_freq_prey'] = (
        'PASS' if lfp_ok else 'FAIL',
        f'{out_prey["low_freq_prey"]:.6f}', '> 0.005')
    print(f"  LL_low_freq_prey: {'PASS' if lfp_ok else 'FAIL'} (lfp={out_prey['low_freq_prey']:.6f}, > 0.005)")

    # ---- 2e. Prey detected=False with no food ----
    ll.reset()
    out_noprey = ll(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                    pred_x=-999.0, pred_y=-999.0, pred_vx=0.0, pred_vy=0.0,
                    foods=[], conspecifics=[])
    no_prey_ok = not out_noprey['prey_detected']
    results['LL_no_prey'] = (
        'PASS' if no_prey_ok else 'FAIL',
        str(out_noprey['prey_detected']), 'False with no food')
    print(f"  LL_no_prey: {'PASS' if no_prey_ok else 'FAIL'} (prey_detected={out_noprey['prey_detected']})")

    # ---- 2f. Conspecific detected within 120px ----
    ll.reset()
    out_cs = ll(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                pred_x=-999.0, pred_y=-999.0, pred_vx=0.0, pred_vy=0.0,
                foods=[],
                conspecifics=[{'x': 480.0, 'y': 300.0, 'speed': 1.0}])
    cs_ok = out_cs['conspecific_dist'] < 999.0
    results['LL_conspecific_dist'] = (
        'PASS' if cs_ok else 'FAIL',
        f'{out_cs["conspecific_dist"]:.1f}', '< 999 (detected within 120px)')
    print(f"  LL_conspecific_dist: {'PASS' if cs_ok else 'FAIL'} (dist={out_cs['conspecific_dist']:.1f}, < 999)")

    # ---- Figure: bar chart across scenarios ----
    scenarios = ['Predator\n(dist=50)', 'Food\n(15px)', 'Conspecific\n(80px)', 'Empty']
    proximities, lfp_vals, hft_vals, cs_dists_norm = [], [], [], []

    # Scenario: predator
    ll.reset()
    o = ll(400, 300, 0.0, 450, 300, 8.0, 0.0, [], [])
    proximities.append(o['proximity'])
    lfp_vals.append(o['low_freq_prey'])
    hft_vals.append(o['high_freq_threat'])
    cs_dists_norm.append(1.0 if o['conspecific_dist'] < 999 else 0.0)

    # Scenario: food 15px ahead
    ll.reset()
    o = ll(400, 300, 0.0, -999, -999, 0, 0, [(415, 300)], [])
    proximities.append(o['proximity'])
    lfp_vals.append(o['low_freq_prey'])
    hft_vals.append(o['high_freq_threat'])
    cs_dists_norm.append(1.0 if o['conspecific_dist'] < 999 else 0.0)

    # Scenario: conspecific within 80px
    ll.reset()
    o = ll(400, 300, 0.0, -999, -999, 0, 0, [],
           [{'x': 480.0, 'y': 300.0, 'speed': 1.0}])
    proximities.append(o['proximity'])
    lfp_vals.append(o['low_freq_prey'])
    hft_vals.append(o['high_freq_threat'])
    cs_dists_norm.append(1.0 if o['conspecific_dist'] < 999 else 0.0)

    # Scenario: empty
    ll.reset()
    o = ll(400, 300, 0.0, -999, -999, 0, 0, [], [])
    proximities.append(o['proximity'])
    lfp_vals.append(o['low_freq_prey'])
    hft_vals.append(o['high_freq_threat'])
    cs_dists_norm.append(1.0 if o['conspecific_dist'] < 999 else 0.0)

    x = np.arange(len(scenarios))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Test 2: SpikingLateralLine Multi-Source Responses', fontsize=14, fontweight='bold')
    ax.bar(x - 1.5 * width, proximities, width, label='proximity', color='steelblue')
    ax.bar(x - 0.5 * width, hft_vals,   width, label='high_freq_threat', color='tomato')
    ax.bar(x + 0.5 * width, lfp_vals,   width, label='low_freq_prey', color='mediumseagreen')
    ax.bar(x + 1.5 * width, cs_dists_norm, width, label='conspecific_detected', color='gold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Output value')
    ax.legend()
    ax.axhline(0.3, color='steelblue', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(0.005, color='mediumseagreen', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test2_lateral_line.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR}/test2_lateral_line.png")
    return all(results[k][0] == 'PASS' for k in results if k.startswith('LL_'))


# ---------------------------------------------------------------------------
# Test 3: SpinalCPG
# ---------------------------------------------------------------------------

def test_cpg():
    """Test 3: SpinalCPG — 96-neuron LIF bout-glide, Renshaw, turn asymmetry."""
    print("\n=== Test 3: SpinalCPG ===")

    # ---- 3a. turn=+1 gives turn_out > 0 ----
    cpg_pos = SpinalCPG(device=DEVICE)
    turn_outs_pos = []
    for _ in range(12):
        _, _, _, turn_out, diag = cpg_pos.step(0.8, turn=+1.0)
        turn_outs_pos.append(turn_out)

    # Find last step with non-glide motor activity
    active_turns = [t for t in turn_outs_pos if abs(t) > 1e-6]
    turn_pos_val = active_turns[-1] if active_turns else 0.0
    turn_pos_ok = turn_pos_val > 0.0
    results['CPG_turn_pos'] = (
        'PASS' if turn_pos_ok else 'FAIL',
        f'{turn_pos_val:.4f}', '> 0 for turn=+1')
    print(f"  CPG_turn_pos: {'PASS' if turn_pos_ok else 'FAIL'} (turn_out={turn_pos_val:.4f}, > 0)")

    # ---- 3b. turn=-1 gives turn_out < 0 ----
    cpg_neg = SpinalCPG(device=DEVICE)
    turn_outs_neg = []
    for _ in range(12):
        _, _, _, turn_out, _ = cpg_neg.step(0.8, turn=-1.0)
        turn_outs_neg.append(turn_out)

    active_turns_neg = [t for t in turn_outs_neg if abs(t) > 1e-6]
    turn_neg_val = active_turns_neg[-1] if active_turns_neg else 0.0
    turn_neg_ok = turn_neg_val < 0.0
    results['CPG_turn_neg'] = (
        'PASS' if turn_neg_ok else 'FAIL',
        f'{turn_neg_val:.4f}', '< 0 for turn=-1')
    print(f"  CPG_turn_neg: {'PASS' if turn_neg_ok else 'FAIL'} (turn_out={turn_neg_val:.4f}, < 0)")

    # ---- 3c. Bout-glide pattern: in_bout transitions to glide_active within 5 steps ----
    cpg_bg = SpinalCPG(device=DEVICE)
    glide_seen = False
    for _ in range(5):
        _, _, _, _, diag = cpg_bg.step(0.8, turn=0.0)
        if diag['glide_active']:
            glide_seen = True
            break
    results['CPG_bout_glide'] = (
        'PASS' if glide_seen else 'FAIL',
        str(glide_seen), 'glide_active=True within 5 steps at drive=0.8')
    print(f"  CPG_bout_glide: {'PASS' if glide_seen else 'FAIL'} (glide_active={glide_seen})")

    # ---- 3d. Renshaw fires after first bout ----
    cpg_rsh = SpinalCPG(device=DEVICE)
    renshaw_vals = []
    for _ in range(15):
        _, _, _, _, diag = cpg_rsh.step(0.8, turn=0.0)
        renshaw_vals.append(diag['renshaw_L'])
    renshaw_peak = max(renshaw_vals)
    rsh_ok = renshaw_peak > 0.0
    results['CPG_renshaw'] = (
        'PASS' if rsh_ok else 'FAIL',
        f'{renshaw_peak:.4f}', '> 0 (builds from MN activity)')
    print(f"  CPG_renshaw: {'PASS' if rsh_ok else 'FAIL'} (renshaw_L max={renshaw_peak:.4f}, > 0)")

    # ---- 3e. Zero drive = zero output ----
    cpg_zero = SpinalCPG(device=DEVICE)
    motor_l, motor_r, _, _, _ = cpg_zero.step(0.0, turn=0.0)
    zero_ok = (motor_l == 0.0 and motor_r == 0.0)
    results['CPG_zero_drive'] = (
        'PASS' if zero_ok else 'FAIL',
        f'L={motor_l:.4f}, R={motor_r:.4f}', 'both = 0 at drive=0')
    print(f"  CPG_zero_drive: {'PASS' if zero_ok else 'FAIL'} (L={motor_l:.4f}, R={motor_r:.4f})")

    # ---- Figure: 3 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Test 3: SpinalCPG Stimulus-Response', fontsize=14, fontweight='bold')

    # Panel 1: turn_out for turn=+1 / -1 / 0
    axes[0].plot(turn_outs_pos, 'b-o', markersize=3, linewidth=1.5, label='turn=+1')
    axes[0].plot(turn_outs_neg, 'r-o', markersize=3, linewidth=1.5, label='turn=-1')
    cpg_z_plot = SpinalCPG(device=DEVICE)
    zero_turns = [cpg_z_plot.step(0.8, 0.0)[3] for _ in range(12)]
    axes[0].plot(zero_turns, 'g-o', markersize=3, linewidth=1.5, label='turn=0')
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[0].set_title('Turn Output vs Turn Command')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('turn_out')
    axes[0].legend(fontsize=8)

    # Panel 2: bout/glide state timeline
    cpg_plot = SpinalCPG(device=DEVICE)
    bout_states, glide_states, speed_vals = [], [], []
    for _ in range(20):
        _, _, spd, _, d = cpg_plot.step(0.8, 0.0)
        bout_states.append(1.0 if d['bout_active'] else 0.0)
        glide_states.append(1.0 if d['glide_active'] else 0.0)
        speed_vals.append(spd)
    steps20 = list(range(20))
    axes[1].fill_between(steps20, bout_states, alpha=0.4, color='blue', label='in_bout')
    axes[1].fill_between(steps20, glide_states, alpha=0.4, color='orange', label='glide_active')
    axes[1].plot(steps20, speed_vals, 'k-', linewidth=1.5, label='speed')
    axes[1].set_title('Bout-Glide State Timeline (drive=0.8)')
    axes[1].set_xlabel('Step')
    axes[1].legend(fontsize=8)

    # Panel 3: Renshaw_L over 15 steps
    axes[2].plot(renshaw_vals, 'purple', linewidth=1.5, marker='o', markersize=3)
    axes[2].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
    axes[2].set_title('Renshaw (V1-like) Inhibition Rate (L)')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('renshaw_L rate')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test3_cpg.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR}/test3_cpg.png")
    return all(results[k][0] == 'PASS' for k in results if k.startswith('CPG_'))


# ---------------------------------------------------------------------------
# Test 4: ReticulospinalSystem
# ---------------------------------------------------------------------------

def test_reticulospinal():
    """Test 4: ReticulospinalSystem — voluntary turn, C-start, T-start, crossed inhibition."""
    print("\n=== Test 4: ReticulospinalSystem ===")

    def _make_sgc(mean_val: float, n: int = 64) -> torch.Tensor:
        return torch.full((n,), mean_val, device=DEVICE)

    def _make_pal_d(left_bias: float, right_bias: float, n: int = 600) -> torch.Tensor:
        t = torch.zeros(n, device=DEVICE)
        half = n // 2
        t[:half]  = left_bias
        t[half:]  = right_bias
        return t

    # ---- 4a. Voluntary right turn > 0 ----
    rs = ReticulospinalSystem(device=DEVICE)
    # Right pallium dominant (right half > left half) → voluntary_turn > 0
    pal_d_right = _make_pal_d(left_bias=0.1, right_bias=0.9)
    sgc_low = _make_sgc(0.0)
    out_vol = rs(sgc_rate=sgc_low, bg_gate=0.8, pal_d_rate=pal_d_right,
                 flee_dir=0.0, goal_speed=1.0, looming=False)
    vol_turn_ok = out_vol['turn'] > 0.0
    results['RS_voluntary_right'] = (
        'PASS' if vol_turn_ok else 'FAIL',
        f'{out_vol["turn"]:.4f}', '> 0 (right pallium dominant)')
    print(f"  RS_voluntary_right: {'PASS' if vol_turn_ok else 'FAIL'} (turn={out_vol['turn']:.4f}, > 0)")

    # ---- 4b. C-start first stage: cstart=True, turn=-1.5 * direction ----
    # flee_dir=+1 → threat on right → left M-cell fires → direction=-1.0 → turn=-1.5
    rs2 = ReticulospinalSystem(device=DEVICE)
    sgc_high = _make_sgc(0.1)   # > 0.05 threshold
    pal_d_neutral = _make_pal_d(0.0, 0.0)
    out_cs1 = rs2(sgc_rate=sgc_high, bg_gate=0.0, pal_d_rate=pal_d_neutral,
                  flee_dir=+1.0, goal_speed=1.0, looming=True)
    cs_triggered = out_cs1['cstart']
    cs_turn_val = out_cs1['turn']
    cs_speed_val = out_cs1['speed']
    cs_ok = cs_triggered and abs(cs_turn_val) == 1.5 and abs(cs_speed_val - 0.3) < 0.01
    results['RS_cstart_stage1'] = (
        'PASS' if cs_ok else 'FAIL',
        f'cstart={cs_triggered}, turn={cs_turn_val:.1f}, speed={cs_speed_val:.1f}',
        'cstart=True, |turn|=1.5, speed=0.3')
    print(f"  RS_cstart_stage1: {'PASS' if cs_ok else 'FAIL'} "
          f"(cstart={cs_triggered}, turn={cs_turn_val:.1f}, speed={cs_speed_val:.1f})")

    # ---- 4c. C-start sequence: 4 steps decreasing |turn|, increasing speed ----
    turns_cs, speeds_cs = [cs_turn_val], [cs_speed_val]
    for _ in range(3):
        out_step = rs2(sgc_rate=sgc_high, bg_gate=0.0, pal_d_rate=pal_d_neutral,
                       flee_dir=+1.0, goal_speed=1.0, looming=True)
        turns_cs.append(out_step['turn'])
        speeds_cs.append(out_step['speed'])

    abs_turns = [abs(t) for t in turns_cs]
    # C-start sequence: |turn| should decrease overall (4→3→2→1 maps to 1.5→1.0→0.2→0.0)
    turn_decreasing = abs_turns[-1] < abs_turns[0]
    speed_increasing = speeds_cs[-1] > speeds_cs[0]
    cs_seq_ok = turn_decreasing and speed_increasing
    results['RS_cstart_sequence'] = (
        'PASS' if cs_seq_ok else 'FAIL',
        f'|turns|={[f"{t:.1f}" for t in abs_turns]}, speeds={[f"{s:.1f}" for s in speeds_cs]}',
        'decreasing |turn|, increasing speed')
    print(f"  RS_cstart_sequence: {'PASS' if cs_seq_ok else 'FAIL'} "
          f"(|turns|={[round(t, 2) for t in abs_turns]}, speeds={[round(s, 2) for s in speeds_cs]})")

    # ---- 4d. T-start first stage: tstart=True, |turn|=0.8 ----
    # flee_dir=-1 → threat on left → T-start dir=+1 → turn=0.8
    rs3 = ReticulospinalSystem(device=DEVICE)
    sgc_sub = _make_sgc(0.03)   # 0.02 < mean < 0.05 → primary T-start
    out_ts1 = rs3(sgc_rate=sgc_sub, bg_gate=0.0, pal_d_rate=pal_d_neutral,
                  flee_dir=-1.0, goal_speed=1.0, looming=True)
    ts_triggered = out_ts1['tstart']
    ts_turn_val = out_ts1['turn']
    ts_ok = ts_triggered and abs(ts_turn_val) == 0.8
    results['RS_tstart_stage1'] = (
        'PASS' if ts_ok else 'FAIL',
        f'tstart={ts_triggered}, |turn|={abs(ts_turn_val):.1f}',
        'tstart=True, |turn|=0.8')
    print(f"  RS_tstart_stage1: {'PASS' if ts_ok else 'FAIL'} "
          f"(tstart={ts_triggered}, |turn|={abs(ts_turn_val):.1f})")

    # ---- 4e. No C-start without looming ----
    rs4 = ReticulospinalSystem(device=DEVICE)
    out_noloom = rs4(sgc_rate=sgc_high, bg_gate=0.0, pal_d_rate=pal_d_neutral,
                     flee_dir=+1.0, goal_speed=1.0, looming=False)
    no_cs_ok = not out_noloom['cstart']
    results['RS_no_cstart_without_loom'] = (
        'PASS' if no_cs_ok else 'FAIL',
        str(out_noloom['cstart']), 'False when looming=False')
    print(f"  RS_no_cstart_without_loom: {'PASS' if no_cs_ok else 'FAIL'} "
          f"(cstart={out_noloom['cstart']})")

    # ---- Figure: 2×2 subplots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Test 4: ReticulospinalSystem Stimulus-Response', fontsize=14, fontweight='bold')

    # Panel 1: C-start turn+speed sequence
    axes[0, 0].plot(turns_cs, 'b-o', linewidth=2, markersize=6, label='turn')
    axes[0, 0].plot(speeds_cs, 'r-s', linewidth=2, markersize=6, label='speed')
    axes[0, 0].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[0, 0].set_title('C-start Sequence (4 steps)')
    axes[0, 0].set_xlabel('C-start Stage')
    axes[0, 0].legend()

    # Panel 2: T-start sequence
    rs_ts_plot = ReticulospinalSystem(device=DEVICE)
    turns_ts, speeds_ts = [], []
    for i in range(4):
        if i == 0:
            o = rs_ts_plot(sgc_sub, 0.0, pal_d_neutral, -1.0, 1.0, True)
        else:
            o = rs_ts_plot(sgc_sub, 0.0, pal_d_neutral, -1.0, 1.0, True)
        turns_ts.append(o['turn'])
        speeds_ts.append(o['speed'])
    axes[0, 1].plot(turns_ts, 'b-o', linewidth=2, markersize=6, label='turn')
    axes[0, 1].plot(speeds_ts, 'r-s', linewidth=2, markersize=6, label='speed')
    axes[0, 1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[0, 1].set_title('T-start Sequence (up to 3 steps)')
    axes[0, 1].set_xlabel('T-start Stage')
    axes[0, 1].legend()

    # Panel 3: Voluntary turn vs bg_gate sweep
    rs_vol = ReticulospinalSystem(device=DEVICE)
    bg_gates = np.linspace(0.0, 1.0, 10)
    vol_turns = []
    for bg in bg_gates:
        o = rs_vol(sgc_low, float(bg), pal_d_right, 0.0, 1.0, False)
        vol_turns.append(o['turn'])
    axes[1, 0].plot(bg_gates, vol_turns, 'g-o', linewidth=2, markersize=5)
    axes[1, 0].set_title('Voluntary Turn vs BG Gate')
    axes[1, 0].set_xlabel('bg_gate')
    axes[1, 0].set_ylabel('turn output')
    axes[1, 0].axhline(0, color='k', linestyle='--', linewidth=0.8)

    # Panel 4: Crossed inhibition — refractory after C-start
    rs_refrac = ReticulospinalSystem(device=DEVICE)
    refrac_L_vals, refrac_R_vals = [], []
    # Trigger C-start
    rs_refrac(sgc_high, 0.0, pal_d_neutral, +1.0, 1.0, True)
    for _ in range(15):
        refrac_L_vals.append(rs_refrac.mauthner_refrac_L)
        refrac_R_vals.append(rs_refrac.mauthner_refrac_R)
        rs_refrac(sgc_high, 0.0, pal_d_neutral, +1.0, 1.0, True)
    axes[1, 1].plot(refrac_L_vals, 'b-', linewidth=1.5, label='refrac_L (own)')
    axes[1, 1].plot(refrac_R_vals, 'r-', linewidth=1.5, label='refrac_R (CoLo crossed)')
    axes[1, 1].set_title('Mauthner Refractory Counters After C-start')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Refractory steps remaining')
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test4_reticulospinal.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR}/test4_reticulospinal.png")
    return all(results[k][0] == 'PASS' for k in results if k.startswith('RS_'))


# ---------------------------------------------------------------------------
# Test 5: SpikingOlfaction
# ---------------------------------------------------------------------------

def test_olfaction():
    """Test 5: SpikingOlfaction — Fick diffusion, bilateral gradient, receptor adaptation."""
    print("\n=== Test 5: SpikingOlfaction ===")

    # ---- 5a. bilateral_diff > 0: food right of fish when heading=π/2 (up) ----
    # Fish at (400,300), heading=π/2 (north). Nares: L=fish_x - d*sin(π/2)=(395,300),
    # R=fish_x + d*sin(π/2)=(405,300). Food at (405,300): 0px from R nare, 10px from L nare.
    olf = SpikingOlfaction(device=DEVICE)
    out_a = olf(fish_x=400.0, fish_y=300.0, fish_heading=math.pi / 2,
                foods=[(405.0, 300.0)], conspecific_injured=False, pred_dist=999.0)
    bdiff_a = out_a['bilateral_diff']
    bdiff_a_ok = bdiff_a > 0.0
    results['OLF_bilateral_diff_right'] = (
        'PASS' if bdiff_a_ok else 'FAIL',
        f'{bdiff_a:.4f}', '> 0 (food right of fish, heading=π/2)')
    print(f"  OLF_bilateral_diff_right: {'PASS' if bdiff_a_ok else 'FAIL'} "
          f"(bilateral_diff={bdiff_a:.4f}, > 0)")

    # ---- 5b. bilateral_diff ≈ 0: food directly ahead (symmetric nostrils) ----
    # Fish at (400,300) heading=0 (east). Nares: L=(400,305), R=(400,295).
    # Food at (450,300): equal distance from both nares.
    olf2 = SpikingOlfaction(device=DEVICE)
    out_b = olf2(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                 foods=[(450.0, 300.0)], conspecific_injured=False, pred_dist=999.0)
    bdiff_b = abs(out_b['bilateral_diff'])
    bdiff_sym_ok = bdiff_b < 0.05
    results['OLF_bilateral_diff_symmetric'] = (
        'PASS' if bdiff_sym_ok else 'FAIL',
        f'{out_b["bilateral_diff"]:.4f}', 'abs < 0.05 (food symmetric ahead)')
    print(f"  OLF_bilateral_diff_symmetric: {'PASS' if bdiff_sym_ok else 'FAIL'} "
          f"(bilateral_diff={out_b['bilateral_diff']:.4f})")

    # ---- 5c. alarm_level > 0 with pred_dist=10 ----
    olf3 = SpikingOlfaction(device=DEVICE)
    out_c = olf3(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                 foods=[], conspecific_injured=False, pred_dist=10.0)
    alarm_c = out_c['alarm_level']
    alarm_c_ok = alarm_c > 0.0
    results['OLF_alarm_pred_dist10'] = (
        'PASS' if alarm_c_ok else 'FAIL',
        f'{alarm_c:.4f}', '> 0 (pred_dist=10, within 50px)')
    print(f"  OLF_alarm_pred_dist10: {'PASS' if alarm_c_ok else 'FAIL'} "
          f"(alarm_level={alarm_c:.4f}, > 0)")

    # ---- 5d. alarm_level > 0 with conspecific_injured=True ----
    olf4 = SpikingOlfaction(device=DEVICE)
    out_d = olf4(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                 foods=[], conspecific_injured=True, pred_dist=50.0)
    alarm_d = out_d['alarm_level']
    alarm_d_ok = alarm_d > 0.0
    results['OLF_alarm_injured'] = (
        'PASS' if alarm_d_ok else 'FAIL',
        f'{alarm_d:.4f}', '> 0 (conspecific_injured=True)')
    print(f"  OLF_alarm_injured: {'PASS' if alarm_d_ok else 'FAIL'} "
          f"(alarm_level={alarm_d:.4f}, > 0)")

    # ---- 5e. alarm_level ≈ 0 at baseline (pred_dist=999, no injury) ----
    olf5 = SpikingOlfaction(device=DEVICE)
    out_e = olf5(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                 foods=[], conspecific_injured=False, pred_dist=999.0)
    alarm_e = out_e['alarm_level']
    alarm_baseline_ok = alarm_e < 0.1
    results['OLF_alarm_baseline'] = (
        'PASS' if alarm_baseline_ok else 'FAIL',
        f'{alarm_e:.4f}', '< 0.1 (no threat)')
    print(f"  OLF_alarm_baseline: {'PASS' if alarm_baseline_ok else 'FAIL'} "
          f"(alarm_level={alarm_e:.4f}, < 0.1)")

    # ---- 5f. Receptor adaptation > 0.8 after 20 repeated steps ----
    olf6 = SpikingOlfaction(device=DEVICE)
    adapt_vals = []
    food_strength_vals = []
    for _ in range(20):
        out_f = olf6(fish_x=400.0, fish_y=300.0, fish_heading=0.0,
                     foods=[(410.0, 300.0)], conspecific_injured=False, pred_dist=999.0)
        adapt_vals.append(out_f['receptor_adapt'])
        food_strength_vals.append(out_f['food_odor_strength'])
    adapt_final = adapt_vals[-1]
    adapt_ok = adapt_final > 0.8
    results['OLF_receptor_adapt'] = (
        'PASS' if adapt_ok else 'FAIL',
        f'{adapt_final:.4f}', '> 0.8 after 20 steps')
    print(f"  OLF_receptor_adapt: {'PASS' if adapt_ok else 'FAIL'} "
          f"(receptor_adapt={adapt_final:.4f}, > 0.8)")

    # ---- 5g. food_odor_strength decreasing with adaptation ----
    food_early = np.mean(food_strength_vals[:3])
    food_late  = np.mean(food_strength_vals[-3:])
    food_dec_ok = food_late < food_early
    results['OLF_food_strength_decreasing'] = (
        'PASS' if food_dec_ok else 'FAIL',
        f'early={food_early:.4f}, late={food_late:.4f}', 'late < early')
    print(f"  OLF_food_strength_decreasing: {'PASS' if food_dec_ok else 'FAIL'} "
          f"(early={food_early:.4f}, late={food_late:.4f})")

    # ---- Figure: 4 panels ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Test 5: SpikingOlfaction Stimulus-Response', fontsize=14, fontweight='bold')

    # Panel 1: bilateral_diff vs food bearing angle
    olf_angle = SpikingOlfaction(device=DEVICE)
    angles = np.linspace(-math.pi, math.pi, 20)
    bdiffs = []
    fish_head = 0.0  # heading east; nares at (400, 305) and (400, 295)
    for ang in angles:
        olf_angle.reset()
        # Place food 30px away at given angle relative to fish centre
        fx = 400.0 + 30.0 * math.cos(ang)
        fy = 300.0 + 30.0 * math.sin(ang)
        o = olf_angle(400.0, 300.0, fish_head, [(fx, fy)], False, 999.0)
        bdiffs.append(o['bilateral_diff'])
    axes[0, 0].plot(np.degrees(angles), bdiffs, 'b-o', markersize=3, linewidth=1.5)
    axes[0, 0].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[0, 0].set_title('Bilateral Diff vs Food Angle (heading=0)')
    axes[0, 0].set_xlabel('Food angle (degrees)')
    axes[0, 0].set_ylabel('C_R - C_L (normalised)')

    # Panel 2: alarm_level vs pred_dist
    olf_alarm = SpikingOlfaction(device=DEVICE)
    pred_dists = np.linspace(0, 100, 20)
    alarm_levels = []
    for pd in pred_dists:
        olf_alarm.reset()
        o = olf_alarm(400.0, 300.0, 0.0, [], False, float(pd))
        alarm_levels.append(o['alarm_level'])
    axes[0, 1].plot(pred_dists, alarm_levels, 'r-o', markersize=3, linewidth=1.5)
    axes[0, 1].set_title('Alarm Level vs Predator Distance')
    axes[0, 1].set_xlabel('pred_dist (px)')
    axes[0, 1].set_ylabel('alarm_level')
    axes[0, 1].axvline(50, color='gray', linestyle='--', linewidth=0.8, label='λ=50px')
    axes[0, 1].legend(fontsize=8)

    # Panel 3: receptor adaptation over 30 steps
    olf_adapt_plot = SpikingOlfaction(device=DEVICE)
    adapt_30, strength_30 = [], []
    for _ in range(30):
        o = olf_adapt_plot(400.0, 300.0, 0.0, [(410.0, 300.0)], False, 999.0)
        adapt_30.append(o['receptor_adapt'])
        strength_30.append(o['food_odor_strength'])
    axes[1, 0].plot(adapt_30, 'orange', linewidth=1.5, label='receptor_adapt')
    axes[1, 0].plot(strength_30, 'green', linewidth=1.5, label='food_odor_strength')
    axes[1, 0].axhline(0.8, color='orange', linestyle='--', linewidth=0.8, label='adapt threshold')
    axes[1, 0].set_title('Receptor Adaptation & Food Strength Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].legend(fontsize=8)

    # Panel 4: food gradient direction vs food bearing
    olf_grad = SpikingOlfaction(device=DEVICE)
    bearings = np.linspace(-math.pi / 2, math.pi / 2, 16)
    grad_dirs = []
    for brg in bearings:
        olf_grad.reset()
        fx = 400.0 + 40.0 * math.cos(brg)
        fy = 300.0 + 40.0 * math.sin(brg)
        o = olf_grad(400.0, 300.0, 0.0, [(fx, fy)], False, 999.0)
        grad_dirs.append(o['food_gradient_dir'])
    axes[1, 1].scatter(np.degrees(bearings), np.degrees(grad_dirs), c='blue', s=30)
    axes[1, 1].plot(np.degrees(bearings), np.degrees(bearings), 'r--',
                    linewidth=1, label='ideal (dir=bearing)')
    axes[1, 1].set_title('Food Gradient Direction vs Food Bearing')
    axes[1, 1].set_xlabel('Food bearing (deg)')
    axes[1, 1].set_ylabel('gradient_dir (deg)')
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'test5_olfaction.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR}/test5_olfaction.png")
    return all(results[k][0] == 'PASS' for k in results if k.startswith('OLF_'))


# ---------------------------------------------------------------------------
# Test 6: Integration smoke test
# ---------------------------------------------------------------------------

def test_integration():
    """Test 6: ZebrafishBrainV2 step() smoke test — no exceptions, goal transitions."""
    print("\n=== Test 6: Integration Smoke Test ===")

    from zebrav2.brain.brain_v2 import ZebrafishBrainV2

    class MockEnv:
        """Minimal mock environment satisfying brain_v2.step() attribute reads."""
        def __init__(self):
            self.brain_L = np.zeros(800, dtype=np.float32)
            self.brain_R = np.zeros(800, dtype=np.float32)
            # Add some food pixels to retinal input
            self.brain_L[50:55] = 0.8
            self.brain_R[50:55] = 0.8
            self.fish_x = 400.0
            self.fish_y = 300.0
            self.fish_heading = 0.0
            self.pred_x = 600.0
            self.pred_y = 300.0
            self._enemy_pixels_total = 5.0
            self.foods = [(420.0, 300.0), (380.0, 280.0)]
            self.all_fish = [
                {'x': 400.0, 'y': 300.0, 'alive': True, 'speed': 0.0, 'heading': 0.0},  # focal
                {'x': 450.0, 'y': 310.0, 'alive': True, 'speed': 1.0, 'heading': 0.1},
                {'x': 370.0, 'y': 290.0, 'alive': True, 'speed': 0.8, 'heading': 0.2},
            ]

    brain = ZebrafishBrainV2(device=DEVICE)
    brain.reset()
    env = MockEnv()

    obs = np.zeros(10, dtype=np.float32)  # dummy obs (not used by brain)
    goals_seen = set()
    exception_raised = False
    out = {'turn': 0.0, 'speed': 1.0, 'goal': 0}  # fallback if exception fires early

    try:
        for _ in range(5):
            out = brain.step(obs, env)
            goals_seen.add(brain.current_goal)
    except Exception as exc:
        exception_raised = True
        print(f"  EXCEPTION: {exc}")

    no_exc_ok = not exception_raised
    results['INT_no_exception'] = (
        'PASS' if no_exc_ok else 'FAIL',
        str(not exception_raised), 'no exception raised')
    print(f"  INT_no_exception: {'PASS' if no_exc_ok else 'FAIL'}")

    # goal is an integer (0-3)
    goal_ok = all(g in (0, 1, 2, 3) for g in goals_seen) if goals_seen else exception_raised
    results['INT_valid_goal'] = (
        'PASS' if goal_ok else 'FAIL',
        str(sorted(goals_seen)), 'goals in {0,1,2,3}')
    print(f"  INT_valid_goal: {'PASS' if goal_ok else 'FAIL'} (goals seen={sorted(goals_seen)})")

    # output dict has required keys
    required_keys = {'turn', 'speed', 'goal'}
    keys_ok = required_keys.issubset(set(out.keys()))
    results['INT_output_keys'] = (
        'PASS' if keys_ok else 'FAIL',
        str(sorted(out.keys())), f'{sorted(required_keys)} present')
    print(f"  INT_output_keys: {'PASS' if keys_ok else 'FAIL'}")

    return all(results[k][0] == 'PASS' for k in results if k.startswith('INT_'))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    """Print pass/fail table and save summary figure."""
    print("\n=== SUMMARY ===")

    n_pass  = sum(1 for v in results.values() if v[0] == 'PASS')
    n_total = len(results)

    fig, ax = plt.subplots(figsize=(13, max(6, n_total * 0.38 + 2)))
    fig.suptitle(
        f'v2 Sensory/Motor Module Tests: {n_pass}/{n_total} PASS '
        f'({100 * n_pass / max(1, n_total):.0f}%)',
        fontsize=14, fontweight='bold')

    col_labels = ['Test', 'Status', 'Value', 'Threshold']
    cell_colors = []
    table_data  = []

    for name, (status, value, threshold) in sorted(results.items()):
        table_data.append([name, status, str(value), str(threshold)])
        if status == 'PASS':
            cell_colors.append(['white', '#d4edda', 'white', 'white'])
        else:
            cell_colors.append(['white', '#f8d7da', 'white', 'white'])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors, loc='center',
                     colWidths=[0.22, 0.08, 0.42, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'summary_sensory_motor.png'), dpi=150)
    plt.close(fig)

    print(f"\n  Results: {n_pass}/{n_total} PASS ({100 * n_pass / max(1, n_total):.0f}%)")
    print(f"  Figures saved to: {PLOT_DIR}/")
    for name, (status, value, threshold) in sorted(results.items()):
        mark = 'OK' if status == 'PASS' else 'XX'
        print(f"  [{mark}] {name}: {value} (threshold: {threshold})")

    if n_pass < n_total:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    print(f"Device: {DEVICE}")
    print("Testing 5 biologically rewritten sensory/motor modules + integration")

    test_retina()
    test_lateral_line()
    test_cpg()
    test_reticulospinal()
    test_olfaction()
    test_integration()
    print_summary()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
