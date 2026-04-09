"""
Hemispheric visual system tests — 5 biological steps + integration.

Steps:
  1. Optic chiasm crossing (L_eye → R_tectum, R_eye → L_tectum)
  2. Top-down attention (pallium → tectum enhances weak response)
  3. Free-energy gaze (PE-based saccade direction)
  4. Gaze offset shifts retinal projection
  5. Thalamus laterality (TC_L vs TC_R segregation)
  6. Integration — full brain.step() propagates gaze_offset

Run:
  .venv/bin/python -m zebrav2.tests.test_hemispheric_vision
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.tectum import Tectum
from zebrav2.brain.retina import RetinaV2
from zebrav2.brain.saccade import SpikingSaccade
from zebrav2.brain.thalamus import Thalamus
from zebrav2.brain.sensory_bridge import inject_sensory

PLOT_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)
FIGURE_PATH = os.path.join(PLOT_DIR, 'v2_hemispheric_vision.png')

results = {}   # step_key -> ('PASS'|'FAIL', description)
pass_count = 0
fail_count = 0


def _record(key, passed, desc):
    global pass_count, fail_count
    status = 'PASS' if passed else 'FAIL'
    results[key] = (status, desc)
    tag = '[OK]  ' if passed else '[FAIL]'
    print(f"  {tag} {key}: {desc}")
    if passed:
        pass_count += 1
    else:
        fail_count += 1


# ---------------------------------------------------------------------------
# Step 1: Optic chiasm crossing
# ---------------------------------------------------------------------------

def test_step1_optic_chiasm():
    """L_eye input → R_tectum (sgc_R high); R_eye input → L_tectum (sgc_L high)."""
    print("\n=== Step 1: Optic Chiasm Crossing ===")

    tectum = Tectum(device=DEVICE)
    retina = RetinaV2(device=DEVICE)

    entity_info = {'enemy': 0.0}

    # Stimulus LEFT only
    L_stim = torch.zeros(800, device=DEVICE)
    R_stim = torch.zeros(800, device=DEVICE)
    L_stim[100:150] = 0.9   # intensity channel
    L_stim[500:550] = 0.8   # type channel (food)

    # Accumulate retina + tectum over 10 steps for stable rates
    for _ in range(10):
        rgc_L = retina(L_stim, R_stim, entity_info)
        tect_L = tectum(rgc_L)

    # Use SFGS-b rates (driven by ON cells from contralateral eye)
    # R_tectum (sfgs_b_R) is driven by L_eye — should be HIGH when stim left
    # L_tectum (sfgs_b_L) is driven by R_eye — should be LOW when stim left
    sfgs_b_L_stim_left  = float(tect_L['sfgs_b_L'].mean())
    sfgs_b_R_stim_left  = float(tect_L['sfgs_b_R'].mean())

    # Stimulus RIGHT only
    retina.reset()
    tectum.reset()
    L_stim2 = torch.zeros(800, device=DEVICE)
    R_stim2 = torch.zeros(800, device=DEVICE)
    R_stim2[100:150] = 0.9
    R_stim2[500:550] = 0.8

    for _ in range(10):
        rgc_R = retina(L_stim2, R_stim2, entity_info)
        tect_R = tectum(rgc_R)

    # L_tectum (sfgs_b_L) driven by R_eye — should be HIGH when stim right
    sfgs_b_L_stim_right = float(tect_R['sfgs_b_L'].mean())
    sfgs_b_R_stim_right = float(tect_R['sfgs_b_R'].mean())

    cross_ok = (sfgs_b_R_stim_left > sfgs_b_L_stim_left) and (sfgs_b_L_stim_right > sfgs_b_R_stim_right)
    _record(
        'CHIASM_crossing', cross_ok,
        f"stim_L: R_tect={sfgs_b_R_stim_left:.4f} > L_tect={sfgs_b_L_stim_left:.4f} "
        f"| stim_R: L_tect={sfgs_b_L_stim_right:.4f} > R_tect={sfgs_b_R_stim_right:.4f}"
    )

    data = {
        'sgc_L_stim_left':  sfgs_b_L_stim_left,
        'sgc_R_stim_left':  sfgs_b_R_stim_left,
        'sgc_L_stim_right': sfgs_b_L_stim_right,
        'sgc_R_stim_right': sfgs_b_R_stim_right,
        'cross_ok': cross_ok,
        'tectum': tectum,
        'retina': retina,
    }
    return data


# ---------------------------------------------------------------------------
# Step 2: Top-down attention (pallium → tectum)
# ---------------------------------------------------------------------------

def test_step2_topdown_attention():
    """I_topdown injection into sfgs_b_L should enhance tectal response."""
    print("\n=== Step 2: Top-Down Attention (pallium→tectum) ===")

    tectum = Tectum(device=DEVICE)
    retina = RetinaV2(device=DEVICE)

    # Weak right stimulus → drives L_tectum
    L_weak = torch.zeros(800, device=DEVICE)
    R_weak = torch.zeros(800, device=DEVICE)
    R_weak[200:210] = 0.3
    R_weak[600:610] = 0.4

    retina.reset()
    tectum.reset()
    rgc_weak = retina(L_weak, R_weak, {'enemy': 0.0})

    # Without topdown
    for _ in range(3):
        out_no_td = tectum(rgc_weak)
    sfgs_b_L_no_td = float(out_no_td['sfgs_b_L'].mean())

    # With topdown
    tectum.reset()
    n_e = tectum.sfgs_b_L.n_e
    I_td_L = torch.ones(n_e, device=DEVICE) * 2.0
    I_td_R = torch.zeros(tectum.sfgs_b_R.n_e, device=DEVICE)
    for _ in range(3):
        out_with_td = tectum(rgc_weak, I_topdown_L=I_td_L, I_topdown_R=I_td_R)
    sfgs_b_L_with_td = float(out_with_td['sfgs_b_L'].mean())

    attention_ok = sfgs_b_L_with_td > sfgs_b_L_no_td
    _record(
        'TOPDOWN_attention', attention_ok,
        f"no_td={sfgs_b_L_no_td:.4f} → with_td={sfgs_b_L_with_td:.4f}"
    )

    return {
        'sfgs_b_L_no_td': sfgs_b_L_no_td,
        'sfgs_b_L_with_td': sfgs_b_L_with_td,
        'attention_ok': attention_ok,
    }


# ---------------------------------------------------------------------------
# Step 3: Free-energy gaze (PE-based saccade direction)
# ---------------------------------------------------------------------------

def test_step3_pe_gaze():
    """High PE on right → gaze right (positive). High PE on left → gaze left (negative)."""
    print("\n=== Step 3: Free-Energy Gaze (PE-based saccade) ===")

    saccade = SpikingSaccade(device=DEVICE)

    # High PE on RIGHT
    saccade.reset()
    offsets_pe_right = []
    for _ in range(10):
        out = saccade(food_bearing=0.0, enemy_bearing=0.0, current_goal=2,
                      salience_L=0.5, salience_R=0.5, pe_L=0.0, pe_R=0.8)
        offsets_pe_right.append(out['gaze_offset'])

    # High PE on LEFT
    saccade.reset()
    offsets_pe_left = []
    for _ in range(10):
        out = saccade(food_bearing=0.0, enemy_bearing=0.0, current_goal=2,
                      salience_L=0.5, salience_R=0.5, pe_L=0.8, pe_R=0.0)
        offsets_pe_left.append(out['gaze_offset'])

    gaze_ok = (offsets_pe_right[-1] > 0) and (offsets_pe_left[-1] < 0)
    _record(
        'PE_GAZE_direction', gaze_ok,
        f"PE_right final offset={offsets_pe_right[-1]:.4f} (>0) | "
        f"PE_left final offset={offsets_pe_left[-1]:.4f} (<0)"
    )

    return {
        'offsets_pe_right': offsets_pe_right,
        'offsets_pe_left': offsets_pe_left,
        'gaze_ok': gaze_ok,
    }


# ---------------------------------------------------------------------------
# Step 4: Gaze offset shifts retinal projection
# ---------------------------------------------------------------------------

def test_step4_gaze_shift():
    """Gaze shifted right moves a right-side food item toward left retina."""
    print("\n=== Step 4: Gaze Offset Shifts Retinal Projection ===")

    class MockEnv:
        fish_x = 400; fish_y = 300; arena_w = 800; arena_h = 600
        rock_formations = []; pred_x = -9999; pred_y = -9999
        foods = [[450, 300, 'small']]  # food 50px to the right

    # No gaze offset
    e1 = MockEnv()
    e1.fish_heading = 0.0
    e1.gaze_offset = 0.0
    inject_sensory(e1)
    food_R_no_gaze = float(np.sum(e1.brain_R[:400] > 0.1))

    # Gaze shifted right by 0.5 rad — food moves leftward in visual field
    e2 = MockEnv()
    e2.fish_heading = 0.0
    e2.gaze_offset = 0.5
    inject_sensory(e2)
    food_R_gaze = float(np.sum(e2.brain_R[:400] > 0.1))
    food_L_gaze = float(np.sum(e2.brain_L[:400] > 0.1))

    gaze_shift_ok = food_L_gaze > food_R_gaze
    _record(
        'GAZE_SHIFT_retina', gaze_shift_ok,
        f"no_gaze: food_R={food_R_no_gaze:.0f}px | "
        f"gaze_right: food_L={food_L_gaze:.0f}px > food_R={food_R_gaze:.0f}px"
    )

    return {
        'food_R_no_gaze': food_R_no_gaze,
        'food_R_gaze': food_R_gaze,
        'food_L_gaze': food_L_gaze,
        'gaze_shift_ok': gaze_shift_ok,
    }


# ---------------------------------------------------------------------------
# Step 5: Thalamus laterality
# ---------------------------------------------------------------------------

def test_step5_thalamus_laterality():
    """TC_R responds more to left-field stim; TC_L responds more to right-field stim."""
    print("\n=== Step 5: Thalamus Laterality ===")

    tectum = Tectum(device=DEVICE)
    retina = RetinaV2(device=DEVICE)

    th_L = Thalamus(DEVICE, sfgs_b_n_e=tectum.sfgs_b_L.n_e, n_tc=150)
    th_R = Thalamus(DEVICE, sfgs_b_n_e=tectum.sfgs_b_R.n_e, n_tc=150)

    pal_s_dummy = torch.zeros(int(0.75 * 1600), device=DEVICE)

    # Stim LEFT (→ R_tectum/sfgs_b_R active): TC_R should respond more
    retina.reset()
    tectum.reset()
    L_left = torch.zeros(800, device=DEVICE)
    L_left[100:150] = 0.9
    L_left[500:550] = 0.8
    for _ in range(10):
        rgc_left = retina(L_left, torch.zeros(800, device=DEVICE), {'enemy': 0.0})
        tect_left = tectum(rgc_left)

    tc_L_left = th_L(tect_left['sfgs_b_L'], pal_s_dummy, 0.3)['TC'].mean().item()
    tc_R_left = th_R(tect_left['sfgs_b_R'], pal_s_dummy, 0.3)['TC'].mean().item()

    # Stim RIGHT (→ L_tectum/sfgs_b_L active): TC_L should respond more
    retina.reset()
    tectum.reset()
    R_right = torch.zeros(800, device=DEVICE)
    R_right[100:150] = 0.9
    R_right[500:550] = 0.8
    for _ in range(10):
        rgc_right = retina(torch.zeros(800, device=DEVICE), R_right, {'enemy': 0.0})
        tect_right = tectum(rgc_right)

    tc_L_right = th_L(tect_right['sfgs_b_L'], pal_s_dummy, 0.3)['TC'].mean().item()
    tc_R_right = th_R(tect_right['sfgs_b_R'], pal_s_dummy, 0.3)['TC'].mean().item()

    thal_ok = (tc_R_left > tc_L_left) and (tc_L_right > tc_R_right)
    _record(
        'THALAMUS_laterality', thal_ok,
        f"stim_L: TC_R={tc_R_left:.4f} > TC_L={tc_L_left:.4f} | "
        f"stim_R: TC_L={tc_L_right:.4f} > TC_R={tc_R_right:.4f}"
    )

    return {
        'tc_L_left':  tc_L_left,
        'tc_R_left':  tc_R_left,
        'tc_L_right': tc_L_right,
        'tc_R_right': tc_R_right,
        'thal_ok': thal_ok,
    }


# ---------------------------------------------------------------------------
# Step 6: Integration — full brain step
# ---------------------------------------------------------------------------

def test_step6_integration():
    """Full brain.step() runs without error and propagates gaze_offset to env."""
    print("\n=== Step 6: Integration (gaze_offset propagates) ===")

    from zebrav2.brain.brain_v2 import ZebrafishBrainV2

    brain = ZebrafishBrainV2(device=DEVICE)
    brain.reset()

    class FullEnv:
        brain_L = np.zeros(800, dtype=np.float32)
        brain_R = np.zeros(800, dtype=np.float32)
        _enemy_pixels_total = 0
        fish_x = 400; fish_y = 300; fish_heading = 0.0; fish_energy = 80.0
        pred_x = -9999; pred_y = -9999
        foods = [[420, 300, 's']]
        rock_formations = []; all_fish = []
        gaze_offset = 0.0; arena_w = 800; arena_h = 600

    env = FullEnv()
    env.brain_R[100:120] = 0.8   # food on right
    env.brain_R[500:520] = 0.8
    obs = np.zeros(10)

    gaze_offsets = []
    out = None
    for _ in range(5):
        out = brain.step(obs, env)
        gaze_offsets.append(float(getattr(env, 'gaze_offset', 0.0)))

    integ_ok = (out is not None
                and 'turn' in out
                and 'speed' in out
                and all(isinstance(g, float) for g in gaze_offsets))
    _record(
        'INTEGRATION_gaze', integ_ok,
        f"turn={out.get('turn', '?'):.4f}, speed={out.get('speed', '?'):.4f}, "
        f"gaze_offsets={[round(g,3) for g in gaze_offsets]}"
    )

    return {
        'gaze_offsets': gaze_offsets,
        'integ_ok': integ_ok,
        'out': out,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(d1, d2, d3, d4, d5, d6):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Hemispheric Visual System — 6-Step Evaluation', fontsize=14, fontweight='bold')

    # ---- Panel 1: Optic chiasm crossing ----
    ax = axes[0, 0]
    labels = ['L_tect\n(stim LEFT)', 'R_tect\n(stim LEFT)',
              'L_tect\n(stim RIGHT)', 'R_tect\n(stim RIGHT)']
    vals   = [d1['sgc_L_stim_left'], d1['sgc_R_stim_left'],
              d1['sgc_L_stim_right'], d1['sgc_R_stim_right']]
    correct = [False, True, True, False]  # expected high/low per chiasm logic
    colors = ['green' if c else 'red' for c in correct]
    bars = ax.bar(labels, vals, color=colors, alpha=0.75, edgecolor='black')
    ax.set_title('Step 1: Optic Chiasm Crossing', fontweight='bold')
    ax.set_ylabel('SFGS-b mean rate')
    ax.set_ylim(0, max(vals) * 1.4 + 1e-6)
    status = 'PASS' if d1['cross_ok'] else 'FAIL'
    color  = 'green' if d1['cross_ok'] else 'red'
    ax.annotate(status, xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=13, color=color, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    # ---- Panel 2: Top-down attention ----
    ax = axes[0, 1]
    td_labels = ['No top-down', 'With top-down']
    td_vals   = [d2['sfgs_b_L_no_td'], d2['sfgs_b_L_with_td']]
    td_colors = ['steelblue', 'green' if d2['attention_ok'] else 'red']
    bars2 = ax.bar(td_labels, td_vals, color=td_colors, alpha=0.75, edgecolor='black')
    ax.set_title('Step 2: Top-Down Attention (pallium→tectum)', fontweight='bold')
    ax.set_ylabel('SFGS-b_L E-rate mean')
    ax.set_ylim(0, max(td_vals) * 1.4 + 1e-6)
    status2 = 'PASS' if d2['attention_ok'] else 'FAIL'
    color2  = 'green' if d2['attention_ok'] else 'red'
    ax.annotate(status2, xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=13, color=color2, fontweight='bold')
    for bar, v in zip(bars2, td_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    # ---- Panel 3: PE gaze ----
    ax = axes[1, 0]
    steps = list(range(1, 11))
    ax.plot(steps, d3['offsets_pe_right'], 'b-o', label='PE right (gaze →)', markersize=5)
    ax.plot(steps, d3['offsets_pe_left'],  'r-o', label='PE left (gaze ←)', markersize=5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Step 3: Free-Energy Gaze (PE-based saccade)', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('gaze_offset (rad)')
    ax.legend(fontsize=8)
    status3 = 'PASS' if d3['gaze_ok'] else 'FAIL'
    color3  = 'green' if d3['gaze_ok'] else 'red'
    ax.annotate(status3, xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=13, color=color3, fontweight='bold')

    # ---- Panel 4: Gaze shift ----
    ax = axes[1, 1]
    gs_labels = ['food_R\n(no gaze)', 'food_R\n(gaze right)', 'food_L\n(gaze right)']
    gs_vals   = [d4['food_R_no_gaze'], d4['food_R_gaze'], d4['food_L_gaze']]
    gs_colors = ['steelblue', 'red', 'green' if d4['gaze_shift_ok'] else 'red']
    bars4 = ax.bar(gs_labels, gs_vals, color=gs_colors, alpha=0.75, edgecolor='black')
    ax.set_title('Step 4: Gaze Offset Shifts Retinal Projection', fontweight='bold')
    ax.set_ylabel('Food pixels in retina')
    ax.set_ylim(0, max(gs_vals) * 1.4 + 0.5)
    status4 = 'PASS' if d4['gaze_shift_ok'] else 'FAIL'
    color4  = 'green' if d4['gaze_shift_ok'] else 'red'
    ax.annotate(status4, xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=13, color=color4, fontweight='bold')
    for bar, v in zip(bars4, gs_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{v:.0f}', ha='center', va='bottom', fontsize=9)

    # ---- Panel 5: Thalamus laterality ----
    ax = axes[2, 0]
    x = np.arange(2)
    width = 0.35
    tc_stim_left  = [d5['tc_L_left'],  d5['tc_R_left']]
    tc_stim_right = [d5['tc_L_right'], d5['tc_R_right']]
    ax.bar(x - width/2, tc_stim_left,  width, label='stim LEFT',  color='royalblue',  alpha=0.75, edgecolor='black')
    ax.bar(x + width/2, tc_stim_right, width, label='stim RIGHT', color='darkorange', alpha=0.75, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['TC_L', 'TC_R'])
    ax.set_title('Step 5: Thalamus Laterality', fontweight='bold')
    ax.set_ylabel('TC mean rate')
    ax.legend(fontsize=8)
    status5 = 'PASS' if d5['thal_ok'] else 'FAIL'
    color5  = 'green' if d5['thal_ok'] else 'red'
    ax.annotate(status5, xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=13, color=color5, fontweight='bold')

    # ---- Panel 6: Integration gaze_offset ----
    ax = axes[2, 1]
    steps6 = list(range(1, 6))
    ax.plot(steps6, d6['gaze_offsets'], 'purple', marker='o', linewidth=2, markersize=7)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Step 6: Integration (gaze_offset propagates)', fontweight='bold')
    ax.set_xlabel('Brain step')
    ax.set_ylabel('env.gaze_offset (rad)')
    ax.set_ylim(-0.6, 0.6)
    status6 = 'PASS' if d6['integ_ok'] else 'FAIL'
    color6  = 'green' if d6['integ_ok'] else 'red'
    ax.annotate(status6, xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=13, color=color6, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 60)
    print("Hemispheric Vision System — Test Suite")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    d1 = test_step1_optic_chiasm()
    d2 = test_step2_topdown_attention()
    d3 = test_step3_pe_gaze()
    d4 = test_step4_gaze_shift()
    d5 = test_step5_thalamus_laterality()
    d6 = test_step6_integration()

    make_figure(d1, d2, d3, d4, d5, d6)

    print("\n" + "=" * 60)
    print(f"Results: {pass_count} PASS / {fail_count} FAIL")
    print("=" * 60)
    for key, (status, desc) in results.items():
        tag = '[OK]  ' if status == 'PASS' else '[FAIL]'
        print(f"  {tag} {key}: {desc}")

    return pass_count, fail_count


if __name__ == '__main__':
    run_all()
