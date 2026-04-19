"""
Issue 7: Neuromodulator cross-talk tests.

Verifies that each neuromodulatory axis responds correctly to its
biological trigger and that downstream gates/gains are affected.

Run: .venv/bin/python -m zebrav2.tests.step05_neuromod_crosstalk
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.brain.neuromod import NeuromodSystem
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


def test_da_reward():
    """DA rises on positive reward, falls on negative RPE."""
    print('\n=== DA: Reward Prediction Error ===')
    nm = NeuromodSystem()

    # Baseline
    out0 = nm.update(reward=0.5, amygdala_alpha=0.0, cms=0.0,
                     flee_active=False, fatigue=0.0, circadian=0.8, current_goal=0)
    da_base = out0['DA']

    # Strong positive reward → high DA (RPE positive)
    nm2 = NeuromodSystem()
    for _ in range(5):
        out_pos = nm2.update(reward=1.0, amygdala_alpha=0.0, cms=0.0,
                             flee_active=False, fatigue=0.0, circadian=0.8, current_goal=0)
    check('Positive reward → DA > 0.5', out_pos['DA'] > 0.5,
          f'DA={out_pos["DA"]:.3f}')

    # After learning value, reward=0 → negative RPE → low DA
    nm3 = NeuromodSystem()
    for _ in range(20):
        nm3.update(reward=1.0, amygdala_alpha=0.0, cms=0.0,
                   flee_active=False, fatigue=0.0, circadian=0.8, current_goal=0)
    out_neg = nm3.update(reward=0.0, amygdala_alpha=0.0, cms=0.0,
                         flee_active=False, fatigue=0.0, circadian=0.8, current_goal=0)
    check('Reward omission after training → DA < 0.5', out_neg['DA'] < 0.5,
          f'DA={out_neg["DA"]:.3f}')


def test_ht5_flee():
    """5-HT falls when flee fraction is high; rises during calm."""
    print('\n=== 5-HT: Flee Suppression ===')

    # Calm fish — 5-HT should stay high
    nm_calm = NeuromodSystem()
    for _ in range(30):
        nm_calm.update(reward=0.0, amygdala_alpha=0.1, cms=0.0,
                       flee_active=False, fatigue=0.1, circadian=0.8, current_goal=0)
    ht5_calm = nm_calm.HT5.item()
    check('Calm (no flee) → 5-HT stays ≥ 0.5', ht5_calm >= 0.45,
          f'5HT={ht5_calm:.3f}')

    # Fleeing fish — 5-HT should fall
    nm_flee = NeuromodSystem()
    for _ in range(40):
        nm_flee.update(reward=0.0, amygdala_alpha=0.9, cms=0.5,
                       flee_active=True, fatigue=0.4, circadian=0.8, current_goal=1)
    ht5_flee = nm_flee.HT5.item()
    check('Sustained flee → 5-HT < 0.5', ht5_flee < 0.5,
          f'5HT={ht5_flee:.3f}')

    check('5-HT calm > 5-HT flee', ht5_calm > ht5_flee,
          f'{ht5_calm:.3f} > {ht5_flee:.3f}')

    # flee_efe_bias: calmer fish have higher bias
    nm_x = NeuromodSystem()
    for _ in range(30):
        nm_x.update(reward=0.0, amygdala_alpha=0.1, cms=0.0,
                    flee_active=False, fatigue=0.0, circadian=0.8, current_goal=0)
    bias_calm = nm_x.get_flee_efe_bias()
    nm_y = NeuromodSystem()
    for _ in range(30):
        nm_y.update(reward=0.0, amygdala_alpha=0.9, cms=0.5,
                    flee_active=True, fatigue=0.4, circadian=0.8, current_goal=1)
    bias_flee = nm_y.get_flee_efe_bias()
    check('Calm → flee_efe_bias > flee state', bias_calm > bias_flee,
          f'{bias_calm:.3f} > {bias_flee:.3f}')


def test_na_arousal():
    """NA rises with amygdala activity (threat) and cms."""
    print('\n=== NA: Threat / Arousal ===')

    nm_low = NeuromodSystem()
    for _ in range(20):
        nm_low.update(reward=0.0, amygdala_alpha=0.0, cms=0.0,
                      flee_active=False, fatigue=0.0, circadian=0.8, current_goal=0)

    nm_high = NeuromodSystem()
    for _ in range(20):
        nm_high.update(reward=0.0, amygdala_alpha=1.0, cms=1.0,
                       flee_active=True, fatigue=0.5, circadian=0.8, current_goal=1)

    na_low = nm_low.NA.item()
    na_high = nm_high.NA.item()
    check('Low threat → NA < 0.6', na_low < 0.6, f'NA={na_low:.3f}')
    check('High threat → NA > 0.6', na_high > 0.6, f'NA={na_high:.3f}')
    check('High threat NA > low threat NA', na_high > na_low,
          f'{na_high:.3f} > {na_low:.3f}')

    # get_gain_all: higher NA → higher gain
    gain_low = nm_low.get_gain_all()
    gain_high = nm_high.get_gain_all()
    check('High NA → higher global gain', gain_high > gain_low,
          f'{gain_high:.3f} > {gain_low:.3f}')


def test_ach_attention():
    """ACh rises with high circadian + low fatigue; ACh gates plasticity."""
    print('\n=== ACh: Attention / Plasticity Gate ===')

    # High circadian (daytime), low fatigue → high ACh
    nm_alert = NeuromodSystem()
    for _ in range(30):
        nm_alert.update(reward=0.0, amygdala_alpha=0.3, cms=0.2,
                        flee_active=False, fatigue=0.0, circadian=1.0, current_goal=0)

    # Night / fatigued → low ACh
    nm_tired = NeuromodSystem()
    for _ in range(30):
        nm_tired.update(reward=0.0, amygdala_alpha=0.1, cms=0.0,
                        flee_active=False, fatigue=0.9, circadian=0.1, current_goal=0)

    ach_alert = nm_alert.ACh.item()
    ach_tired = nm_tired.ACh.item()
    check('Alert (high circ, low fatigue) → ACh > 0.4', ach_alert > 0.4,
          f'ACh={ach_alert:.3f}')
    check('Tired (low circ, high fatigue) → ACh < 0.4', ach_tired < 0.4,
          f'ACh={ach_tired:.3f}')
    check('Alert ACh > tired ACh', ach_alert > ach_tired,
          f'{ach_alert:.3f} > {ach_tired:.3f}')

    # Plasticity gate scales with ACh
    gate_alert = nm_alert.get_plasticity_gate()
    gate_tired = nm_tired.get_plasticity_gate()
    check('Alert → higher plasticity gate', gate_alert > gate_tired,
          f'{gate_alert:.3f} > {gate_tired:.3f}')


def test_reset_clears_state():
    """reset() returns all axes to baseline."""
    print('\n=== Reset Clears State ===')
    nm = NeuromodSystem()
    for _ in range(50):
        nm.update(reward=1.0, amygdala_alpha=1.0, cms=1.0,
                  flee_active=True, fatigue=0.5, circadian=0.8, current_goal=1)
    nm.reset()
    check('DA reset to 0.5', abs(nm.DA.item() - 0.5) < 1e-5, f'DA={nm.DA.item():.5f}')
    check('NA reset to 0.3', abs(nm.NA.item() - 0.3) < 1e-5, f'NA={nm.NA.item():.5f}')
    check('5-HT reset to 0.5', abs(nm.HT5.item() - 0.5) < 1e-5, f'5HT={nm.HT5.item():.5f}')
    check('ACh reset to 0.5', abs(nm.ACh.item() - 0.5) < 1e-5, f'ACh={nm.ACh.item():.5f}')


if __name__ == '__main__':
    print('=' * 60)
    print('Step 05: Neuromodulator Cross-Talk Tests')
    print('=' * 60)
    test_da_reward()
    test_ht5_flee()
    test_na_arousal()
    test_ach_attention()
    test_reset_clears_state()
    print(f'\nResult: {passes}/{passes+fails} passed')
    import sys
    sys.exit(0 if fails == 0 else 1)
