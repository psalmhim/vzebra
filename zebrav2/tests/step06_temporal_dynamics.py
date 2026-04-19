"""
Issue 6: Temporal dynamics tests — STDP weight changes, eligibility traces.

Verifies that:
  - Causal pre→post pairing causes LTP (weight increase)
  - Anti-causal post→pre pairing causes LTD (weight decrease)
  - DA as third factor scales the weight update magnitude
  - Eligibility trace decays to zero without spikes
  - Homeostatic scaling normalizes runaway weights

Run: .venv/bin/python -m zebrav2.tests.step06_temporal_dynamics
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from zebrav2.brain.plasticity import EligibilitySTDP
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


def make_stdp(n_post=10, n_pre=10):
    W = nn.Parameter(torch.zeros(n_post, n_pre, device=DEVICE))
    stdp = EligibilitySTDP(W, device=DEVICE, A_plus=0.01, A_minus=0.01)
    return stdp, W


def test_causal_ltp():
    """Pre fires 1ms before post → LTP (weight increases with positive DA)."""
    print('\n=== Causal Pairing → LTP ===')
    stdp, W = make_stdp()
    w_before = W.data.clone()

    pre = torch.zeros(10, device=DEVICE)
    post = torch.zeros(10, device=DEVICE)
    pre[0] = 1.0  # pre fires at t=0

    # Run 5ms: pre fires at step 0, post fires at step 2
    for t in range(5):
        pre_sp = pre if t == 0 else torch.zeros(10, device=DEVICE)
        post_sp = post.clone()
        if t == 2:
            post_sp[0] = 1.0  # post fires 2ms after pre
        stdp.update_traces(pre_sp, post_sp)

    stdp.consolidate(DA=1.0, ACh=1.0, eta=0.01)
    w_after = W.data.clone()
    delta = (w_after - w_before).sum().item()
    check('Causal pairing (pre→post) → positive weight delta', delta > 0,
          f'ΔW_sum={delta:.6f}')
    check('LTP at [0,0] synapse', W.data[0, 0].item() > 0,
          f'W[0,0]={W.data[0,0].item():.6f}')


def test_anticausal_ltd():
    """Post fires before pre → LTD (weight decreases with positive DA)."""
    print('\n=== Anti-Causal Pairing → LTD ===')
    stdp, W = make_stdp()

    pre = torch.zeros(10, device=DEVICE)
    post = torch.zeros(10, device=DEVICE)

    # Post fires at t=0, pre fires at t=3 → anti-causal
    for t in range(6):
        pre_sp = torch.zeros(10, device=DEVICE)
        post_sp = torch.zeros(10, device=DEVICE)
        if t == 0:
            post_sp[0] = 1.0
        if t == 3:
            pre_sp[0] = 1.0
        stdp.update_traces(pre_sp, post_sp)

    w_before = W.data.clone()
    stdp.consolidate(DA=1.0, ACh=1.0, eta=0.01)
    w_after = W.data.clone()
    delta = (w_after - w_before).sum().item()
    check('Anti-causal pairing (post→pre) → negative weight delta', delta < 0,
          f'ΔW_sum={delta:.6f}')


def test_da_scales_update():
    """DA=1.0 produces larger weight update than DA=0.1."""
    print('\n=== DA Scales Weight Update ===')

    def run_pairing(da_val):
        stdp, W = make_stdp()
        for t in range(4):
            pre_sp = torch.ones(10, device=DEVICE) if t == 0 else torch.zeros(10, device=DEVICE)
            post_sp = torch.ones(10, device=DEVICE) if t == 2 else torch.zeros(10, device=DEVICE)
            stdp.update_traces(pre_sp, post_sp)
        stdp.consolidate(DA=da_val, ACh=1.0, eta=0.01)
        return W.data.abs().sum().item()

    delta_high_da = run_pairing(1.0)
    delta_low_da = run_pairing(0.1)
    check('DA=1.0 → larger weight change than DA=0.1', delta_high_da > delta_low_da,
          f'high={delta_high_da:.6f}, low={delta_low_da:.6f}')


def test_eligibility_decays():
    """Without spikes, eligibility trace decays toward zero.
    Uses causal pairing (pre then post) to create a non-zero trace first."""
    print('\n=== Eligibility Trace Decay ===')
    stdp, W = make_stdp()

    # Causal pairing: pre fires at t=0, post fires at t=2 → LTP eligibility
    for t in range(5):
        pre_sp = torch.ones(10, device=DEVICE) if t == 0 else torch.zeros(10, device=DEVICE)
        post_sp = torch.ones(10, device=DEVICE) if t == 2 else torch.zeros(10, device=DEVICE)
        stdp.update_traces(pre_sp, post_sp)
    e_after_burst = stdp.e_trace.abs().sum().item()

    # 50 silent steps
    for _ in range(50):
        stdp.update_traces(torch.zeros(10, device=DEVICE), torch.zeros(10, device=DEVICE))
    e_after_silence = stdp.e_trace.abs().sum().item()

    # TAU_ELIG = 500ms, DT = 1ms → after 50ms, expected ratio ≈ exp(-50/500) ≈ 0.905
    import math as _math
    expected_ratio = _math.exp(-50 * 0.001 / 0.500)
    actual_ratio = e_after_silence / max(e_after_burst, 1e-9)
    check('Eligibility trace decays after burst', e_after_silence < e_after_burst,
          f'burst={e_after_burst:.6f}, silence={e_after_silence:.6f}')
    check(f'Trace decay matches tau_elig (ratio ≈ {expected_ratio:.3f} ± 0.05)',
          abs(actual_ratio - expected_ratio) < 0.05,
          f'actual_ratio={actual_ratio:.4f}')


def test_homeostatic_scaling():
    """Homeostatic scaling normalizes overactive postsynaptic neurons."""
    print('\n=== Homeostatic Scaling ===')
    stdp, W = make_stdp()

    # Initialize W to large values
    with torch.no_grad():
        W.data.fill_(1.5)

    w_before = W.data.mean().item()
    # 200 steps of high-rate post activity → homeostatic scale should reduce W
    for _ in range(200):
        post_sp = torch.ones(10, device=DEVICE)  # always spiking (high rate)
        stdp.homeostatic_scale(post_sp, alpha=1e-3)

    w_after = W.data.mean().item()
    check('Homeostatic scaling reduces overactive weights', w_after < w_before,
          f'{w_before:.4f} → {w_after:.4f}')
    check('Weights remain within [w_min, w_max]',
          W.data.min().item() >= stdp.w_min and W.data.max().item() <= stdp.w_max,
          f'min={W.data.min().item():.3f}, max={W.data.max().item():.3f}')


def test_ach_gates_plasticity():
    """ACh multiplier modulates consolidation — high ACh → more learning."""
    print('\n=== ACh Gates Plasticity ===')

    def run_ach(ach_val):
        stdp, W = make_stdp()
        for t in range(4):
            pre_sp = torch.ones(10, device=DEVICE) if t == 0 else torch.zeros(10, device=DEVICE)
            post_sp = torch.ones(10, device=DEVICE) if t == 2 else torch.zeros(10, device=DEVICE)
            stdp.update_traces(pre_sp, post_sp)
        stdp.consolidate(DA=0.8, ACh=ach_val, eta=0.01)
        return W.data.abs().sum().item()

    high = run_ach(2.0)
    low = run_ach(0.1)
    check('High ACh → larger weight update', high > low,
          f'high={high:.6f}, low={low:.6f}')


def test_stdp_reset():
    """reset() zeroes all traces."""
    print('\n=== STDP Reset ===')
    stdp, W = make_stdp()
    for _ in range(10):
        stdp.update_traces(torch.ones(10, device=DEVICE), torch.ones(10, device=DEVICE))
    stdp.reset()
    check('e_trace zeroed after reset', stdp.e_trace.abs().sum().item() < 1e-9,
          f'sum={stdp.e_trace.abs().sum().item():.2e}')
    check('pre_trace zeroed after reset', stdp.pre_trace.abs().sum().item() < 1e-9)
    check('post_trace zeroed after reset', stdp.post_trace.abs().sum().item() < 1e-9)


if __name__ == '__main__':
    print('=' * 60)
    print('Step 06: Temporal Dynamics (STDP) Tests')
    print('=' * 60)
    test_causal_ltp()
    test_anticausal_ltd()
    test_da_scales_update()
    test_eligibility_decays()
    test_homeostatic_scaling()
    test_ach_gates_plasticity()
    test_stdp_reset()
    print(f'\nResult: {passes}/{passes+fails} passed')
    import sys
    sys.exit(0 if fails == 0 else 1)
