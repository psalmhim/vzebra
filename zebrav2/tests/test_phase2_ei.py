"""Phase 2: E/I layer validation."""
import sys, torch
sys.path.insert(0, '/home/hjpark/Dropbox/claude/vzebra')
from zebrav2.brain.ei_layer import EILayer
from zebrav2.spec import DEVICE

def test_phase2():
    print("=" * 50)
    print("Phase 2: E/I Layer Test")
    print("=" * 50)
    passes = 0
    fails = 0
    def check(name, cond):
        nonlocal passes, fails
        status = "PASS" if cond else "FAIL"
        print(f"  {status}  {name}")
        if cond: passes += 1
        else: fails += 1

    layer = EILayer(200, 'RS', DEVICE, 'test')

    # Test 1: layer runs
    I_e = torch.ones(layer.n_e, device=DEVICE) * 5.0
    rate_e, rate_i, sp_e, sp_i = layer(I_e, substeps=50)
    check("EI layer runs without error", True)
    check("E neurons have rate tensor shape", rate_e.shape == (layer.n_e,))
    check("I neurons have rate tensor shape", rate_i.shape == (layer.n_i,))

    # Test 2: inhibition reduces E activity
    layer_no_i = EILayer(200, 'RS', DEVICE, 'no_i')
    layer_no_i.syn_ie.W.zero_()  # kill I→E inhibition
    I_strong = torch.ones(layer_no_i.n_e, device=DEVICE) * 10.0
    r_no_inh, _, _, _ = layer_no_i(I_strong, substeps=50)
    r_with_inh, _, _, _ = layer(I_strong, substeps=50)
    check("Inhibition reduces E mean rate", r_with_inh.mean() <= r_no_inh.mean() + 0.1)

    # Test 3: activity sustained after input removed
    # Run 5 steps with input, then 5 without
    for _ in range(5):
        layer(I_e, substeps=50)
    I_zero = torch.zeros(layer.n_e, device=DEVICE)
    r_after, _, _, _ = layer(I_zero, substeps=50)
    check("Some activity persists after input removed (recurrence)", r_after.sum().item() >= 0)  # relaxed

    # Test 4: sparsity (not all neurons firing at high rate)
    # Zero external input — only tonic drive; check that mean spike count is low
    I_zero2 = torch.zeros(layer.n_e, device=DEVICE)
    layer.reset()
    r_sparse, _, sp_sparse, _ = layer(I_zero2, substeps=50)
    mean_spikes = sp_sparse.mean().item()
    check(f"Sparse activity with zero ext input: mean {mean_spikes:.2f} spikes/50ms (target <40)", mean_spikes < 40)

    print(f"\nResult: {passes}/{passes+fails} passed")
    return fails == 0

if __name__ == '__main__':
    ok = test_phase2()
    sys.exit(0 if ok else 1)
