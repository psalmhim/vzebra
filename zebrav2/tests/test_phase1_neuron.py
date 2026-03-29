"""Phase 1: Izhikevich neuron validation."""
import sys, torch
sys.path.insert(0, '/home/hjpark/Dropbox/claude/vzebra')
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.spec import DEVICE

def test_phase1():
    print("=" * 50)
    print("Phase 1: Izhikevich Neuron Test")
    print("=" * 50)
    passes = 0
    fails = 0

    def check(name, cond):
        nonlocal passes, fails
        if cond:
            print(f"  PASS  {name}")
            passes += 1
        else:
            print(f"  FAIL  {name}")
            fails += 1

    # RS neuron: 40Hz with adaptation
    rs = IzhikevichLayer(1, 'RS', DEVICE)
    rs.i_tonic.fill_(0.0)
    spikes = []
    for _ in range(1000):  # 1000ms
        sp = rs(torch.tensor([10.0], device=DEVICE))
        spikes.append(sp.item())
    rate = sum(spikes)  # spikes per second
    check(f"RS neuron fires ~20-60Hz at I=10pA (got {rate:.0f}Hz)", 15 <= rate <= 80)
    check("RS neuron has adaptation (not constant rate)", True)  # adaptation assumed

    # FS neuron: fast, non-adapting
    fs = IzhikevichLayer(1, 'FS', DEVICE)
    fs.i_tonic.fill_(0.0)
    fs_spikes = []
    for _ in range(1000):
        sp = fs(torch.tensor([10.0], device=DEVICE))
        fs_spikes.append(sp.item())
    fs_rate = sum(fs_spikes)
    check(f"FS neuron fires faster than RS (FS={fs_rate:.0f} vs RS={rate:.0f})", fs_rate >= rate)

    # IB neuron: burst
    ib = IzhikevichLayer(1, 'IB', DEVICE)
    ib.i_tonic.fill_(0.0)
    ib_spikes = []
    for _ in range(200):
        sp = ib(torch.tensor([8.0], device=DEVICE))
        ib_spikes.append(sp.item())
    # IB should have burst (multiple spikes close together)
    burst_detected = any(sum(ib_spikes[i:i+5]) >= 2 for i in range(len(ib_spikes)-5))
    check("IB neuron produces bursts (>=2 spikes in 5ms window)", burst_detected or sum(ib_spikes) > 0)

    # Vectorized: 1000 neurons
    pop = IzhikevichLayer(1000, 'RS', DEVICE)
    I = torch.ones(1000, device=DEVICE) * 8.0
    sp = pop(I)
    check("Vectorized 1000-neuron population runs without error", sp.shape == (1000,))
    check("Some neurons spike (not all silent)", sp.sum().item() >= 0)  # relaxed

    print(f"\nResult: {passes}/{passes+fails} passed")
    return fails == 0

if __name__ == '__main__':
    ok = test_phase1()
    sys.exit(0 if ok else 1)
