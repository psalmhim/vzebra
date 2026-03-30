"""
GPU benchmark and CUDA migration validation.

Tests v2 brain on available device (MPS/CUDA/CPU) and reports timing.
On CUDA: verifies all modules work without float64 errors.

Run: .venv/bin/python -u -m zebrav2.tools.gpu_benchmark
"""
import os, sys, time
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv


def benchmark(device, n_steps=50):
    """Run n_steps and return timing stats."""
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=n_steps)
    brain = ZebrafishBrainV2(device=device)
    obs, info = env.reset(seed=42)
    brain.reset()

    # Warmup (first step is slow due to JIT/compilation)
    inject_sensory(env)
    brain.step(obs, env)
    obs, _, _, _, info = env.step(np.array([0, 1], dtype=np.float32))
    env._eaten_now = 0

    times = []
    for t in range(n_steps):
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(brain.current_goal == 1, 0.0)
        inject_sensory(env)
        t0 = time.time()
        out = brain.step(obs, env)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        obs, _, term, trunc, info = env.step(
            np.array([out['turn'], out['speed']], dtype=np.float32))
        env._eaten_now = info.get('food_eaten_this_step', 0)
        if term or trunc:
            break
    env.close()
    del brain
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'device': str(device),
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'steps_per_sec': 1.0 / np.mean(times),
        'steps_run': len(times),
    }


def main():
    print("=" * 60)
    print("  ZebrafishBrainV2 GPU Benchmark")
    print("=" * 60)

    # Check available devices
    devices = []
    devices.append(('CPU', torch.device('cpu')))
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append(('MPS (Apple Silicon)', torch.device('mps')))
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append((f'CUDA:{i} ({name})', torch.device(f'cuda:{i}')))

    print(f"\n  Available devices: {len(devices)}")
    for name, dev in devices:
        print(f"    {name}: {dev}")
    print(f"  Current default: {DEVICE}")
    print()

    # Benchmark each device
    results = []
    for name, dev in devices:
        print(f"  Benchmarking {name}...", end=' ', flush=True)
        try:
            r = benchmark(dev, n_steps=20)
            results.append((name, r))
            print(f"{r['mean_ms']:.0f} ms/step ({r['steps_per_sec']:.1f} steps/s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((name, {'error': str(e)}))

    # Summary
    print("\n" + "=" * 60)
    print(f"  {'Device':<30s} {'ms/step':>10s} {'steps/s':>10s} {'speedup':>10s}")
    print("  " + "-" * 56)
    baseline = None
    for name, r in results:
        if 'error' in r:
            print(f"  {name:<30s} {'FAILED':>10s}")
            continue
        if baseline is None:
            baseline = r['mean_ms']
        speedup = baseline / r['mean_ms']
        print(f"  {name:<30s} {r['mean_ms']:>8.0f}ms {r['steps_per_sec']:>8.1f} {speedup:>8.1f}x")

    # CUDA migration notes
    print("\n  CUDA Migration Notes:")
    print("  - All modules auto-detect device via zebrav2/spec.py")
    print("  - No code changes needed: set CUDA_VISIBLE_DEVICES=0")
    print("  - Known MPS issues (fixed): float64, adaptive_avg_pool1d")
    print("  - For multi-GPU: set CUDA_VISIBLE_DEVICES to specific GPU")
    print("  - Expected CUDA speedup: 3-5x over MPS")


if __name__ == '__main__':
    main()
