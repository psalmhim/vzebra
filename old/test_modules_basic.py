# ============================================================
# SCRIPT: test_modules_basic.py
# AUTHOR: HJ Park & GPT-5 (PC-SNN Project)
# VERSION: v13.1b (2025-11-12)
#
# PURPOSE:
#     Quick regression / smoke test for all PC-SNN modules.
#     Ensures each subsystem executes one forward step and returns finite values.
#
# REASON FOR UPDATE:
#     Added after modularization (v13.1b) to confirm interoperability
#     and correctness of independent components before full simulation runs.
# ============================================================

import torch
import numpy as np
from modules import (
    RetinaPC, AudioPC, VisualCortexPC, WorkingMemory,
    DopamineSystem, BasalGanglia, OpticTectum, ThalamusRelay
)

def finite(x):
    if isinstance(x, torch.Tensor):
        return torch.isfinite(x).all().item()
    return np.isfinite(x).all()

def test_retina(device):
    retina = RetinaPC(device=device)
    x = torch.rand(1, 784, device=device)
    pred, F, Pi = retina.step(x)
    print(f"[RetinaPC] F={F:.4f}, Pi={Pi:.3f}, finite={finite(pred)}")

def test_audio(device):
    audio = AudioPC(device=device)
    tone = torch.rand(1, 16, device=device)
    pred, F, Pi = audio.step(tone)
    print(f"[AudioPC] F={F:.4f}, Pi={Pi:.3f}, finite={finite(pred)}")

def test_cortex_memory(device):
    v1L = VisualCortexPC(device=device)
    v1R = VisualCortexPC(device=device)
    rL = torch.rand(1, 64, device=device)
    rR = torch.rand(1, 64, device=device)
    zL = v1L(rL, rR)
    mem = WorkingMemory(device=device)
    m = mem.step(zL)
    print(f"[Cortex+Memory] zL.mean={zL.mean():.4f}, mem.mean={m.mean():.4f}")

def test_dopamine():
    dopa = DopamineSystem()
    valL, valR, rpe = dopa.update(0.1, 0.2, 0.3, 0.1, cms=0.05)
    print(f"[Dopamine] vL={valL:+.3f}, vR={valR:+.3f}, RPE={rpe:+.3f}")

def test_basalganglia():
    bg = BasalGanglia(mode="alternating")
    out = bg.step(valL=-0.3, valR=0.4, dopa=0.5, rpe=0.2, cms=0.1)
    print(f"[BasalGanglia] output={out:+.3f}")

def test_tectum():
    tectum = OpticTectum()
    eye = tectum.step(valL=-0.2, valR=0.3, Fmean=0.05, bg_gate=0.5, dopa=0.6)
    print(f"[OpticTectum] eye_pos={eye:+.3f}")

def test_thalamus():
    thal = ThalamusRelay()
    cms = thal.step(Fv=0.02, Fa=0.03)
    print(f"[ThalamusRelay] CMS={cms:+.3f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing modules on device: {device}\n" + "-" * 60)
    test_retina(device)
    test_audio(device)
    test_cortex_memory(device)
    test_dopamine()
    test_basalganglia()
    test_tectum()
    test_thalamus()
    print("-" * 60 + "\nAll module tests completed.\n")

if __name__ == "__main__":
    main()

