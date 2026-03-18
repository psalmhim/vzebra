# PC-SNN Modules (v13.1b)

**Predictive-Coding Spiking Neural Network — Zebrafish Brain Simulation**

Each subsystem implements a biologically inspired predictive-coding loop.
All modules are interoperable and independently testable.

---

| Module | Function | Biological Analogue | Key Signals |
|---------|-----------|---------------------|--------------|
| `retina_pc.py` | Visual encoder / prediction error minimization | Retina + LGN | Free energy (Fv), precision (ΠL/R) |
| `audio_pc.py` | Predictive auditory pathway | Torus semicircularis / auditory thalamus | Free energy (Fa), precision (ΠA) |
| `cortex_memory.py` | Cortical feature integration and short-term memory | V1/V2 ↔ pretectum | Latent z, working memory trace |
| `dopamine_system.py` | Reward prediction error and valence | Dopaminergic nuclei (VTA/SNc) | RPE, valenceL/R |
| `basal_ganglia.py` | Motor gating / saccade alternation | Basal ganglia loops | Gate (BG), oscillatory state |
| `optic_tectum.py` | Gaze control and rebound dynamics | Tectum / superior colliculus | Eye position / velocity |
| `thalamus_relay.py` | Cross-modal novelty detection (CMS) | Thalamus / habenula relay | Cross-modal salience |
| `__init__.py` | Unified import interface | — | — |

---

### Version Notes
- **v13.1b (2025-11-12):**
  - First fully modular release.
  - All modules annotated with update headers.
  - Compatible with `pcsnn_v13_1b_alternation_fix.py`.
  - Ready for CMS-driven novelty experiments (alternating or exploratory mode).

---

### Integration Example
```python
from modules import (
    RetinaPC, AudioPC, VisualCortexPC, WorkingMemory,
    DopamineSystem, BasalGanglia, OpticTectum, ThalamusRelay
)

retinaL, retinaR = RetinaPC(), RetinaPC()
audio = AudioPC()
v1L, v1R = VisualCortexPC(), VisualCortexPC()
mem = WorkingMemory()
dopa = DopamineSystem()
bg = BasalGanglia(mode="novelty_driven")
tectum = OpticTectum()
thal = ThalamusRelay()

