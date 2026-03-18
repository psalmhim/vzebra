# ============================================================
# PC-SNN v1 → v30 Test Environment Suite
# Author: H.J. Park & GPT-5
# (2025-11-22)
#
# Run this file to test every subsystem of the multimodal
# predictive-coding active-inference agent, WITHOUT constructing
# an agent class.
# ============================================================

import torch

# Import all modules
from modules.retina_pc import RetinaPC
from modules.audio_pc import AudioPC
from modules.thalamus_relay import ThalamusRelay
from modules.dopamine_tectal_system import DopamineTectalSystem
from modules.dopamine_hierarchical_efe_system import DopamineHierarchicalEFESystem
from modules.working_memory import WorkingMemory
from modules.cortex_memory import CortexMemory
from modules.meta_memory import MetaMemory
from modules.visual_cortex_pc import VisualCortexPC
from modules.body_pc import BodyPC
from modules.interoceptive_system import InteroceptiveSystem
from modules.metabolic_system import MetabolicSystem
from modules.meta_precision_field import MetaPrecisionField
from modules.deep_inference_field import DeepInferenceField
from modules.outcome_predictor import OutcomePredictor
from modules.temporal_inference_field import TemporalInferenceField
from modules.basal_ganglia import BasalGanglia
from modules.optic_tectum import OpticTectum
from modules.motor_eye import MotorEye
from modules.motor_tail import MotorTail


# ============================================================
# Utility
# ============================================================
def show(title, obj):
    print(f"\n--- {title} ---")
    print(obj)


# ============================================================
# BLOCK A — Vision (v1→v10)
# ============================================================
def test_retina():
    retina = RetinaPC(n_in=784, n_out=64)
    x = torch.randn(1, 784)
    pred, F, Pi = retina.step(x, rpe=0.0)
    show("RetinaPC", {"pred_shape": pred.shape, "FE": F, "Pi": Pi})


def test_visual_cortex():
    vc = VisualCortexPC()
    z = torch.randn(1, 64)
    ctx = torch.randn(1, 64)
    template = torch.randn(1, 64)
    z2, err = vc.forward(z, ctx=ctx, template=template)
    show("VisualCortexPC", {"latent_mean": z2.mean().item(), "err": float(err.abs().mean())})


def test_optic_tectum():
    ot = OpticTectum()
    pos = ot.step(valL=0.1, valR=0.3, Fmean=0.2, bg_gate=0.5, dopa=0.6)
    show("OpticTectum", {"eye_position": pos})


# ============================================================
# BLOCK B — Audio + Multimodal (v10→v15)
# ============================================================
def test_audio_pc():
    audio = AudioPC()
    tone = torch.randn(1, 16)
    pred, F, Pi = audio.step(tone)
    show("AudioPC", {"FE": F, "Pi": Pi})


def test_thalamus():
    th = ThalamusRelay(mode="dual")
    cms = th.step(Fv=0.2, Fa=0.5)
    show("ThalamusRelay", {"CMS": cms})


def test_fast_dopa():
    dsys = DopamineTectalSystem(mode="dual")
    valL, valR, rpe, dopa = dsys.step(Fv=0.3, Fa=0.1, cms=0.05)
    show("Fast Dopamine", {"valL": valL, "valR": valR, "rpe": rpe, "dopa": dopa})


# ============================================================
# BLOCK C — Memory & Cortex (v15→v20)
# ============================================================
def test_working_memory():
    wm = WorkingMemory()
    z = torch.randn(1, 64)
    m, a = wm.step(z, dopa=0.8, cms=0.2, rpe=0.01)
    show("WorkingMemory", {"mem_mean": float(m.mean()), "alpha_eff": a})


def test_cortex_memory():
    cm = CortexMemory()
    for _ in range(10):
        cm.step(torch.randn(1, 64))
    show("CortexMemory", {"template_mean": float(cm.template.mean())})


def test_meta_memory():
    mm = MetaMemory()
    fused = mm.step(torch.randn(1, 64), torch.randn(1, 16), torch.randn(1, 64))
    show("MetaMemory", {"fused": float(fused.mean())})


def test_visual_cortex_with_memory():
    vc = VisualCortexPC()
    z = torch.randn(1, 64)
    ctx = torch.randn(1, 64)
    template = torch.randn(1, 64)
    z2, _ = vc.forward(z, ctx=ctx, template=template)
    show("Cortex Integration", {"latent": float(z2.mean())})


# ============================================================
# BLOCK D — Body & Interoception (v20→v25)
# ============================================================
def test_body_pc():
    bp = BodyPC()
    pred, F = bp.step(torch.randn(1, 16))
    show("BodyPC", {"FE": F})


def test_interoception():
    intero = InteroceptiveSystem()
    Fint, hunger, err = intero.step(torch.tensor([0.2]))
    show("Interoception", {"FE": float(Fint), "hunger": float(hunger), "err": float(err)})


def test_metabolic():
    metab = MetabolicSystem()
    for _ in range(10):
        e, d = metab.step()
    show("MetabolicSystem", {"energy": float(e), "dopa_baseline": float(d)})


# ============================================================
# BLOCK E — Precision & Deep Inference (v25→v28)
# ============================================================
def test_meta_precision():
    mp = MetaPrecisionField()
    prec = mp.step(torch.randn(1, 64))
    show("MetaPrecisionField", {"precision": float(prec)})


def test_deep_inference():
    dif = DeepInferenceField()
    z = dif.step(torch.randn(1, 64))
    show("DeepInferenceField", {"latent": float(z.mean())})


def test_outcome_predictor():
    op = OutcomePredictor()
    future = op.step(torch.randn(1, 64))
    show("OutcomePredictor", {"future_FE_mean": float(future.mean())})


# ============================================================
# BLOCK F — Temporal AI + Motor (v28→v30)
# ============================================================
def test_temporal_inference():
    tif = TemporalInferenceField(mode="standard")
    s = torch.randn(64)
    action, efe, qpi = tif.step(s, dopa_gain=1.0)
    show("TIF", {"action": action, "EFE": efe, "q_pi": qpi})


def test_basal_ganglia():
    bg = BasalGanglia(mode="novelty")
    out = bg.step(valL=0.1, valR=0.3, dopa=0.6, rpe=0.2, cms=0.5)
    show("BasalGanglia", {"output": out})


def test_motor_eye():
    me = MotorEye()
    cmd = me.step(eye_pos=0.5)
    show("MotorEye", {"motor_cmd": cmd})


def test_motor_tail():
    mt = MotorTail()
    for i in range(3):
        out = mt.step(drive=0.3)
    show("MotorTail", {"tail_output": out})


# ============================================================
# MAIN — Run all tests
# ============================================================
if __name__ == "__main__":

    print("\n================ PC-SNN v1→v30 Test Suite ================\n")

    # A — Vision
    test_retina()
    test_visual_cortex()
    test_optic_tectum()

    # B — Audio + Multimodal
    test_audio_pc()
    test_thalamus()
    test_fast_dopa()

    # C — Memory
    test_working_memory()
    test_cortex_memory()
    test_meta_memory()
    test_visual_cortex_with_memory()

    # D — Interoception
    test_body_pc()
    test_interoception()
    test_metabolic()

    # E — Deep inference
    test_meta_precision()
    test_deep_inference()
    test_outcome_predictor()

    # F — Temporal AI + Motor
    test_temporal_inference()
    test_basal_ganglia()
    test_motor_eye()
    test_motor_tail()

    print("\n================== All Tests Complete ==================\n")
