# ============================================================
# SCRIPT: pcsnn_v13_4_multisensory_tri.py
# AUTHOR: H.J. Park & GPT-5
# VERSION: v13.4 (2025-11-14)
#
# PURPOSE:
#     Multisensory predictive-coding simulation (Visual + Audio + Body)
#     with cross-modal salience (CMS) and dopamine gating.
#
# UPDATES:
#     • Adds BodyPC for tri-modal integration (V+A+B)
#     • Unified module interface via BaseModule
#     • Compatible with all v13.3 cortical modules
#
# DEPENDENCIES:
#     from modules import *
# ============================================================

import torch
import matplotlib.pyplot as plt
from modules import (
    RetinaPC, AudioPC, BodyPC,
    VisualCortexPC, WorkingMemory,
    ThalamusRelay, DopamineSystem, BasalGanglia
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------------------------------------
# Instantiate modules
# ------------------------------------------------------------
retina = RetinaPC(n_in=64, n_out=32, device=device)
audio = AudioPC(n_in=16, n_out=8, device=device)
body  = BodyPC(n_in=8,  n_out=8,  device=device)
v1L   = VisualCortexPC(n_in=32, n_latent=32, device=device)
v1R   = VisualCortexPC(n_in=32, n_latent=32, device=device)
mem   = WorkingMemory(n_latent=32, device=device)
thal  = ThalamusRelay(device=device, mode="tri")
dopa = DopamineSystem(device=device, mode="tri")   # or "dual" / "policy"
bg    = BasalGanglia(mode="alternating")

# ------------------------------------------------------------
# Simulation setup
# ------------------------------------------------------------
T = 350
Fv_hist, Fa_hist, Fb_hist = [], [], []
RPE_hist, CMS_hist, BG_hist, Eye_hist = [], [], [], []

for t in range(T):
    # ---------------- sensory input ----------------
    img = torch.randn(1, 64, device=device)
    tone = torch.randn(1, 16, device=device)
    prop = torch.randn(1, 8,  device=device)

    # --- retina, audio, body predictive updates ---
    pred_v, Fv, Pi_v = retina.step(img)
    pred_a, Fa, Pi_a = audio.step(tone)
    pred_b, Fb, Pi_b = body.step(prop)

    # --- thalamic cross-modal salience (tri-modal) ---
    cms = thal.step(Fv, Fa, Fb)

    # --- dopamine system updates ---
    valL, valR, rpe, dopa_out = dopa.step(Fv, Fa, Fb=Fb, cms=cms)


    # --- cortical reconstruction & working memory ---
    zL, _ = v1L(pred_v, pred_v)
    zR, _ = v1R(pred_v, pred_v)
    ctx = mem.step((zL + zR) / 2, dopa=dopa_out, cms=cms)

    # --- basal ganglia control (eye or saccade output) ---
    eye = bg.step(valL=Fv, valR=Fa, dopa=dopa_out, rpe=rpe, cms=cms)

    # ---------------- record data ----------------
    Fv_hist.append(Fv)
    Fa_hist.append(Fa)
    Fb_hist.append(Fb)
    RPE_hist.append(rpe)
    CMS_hist.append(cms)
    BG_hist.append(bg.state)
    Eye_hist.append(eye)

    if t % 50 == 0:
        print(f"Step {t:03d}: "
              f"Fv={Fv:.3f} Fa={Fa:.3f} Fb={Fb:.3f} "
              f"Πv={Pi_v:.3f} Πa={Pi_a:.3f} Πb={Pi_b:.3f} "
              f"RPE={rpe:+.3f} CMS={cms:+.3f} BG={bg.state:+.3f} Eye={eye:+.2f}")

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(Fv_hist, label="F_visual")
plt.plot(Fa_hist, label="F_audio")
plt.plot(Fb_hist, label="F_body")
plt.plot(CMS_hist, label="CMS", linestyle="--")
plt.plot(RPE_hist, label="RPE", linestyle=":")
plt.plot(Eye_hist, label="Eye", alpha=0.7)
plt.legend()
plt.title("PC-SNN v13.4 Tri-modal Integration (Visual + Auditory + Body)")
plt.xlabel("Time steps")
plt.ylabel("Activity / Free energy")
plt.tight_layout()
plt.savefig("plots/v13_4_multisensory_tri.png")
print("[Saved] plots/v13_4_multisensory_tri.png")

