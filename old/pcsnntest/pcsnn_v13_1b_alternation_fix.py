# ============================================================
# pcsnn_v13_1b_alternation_fix.py
# Modular Zebrafish PC-SNN with thalamic novelty & BG alternation
# ============================================================
# Structure:
#   modules/
#     retina_pc.py
#     audio_pc.py
#     cortex_memory.py
#     dopamine_system.py
#     thalamus_relay.py
#     basal_ganglia.py
#     optic_tectum.py
#   plots/
# ============================================================

import torch, os, math
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---- Import neural subsystems ----
from modules.retina_pc import RetinaPC
from modules.audio_pc import AudioPC
from modules.cortex_memory import VisualCortexPC, WorkingMemory
from modules.dopamine_system import DopamineSystem
from modules.thalamus_relay import ThalamusRelay
from modules.basal_ganglia import BasalGanglia
from modules.optic_tectum import OpticTectum

# ============================================================
# Predictive-Coding Zebrafish Agent
# ============================================================
class PCSNNAI_Agent:
    def __init__(self, device="cpu", mode="alternating"):
        self.device = device
        self.mode = mode

        # core modules
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(device=device)
        self.audio = AudioPC(device=device)
        self.thal = ThalamusRelay()
        self.dopa = DopamineSystem()
        self.bg = BasalGanglia(mode=mode)
        self.tectum = OpticTectum()

        # simple recognizers (placeholders)
        self.recognizerL = torch.nn.Linear(64, 10, device=device)
        self.recognizerR = torch.nn.Linear(64, 10, device=device)

    def perceive_and_act(self, img, label, t):
        # ---- visual preprocessing with gaze feedback ----
        x = img.view(28, 28)
        shift = int(1 * self.tectum.eye_pos)
        x = torch.roll(x, shifts=shift, dims=1)
        if t % 60 == 0:
            x = 1.0 - x  # polarity change every 60 steps
        precision_bias = 0.5 + 0.5 * abs(self.tectum.eye_pos)
        x = x.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])

        # ---- retinal predictive coding ----
        rL, F_L, Pi_L = self.retinaL.step(xL, rpe=self.dopa.rpe, valence=self.dopa.valL, precision_bias=precision_bias)
        rR, F_R, Pi_R = self.retinaR.step(xR, rpe=self.dopa.rpe, valence=self.dopa.valR, precision_bias=precision_bias)

        # ---- cortical processing ----
        v1L = self.v1L(rL, rR)
        v1R = self.v1R(rR, rL)
        latent = (v1L + v1R) / 2
        _ = self.mem.step(latent)

        # ---- auditory predictive coding ----
        tone = torch.sin(torch.linspace(0, math.pi, 16) + 0.08 * t) + 0.05 * torch.randn(16)
        a_pred, F_audio, Pi_aud = self.audio.step(tone, rpe=self.dopa.rpe)

        # ---- thalamic novelty detection ----
        crossmodal_sal = self.thal.step(F_L + F_R, F_audio)

        # ---- recognition + reward mapping ----
        predL = self.recognizerL(v1L).argmax().item()
        predR = self.recognizerR(v1R).argmax().item()
        def val_map(p): return +0.5 if p <= 2 else (-0.5 if p >= 7 else 0.0)
        rewardL, rewardR = val_map(predL), val_map(predR)

        # ---- dopamine, BG, tectum integration ----
        valL, valR, rpe = self.dopa.update(
            self.dopa.valL, self.dopa.valR, rewardL, rewardR, crossmodal_salience=crossmodal_sal)
        dopa_level = self.dopa.dopamine_level()
        bg_gate = self.bg.step(valL, valR, dopa_level, rpe)
        Fmean = (F_L + F_R) / 2
        eye = self.tectum.step(valL, valR, Fmean, bg_gate, dopa_level)

        return (F_L, F_R, F_audio, Pi_L, Pi_R, Pi_aud,
                valL, valR, rpe, eye, bg_gate, dopa_level, crossmodal_sal)

# ============================================================
# Simulation Loop
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)

    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)
    agent = PCSNNAI_Agent(device=device, mode="alternating")  # switch to 'exploratory' as needed

    FL, FR, FA, PiL, PiR, PiA, vL, vR, RPE, Eye, BGgate, Dopa, CMS = [], [], [], [], [], [], [], [], [], [], [], [], []

    for t, (img, label) in enumerate(loader):
        fL, fR, fA, piL, piR, piA, valL, valR, rpe, eye, bg_gate, dopa, cms = agent.perceive_and_act(img, label.item(), t)
        if t % 50 == 0:
            print(f"Step {t:03d}: Fv={fL:.3f}/{fR:.3f} Fa={fA:.3f} ΠL={piL:.3f} ΠR={piR:.3f} ΠA={piA:.3f} "
                  f"vL={valL:+.2f} vR={valR:+.2f} RPE={rpe:+.3f} CMS={cms:+.3f} BG={bg_gate:+.2f} Eye={eye:+.2f}")
        FL.append(fL); FR.append(fR); FA.append(fA)
        PiL.append(piL); PiR.append(piR); PiA.append(piA)
        vL.append(valL); vR.append(valR); RPE.append(rpe); Eye.append(eye)
        BGgate.append(bg_gate); Dopa.append(dopa); CMS.append(cms)
        if t >= 300: break

    # ---- Visualization ----
    plt.figure(figsize=(10,9))
    plt.subplot(5,1,1); plt.plot(FL, label="F Left"); plt.plot(FR, label="F Right"); plt.plot(FA, label="F Audio")
    plt.legend(); plt.ylabel("Free Energy")

    plt.subplot(5,1,2); plt.plot(vL, label="Val L"); plt.plot(vR, label="Val R")
    plt.legend(); plt.ylabel("Valence")

    plt.subplot(5,1,3); plt.plot(RPE, label="RPE"); plt.plot(Dopa, label="DA"); plt.plot(CMS, label="Crossmodal Salience")
    plt.legend(); plt.ylabel("RPE/DA/CMS")

    plt.subplot(5,1,4); plt.plot(BGgate, label="BG Gate"); plt.plot(Eye, label="Eye Pos")
    plt.legend(); plt.ylabel("Motor (Eye/BG)")

    plt.subplot(5,1,5); plt.plot(PiL, label="ΠL"); plt.plot(PiR, label="ΠR"); plt.plot(PiA, label="ΠAudio")
    plt.legend(); plt.ylabel("Precision")

    plt.tight_layout()
    plt.savefig("plots/v13_1b_alternation_fix.png")
    print("[Saved] plots/v13_1b_alternation_fix.png")

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()

