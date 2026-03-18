# ============================================================
# pcsnn_dopamine_valence_pursuit_v13_thalamic_multisensory.py
# Zebrafish PC-SNN with thalamic relay and cross-modal (audio)
# predictive integration driving adaptive saccades
# ============================================================

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt, os, math

# ============================================================
# 1. Retina predictive coding (same stabilized version)
# ============================================================
class RetinaPC(nn.Module):
    def __init__(self, n_in=784, n_out=64, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(n_in, n_out, device=device))
        self.Pi = torch.ones(1, n_out, device=device) * 0.1
        self.prev_pred = torch.zeros(1, n_out, device=device)
        self.device = device
    def step(self, x, rpe=0.0, valence=0.0, precision_bias=1.0):
        pred = x @ self.W
        err = pred - self.prev_pred
        self.prev_pred = pred.detach()
        F_val = 0.5 * (err ** 2).mean()
        mod = torch.exp(-torch.abs(torch.tensor(rpe + valence, device=self.device)))
        lr = 0.002 * (0.5 / (1 + self.Pi.mean()))
        dW = lr * (x.T @ err) * mod * precision_bias
        self.W.data += dW
        self.W.data /= (1.0 + self.W.data.abs().mean())
        self.Pi += 0.003 * (err.abs().mean() * precision_bias - self.Pi)
        self.Pi = torch.clamp(self.Pi, 0.01, 0.5)
        return pred.detach(), F_val.item(), self.Pi.mean().item()

# ============================================================
# 2. Audio predictive coding (simplified cross-modal PC unit)
# ============================================================
class AudioPC(nn.Module):
    def __init__(self, n_in=16, n_lat=16, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(n_in, n_lat, device=device))
        self.latent = torch.zeros(1, n_lat, device=device)
        self.Pi = torch.ones(1, n_lat, device=device) * 0.1
        self.device = device
    def step(self, a_tone, rpe=0.0):
        a_in = a_tone.view(1, -1).to(self.device)
        pred = a_in @ self.W
        err = pred - self.latent
        self.latent = 0.9 * self.latent + 0.1 * pred
        F_val = 0.5 * (err ** 2).mean()
        self.W.data += 0.001 * (a_in.T @ err) * torch.exp(-torch.abs(torch.tensor(rpe, device=self.device)))
        self.W.data /= (1.0 + self.W.data.abs().mean())
        self.Pi += 0.002 * (err.abs().mean() - self.Pi)
        self.Pi = torch.clamp(self.Pi, 0.01, 0.5)
        return pred.detach(), F_val.item(), self.Pi.mean().item()

# ============================================================
# 3. Visual cortex, working memory (same)
# ============================================================
class VisualCortexPC(nn.Module):
    def __init__(self, n_in=64, n_out=64, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.05 * torch.randn(n_in, n_out, device=device))
        self.cross_gain = 0.2
    def forward(self, x_self, x_contra):
        inhib = self.cross_gain * x_contra.mean(0, keepdim=True)
        return F.relu(x_self @ self.W - inhib)

class WorkingMemory(nn.Module):
    def __init__(self, n_in=64, n_mem=32, tau=0.9, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.05 * torch.randn(n_in, n_mem, device=device))
        self.state = torch.zeros(1, n_mem, device=device)
        self.tau = tau
    def step(self, x):
        self.state = self.tau * self.state + (1 - self.tau) * (x @ self.W)
        return self.state

# ============================================================
# 4. Thalamic relay: integrates multimodal salience
# ============================================================
class ThalamusRelay:
    def __init__(self):
        self.state = 0.0
        self.alpha = 0.8
    def step(self, visual_err, audio_err):
        # combine modalities; treat mismatch as salience
        sal = 0.5 * visual_err + 0.5 * audio_err
        self.state = self.alpha * self.state + (1 - self.alpha) * sal
        return self.state

# ============================================================
# 5. Dopamine / Valence with cross-modal modulation
# ============================================================
class DopamineSystem:
    def __init__(self):
        self.rpe = 0.0
        self.valL = 0.0
        self.valR = 0.0
        self.dopa_phase = 0.0
        self.osc_freq = 0.6
        self.val_decay = 0.02
        self.phase_jitter = 0.03
    def update(self, pred_val_L, pred_val_R, reward_L, reward_R, crossmodal_salience=0.0):
        rpe_L = reward_L - pred_val_L
        rpe_R = reward_R - pred_val_R
        base_rpe = (rpe_L + rpe_R) / 2
        # add cross-modal salience term (novelty)
        self.rpe = base_rpe + 0.3 * crossmodal_salience
        # reciprocal inhibition + centering
        self.valL = (1 - self.val_decay) * self.valL + 0.1 * (rpe_L - 0.5 * rpe_R)
        self.valR = (1 - self.val_decay) * self.valR + 0.1 * (rpe_R - 0.5 * rpe_L)
        mean = 0.5 * (self.valL + self.valR)
        self.valL -= mean; self.valR -= mean
        self.valL = float(torch.clamp(torch.tensor(self.valL), -0.5, 0.5))
        self.valR = float(torch.clamp(torch.tensor(self.valR), -0.5, 0.5))
        self.dopa_phase += self.osc_freq + self.phase_jitter * (torch.rand(1).item() - 0.5)
        return self.valL, self.valR, self.rpe
    def dopamine_level(self):
        return 0.5 + 0.4 * math.sin(self.dopa_phase)

# ============================================================
# 6. Basal ganglia + Optic tectum (same stabilized)
# ============================================================
class BasalGanglia:
    def __init__(self):
        self.state = 0.0
        self.momentum = 0.0
        self.leak = 0.12
        self.k_gate = 1.5
        self.max_state = 0.6
        self.noise = 0.03
    def step(self, valL, valR, dopa, rpe):
        drive = (valR - valL) + 0.5 * (dopa - 0.5)
        switch_prob = torch.sigmoid(torch.tensor(abs(rpe) * 5.0))
        if torch.rand(1).item() < switch_prob:
            drive *= -1.0
        self.momentum = 0.7 * self.momentum + 0.3 * drive
        self.state += self.momentum
        self.state -= self.leak * self.state * (1.0 + (0.6 - dopa))
        self.state += self.noise * torch.randn(1).item()
        self.state = float(torch.clamp(torch.tensor(self.state), -self.max_state, self.max_state))
        return float(torch.tanh(torch.tensor(self.k_gate * self.state)))

class OpticTectum:
    def __init__(self):
        self.eye_pos = 0.0
        self.eye_vel = 0.0
        self.center_pull = 0.15
        self.bg_gain = 0.3
    def step(self, valL, valR, Fmean, bg_gate, dopa):
        drive = torch.tanh(torch.tensor(1.5 * (valR - valL))).item() - 0.3 * Fmean + self.bg_gain * bg_gate
        self.eye_vel = 0.9 * self.eye_vel + 0.1 * drive
        extra_center = 0.05 if dopa < 0.4 else 0.0
        self.eye_pos += self.eye_vel - (self.center_pull + extra_center) * self.eye_pos
        self.eye_pos = float(torch.clamp(torch.tensor(self.eye_pos), -1.0, 1.0))
        return self.eye_pos

# ============================================================
# 7. Agent integrating vision, audio, and thalamus
# ============================================================
class PCSNNAI_Agent:
    def __init__(self, device="cpu"):
        self.device = device
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(device=device)
        self.audio = AudioPC(device=device)
        self.thal = ThalamusRelay()
        self.dopa = DopamineSystem()
        self.bg = BasalGanglia()
        self.tectum = OpticTectum()
        self.recognizerL = nn.Linear(64, 10, device=device)
        self.recognizerR = nn.Linear(64, 10, device=device)

    def perceive_and_act(self, img, label, t):
        # --- Visual preprocessing with eye feedback ---
        x = img.view(28, 28)
        shift = int(1 * self.tectum.eye_pos)
        x = torch.roll(x, shifts=shift, dims=1)
        if t % 60 == 0:
            x = 1.0 - x
        precision_bias = 0.5 + 0.5 * abs(self.tectum.eye_pos)
        x = x.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])

        # --- Retinal and cortical inference ---
        rL, F_L, Pi_L = self.retinaL.step(xL, rpe=self.dopa.rpe, valence=self.dopa.valL, precision_bias=precision_bias)
        rR, F_R, Pi_R = self.retinaR.step(xR, rpe=self.dopa.rpe, valence=self.dopa.valR, precision_bias=precision_bias)
        v1L = self.v1L(rL, rR)
        v1R = self.v1R(rR, rL)
        latent = (v1L + v1R) / 2
        _ = self.mem.step(latent)

        # --- Audio input: oscillating tone pattern with small noise ---
        tone = torch.sin(torch.linspace(0, math.pi, 16) + 0.05 * t) + 0.05 * torch.randn(16)
        a_pred, F_audio, Pi_aud = self.audio.step(tone, rpe=self.dopa.rpe)

        # --- Thalamic relay combines modalities ---
        crossmodal_sal = self.thal.step(F_L + F_R, F_audio)

        # --- Recognition and rewards ---
        predL = self.recognizerL(v1L).argmax().item()
        predR = self.recognizerR(v1R).argmax().item()
        def val_map(p): return +0.5 if p <= 2 else (-0.5 if p >= 7 else 0.0)
        rewardL, rewardR = val_map(predL), val_map(predR)

        # --- Dopamine + BG + Tectum integration ---
        valL, valR, rpe = self.dopa.update(self.dopa.valL, self.dopa.valR, rewardL, rewardR,
                                           crossmodal_salience=crossmodal_sal)
        dopa_level = self.dopa.dopamine_level()
        bg_gate = self.bg.step(valL, valR, dopa_level, rpe)
        Fmean = (F_L + F_R) / 2
        eye = self.tectum.step(valL, valR, Fmean, bg_gate, dopa_level)

        return (F_L, F_R, F_audio, Pi_L, Pi_R, Pi_aud,
                valL, valR, rpe, eye, bg_gate, dopa_level, crossmodal_sal)

# ============================================================
# 8. Main simulation
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)
    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)
    agent = PCSNNAI_Agent(device=device)

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

    plt.figure(figsize=(10,9))
    plt.subplot(5,1,1); plt.plot(FL, label="F Left"); plt.plot(FR, label="F Right"); plt.plot(FA, label="F Audio"); plt.legend(); plt.ylabel("Free Energy")
    plt.subplot(5,1,2); plt.plot(vL, label="Val L"); plt.plot(vR, label="Val R"); plt.legend(); plt.ylabel("Valence")
    plt.subplot(5,1,3); plt.plot(RPE, label="RPE"); plt.plot(Dopa, label="DA"); plt.plot(CMS, label="Crossmodal Salience"); plt.legend(); plt.ylabel("RPE/DA/CMS")
    plt.subplot(5,1,4); plt.plot(BGgate, label="BG Gate"); plt.plot(Eye, label="Eye Pos"); plt.legend(); plt.ylabel("Motor")
    plt.subplot(5,1,5); plt.plot(PiL, label="ΠL"); plt.plot(PiR, label="ΠR"); plt.plot(PiA, label="ΠAudio"); plt.legend(); plt.ylabel("Precision")
    plt.tight_layout(); plt.savefig("plots/v13_thalamic_multisensory.png")
    print("[Saved] plots/v13_thalamic_multisensory.png")

if __name__ == "__main__":
    main()
