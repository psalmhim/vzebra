# ============================================================
# pcsnn_dopamine_valence_pursuit_v12_2c_balanced.py
# Stabilized alternation: valence zero-centering, dwell habituation,
# nonlinear tectal drive, and BG sign-cross saccade pulses
# ============================================================

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt, os, math

# ---------------- Retina (as in v12_2b: normalized + clamped Π) -------------
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

# ---------------- V1 / WM (unchanged) ---------------------------------------
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

# ---------------- Dopamine with reciprocal inhibition + zero-centering -------
class DopamineSystem:
    def __init__(self):
        self.rpe = 0.0
        self.valL = 0.0
        self.valR = 0.0
        self.dopa_phase = 0.0
        self.osc_freq = 0.6
        self.val_decay = 0.02
        self.phase_jitter = 0.03
    def update(self, pred_val_L, pred_val_R, reward_L, reward_R):
        rpe_L = reward_L - pred_val_L
        rpe_R = reward_R - pred_val_R
        self.rpe = (rpe_L + rpe_R) / 2
        self.valL = (1 - self.val_decay) * self.valL + 0.1 * (rpe_L - 0.5 * rpe_R)
        self.valR = (1 - self.val_decay) * self.valR + 0.1 * (rpe_R - 0.5 * rpe_L)
        # zero-center valences so they sum ~0 (removes chronic bias)
        m = 0.5 * (self.valL + self.valR)
        self.valL -= m; self.valR -= m
        # clamp
        self.valL = float(torch.clamp(torch.tensor(self.valL), -0.5, 0.5))
        self.valR = float(torch.clamp(torch.tensor(self.valR), -0.5, 0.5))
        self.dopa_phase += self.osc_freq + self.phase_jitter * (torch.rand(1).item() - 0.5)
        return self.valL, self.valR, self.rpe
    def dopamine_level(self):
        return 0.5 + 0.4 * math.sin(self.dopa_phase)

# ---------------- Basal ganglia (same as v12_2b) ----------------------------
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

# ---------------- Optic tectum with nonlinear salience + saccade pulse -------
class OpticTectum:
    def __init__(self):
        self.eye_pos = 0.0
        self.eye_vel = 0.0
        self.center_pull = 0.15
        self.bg_gain = 0.30   # a bit stronger than 0.25 but still mild
    def step(self, valL, valR, Fmean, bg_gate, dopa, pulse=0.0):
        # compress salience; prevents a big valence diff from pinning eye
        sal = float(torch.tanh(torch.tensor(1.5 * (valR - valL))))
        drive = sal - 0.3 * Fmean + self.bg_gain * bg_gate
        # include a brief saccade pulse when BG changes sign
        self.eye_vel = 0.88 * self.eye_vel + 0.10 * drive + 0.25 * pulse
        extra_center = 0.05 if dopa < 0.4 else 0.0
        self.eye_pos += self.eye_vel - (self.center_pull + extra_center) * self.eye_pos
        self.eye_pos = float(torch.clamp(torch.tensor(self.eye_pos), -1.0, 1.0))
        return self.eye_pos

# ---------------- Agent ------------------------------------------------------
class PCSNNAI_Agent:
    def __init__(self, device="cpu"):
        self.device = device
        self.retinaL = RetinaPC(device=device); self.retinaR = RetinaPC(device=device)
        self.v1L = VisualCortexPC(device=device); self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(device=device)
        self.dopa = DopamineSystem(); self.bg = BasalGanglia(); self.tectum = OpticTectum()
        self.recognizerL = nn.Linear(64, 10, device=device); self.recognizerR = nn.Linear(64, 10, device=device)
        self.prev_bg = 0.0
        self.dwell = 0  # for dwell-based habituation

    def perceive_and_act(self, img, label, t):
        # visual feedback: mild roll; polarity flip every 60 steps
        x = img.view(28, 28)
        shift = int(1 * self.tectum.eye_pos)
        x = torch.roll(x, shifts=shift, dims=1)
        if t % 60 == 0: x = 1.0 - x
        precision_bias = 0.5 + 0.5 * abs(self.tectum.eye_pos)

        x = x.view(1, -1).to(self.device); xL, xR = x, torch.flip(x, dims=[1])

        rL, F_L, Pi_L = self.retinaL.step(xL, rpe=self.dopa.rpe, valence=self.dopa.valL, precision_bias=precision_bias)
        rR, F_R, Pi_R = self.retinaR.step(xR, rpe=self.dopa.rpe, valence=self.dopa.valR, precision_bias=precision_bias)

        v1L = self.v1L(rL, rR); v1R = self.v1R(rR, rL)
        _ = self.mem.step((v1L + v1R) / 2)

        # simple recognizers -> rewards
        predL = self.recognizerL(v1L).argmax().item()
        predR = self.recognizerR(v1R).argmax().item()
        def val_map(p): return +0.5 if p <= 2 else (-0.5 if p >= 7 else 0.0)
        rewardL, rewardR = val_map(predL), val_map(predR)

        valL, valR, rpe = self.dopa.update(self.dopa.valL, self.dopa.valR, rewardL, rewardR)

        # dwell-based habituation: if eye stays to one side, reduce that side's valence
        side = -1 if self.tectum.eye_pos < -0.5 else (1 if self.tectum.eye_pos > 0.5 else 0)
        self.dwell = self.dwell + 1 if side != 0 else 0
        hab = min(0.05, 0.002 * self.dwell)  # grows with dwell, capped
        if side == -1:   # left dwell
            valL -= hab; valR += 0.5 * hab
        elif side == 1:  # right dwell
            valR -= hab; valL += 0.5 * hab
        # write back (keeps zero-centered nature)
        self.dopa.valL, self.dopa.valR = valL, valR

        dopa_level = self.dopa.dopamine_level()
        bg_gate = self.bg.step(valL, valR, dopa_level, rpe)

        # saccade pulse when BG changes sign
        pulse = 0.0
        if bg_gate * self.prev_bg <= 0 and abs(bg_gate) > 0.4:
            pulse = 1.0 if bg_gate > 0 else -1.0
        self.prev_bg = bg_gate

        Fmean = (F_L + F_R) / 2
        eye = self.tectum.step(valL, valR, Fmean, bg_gate, dopa_level, pulse=pulse)

        return F_L, F_R, Pi_L, Pi_R, valL, valR, rpe, eye, bg_gate, dopa_level

# ---------------- Main -------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)
    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)
    agent = PCSNNAI_Agent(device=device)

    FL, FR, PiL, PiR, vL, vR, RPE, Eye, BGgate, Dopa = [], [], [], [], [], [], [], [], [], []
    for t, (img, label) in enumerate(loader):
        fL, fR, piL, piR, valL, valR, rpe, eye, bg_gate, dopa = agent.perceive_and_act(img, label.item(), t)
        if t % 50 == 0:
            print(f"Step {t:03d}: FL={fL:.3f} FR={fR:.3f} ΠL={piL:.3f} ΠR={piR:.3f} "
                  f"vL={valL:+.2f} vR={valR:+.2f} RPE={rpe:+.3f} BG={bg_gate:+.2f} Eye={eye:+.2f}")
        FL.append(fL); FR.append(fR); PiL.append(piL); PiR.append(piR)
        vL.append(valL); vR.append(valR); RPE.append(rpe); Eye.append(eye)
        BGgate.append(bg_gate); Dopa.append(dopa)
        if t >= 300: break

    plt.figure(figsize=(10,8))
    plt.subplot(4,1,1); plt.plot(FL, label="Free Energy L"); plt.plot(FR, label="Free Energy R"); plt.legend(); plt.ylabel("F")
    plt.subplot(4,1,2); plt.plot(PiL, label="ΠL"); plt.plot(PiR, label="ΠR"); plt.plot(vL, label="Val L"); plt.plot(vR, label="Val R"); plt.legend(); plt.ylabel("Precision/Valence")
    plt.subplot(4,1,3); plt.plot(RPE, label="RPE"); plt.plot(Dopa, label="Dopamine"); plt.legend(); plt.ylabel("RPE/DA")
    plt.subplot(4,1,4); plt.plot(BGgate, label="BG Gating"); plt.plot(Eye, label="Eye Position"); plt.legend(); plt.xlabel("Step")
    plt.tight_layout(); plt.savefig("plots/v12_2c_balanced.png")
    print("[Saved] plots/v12_2c_balanced.png")

if __name__ == "__main__":
    main()