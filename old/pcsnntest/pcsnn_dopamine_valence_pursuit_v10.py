# pcsnn_dopamine_valence_pursuit_v10.py
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt, os

# -------------------------------------------------------------
# Retina (predictive coding)
# -------------------------------------------------------------
class RetinaPC(nn.Module):
    def __init__(self, n_in=784, n_out=64, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(n_in, n_out, device=device))
        self.Pi = torch.ones(1, n_out, device=device) * 0.1
        self.prev_pred = torch.zeros(1, n_out, device=device)
        self.device = device

    def step(self, x, rpe=0.0, valence=0.0):
        # Project to latent
        pred = x @ self.W                           # [1,64]
        # Latent error (change from previous prediction)
        err = pred - self.prev_pred
        self.prev_pred = pred.detach()              # store for next step
        # Free energy and modulation
        F_val = 0.5 * (err ** 2).mean()
        mod = torch.exp(-torch.abs(torch.tensor(rpe + valence, device=self.device)))
        # Hebbian-like weight update
        self.W.data += 0.005 * (x.T @ err) * mod
        # Update precision
        self.Pi += 0.01 * (err.abs().mean() - self.Pi)
        return pred.detach(), F_val.item(), self.Pi.mean().item()

# -------------------------------------------------------------
# Visual cortex (cross inhibition)
# -------------------------------------------------------------
class VisualCortexPC(nn.Module):
    def __init__(self, n_in=64, n_out=64, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.05 * torch.randn(n_in, n_out, device=device))
        self.cross_gain = 0.2
        self.device = device
    def forward(self, x_self, x_contra):
        inhib = self.cross_gain * x_contra.mean(0, keepdim=True)
        return F.relu(x_self @ self.W - inhib)

# -------------------------------------------------------------
# Working memory
# -------------------------------------------------------------
class WorkingMemory(nn.Module):
    def __init__(self, n_in=64, n_mem=32, tau=0.9, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.05 * torch.randn(n_in, n_mem, device=device))
        self.state = torch.zeros(1, n_mem, device=device)
        self.tau = tau
        self.device = device
    def step(self, x):
        self.state = self.tau * self.state + (1 - self.tau) * (x @ self.W)
        return self.state

# -------------------------------------------------------------
# Dopamine / Valence system
# -------------------------------------------------------------
class DopamineSystem:
    def __init__(self):
        self.rpe = 0.0
        self.valL = 0.0
        self.valR = 0.0
    def update(self, pred_val_L, pred_val_R, reward_L, reward_R):
        rpe_L = reward_L - pred_val_L
        rpe_R = reward_R - pred_val_R
        self.rpe = (rpe_L + rpe_R) / 2
        self.valL += 0.1 * rpe_L
        self.valR += 0.1 * rpe_R
        return self.valL, self.valR, self.rpe

# -------------------------------------------------------------
# Eye controller (valence asymmetry)
# -------------------------------------------------------------
class EyeController:
    def __init__(self):
        self.eye = 0.0
    def update(self, valL, valR, Fmean):
        delta = 0.05 * ((valR - valL) - Fmean)
        self.eye += delta
        self.eye = float(torch.clamp(torch.tensor(self.eye), -1.0, 1.0))
        return self.eye

# -------------------------------------------------------------
# Full agent with dual recognizers
# -------------------------------------------------------------
class PCSNNAI_Agent:
    def __init__(self, device="cpu"):
        self.device = device
        self.retinaL = RetinaPC(device=device)
        self.retinaR = RetinaPC(device=device)
        self.v1L = VisualCortexPC(device=device)
        self.v1R = VisualCortexPC(device=device)
        self.mem = WorkingMemory(device=device)
        self.dopa = DopamineSystem()
        self.eye = EyeController()
        self.recognizerL = nn.Linear(64, 10, device=device)
        self.recognizerR = nn.Linear(64, 10, device=device)

    def perceive_and_act(self, img, label):
        x = img.view(1, -1).to(self.device)
        xL, xR = x, torch.flip(x, dims=[1])

        rL, F_L, Pi_L = self.retinaL.step(xL, rpe=self.dopa.rpe, valence=self.dopa.valL)
        rR, F_R, Pi_R = self.retinaR.step(xR, rpe=self.dopa.rpe, valence=self.dopa.valR)

        v1L = self.v1L(rL, rR)
        v1R = self.v1R(rR, rL)
        latent = (v1L + v1R) / 2
        mem_state = self.mem.step(latent)

        logitL = self.recognizerL(v1L)
        logitR = self.recognizerR(v1R)
        predL, predR = logitL.argmax().item(), logitR.argmax().item()

        def val_map(pred):
            if pred <= 2: return +0.5
            elif pred >= 7: return -0.5
            else: return 0.0
        rewardL, rewardR = val_map(predL), val_map(predR)

        valL, valR, rpe = self.dopa.update(self.dopa.valL, self.dopa.valR,
                                           rewardL, rewardR)
        Fmean = (F_L + F_R) / 2
        eye = self.eye.update(valL, valR, Fmean)

        return F_L, F_R, Pi_L, Pi_R, valL, valR, rpe, eye

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs("plots", exist_ok=True)
    data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=1, shuffle=True)

    agent = PCSNNAI_Agent(device=device)
    FL, FR, PiL, PiR, vL, vR, RPE, Eye = [], [], [], [], [], [], [], []

    for t, (img, label) in enumerate(loader):
        fL, fR, piL, piR, valL, valR, rpe, eye = agent.perceive_and_act(img, label.item())
        if t % 50 == 0:
            print(f"Step {t:03d}: FL={fL:.3f} FR={fR:.3f} ΠL={piL:.3f} ΠR={piR:.3f} "
                  f"vL={valL:+.2f} vR={valR:+.2f} RPE={rpe:+.3f} Eye={eye:+.2f}")
        FL.append(fL); FR.append(fR); PiL.append(piL); PiR.append(piR)
        vL.append(valL); vR.append(valR); RPE.append(rpe); Eye.append(eye)
        if t >= 300: break

    plt.figure(figsize=(10,7))
    plt.subplot(3,1,1)
    plt.plot(FL, label="Free Energy L"); plt.plot(FR, label="Free Energy R")
    plt.legend(); plt.ylabel("F")

    plt.subplot(3,1,2)
    plt.plot(PiL, label="ΠL"); plt.plot(PiR, label="ΠR")
    plt.plot(vL, label="Val L"); plt.plot(vR, label="Val R"); plt.legend(); plt.ylabel("Precision/Valence")

    plt.subplot(3,1,3)
    plt.plot(RPE, label="RPE"); plt.plot(Eye, label="Eye"); plt.legend(); plt.xlabel("Step")
    plt.tight_layout(); plt.savefig("plots/v10_dual_dynamics.png")
    print("[Saved] plots/v10_dual_dynamics.png")

if __name__ == "__main__":
    main()
