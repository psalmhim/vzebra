# pcsnn_detection_pursuit.py  (corrected)
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

FIELD = 56
OBJ   = 28
k_eye = 0.05
STEPS = 150
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# MNIST loader
# ------------------------------------------------------------
transform = transforms.ToTensor()
mnist = datasets.MNIST("./data", train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=1, shuffle=True)

# ------------------------------------------------------------
# Scene and retina helpers
# ------------------------------------------------------------
def make_scene(img):
    scene = torch.zeros(1, 1, FIELD, FIELD)
    x = np.random.randint(0, FIELD - OBJ)
    y = np.random.randint(0, FIELD - OBJ)
    scene[0, 0, y:y+OBJ, x:x+OBJ] = img
    obj_center = x + OBJ/2 - FIELD/2
    return scene.to(device), obj_center

def retinal_sample(scene, theta_L, theta_R):
    shift_L = int(np.clip(theta_L * 10, -5, 5))
    shift_R = int(np.clip(theta_R * 10, -5, 5))
    left  = torch.roll(scene[:, :, :, :FIELD//2],  shifts=shift_L, dims=3)
    right = torch.roll(scene[:, :, :, FIELD//2:], shifts=shift_R, dims=3)
    return left, right

# ------------------------------------------------------------
# Predictive-coding unit
# ------------------------------------------------------------
class PCUnit(nn.Module):
    def __init__(self, n_in, n_out, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.1 * torch.randn(n_in, n_out, device=device))
        self.v_b = torch.zeros(n_out, device=device)
        self.v_s = torch.zeros(n_out, device=device)

    def step(self, x):
        # x : [B, n_in]
        pred = x @ self.W                        # [B, n_out]
        err  = pred.mean(0) - self.v_b           # [n_out]
        self.v_b = 0.9 * self.v_b + 0.1 * pred.mean(0)
        self.v_s = 0.9 * self.v_s + 0.1 * err
        return self.v_s.clone()                  # [n_out]

# ------------------------------------------------------------
# Network
# ------------------------------------------------------------
Npix = FIELD * (FIELD//2)       # 56*28 = 1568
retina_L = PCUnit(Npix, 64, device)
retina_R = PCUnit(Npix, 64, device)
tectum_L = PCUnit(64, 32, device)
tectum_R = PCUnit(64, 32, device)

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------
theta_L = theta_R = 0.0
F_log, eyeL_log, eyeR_log = [], [], []

for img, lbl in loader:
    img = img.to(device)
    scene, obj_x = make_scene(img)
    for t in range(STEPS):
        left, right = retinal_sample(scene, theta_L, theta_R)
        xL = left.view(1, -1)   # [1,1568]
        xR = right.view(1, -1)

        rL = retina_L.step(xL)
        rR = retina_R.step(xR)
        sL = tectum_L.step(rL.unsqueeze(0))
        sR = tectum_R.step(rR.unsqueeze(0))

        F = ((rL - rR)**2).mean().item()
        F_log.append(F)
        eyeL_log.append(theta_L)
        eyeR_log.append(theta_R)

        dtheta = k_eye * (sR.mean().item() - sL.mean().item())
        theta_L += dtheta
        theta_R -= dtheta
    break

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
Path("plots").mkdir(exist_ok=True)
plt.figure(figsize=(8,4))
plt.subplot(211)
plt.plot(F_log)
plt.ylabel("Free-energy-like error")
plt.subplot(212)
plt.plot(eyeL_log,label="θ_L"); plt.plot(eyeR_log,label="θ_R")
plt.legend(); plt.xlabel("Time"); plt.ylabel("Eye angle")
plt.tight_layout()
plt.savefig("plots/detection_pursuit.png",dpi=300)
print("Saved: plots/detection_pursuit.png")
