import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Global parameters
# ============================================================
FIELD = 56
OBJ = 28
STEPS = 150
k_eye = 0.05
k_tail = 0.1
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ============================================================
# Load MNIST
# ============================================================
transform = transforms.ToTensor()
mnist = datasets.MNIST("./data", train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=1, shuffle=True)

# ============================================================
# Scene utilities
# ============================================================
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

# ============================================================
# Predictive-coding unit
# ============================================================
class PCUnit(nn.Module):
    def __init__(self, n_in, n_out, device="cpu"):
        super().__init__()
        self.W = nn.Parameter(0.1 * torch.randn(n_in, n_out, device=device))
        self.v_b = torch.zeros(n_out, device=device)
        self.v_s = torch.zeros(n_out, device=device)
    def step(self, x):
        pred = x @ self.W
        err = pred.mean(0) - self.v_b
        self.v_b = 0.9 * self.v_b + 0.1 * pred.mean(0)
        self.v_s = 0.9 * self.v_s + 0.1 * err
        return self.v_s.clone()

# ============================================================
# Build network
# ============================================================

Npix = FIELD * (FIELD//2)  # 1568
retina_L = PCUnit(Npix, 64, device)
retina_R = PCUnit(Npix, 64, device)
tectum_L = PCUnit(64, 32, device)
tectum_R = PCUnit(64, 32, device)
pallium  = PCUnit(32, 10, device)    # fixed: input 32 (not 64)
motor    = PCUnit(10, 3, device)
readout  = nn.Linear(10, 10, device=device)


# ============================================================
# Valence mapping
# ============================================================
def digit_to_valence(label):
    if label in [0,1,2]:
        return 0  # approach
    elif label in [7,8,9]:
        return 2  # flee
    else:
        return 1  # neutral

# ============================================================
# Simulation
# ============================================================
theta_L = theta_R = 0.0
tail_force = 0.0
F_log, eyeL_log, eyeR_log, tail_log, recog_log = [], [], [], [], []

for img, lbl in loader:
    label = lbl.item()
    scene, obj_x = make_scene(img.to(device))
    valence_true = digit_to_valence(label)
    for t in range(STEPS):
        left, right = retinal_sample(scene, theta_L, theta_R)
        xL = left.view(1, -1)
        xR = right.view(1, -1)

        # visual pathway
        rL = retina_L.step(xL)
        rR = retina_R.step(xR)
        sL = tectum_L.step(rL.unsqueeze(0))
        sR = tectum_R.step(rR.unsqueeze(0))

        # binocular integration (average features)
        fused = (sL + sR).unsqueeze(0)

        # recognition (pallium)
        pall_act = pallium.step(fused)
        logits = readout(pall_act.unsqueeze(0))
        pred_digit = logits.argmax(1).item()

        # valence & motor response
        val_pred = digit_to_valence(pred_digit)
        motor_act = motor.step(pall_act.unsqueeze(0))


        # eye + tail dynamics
        F = ((rL - rR)**2).mean().item()
        F_log.append(F)
        dtheta = k_eye * (sR.mean().item() - sL.mean().item())
        theta_L += dtheta; theta_R -= dtheta
        tail_force += k_tail * (1 if val_pred==0 else (-1 if val_pred==2 else 0))

        eyeL_log.append(theta_L)
        eyeR_log.append(theta_R)
        tail_log.append(tail_force)
        recog_log.append(pred_digit)

    break  # one trial

# ============================================================
# Save figure
# ============================================================
Path("plots").mkdir(exist_ok=True)
plt.figure(figsize=(9,6))
plt.subplot(311)
plt.plot(F_log)
plt.ylabel("Free-energy-like error")

plt.subplot(312)
plt.plot(eyeL_log,label="θ_L"); plt.plot(eyeR_log,label="θ_R")
plt.legend(); plt.ylabel("Eye angles")

plt.subplot(313)
plt.plot(tail_log,label="Tail force",color='g')
plt.ylabel("Tail motor")
plt.xlabel("Time step")
plt.tight_layout()
plt.savefig("plots/recognition_valence_motor.png",dpi=300)
print("Saved: plots/recognition_valence_motor.png")
print(f"True label={label}, predicted={recog_log[-1]}, valence_true={valence_true}")
