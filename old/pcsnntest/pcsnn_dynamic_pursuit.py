import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Device setup
# ============================================================
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# ============================================================
# Parameters
# ============================================================
FIELD, OBJ = 56, 28
STEPS = 200
k_eye = 0.05  # eye gain
k_obj = 0.15  # object drift rate
device = device

# ============================================================
# Predictive-coding unit (simplified)
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
# Scene generator
# ============================================================
def make_scene(x_pos, obj_intensity=1.0):
    """Generate a simple horizontal bar object at position x_pos (in pixels)."""
    scene = torch.zeros(1, 1, FIELD, FIELD, device=device)
    x = int(np.clip(x_pos, 0, FIELD - OBJ))
    y = FIELD // 2 - OBJ // 2
    scene[0, 0, y:y+OBJ, x:x+OBJ] = obj_intensity
    return scene

def retinal_sample(scene, theta_L, theta_R):
    shift_L = int(np.clip(theta_L * 10, -5, 5))
    shift_R = int(np.clip(theta_R * 10, -5, 5))
    left  = torch.roll(scene[:, :, :, :FIELD//2],  shifts=shift_L, dims=3)
    right = torch.roll(scene[:, :, :, FIELD//2:], shifts=shift_R, dims=3)
    return left, right

# ============================================================
# Build minimal visual system
# ============================================================
Npix = FIELD * (FIELD//2)
retina_L = PCUnit(Npix, 64, device)
retina_R = PCUnit(Npix, 64, device)
tectum_L = PCUnit(64, 32, device)
tectum_R = PCUnit(64, 32, device)

# ============================================================
# Simulation loop
# ============================================================
theta_L = theta_R = 0.0
x_obj = 10.0             # start position
dx_obj = +k_obj          # drift speed
F_log, eyeL_log, eyeR_log, x_log = [], [], [], []

for t in range(STEPS):
    # move object smoothly across screen, bounce at borders
    x_obj += dx_obj
    if x_obj < 0 or x_obj > FIELD - OBJ:
        dx_obj *= -1
        x_obj = np.clip(x_obj, 0, FIELD - OBJ)

    scene = make_scene(x_obj)
    left, right = retinal_sample(scene, theta_L, theta_R)
    xL = left.view(1, -1)
    xR = right.view(1, -1)

    rL = retina_L.step(xL)
    rR = retina_R.step(xR)
    sL = tectum_L.step(rL.unsqueeze(0))
    sR = tectum_R.step(rR.unsqueeze(0))

    # compute binocular prediction error and adjust eyes
    F = ((rL - rR)**2).mean().item()
    dtheta = k_eye * (sR.mean().item() - sL.mean().item())
    theta_L += dtheta
    theta_R -= dtheta

    # log
    F_log.append(F)
    eyeL_log.append(theta_L)
    eyeR_log.append(theta_R)
    x_log.append(x_obj - FIELD/2)  # centered position

# ============================================================
# Plot results
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
plt.plot(x_log,label="Object position")
plt.plot(eyeL_log,label="Eye L",alpha=0.7)
plt.plot([-x for x in eyeR_log],label="Eye R",alpha=0.7)
plt.legend(); plt.ylabel("Position / deg")
plt.xlabel("Time step")

plt.tight_layout()
plt.savefig("plots/dynamic_pursuit.png",dpi=300)
print("Saved: plots/dynamic_pursuit.png")
