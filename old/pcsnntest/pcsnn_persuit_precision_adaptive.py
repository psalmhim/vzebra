# pcsnn_precision_adaptive.py
import torch, matplotlib.pyplot as plt, os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Retina module with adaptive precision
# -------------------------------------------------------
class Retina(torch.nn.Module):
    def __init__(self, n_in=1, n_out=32, tau=0.9, eta=0.01,
                 pi_min=0.01, pi_max=2.0, device="cpu"):
        super().__init__()
        self.W = torch.randn(n_in, n_out, device=device) * 0.05
        self.v_b = torch.zeros(n_out, device=device)
        self.v_s = torch.zeros(n_out, device=device)
        self.Pi = torch.ones(n_out, device=device) * 0.1
        self.tau, self.eta = tau, eta
        self.pi_min, self.pi_max = pi_min, pi_max
        self.device = device

    def step(self, x):
        if x.dim() > 1:
            x = x.view(-1)
        pred = x @ self.W
        err = (x.unsqueeze(0) - pred).view(-1)

        # Precision update: relative scaling, low-pass filtered
        abs_err = err.abs()
        delta_pi = self.eta * (abs_err / (abs_err.mean() + 1e-6) - self.Pi)
        self.Pi = torch.clamp(self.Pi + delta_pi, self.pi_min, self.pi_max)

        # Weighted state update
        weighted_err = self.Pi * err
        self.v_b = self.tau * self.v_b + (1 - self.tau) * pred
        self.v_s = self.tau * self.v_s + (1 - self.tau) * weighted_err

        F = (self.Pi * err**2).mean().item()
        return self.v_s.mean(), F, self.Pi.mean().item()

# -------------------------------------------------------
# Eye system and object simulation
# -------------------------------------------------------
class ZebrafishVision:
    def __init__(self, device="cpu"):
        self.eyeL = Retina(device=device)
        self.eyeR = Retina(device=device)
        self.theta_L = torch.tensor(0.0, device=device)
        self.theta_R = torch.tensor(0.0, device=device)
        self.k_eye = 0.05  # eye movement gain

    def step(self, obj_pos):
        # Simulate visual inputs to each eye (retinal disparity)
        xL = torch.tensor([[obj_pos - self.theta_L.item()]], device=device)
        xR = torch.tensor([[obj_pos - self.theta_R.item()]], device=device)
        # Update retinal predictions
        rL, FL, PiL = self.eyeL.step(xL)
        rR, FR, PiR = self.eyeR.step(xR)
        # Compute prediction-error difference (vergence signal)
        dtheta = (rR - rL)
        gain = 0.5 * (PiL + PiR)  # average sensory precision
        # Update eyes using adaptive gain
        self.theta_L += self.k_eye * gain * dtheta
        self.theta_R -= self.k_eye * gain * dtheta
        F = 0.5 * (FL + FR)
        return F, self.theta_L.item(), self.theta_R.item(), PiL, PiR

# -------------------------------------------------------
# Simulation
# -------------------------------------------------------
T = 200
obj_traj = torch.cat([
    torch.linspace(-10, 0, T // 2),
    torch.linspace(0, -10, T // 2)
]).to(device)

fish = ZebrafishVision(device=device)

F_hist, PiL_hist, PiR_hist, eyeL_hist, eyeR_hist = [], [], [], [], []
for t in range(T):
    F, eL, eR, PiL, PiR = fish.step(obj_traj[t])
    F_hist.append(F)
    PiL_hist.append(PiL)
    PiR_hist.append(PiR)
    eyeL_hist.append(eL)
    eyeR_hist.append(eR)

# -------------------------------------------------------
# Plot results
# -------------------------------------------------------
plt.figure(figsize=(9, 8))
plt.subplot(3, 1, 1)
plt.plot(F_hist)
plt.ylabel("Free-energy-like error")

plt.subplot(3, 1, 2)
plt.plot(PiL_hist, label="Precision L")
plt.plot(PiR_hist, label="Precision R")
plt.legend(); plt.ylabel("Adaptive precision")

plt.subplot(3, 1, 3)
plt.plot(eyeL_hist, label="θ_L")
plt.plot(eyeR_hist, label="θ_R")
plt.plot(obj_traj.cpu(), label="Object", alpha=0.5)
plt.legend(); plt.xlabel("Time step"); plt.ylabel("Eye / Object position")

os.makedirs("plots", exist_ok=True)
plt.tight_layout()
plt.savefig("plots/precision_adaptive.png")
print("[Saved] plots/precision_adaptive.png")
