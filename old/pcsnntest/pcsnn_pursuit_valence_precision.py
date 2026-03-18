# pcsnn_pursuit_valence_precision.py
import torch, matplotlib.pyplot as plt, os
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Retina with valence-modulated precision
# -------------------------------------------------------
class Retina(torch.nn.Module):
    def __init__(self, n_in=1, n_out=32, tau=0.9, eta=0.01,
                 pi_min=0.01, pi_max=2.0, beta_valence=0.5, device="cpu"):
        super().__init__()
        self.W = torch.randn(n_in, n_out, device=device) * 0.05
        self.v_b = torch.zeros(n_out, device=device)
        self.v_s = torch.zeros(n_out, device=device)
        self.Pi  = torch.ones(n_out, device=device) * 0.1
        self.tau, self.eta = tau, eta
        self.pi_min, self.pi_max = pi_min, pi_max
        self.beta = beta_valence
        self.device = device

    def step(self, x, valence=0.0):
        if x.dim() > 1: x = x.view(-1)
        pred = x @ self.W
        err  = (x.unsqueeze(0) - pred).view(-1)
        abs_err = err.abs()
        rel = abs_err / (abs_err.mean() + 1e-6)

        # --- valence-modulated precision update ---
        # inside Retina.step(), replace precision update and weighting parts
        target = (1 + self.beta * valence) * rel
        delta  = self.eta * (target - self.Pi)
        self.Pi = torch.clamp(self.Pi + delta, self.pi_min, self.pi_max)

        # nonlinear transformation for motor gain stability
        gain_scale = torch.tanh(self.Pi)  # bounds 0–1
        weighted_err = gain_scale * err
        self.v_b = self.tau * self.v_b + (1 - self.tau) * pred
        self.v_s = self.tau * self.v_s + (1 - self.tau) * weighted_err

        F = (self.Pi * err**2).mean().item()
        return self.v_s.mean(), F, self.Pi.mean().item()

# -------------------------------------------------------
# Zebrafish vision-motor loop
# -------------------------------------------------------
class ZebrafishVision:
    def __init__(self, device="cpu"):
        self.eyeL = Retina(device=device)
        self.eyeR = Retina(device=device)
        self.theta_L = torch.tensor(0.0, device=device)
        self.theta_R = torch.tensor(0.0, device=device)
        self.k_eye_L = 0.05
        self.k_eye_R = 0.06

    def step(self, obj_pos, valence=0.0):
        xL = torch.tensor([[obj_pos - self.theta_L.item()]], device=device)
        xR = torch.tensor([[obj_pos - self.theta_R.item()]], device=device)
        rL, FL, PiL = self.eyeL.step(xL, valence)
        rR, FR, PiR = self.eyeR.step(xR, valence)

        dtheta = (rR - rL)
        gain = 0.5 * (PiL + PiR)
        self.theta_L += self.k_eye_L * gain * dtheta
        self.theta_R -= self.k_eye_R * gain * dtheta

        # damping for eye motion
        self.theta_L *= 0.98
        self.theta_R *= 0.98

        F = 0.5 * (FL + FR)
        return F, self.theta_L.item(), self.theta_R.item(), PiL, PiR

# -------------------------------------------------------
# Simulation setup
# -------------------------------------------------------
T = 200
obj_traj = torch.cat([
    torch.linspace(-10, 0, T//2),
    torch.linspace(0, -10, T//2)
]).to(device)

# --- valence schedule ---
valence = torch.zeros(T)
valence[:T//3]  =  1.0   # reward → strong attention
valence[T//3:2*T//3] = 0.0
valence[2*T//3:] = -1.0  # threat → reduced precision

fish = ZebrafishVision(device=device)

F_hist, PiL_hist, PiR_hist, eyeL_hist, eyeR_hist = [], [], [], [], []
for t in range(T):
    F, eL, eR, PiL, PiR = fish.step(obj_traj[t], valence[t])
    F_hist.append(F)
    PiL_hist.append(PiL)
    PiR_hist.append(PiR)
    eyeL_hist.append(eL)
    eyeR_hist.append(eR)

# -------------------------------------------------------
# Plot results
# -------------------------------------------------------
plt.figure(figsize=(9,8))
plt.subplot(3,1,1)
plt.plot(F_hist)
plt.ylabel("Free-energy-like error")

plt.subplot(3,1,2)
plt.plot(PiL_hist,label="Precision L")
plt.plot(PiR_hist,label="Precision R")
plt.plot(valence*0.5+0.5,label="Valence (scaled)",linestyle="--",color="gray")
plt.legend(); plt.ylabel("Adaptive precision")

plt.subplot(3,1,3)
plt.plot(eyeL_hist,label="θ_L")
plt.plot(eyeR_hist,label="θ_R")
plt.plot(obj_traj.cpu(),label="Object",alpha=0.5)
plt.legend(); plt.xlabel("Time step"); plt.ylabel("Eye / Object position")

os.makedirs("plots",exist_ok=True)
plt.tight_layout()
plt.savefig("plots/pursuit_valence_precision.png")
print("[Saved] plots/pursuit_valence_precision.png")
