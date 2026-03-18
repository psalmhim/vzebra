# pcsnn_pursuit_dopamine_valence.py
import torch, matplotlib.pyplot as plt, os
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Retina with learned valence-modulated precision
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

        target = (1 + self.beta * valence) * rel
        delta  = self.eta * (target - self.Pi)
        self.Pi = torch.clamp(self.Pi + delta, self.pi_min, self.pi_max)

        # Nonlinear bounded gain
        weighted_err = torch.tanh(self.Pi) * err
        self.v_b = self.tau * self.v_b + (1 - self.tau) * pred
        self.v_s = self.tau * self.v_s + (1 - self.tau) * weighted_err
        F = (self.Pi * err**2).mean().item()
        return self.v_s.mean(), F, self.Pi.mean().item()

# -------------------------------------------------------
# Dopamine / valence module
# -------------------------------------------------------
class ValenceSystem:
    def __init__(self, lr=0.05, decay=0.995):
        self.v = 0.0    # expected value
        self.lr = lr    # learning rate (dopaminergic plasticity)
        self.decay = decay

    def update(self, reward):
        # Reward-prediction error δ = r - v
        delta = reward - self.v
        self.v += self.lr * delta
        self.v *= self.decay  # mild forgetting
        return self.v, delta

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

    def step(self, obj_pos, valence):
        xL = torch.tensor([[obj_pos - self.theta_L.item()]], device=device)
        xR = torch.tensor([[obj_pos - self.theta_R.item()]], device=device)
        rL, FL, PiL = self.eyeL.step(xL, valence)
        rR, FR, PiR = self.eyeR.step(xR, valence)

        dtheta = (rR - rL)
        gain = 0.5 * (PiL + PiR)
        self.theta_L += self.k_eye_L * gain * dtheta
        self.theta_R -= self.k_eye_R * gain * dtheta

        # damping for realism
        self.theta_L *= 0.98
        self.theta_R *= 0.98
        F = 0.5 * (FL + FR)
        return F, self.theta_L.item(), self.theta_R.item(), PiL, PiR

# -------------------------------------------------------
# Simulation
# -------------------------------------------------------
T = 200
obj_traj = torch.cat([
    torch.linspace(-10, 0, T//2),
    torch.linspace(0, -10, T//2)
]).to(device)

fish = ZebrafishVision(device=device)
dopamine = ValenceSystem()

F_hist, PiL_hist, PiR_hist, eyeL_hist, eyeR_hist, val_hist, rpe_hist = [], [], [], [], [], [], []

for t in range(T):
    # Reward based on proximity of eyes to object (better centering → higher reward)
    eye_center = 0.5*(fish.theta_L + fish.theta_R)
    err_to_obj = torch.abs(obj_traj[t] - eye_center)
    reward = float(torch.exp(-0.1 * err_to_obj)) - 0.5  # between -0.5 and +0.5

    valence, rpe = dopamine.update(reward)
    F, eL, eR, PiL, PiR = fish.step(obj_traj[t], valence)

    F_hist.append(F)
    PiL_hist.append(PiL)
    PiR_hist.append(PiR)
    eyeL_hist.append(eL)
    eyeR_hist.append(eR)
    val_hist.append(valence)
    rpe_hist.append(rpe)

# -------------------------------------------------------
# Plot
# -------------------------------------------------------
plt.figure(figsize=(9,9))
plt.subplot(4,1,1)
plt.plot(F_hist); plt.ylabel("Free-energy-like error")

plt.subplot(4,1,2)
plt.plot(PiL_hist,label="Precision L")
plt.plot(PiR_hist,label="Precision R")
plt.plot(val_hist,label="Valence",linestyle="--",color="gray")
plt.legend(); plt.ylabel("Adaptive precision / valence")

plt.subplot(4,1,3)
plt.plot(rpe_hist,label="RPE (dopamine)")
plt.legend(); plt.ylabel("Reward-prediction error")

plt.subplot(4,1,4)
plt.plot(eyeL_hist,label="θ_L")
plt.plot(eyeR_hist,label="θ_R")
plt.plot(obj_traj.cpu(),label="Object",alpha=0.5)
plt.legend(); plt.xlabel("Time step"); plt.ylabel("Eye / Object position")

os.makedirs("plots",exist_ok=True)
plt.tight_layout()
plt.savefig("plots/pursuit_dopamine_valence.png")
print("[Saved] plots/pursuit_dopamine_valence.png")
