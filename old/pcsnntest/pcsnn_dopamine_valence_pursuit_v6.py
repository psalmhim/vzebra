# ---------------------------------------------------------------
# pcsnn_dopamine_valence_pursuit_v6.py
# ---------------------------------------------------------------
# Dual-retina predictive-coding SNN with shared dopamine system
# and emergent lateralized valence–precision competition
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------
# Retina Module
# ---------------------------------------------------------------
class Retina(nn.Module):
    def __init__(self, in_dim=28*28, hid_dim=64, eta=0.005, name="Retina"):
        super().__init__()
        self.W = nn.Parameter(0.1 * torch.randn(in_dim, hid_dim, device=device))
        self.Pi = torch.ones(in_dim, device=device) * 0.1
        self.eta = eta
        self.name = name
        self.step_count = 0

    def step(self, x, valence=0.0, rpe=0.0):
        x = x / (x.norm() + 1e-6)
        latent = x @ self.W
        pred = latent @ self.W.T
        err = x - pred

        # Adaptive precision
        self.Pi = torch.clamp(self.Pi + self.eta * (err.abs() - self.Pi), 0.02, 0.3)
        F_val = ((self.Pi * (err ** 2)).mean() / (x.var() + 1e-6)).item()

        # Dopamine & valence modulation
        mod = torch.exp(-torch.abs(torch.as_tensor(rpe + valence, device=x.device)))

        # Decaying learning rate
        lr = 0.01 * torch.exp(-0.001 * torch.tensor(self.step_count, device=x.device))
        self.step_count += 1

        dW = (x.T @ (err @ self.W)) * mod
        self.W.data += lr * dW

        return latent.detach(), F_val, self.Pi.mean().item()

# ---------------------------------------------------------------
# Shared Dopamine System
# ---------------------------------------------------------------
class DopamineSystem:
    def __init__(self, alpha=0.1, n_cat=10):
        self.values = torch.zeros(n_cat, device=device)
        self.alpha = alpha
        self.rpe = 0.0

    def update(self, reward, category):
        v = self.values[category]
        rpe = reward - v
        self.rpe = float(torch.clamp(rpe, -1.0, 1.0))
        self.values[category] = v + self.alpha * self.rpe

# ---------------------------------------------------------------
# Recognizer: shared classifier
# ---------------------------------------------------------------
class Recognizer(nn.Module):
    def __init__(self, hid_dim=64, n_class=10):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_class)
        self.to(device)

    def forward(self, x):
        return self.fc(x) / 2.0  # soft temperature

# ---------------------------------------------------------------
# Dual-hemisphere Agent
# ---------------------------------------------------------------
class DualRetinaAgent:
    def __init__(self, retinaL, retinaR, recognizer, dopamine):
        self.retinaL = retinaL
        self.retinaR = retinaR
        self.recognizer = recognizer
        self.dopa = dopamine
        self.eye = 0.0  # horizontal position (-1=L, +1=R)

    def perceive_and_act(self, x, label):
        x = x.to(device).view(1, -1)

        # Each retina perceives same stimulus, modulated by lateral bias
        xL = x * (1 - 0.5 * max(self.eye, 0))  # right bias dims left input
        xR = x * (1 - 0.5 * max(-self.eye, 0)) # left bias dims right input

        latentL, FL, PiL = self.retinaL.step(xL, valence=0.0, rpe=self.dopa.rpe)
        latentR, FR, PiR = self.retinaR.step(xR, valence=0.0, rpe=self.dopa.rpe)

        # Combine latent signals (weighted by precision)
        wL, wR = PiL / (PiL + PiR + 1e-6), PiR / (PiL + PiR + 1e-6)
        latent = (wL * latentL + wR * latentR)

        logits = self.recognizer(latent)
        pred = logits.argmax(1).item()

        # Reward and dopamine update
        reward = 0.5 if pred == label else -0.5
        self.dopa.update(reward, label)
        valence = torch.tanh(self.dopa.values[label]).item()

        # Lateral eye control: biased by relative precision
        delta_eye = 0.05 * torch.tanh(torch.tensor(PiR - PiL + valence)).item()
        self.eye += delta_eye
        self.eye += 0.01 * (-self.eye)
        self.eye = float(torch.clamp(torch.as_tensor(self.eye), -1.0, 1.0))

        align = torch.cos(torch.tensor(self.eye * 3.14 / 2)).item()

        return (FL+FR)/2, (PiL+PiR)/2, valence, self.dopa.rpe, pred, reward, self.eye, align, PiL, PiR

# ---------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------
def main():
    retinaL = Retina(name="Left").to(device)
    retinaR = Retina(name="Right").to(device)
    recognizer = Recognizer().to(device)
    dopamine = DopamineSystem()
    agent = DualRetinaAgent(retinaL, retinaR, recognizer, dopamine)

    steps = 300
    F_hist, Pi_hist, val_hist, rpe_hist, eye_hist = [], [], [], [], []
    PiL_hist, PiR_hist = [], []

    for step in range(steps):
        img = torch.rand(1, 28, 28, device=device)
        label = torch.randint(0, 10, (1,)).item()

        F, Pi, val, rpe, pred, rew, eye, align, PiL, PiR = agent.perceive_and_act(img, label)

        F_hist.append(F)
        Pi_hist.append(Pi)
        val_hist.append(val)
        rpe_hist.append(rpe)
        eye_hist.append(eye)
        PiL_hist.append(PiL)
        PiR_hist.append(PiR)

        if step % 50 == 0:
            print(f"Step {step:03d}: F={F:.3f} Pi={Pi:.3f} "
                  f"Val={val:+.2f} RPE={rpe:+.3f} Eye={eye:+.2f} "
                  f"PiL={PiL:.3f} PiR={PiR:.3f}")

    # ---------------------------------------------------------------
    # Plot dynamics
    # ---------------------------------------------------------------
    fig, axs = plt.subplots(5, 1, figsize=(8, 9))

    axs[0].plot(F_hist, label="Free Energy"); axs[0].legend()
    axs[0].set_ylabel("Free Energy")

    axs[1].plot(PiL_hist, label="Pi Left"); axs[1].plot(PiR_hist, label="Pi Right")
    axs[1].legend(); axs[1].set_ylabel("Precision (Left vs Right)")

    axs[2].plot(val_hist, label="Valence"); axs[2].plot(rpe_hist, label="RPE")
    axs[2].legend(); axs[2].set_ylabel("Valence / RPE")

    axs[3].plot(eye_hist, label="Eye pos")
    axs[3].axhline(0, color='gray', linestyle='--')
    axs[3].legend(); axs[3].set_ylabel("Eye position")

    axs[4].plot(PiR_hist, label="Right Precision")
    axs[4].plot(PiL_hist, label="Left Precision")
    axs[4].legend(); axs[4].set_ylabel("Precision trace"); axs[4].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig("plots/pursuit_dualretina_v6.png")
    print("[Saved] plots/pursuit_dualretina_v6.png")

    # ---------------------------------------------------------------
    # Plot category-specific valence
    # ---------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(range(10), dopamine.values.cpu().numpy())
    plt.xlabel("Digit category")
    plt.ylabel("Learned valence")
    plt.title("Category-specific dopaminergic valence (shared hemispheres)")
    plt.tight_layout()
    plt.savefig("plots/category_valence_dual_v6.png")
    print("[Saved] plots/category_valence_dual_v6.png")

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
