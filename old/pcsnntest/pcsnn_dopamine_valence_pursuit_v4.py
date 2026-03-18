# pcsnn_dopamine_valence_pursuit_v5.py
# ---------------------------------------------------------------
# Predictive-coding SNN with category-specific dopaminergic valence
# and homeostatic eye pursuit control
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------
# Retina: predictive-coding sensory layer
# ---------------------------------------------------------------
class Retina(nn.Module):
    def __init__(self, in_dim=28*28, hid_dim=64, eta=0.005):
        super().__init__()
        self.W = nn.Parameter(0.1 * torch.randn(in_dim, hid_dim, device=device))
        self.Pi = torch.ones(in_dim, device=device) * 0.1
        self.eta = eta
        self.step_count = 0

    def step(self, x, valence=0.0, rpe=0.0):
        x = x / (x.norm() + 1e-6)
        latent = x @ self.W
        pred = latent @ self.W.T
        err = x - pred

        # Adaptive precision (bounded)
        self.Pi = torch.clamp(
            self.Pi + self.eta * (err.abs() - self.Pi),
            0.02, 0.3
        )

        # Free-energy-like cost
        F_val = ((self.Pi * (err ** 2)).mean() / (x.var() + 1e-6)).item()

        # Dopamine + valence modulation
        modulator = torch.exp(-torch.abs(torch.tensor(rpe + valence, device=x.device)))

        # Decaying learning rate
        lr = 0.01 * torch.exp(-0.001 * torch.tensor(self.step_count, device=x.device))
        self.step_count += 1

        # Hebbian update
        dW = (x.T @ (err @ self.W)) * modulator
        self.W.data += lr * dW

        return latent.detach(), F_val, self.Pi.mean().item()

# ---------------------------------------------------------------
# Dopamine system with category-specific valence
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
# Recognizer: maps latent code → categorical prediction
# ---------------------------------------------------------------
class Recognizer(nn.Module):
    def __init__(self, hid_dim=64, n_class=10):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_class)
        self.to(device)

    def forward(self, x):
        logits = self.fc(x) / 2.0
        return logits

# ---------------------------------------------------------------
# Agent: integrates Retina + Dopamine + Recognizer + Motor control
# ---------------------------------------------------------------
class Agent:
    def __init__(self, retina, recognizer, dopamine):
        self.retina = retina
        self.recognizer = recognizer
        self.dopa = dopamine
        self.eye = 0.0

    def perceive_and_act(self, x, label):
        x = x.to(device).view(1, -1)
        latent, F_val, Pi_val = self.retina.step(x, valence=0.0, rpe=self.dopa.rpe)

        logits = self.recognizer(latent)
        pred = logits.argmax(1).item()

        reward = 0.5 if pred == label else -0.5
        self.dopa.update(reward, label)
        valence = torch.tanh(self.dopa.values[label]).item()

        # Eye control with homeostatic decay
        self.eye += 0.05 * (torch.sign(torch.tensor(valence)) - self.eye)
        self.eye += 0.01 * (-self.eye)
        self.eye = float(torch.clamp(torch.as_tensor(self.eye), -1.0, 1.0))

        align = torch.cos(torch.tensor(self.eye * 3.14 / 2)).item()

        return F_val, Pi_val, valence, self.dopa.rpe, pred, reward, self.eye, align

# ---------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------
def main():
    retina = Retina().to(device)
    recognizer = Recognizer().to(device)
    dopa = DopamineSystem()
    agent = Agent(retina, recognizer, dopa)

    steps = 300
    F_hist, Pi_hist, val_hist, rpe_hist, eye_hist = [], [], [], [], []

    for step in range(steps):
        img = torch.rand(1, 28, 28, device=device)
        label = torch.randint(0, 10, (1,)).item()

        F, Pi, val, rpe, pred, rew, eye, align = agent.perceive_and_act(img, label)

        F_hist.append(F)
        Pi_hist.append(Pi)
        val_hist.append(val)
        rpe_hist.append(rpe)
        eye_hist.append(eye)

        if step % 50 == 0:
            print(f"Step {step:03d}: F={F:.3f} Pi={Pi:.3f} "
                  f"Val={val:+.2f} RPE={rpe:+.3f} Pred={pred} Rew={rew:+.1f} "
                  f"Eye={eye:+.2f} Align={align:+.2f}")

    # ---------------------------------------------------------------
    # Plot temporal dynamics
    # ---------------------------------------------------------------
    fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    axs[0].plot(F_hist, label="Free Energy")
    axs[0].set_ylabel("Free Energy"); axs[0].legend()

    axs[1].plot(Pi_hist, label="Precision (Pi)")
    axs[1].set_ylabel("Precision"); axs[1].legend()

    axs[2].plot(val_hist, label="Valence")
    axs[2].plot(rpe_hist, label="RPE")
    axs[2].set_ylabel("Valence / RPE"); axs[2].legend()

    axs[3].plot(eye_hist)
    axs[3].set_ylabel("Eye position"); axs[3].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig("plots/pursuit_dynamics_v5.png")
    print("[Saved] plots/pursuit_dynamics_v5.png")

    # ---------------------------------------------------------------
    # Plot category-specific valence map
    # ---------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(range(10), dopa.values.cpu().numpy())
    plt.xlabel("Digit category"); plt.ylabel("Learned valence")
    plt.title("Category-specific dopaminergic valence")
    plt.tight_layout()
    plt.savefig("plots/category_valence_v5.png")
    print("[Saved] plots/category_valence_v5.png")

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
