"""
Zebrafish-style PC-SNN with dopamine-modulated predictive coding
and category-specific valence learning on MNIST.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# Dopaminergic module
# ============================================================

class DopamineSystem:
    def __init__(self, n_cat=10, alpha=0.05, gamma=0.9):
        self.valence = torch.zeros(n_cat)
        self.rpe = 0.0
        self.alpha = alpha
        self.gamma = gamma

    def update(self, pred, reward):
        v_old = self.valence[pred]
        self.rpe = reward - v_old
        self.valence[pred] += self.alpha * self.rpe
        self.valence = torch.clamp(self.valence, -1, 1)
        return self.valence[pred].item(), self.rpe


# ============================================================
# Retina: Predictive Coding with Dopaminergic Plasticity
# ============================================================

class Retina(torch.nn.Module):
    def __init__(self, n_in=784, n_out=64,
                 tau=0.9, eta=0.01,
                 pi_min=0.01, pi_max=2.0,
                 beta_valence=0.5,
                 lr_dopa=0.001, device="cpu"):
        super().__init__()
        self.device = device
        self.n_in, self.n_out = n_in, n_out

        # Encoder–Decoder weights
        self.W_enc = torch.randn(n_in, n_out, device=device) * 0.05
        self.W_dec = torch.randn(n_out, n_in, device=device) * 0.05

        # Latent activity & precision
        self.v_lat = torch.zeros(n_out, device=device)
        self.Pi = torch.ones(n_in, device=device) * 0.1

        # Parameters
        self.tau, self.eta = tau, eta
        self.pi_min, self.pi_max = pi_min, pi_max
        self.beta = beta_valence
        self.lr_dopa = lr_dopa

    def step(self, x, valence=0.0, rpe=0.0):
        """Predictive coding retina with dopamine-modulated plasticity."""
        # 1. Encode → latent z
        z = torch.tanh(x @ self.W_enc)             # [1, n_out]

        # 2. Decode → reconstruct x̂
        pred = z @ self.W_dec                      # [1, n_in]

        # 3. Compute sensory prediction error
        err = (x - pred).view(-1)
        abs_err = err.abs()
        rel = abs_err / (abs_err.mean() + 1e-6)

        # 4. Update precision (attention)
        target = (1 + self.beta * valence) * rel
        delta = self.eta * (target - self.Pi)
        if not isinstance(rpe, torch.Tensor):
            rpe_t = torch.tensor(rpe, device=self.device, dtype=torch.float32)
        else:
            rpe_t = rpe.to(self.device)

        delta *= torch.exp(-torch.abs(rpe_t))      # dopamine dampens volatility

        self.Pi = torch.clamp(self.Pi + delta, self.pi_min, self.pi_max)

        # 5. Free-energy-like term
        F = (self.Pi * err**2).mean().item()

        # 6. Latent state update
        self.v_lat = self.tau * self.v_lat + (1 - self.tau) * z.squeeze()

        # 7. Dopamine-modulated bidirectional plasticity
        if abs(rpe) > 1e-5:
            dW_enc = self.lr_dopa * rpe * (x.T @ z)
            dW_dec = self.lr_dopa * rpe * (z.T @ err.view(1, -1))
            self.W_enc += dW_enc.clamp(-0.05, 0.05)
            self.W_dec += dW_dec.clamp(-0.05, 0.05)

        return self.v_lat, F, self.Pi.mean().item()


# ============================================================
# Simple Recognizer (Classifier)
# ============================================================

class Recognizer(torch.nn.Module):
    def __init__(self, n_in=64, n_class=10):
        super().__init__()
        self.fc = torch.nn.Linear(n_in, n_class)

    def forward(self, x):
        return self.fc(x)


# ============================================================
# Zebrafish Agent
# ============================================================

class ZebrafishAgent:
    def __init__(self, retina, recognizer, dopa, k_eye=0.05):
        self.retina = retina
        self.recognizer = recognizer
        self.dopa = dopa
        self.theta_L = 0.0
        self.theta_R = 0.0
        self.k_eye = k_eye

    def perceive_and_act(self, img, label):
        x = img.view(1, -1).to(device)

        v_cat = float(self.dopa.valence[label])

        # Retina: predictive coding + dopaminergic plasticity
        retinal_act, F, Pi = self.retina.step(x, valence=v_cat, rpe=self.dopa.rpe)

        # Recognition
        logits = self.recognizer(retinal_act)
        pred = logits.argmax().item()

        # Reward by digit category
        if pred <= 2:
            reward = 0.5
        elif pred >= 7:
            reward = -0.5
        else:
            reward = 0.0

        # Dopaminergic update
        val, rpe = self.dopa.update(pred, reward)

        # Motor adaptation
        motor_signal = self.k_eye * val
        self.theta_L += motor_signal
        self.theta_R -= motor_signal

        return F, Pi, val, rpe, pred, reward


# ============================================================
# Setup dataset and training loop
# ============================================================

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(mnist, batch_size=1, shuffle=True)

retina = Retina(device=device)
recognizer = Recognizer().to(device)
dopa = DopamineSystem()
agent = ZebrafishAgent(retina, recognizer, dopa)
print("Retina device:", next(retina.parameters(), torch.zeros(1, device=device)).device)
print("Recognizer device:", next(recognizer.parameters()).device)

F_hist, Pi_hist, val_hist, rpe_hist, rew_hist = [], [], [], [], []

os.makedirs("plots", exist_ok=True)
n_iter = 300

for i, (img, label) in enumerate(loader):
    if i >= n_iter: break
    F, Pi, val, rpe, pred, reward = agent.perceive_and_act(img, label.item())

    F_hist.append(F)
    Pi_hist.append(Pi)
    val_hist.append(val)
    rpe_hist.append(rpe)
    rew_hist.append(reward)

    if (i + 1) % 50 == 0:
        print(f"Step {i+1:03d}: F={F:.3f}, Pi={Pi:.3f}, Val={val:.2f}, RPE={rpe:.3f}, Pred={pred}, Rew={reward}")

        # Save visualizations of evolving encoder weights
        W_show = agent.retina.W_enc.detach().cpu().numpy().T[:16]  # show first 16 features
        fig, axes = plt.subplots(4, 4, figsize=(4, 4))
        for j, ax in enumerate(axes.flat):
            ax.imshow(W_show[j].reshape(28, 28), cmap="bwr", vmin=-0.1, vmax=0.1)
            ax.axis("off")
        plt.suptitle(f"Encoder weights at step {i+1}")
        plt.tight_layout()
        plt.savefig(f"plots/encoder_step_{i+1:03d}.png")
        plt.close(fig)

# ============================================================
# Plot learning curves
# ============================================================

plt.figure(figsize=(8, 10))
plt.subplot(4,1,1)
plt.plot(F_hist)
plt.ylabel("Free-energy-like error")

plt.subplot(4,1,2)
plt.plot(Pi_hist, label="Precision")
plt.plot(val_hist, '--', label="Valence")
plt.legend()
plt.ylabel("Precision / Valence")

plt.subplot(4,1,3)
plt.plot(rpe_hist, label="RPE (dopamine)")
plt.legend()
plt.ylabel("Reward prediction error")

plt.subplot(4,1,4)
plt.plot(rew_hist, color='gray', label="Reward")
plt.legend()
plt.xlabel("Sample #")
plt.ylabel("Reward signal")

plt.tight_layout()
plt.savefig("plots/dopamine_valence_learning.png")
plt.show()
