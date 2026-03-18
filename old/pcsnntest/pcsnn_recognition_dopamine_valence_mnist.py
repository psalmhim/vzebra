import torch, torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt, os, random

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# Retina (visual predictive layer with precision modulation)
# -------------------------------------------------------
class Retina(torch.nn.Module):
    def __init__(self, n_in=784, n_out=64, tau=0.9, eta=0.01,
                 pi_min=0.01, pi_max=2.0, beta_valence=0.5, device="cpu"):
        super().__init__()
        self.W_enc = torch.randn(n_in, n_out, device=device) * 0.05
        self.W_dec = torch.randn(n_out, n_in, device=device) * 0.05
        self.v_lat = torch.zeros(n_out, device=device)
        self.Pi = torch.ones(n_in, device=device) * 0.1
        self.tau, self.eta = tau, eta
        self.pi_min, self.pi_max = pi_min, pi_max
        self.beta = beta_valence
        self.device = device

    def step(self, x, valence=0.0):
        # Encode input into latent representation
        z = torch.tanh(x @ self.W_enc)  # [1, n_out]
        # Decode back to reconstruct the input
        pred = z @ self.W_dec           # [1, n_in]
        # Compute reconstruction error in input space
        err = (x - pred).view(-1)
        abs_err = err.abs()
        rel = abs_err / (abs_err.mean() + 1e-6)

        target = (1 + self.beta * valence) * rel
        delta = self.eta * (target - self.Pi)
        self.Pi = torch.clamp(self.Pi + delta, self.pi_min, self.pi_max)

        # Precision-weighted update (like predictive coding gain)
        weighted_err = torch.tanh(self.Pi) * err
        F = (self.Pi * err**2).mean().item()

        # Update latent state slowly (integrator)
        self.v_lat = self.tau * self.v_lat + (1 - self.tau) * z.squeeze()
        return self.v_lat, F, self.Pi.mean().item()


# -------------------------------------------------------
# Simple Pallium-like classifier (recognition network)
# -------------------------------------------------------
class Recognizer(torch.nn.Module):
    def __init__(self, n_in=64, n_classes=10):
        super().__init__()
        self.fc = torch.nn.Linear(n_in, n_classes)
    def forward(self, x):
        return self.fc(x)

# -------------------------------------------------------
# Dopaminergic valence learning
# -------------------------------------------------------
class DopamineSystem:
    def __init__(self, n_categories=10, lr=0.05, decay=0.995):
        self.valence = torch.zeros(n_categories)
        self.lr = lr; self.decay = decay
        self.rpe = 0.0
    def update(self, category, reward):
        v = self.valence[category]
        delta = reward - v
        self.valence[category] = v + self.lr * delta
        self.valence *= self.decay
        self.rpe = delta
        return float(self.valence[category]), float(delta)

# -------------------------------------------------------
# Zebrafish perception–action–valence loop
# -------------------------------------------------------
class ZebrafishAgent:
    def __init__(self, device="cpu"):
        self.retina = Retina(device=device)
        self.recognizer = Recognizer().to(device)
        self.dopa = DopamineSystem()
        self.theta_L = 0.0; self.theta_R = 0.0
        self.k_eye = 0.02

    def perceive_and_act(self, img, label):
        x = img.view(1, -1).to(device)
        # Retrieve category valence
        v_cat = float(self.dopa.valence[label])
        retinal_act, F, Pi = self.retina.step(x, valence=v_cat)
        # Classify object
        logits = self.recognizer(retinal_act)
        pred = logits.argmax().item()

        # --- define reward ---
        # approach good (0–2), neutral (3–6), flee bad (7–9)
        if pred <= 2:  reward = 0.5
        elif pred >= 7: reward = -0.5
        else: reward = 0.0

        # --- dopaminergic learning ---
        val, rpe = self.dopa.update(pred, reward)

        # --- simple motor adaptation (eye motion) ---
        motor_signal = self.k_eye * val
        self.theta_L += motor_signal; self.theta_R -= motor_signal

        return F, Pi, val, rpe, pred, reward

# -------------------------------------------------------
# Simulation
# -------------------------------------------------------
transform = transforms.ToTensor()
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

agent = ZebrafishAgent(device=device)
F_hist, Pi_hist, Val_hist, RPE_hist, Reward_hist = [], [], [], [], []

for step, (img, label) in enumerate(loader):
    F, Pi, val, rpe, pred, reward = agent.perceive_and_act(img, label.item())
    F_hist.append(F); Pi_hist.append(Pi)
    Val_hist.append(val); RPE_hist.append(rpe); Reward_hist.append(reward)
    if step > 300: break

# -------------------------------------------------------
# Plot
# -------------------------------------------------------
plt.figure(figsize=(10,9))
plt.subplot(4,1,1)
plt.plot(F_hist); plt.ylabel("Free-energy-like error")

plt.subplot(4,1,2)
plt.plot(Pi_hist,label="Precision")
plt.plot(Val_hist,label="Valence",linestyle="--")
plt.legend(); plt.ylabel("Precision / Valence")

plt.subplot(4,1,3)
plt.plot(RPE_hist,label="RPE (dopamine)")
plt.legend(); plt.ylabel("Reward prediction error")

plt.subplot(4,1,4)
plt.plot(Reward_hist,label="Reward",color="gray")
plt.legend(); plt.xlabel("Sample #"); plt.ylabel("Reward signal")

os.makedirs("plots", exist_ok=True)
plt.tight_layout()
plt.savefig("plots/recognition_dopamine_valence_mnist.png")
print("[Saved] plots/recognition_dopamine_valence_mnist.png")
