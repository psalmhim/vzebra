# ===============================================================
# pcsnn_dopamine_valence_competitive_v2.py
# ===============================================================
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt, os, random

# ===============================================================
# Retina: predictive coding with dopaminergic precision modulation
# ===============================================================
class Retina(nn.Module):
    def __init__(self, n_in=784, n_lat=64, eta=0.01, tau=0.9,
                 pi_min=0.01, pi_max=2.0, beta_val=0.5,
                 curiosity_decay=0.999, device="cpu"):
        super().__init__()
        self.device = device
        self.W_enc = nn.Parameter(torch.randn(n_in, n_lat, device=device)*0.05)
        self.W_dec = nn.Parameter(torch.randn(n_lat, n_in, device=device)*0.05)
        self.Pi = torch.ones(n_in, device=device)*0.1
        self.v_lat = torch.zeros(n_lat, device=device)
        self.eta = eta; self.tau = tau
        self.pi_min, self.pi_max = pi_min, pi_max
        self.beta_val = beta_val
        self.curiosity_decay = curiosity_decay

    def step(self, x, valence=0.0, rpe=0.0):
        rpe_t = torch.tensor(rpe, device=self.device)
        z = torch.tanh(x @ self.W_enc)
        pred = z @ self.W_dec
        err = (x - pred).view(-1)
        abs_err = err.abs()
        rel = abs_err / (abs_err.mean() + 1e-6)

        target = (1 + self.beta_val * valence) * rel
        delta = self.eta * (target - self.Pi)
        delta *= torch.exp(-torch.abs(rpe_t))
        self.Pi = torch.clamp(self.Pi * self.curiosity_decay + delta,
                              self.pi_min, self.pi_max)

        F_val = (self.Pi * err**2).mean().item()
        self.v_lat = self.tau * self.v_lat + (1 - self.tau) * z.squeeze()
        return self.v_lat.unsqueeze(0), F_val, self.Pi.mean().item()

# ===============================================================
# Dopamine system with decay, exploration noise, and RPE learning
# ===============================================================
class DopamineSystem:
    def __init__(self, n_cat=10, lr_val=0.02, decay=0.995):
        self.valence = torch.zeros(n_cat)
        self.rpe = 0.0
        self.lr_val = lr_val
        self.decay = decay

    def update(self, pred, reward):
        self.valence *= self.decay  # homeostatic decay
        val = self.valence[pred].item()
        rpe = reward - val
        self.valence[pred] += self.lr_val * rpe
        self.rpe = float(rpe)
        return val, self.rpe

# ===============================================================
# Simple recognizer network
# ===============================================================
class Recognizer(nn.Module):
    def __init__(self, n_lat=64, n_out=10):
        super().__init__()
        self.fc = nn.Linear(n_lat, n_out)
    def forward(self, x): return self.fc(x)

# ===============================================================
# Motor control (eye bias)
# ===============================================================
class MotorSystem:
    def __init__(self, k_eye=0.1):
        self.k_eye = k_eye
        self.eye = 0.0
    def update(self, diff_val):
        self.eye += self.k_eye * diff_val
        self.eye = float(torch.tanh(torch.tensor(self.eye)))  # smooth limit
        return self.eye

# ===============================================================
# Combine left and right visual input
# ===============================================================
def combine_inputs(imgL, imgR, eye_bias):
    bias_L = max(0, 1 - eye_bias)
    bias_R = max(0, 1 + eye_bias)
    return (bias_L * imgL + bias_R * imgR) / (bias_L + bias_R + 1e-6)

# ===============================================================
# Zebrafish agent integrating dual retina, dopamine, and motor
# ===============================================================
class ZebrafishAgent:
    def __init__(self, device="cpu"):
        self.device = device
        self.retinaL = Retina(device=device)
        self.retinaR = Retina(device=device)
        self.recognizer = Recognizer().to(device)
        self.dopa = DopamineSystem()
        self.motor = MotorSystem()

    def perceive_and_act(self, imgL, labL, imgR, labR):
        xL = imgL.view(1, -1).to(self.device)
        xR = imgR.view(1, -1).to(self.device)

        # valence per seen label
        vL_cat = float(self.dopa.valence[labL]) if labL < len(self.dopa.valence) else 0.0
        vR_cat = float(self.dopa.valence[labR]) if labR < len(self.dopa.valence) else 0.0

        # retina predictive coding
        rL, F_L, Pi_L = self.retinaL.step(xL, valence=vL_cat, rpe=self.dopa.rpe)
        rR, F_R, Pi_R = self.retinaR.step(xR, valence=vR_cat, rpe=self.dopa.rpe)

        # combine fields
        fused = combine_inputs(rL, rR, self.motor.eye)
        logits = self.recognizer(fused)
        pred = logits.argmax().item()

        # reward mapping by category
        if pred in [0,1,2]:
            reward = 0.5
        elif pred in [7,8,9]:
            reward = -0.5
        else:
            reward = -0.1

        val, rpe = self.dopa.update(pred, reward)
        self.dopa.rpe += 0.05 * random.uniform(-1,1)  # exploration noise

        # compute relative valence bias
        diff_val = (vR_cat - vL_cat) + 0.2 * (Pi_R - Pi_L)
        eye_bias = self.motor.update(diff_val)

        return F_L + F_R, (Pi_L + Pi_R)/2, val, rpe, pred, reward, eye_bias, vL_cat, vR_cat

# ===============================================================
# Main simulation
# ===============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    transform = transforms.ToTensor()
    loader = DataLoader(datasets.MNIST("./data", train=True, download=True,
                                       transform=transform),
                        batch_size=2, shuffle=True)

    agent = ZebrafishAgent(device=device)
    os.makedirs("plots", exist_ok=True)

    Fs, Pis, Vals, RPEs, Eyes = [], [], [], [], []

    for step, (imgs, labels) in enumerate(loader):
        imgL, imgR = imgs[0], imgs[1]
        labL, labR = labels[0].item(), labels[1].item()

        F, Pi, val, rpe, pred, rew, eye_bias, vL_cat, vR_cat = \
            agent.perceive_and_act(imgL, labL, imgR, labR)

        if step % 50 == 0:
            print(f"Step {step:03d}: F={F:.3f} Pi={Pi:.3f} Val={val:+.2f} "
                  f"RPE={rpe:+.3f} Pred={pred} Rew={rew:+.1f} Eye={eye_bias:+.2f} "
                  f"(vL={vL_cat:+.2f}, vR={vR_cat:+.2f})")
        Fs.append(F); Pis.append(Pi); Vals.append(val); RPEs.append(rpe); Eyes.append(eye_bias)
        if step >= 300: break

    # ----------------- Visualization -----------------
    plt.figure(figsize=(8,6))
    plt.subplot(4,1,1); plt.plot(Fs); plt.ylabel("Free Energy")
    plt.subplot(4,1,2); plt.plot(Pis); plt.ylabel("Precision (Pi)")
    plt.subplot(4,1,3); plt.plot(Vals,label="Valence"); plt.plot(RPEs,label="RPE"); plt.legend()
    plt.subplot(4,1,4); plt.plot(Eyes,label="Eye bias"); plt.ylabel("Eye"); plt.xlabel("Step")
    plt.tight_layout(); plt.savefig("plots/dopamine_competitive_v2.png")
    print("[Saved] plots/dopamine_competitive_v2.png")

    # learned category valences
    plt.figure(figsize=(6,4))
    plt.bar(range(10), agent.dopa.valence.numpy())
    plt.xlabel("Digit"); plt.ylabel("Learned valence")
    plt.title("Category-specific dopaminergic valence (v2)")
    plt.savefig("plots/category_valence_v2.png")
    print("[Saved] plots/category_valence_v2.png")

if __name__ == "__main__":
    main()
