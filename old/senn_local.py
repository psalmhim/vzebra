import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================================================
#  Senn 2023 Local Spiking Predictive Layer
# =========================================================
class SennLocalLayer(nn.Module):
    def __init__(self, in_dim, out_dim, tau_mem=20.0, tau_e=100.0, lr=1e-3):
        super().__init__()
        self.W = nn.Parameter(0.1 * torch.randn(in_dim, out_dim))
        self.v = torch.zeros(out_dim)
        self.tau_mem = tau_mem
        self.tau_e = tau_e
        self.lr = lr
        self.register_buffer("elig", torch.zeros(in_dim, out_dim))

    def forward(self, x):
        device = x.device
        self.v = self.v.to(device)
        self.elig = self.elig.to(device)

        I = x @ self.W
        self.v = self.v * (1 - 1 / self.tau_mem) + I.mean(0)
        rho = torch.sigmoid(self.v)
        z = torch.bernoulli(rho)
        delta = (z - rho).detach()

        # --- batch-aware eligibility update ---
        delta_batched = delta.unsqueeze(0).expand(x.size(0), -1)  # (B, out_dim)
        self.elig = (1 - 1 / self.tau_e) * self.elig + x.T @ delta_batched
        # --------------------------------------

        self.W.data += self.lr * self.elig / x.size(0)
        return z, rho




# =========================================================
#  Multi-layer Local Predictive SNN
# =========================================================
class LocalSNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1 = SennLocalLayer(784, 400)
        self.l2 = SennLocalLayer(400, 200)
        self.l3 = SennLocalLayer(200, 100)
        self.readout = nn.Linear(100, 10).to(device)

    def forward_local(self, x):
        z1, _ = self.l1(x)
        z2, _ = self.l2(z1.unsqueeze(0))
        z3, _ = self.l3(z2.unsqueeze(0))
        return z3

    def forward(self, x):
        z3 = self.forward_local(x)
        return self.readout(z3)


# =========================================================
#  Training utilities
# =========================================================
def train_local(model, loader, device, epochs=5):
    model.train()
    for ep in range(epochs):
        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)
            _ = model.forward_local(x)
        print(f"[Local unsupervised] Epoch {ep+1} done.")


def train_readout(model, loader, device, epochs=3):
    opt = torch.optim.Adam(model.readout.parameters(), lr=0.001)
    for ep in range(epochs):
        acc, total = 0, 0
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            z3 = model.forward_local(x)
            logits = model.readout(z3)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            acc += (pred == y).sum().item()
            total += y.size(0)
        print(f"[Readout] Ep {ep+1}: train acc={(acc/total)*100:.2f}%")


def test(model, loader, device):
    model.eval()
    acc, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            z3 = model.forward_local(x)
            logits = model.readout(z3)
            pred = logits.argmax(1)
            acc += (pred == y).sum().item()
            total += y.size(0)
    print(f"Test accuracy = {(acc/total)*100:.2f}%")
    return acc / total


# =========================================================
#  Main routine
# =========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = transforms.ToTensor()
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = LocalSNN(device).to(device)
    print("Model initialized.")

    # ------------------ Local unsupervised learning ------------------
    train_local(model, train_loader, device, epochs=10)

    # ------------------ Train readout (supervised) -------------------
    train_readout(model, train_loader, device, epochs=3)

    # ------------------ Evaluate ------------------------------------
    test(model, test_loader, device)


if __name__ == "__main__":
    main()
