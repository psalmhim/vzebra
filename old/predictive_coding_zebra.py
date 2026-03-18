import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Two-compartment predictive coding neuron (Senn/Urbanczik-Senn)
# ============================================================

class TwoCompPCNeuron(nn.Module):
    def __init__(self, n_in, n_out, n_fb,
                 tau_b=0.9, tau_a=0.9, eta=1e-5, device="cpu"):
        super().__init__()
        self.device = device
        self.n_in  = n_in
        self.n_out = n_out
        self.n_fb  = n_fb
        self.eta   = eta

        self.W_ff = nn.Parameter(0.05 * torch.randn(n_in, n_out, device=device))
        self.W_fb = nn.Parameter(0.05 * torch.randn(n_fb, n_out, device=device))

        self.v_b = None
        self.v_a = None
        self.v   = None
        self.spk = None

        self.tau_b = tau_b
        self.tau_a = tau_a

    def reset_state(self, batch):
        self.v_b = torch.zeros(batch, self.n_out, device=self.device)
        self.v_a = torch.zeros(batch, self.n_out, device=self.device)
        self.v   = torch.zeros(batch, self.n_out, device=self.device)
        self.spk = torch.zeros(batch, self.n_out, device=self.device)

    def forward(self, ff_input, fb_input):
        # Basal (feedforward)
        I_b = ff_input @ self.W_ff
        self.v_b = self.tau_b * self.v_b + (1 - self.tau_b) * I_b
        self.v_b.clamp_(-10, 10)

        # Apical (feedback) scaled for stability
        I_a = (fb_input @ self.W_fb) * 0.05
        self.v_a = self.tau_a * self.v_a + (1 - self.tau_a) * I_a
        self.v_a.clamp_(-10, 10)

        # Soma
        self.v = (self.v_b - self.v_a).clamp(-10, 10)

        # Stable spike function
        self.spk = torch.sigmoid(self.v.clamp(-5, 5))

        # Local error
        error = self.v_b - self.v_a

        # Local normalized Hebbian update
        dW_ff = ff_input.t() @ error
        dW_fb = fb_input.t() @ error

        dW_ff = dW_ff / (1e-3 + dW_ff.norm())
        dW_fb = dW_fb / (1e-3 + dW_fb.norm())

        self.W_ff.data += self.eta * dW_ff
        self.W_fb.data -= self.eta * dW_fb

        # Clamp weights to avoid explosion
        self.W_ff.data.clamp_(-0.2, 0.2)
        self.W_fb.data.clamp_(-0.2, 0.2)

        return self.spk, error



# ============================================================
# Predictive Coding Visual Hierarchy (TeO → PrT → Pallium)
# ============================================================

class ZebraPCNet(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        self.teo = TwoCompPCNeuron(784, 400, n_fb=200, device=device)
        self.prt = TwoCompPCNeuron(400, 300, n_fb=200, device=device)
        self.pal = TwoCompPCNeuron(300, 200, n_fb=1,   device=device)

        # Supervised output head
        self.readout = nn.Linear(200, 10).to(device)

    def reset(self, batch):
        self.teo.reset_state(batch)
        self.prt.reset_state(batch)
        self.pal.reset_state(batch)

    def forward_pc(self, x, T=6):
        B = x.size(0)
        self.reset(B)

        fb = torch.zeros(B, 200, device=self.device)

        for _ in range(T):
            teo_spk, _ = self.teo(x, fb)
            prt_spk, _ = self.prt(teo_spk, fb)
            pal_spk, _ = self.pal(prt_spk, torch.zeros(B, 1, device=self.device))

            # delayed stable feedback
            fb = pal_spk.detach()

        return pal_spk


    def forward(self, x):
        pal_spk = self.forward_pc(x)
        return self.readout(pal_spk)


# ============================================================
# SUPERVISED TRAINING FUNCTIONS (only train readout head)
# ============================================================

def train_supervised(model, loader, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0

    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (logits.argmax(1) == y).sum().item()

    return loss_sum / total, 100 * correct / total


def test_supervised(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()

    return loss_sum / total, 100 * correct / total


# ============================================================
# MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tf = transforms.Compose([transforms.ToTensor()])

    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    mnist_test  = datasets.MNIST("./data", train=False, transform=tf)

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader  = DataLoader(mnist_test,  batch_size=128)

    model = ZebraPCNet(device=device)

    # ============================================================
    # (PHASE 1) UNSUPERVISED PREDICTIVE CODING PRETRAINING
    # ============================================================
    print("Phase 1: Unsupervised PC pretraining...")

    for epoch in range(5):
        for x, _ in train_loader:
            x = x.view(x.size(0), -1).to(device)
            model.forward_pc(x)  # only local learning
        print(f"PC Epoch {epoch+1} done.")

    with torch.no_grad():
        print("TeO | FF:", model.teo.W_ff.norm().item(),
            "FB:", model.teo.W_fb.norm().item())

        print("PrT | FF:", model.prt.W_ff.norm().item(),
            "FB:", model.prt.W_fb.norm().item())

        print("Pal | FF:", model.pal.W_ff.norm().item(),
            "FB:", model.pal.W_fb.norm().item())

        # Optional: measure mean spikes
        sample = next(iter(train_loader))[0][:64].view(64, -1).to(device)
        pal_spk = model.forward_pc(sample)
        print("Mean spikes → TeO:", model.teo.spk.mean().item(),
            "PrT:", model.prt.spk.mean().item(),
            "Pal:", model.pal.spk.mean().item())

    # ============================================================
    # PHASE 2: FREEZE PC layers (TeO, PrT, Pallium)
    # ============================================================
    for p in model.teo.parameters(): p.requires_grad = False
    for p in model.prt.parameters(): p.requires_grad = False
    for p in model.pal.parameters(): p.requires_grad = False

    # ============================================================
    # (PHASE 3) SUPERVISED TRAINING OF READOUT
    # ============================================================
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=1e-3)

    print("Phase 2: Supervised training begins...")

    for epoch in range(1, 6):
        tr_loss, tr_acc = train_supervised(model, train_loader, optimizer, device)
        te_loss, te_acc = test_supervised(model, test_loader, device)

        print(f"Epoch {epoch}: "
              f"train_acc={tr_acc:.2f}%, test_acc={te_acc:.2f}%")
