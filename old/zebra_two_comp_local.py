import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==============================================================
#  Surrogate gradient for spikes
# ==============================================================
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        out = (v > 0).float()
        ctx.save_for_backward(v)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        grad = 0.3 * torch.clamp(1 - v.abs(), min=0)
        return grad_output * grad
spike_fn = SpikeFn.apply


# ==============================================================
#  Two-compartment predictive neuron layer
# ==============================================================
class TwoCompLayer(nn.Module):
    """
    Basal + apical → soma predictive neuron
    Local learning: ΔW_ff ∝ (v_s − v_b) · x_pre
    """
    def __init__(self, n_in, n_out, tau=0.8, fb_dim=None,
                 use_feedback=False, device="cpu",
                 lr=3e-4, alpha=0.05, fb_gain=1.0):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.device = device
        self.tau = tau
        self.lr = lr
        self.alpha = alpha
        self.fb_gain = fb_gain

        # feedforward weights
        self.W_ff = nn.Parameter(0.1 * torch.randn(n_in, n_out, device=device))

        # optional feedback transformation
        self.use_feedback = use_feedback
        if use_feedback and fb_dim is not None:
            self.W_fb = nn.Parameter(0.1 * torch.randn(fb_dim, n_out, device=device))
        else:
            self.W_fb = None

        # neuron states
        self.v_b = torch.zeros(1, n_out, device=device)
        self.v_a = torch.zeros(1, n_out, device=device)
        self.v_s = torch.zeros(1, n_out, device=device)
        self.spikes = torch.zeros(1, n_out, device=device)

        # filtered traces for local plasticity
        self.rbar_pre = torch.zeros(1, n_in, device=device)
        self.ebar_post = torch.zeros(1, n_out, device=device)

    def reset_state(self, B):
        dev = self.device
        self.v_b = torch.zeros(B, self.n_out, device=dev)
        self.v_a = torch.zeros(B, self.n_out, device=dev)
        self.v_s = torch.zeros(B, self.n_out, device=dev)
        self.spikes = torch.zeros(B, self.n_out, device=dev)
        self.rbar_pre = torch.zeros(B, self.n_in, device=dev)
        self.ebar_post = torch.zeros(B, self.n_out, device=dev)

    def step(self, x, fb=None):
        # ----- Basal dendrite -----
        pred_b = x @ self.W_ff
        self.v_b = self.tau * self.v_b + (1 - self.tau) * pred_b

        # ----- Apical dendrite -----
        if self.use_feedback and fb is not None:
            if self.W_fb is not None:
                pred_a = self.fb_gain * (fb @ self.W_fb)
            else:
                # direct feedback (teaching signal)
                pred_a = self.fb_gain * fb
        else:
            pred_a = torch.zeros_like(self.v_a)
        self.v_a = self.tau * self.v_a + (1 - self.tau) * pred_a

        # ----- Soma -----
        self.v_s = self.tau * self.v_s + (1 - self.tau) * (self.v_b - self.v_a)
        self.spikes = spike_fn(self.v_s)
        mismatch = self.v_s - self.v_b

        # ----- Local learning -----
        self.rbar_pre  = (1 - self.alpha) * self.rbar_pre  + self.alpha * x
        self.ebar_post = (1 - self.alpha) * self.ebar_post + self.alpha * mismatch
        dW = self.rbar_pre.transpose(1, 0) @ self.ebar_post / x.size(0)
        self.W_ff.data += self.lr * dW

        return self.spikes


# ==============================================================
#  Zebrafish predictive SNN
# ==============================================================
class ZebrafishSNN(nn.Module):
    """
    Retina (784) → TeO (400) → PrT (300) → Pallium (200)
    → Motor (50) → Action (3)
    Feedback: Pallium→TeO, plus apical teaching at Pallium.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        # perception pathway
        self.teo = TwoCompLayer(784, 400, tau=0.8, device=device,
                                use_feedback=True, fb_dim=200)
        self.prt = TwoCompLayer(400, 300, tau=0.8, device=device)
        # top layer with apical teaching
        self.pal = TwoCompLayer(300, 200, tau=0.8, device=device,
                                use_feedback=True, fb_gain=0.5)
        self.mot = TwoCompLayer(200, 50, tau=0.8, device=device)

        # linear readouts
        self.action_head = nn.Linear(50, 3)
        self.perception_head = nn.Linear(200, 10)

    def reset_state(self, B):
        self.teo.reset_state(B)
        self.prt.reset_state(B)
        self.pal.reset_state(B)
        self.mot.reset_state(B)
        self.prev_pal = torch.zeros(B, 200, device=self.device)

    def forward(self, x, labels=None, T=10):
        B = x.size(0)
        self.reset_state(B)

        for t in range(T):
            fb_teo = self.prev_pal
            teo_spk = self.teo.step(x, fb=fb_teo)
            prt_spk = self.prt.step(teo_spk)

            # ---- Pallium with apical teaching ----
            if labels is not None:
                logits = self.perception_head(self.pal.v_s)
                err = F.one_hot(labels, num_classes=10).float() - F.softmax(logits, dim=1)
                fb_top = err @ self.perception_head.weight  # (B,10)·(10,200)→(B,200)
            else:
                fb_top = None

            pal_spk = self.pal.step(prt_spk, fb=fb_top)
            self.prev_pal = pal_spk.detach()
            mot_spk = self.mot.step(pal_spk)

        digit_logits = self.perception_head(self.pal.v_s)
        action_logits = self.action_head(self.mot.v_s)
        return digit_logits, action_logits


# ==============================================================
#  Utilities
# ==============================================================
def mnist_to_action(y):
    act = torch.zeros_like(y)
    act[(y >= 0) & (y <= 3)] = 0
    act[(y >= 4) & (y <= 7)] = 1
    act[(y >= 8)] = 2
    return act


def train_local_with_teaching(model, loader, device, epochs=5):
    model.train()
    for ep in range(epochs):
        for imgs, labels in loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            labels = labels.to(device)
            _ = model(imgs, labels=labels)   # apical teaching active
        print(f"[Local+Teaching] Epoch {ep+1} complete.")


def train_readout(model, loader, device, epochs=3):
    opt = torch.optim.Adam(
        list(model.perception_head.parameters()) +
        list(model.action_head.parameters()), lr=1e-3)
    for ep in range(epochs):
        acc, total = 0, 0
        for imgs, labels in loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            labels = labels.to(device)
            actions = mnist_to_action(labels).to(device)
            z_digit, z_act = model(imgs)
            loss = F.cross_entropy(z_digit, labels) + 0.3 * F.cross_entropy(z_act, actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc += (z_digit.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"[Readout] Epoch {ep+1}: train_acc={(acc/total)*100:.2f}%")


def test(model, loader, device):
    model.eval()
    acc, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            labels = labels.to(device)
            digit_logits, _ = model(imgs)
            pred = digit_logits.argmax(1)
            acc += (pred == labels).sum().item()
            total += labels.size(0)
    print(f"Test accuracy = {acc/total*100:.2f}%")
    return acc / total


# ==============================================================
#  Main
# ==============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    transform = transforms.ToTensor()
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)

    model = ZebrafishSNN(device=device).to(device)

    # Phase 1 – local learning with apical teaching
    train_local_with_teaching(model, train_loader, device, epochs=5)

    # Phase 2 – optional readout fine-tuning
    train_readout(model, train_loader, device, epochs=3)

    test(model, test_loader, device)


if __name__ == "__main__":
    main()
