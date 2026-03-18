import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==============================================================
# Utility: Surrogate gradient for spikes
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
# Basic LIF neuron with recurrent + feedback + feedforward
# ==============================================================
class LIFLayer(nn.Module):
    def __init__(self, n_in, n_out, tau=0.8, dt=1.0,
                 use_recurrent=False,
                 use_feedback=False,
                 fb_dim=None,
                 device="cpu"):
        super().__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        
        # Membrane decay
        self.alpha = torch.tensor(tau, device=device)
        self.dt = dt
        
        # Trainable feedforward
        self.W_ff = nn.Parameter(0.1 * torch.randn(n_in, n_out, device=device))
        
        # Optional recurrent
        self.use_recurrent = use_recurrent
        if use_recurrent:
            self.W_rec = nn.Parameter(0.1 * torch.randn(n_out, n_out, device=device))
        
        # Optional feedback
        self.use_feedback = use_feedback
        if use_feedback:
            assert fb_dim is not None
            self.W_fb = nn.Parameter(0.1 * torch.randn(fb_dim, n_out, device=device))
        
        # Neuron state
        self.v = torch.zeros(1, n_out, device=device)
        self.spikes = torch.zeros(1, n_out, device=device)
        
    def reset_state(self, batch_size):
        self.v = torch.zeros(batch_size, self.n_out, device=self.device)
        self.spikes = torch.zeros(batch_size, self.n_out, device=self.device)
    
    def step(self, x, fb=None):
        """
        x: (B, n_in)
        fb: (B, fb_dim) or None
        """
        I = x @ self.W_ff   # feedforward input
        
        if self.use_recurrent:
            I = I + (self.spikes @ self.W_rec)
        
        if self.use_feedback and fb is not None:
            I = I + (fb @ self.W_fb)
        
        # membrane update
        self.v = self.alpha * self.v + I
        
        # spike
        self.spikes = spike_fn(self.v)
        
        # reset
        self.v = self.v - self.spikes
        
        return self.spikes


# ==============================================================
# Perception–Action zebrafish architecture
# ==============================================================
class ZebrafishSNN(nn.Module):
    """
    Retina (784) → TeO (400) → PrT (300) → Pallium (200)
    
    Heads:
        - Digit Head: 10-way classifier
        - Action Head: 3-way behavior classifier
        
    Feedback:
        Pallium spikes (t-1) → TeO (t)
    """
    
    def __init__(self, device="cpu"):
        super().__init__()
        
        self.device = device
        
        # ----------------------------------------------------------
        # Visual pathway (perception)
        # ----------------------------------------------------------
        self.teo = LIFLayer(784, 400, tau=0.8, device=device,
                            use_recurrent=False,
                            use_feedback=True,
                            fb_dim=200)

        self.prt = LIFLayer(400, 300, tau=0.8, device=device,
                            use_recurrent=False)

        self.pal = LIFLayer(300, 200, tau=0.8, device=device,
                            use_recurrent=False)
        
        # ----------------------------------------------------------
        # Motor / action pathway
        # ----------------------------------------------------------
        self.mot = LIFLayer(200, 50, tau=0.8, device=device,
                            use_recurrent=False)
        
        # Action classifier (3-way)
        self.action_head = nn.Linear(50, 3)
        
        # Digit classifier (10-way)
        self.perception_head = nn.Linear(200, 10)
    
    def reset_state(self, batch_size):
        self.teo.reset_state(batch_size)
        self.prt.reset_state(batch_size)
        self.pal.reset_state(batch_size)
        self.mot.reset_state(batch_size)
        self.prev_pal_spikes = torch.zeros(batch_size, 200, device=self.device)
    
    def forward(self, x):
        """
        x: (B, 784)
        Simulate multiple SNN steps (T=10 by default)
        Output:
            digit_logits, action_logits
        """
        B = x.size(0)
        self.reset_state(B)
        
        T = 10  # number of SNN time steps
        
        for t in range(T):
            # previous-timestep feedback (biologically realistic)
            fb = self.prev_pal_spikes
            
            teo_spk = self.teo.step(x, fb=fb)
            prt_spk = self.prt.step(teo_spk)
            pal_spk = self.pal.step(prt_spk)
            
            # update feedback state
            self.prev_pal_spikes = pal_spk.detach()
            
            # motor → action
            mot_spk = self.mot.step(pal_spk)
        
        # Final readouts (use last spikes)
        digit_logits = self.perception_head(pal_spk)
        action_logits = self.action_head(mot_spk)
        
        return digit_logits, action_logits


# ==============================================================
# TRAINING LOOP
# ==============================================================
def mnist_to_action(y):
    """
    Map MNIST labels to:
        0–3 → 0 (indifferent)
        4–7 → 1 (approach)
        8–9 → 2 (flee)
    """
    action = torch.zeros_like(y)
    action[(y >= 0) & (y <= 3)] = 0
    action[(y >= 4) & (y <= 7)] = 1
    action[(y >= 8)] = 2
    return action


def train(model, loader, opt, device, beta=0.3):
    model.train()
    total_correct = 0
    total_samples = 0
    
    for imgs, labels in loader:
        imgs = imgs.view(imgs.size(0), -1).to(device)
        labels = labels.to(device)
        actions = mnist_to_action(labels).to(device)
        
        opt.zero_grad()
        
        digit_logits, action_logits = model(imgs)
        
        loss_digit = F.cross_entropy(digit_logits, labels)
        loss_action = F.cross_entropy(action_logits, actions)
        
        loss = loss_digit + beta * loss_action
        loss.backward()
        
        opt.step()
        
        # accuracy
        pred = digit_logits.argmax(1)
        total_correct += (pred == labels).sum().item()
        total_samples += labels.size(0)
    
    return total_correct / total_samples


def test(model, loader, device):
    model.eval()
    total_correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            labels = labels.to(device)
            
            digit_logits, _ = model(imgs)
            pred = digit_logits.argmax(1)
            total_correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return total_correct / total


# ==============================================================
# Main
# ==============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("Using device:", device)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)
    
    model = ZebrafishSNN(device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(1, 11):
        acc = train(model, train_loader, opt, device)
        test_acc = test(model, test_loader, device)
        print(f"Epoch {epoch}: train_acc={acc*100:.2f}%, test_acc={test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
