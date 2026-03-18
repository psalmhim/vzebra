import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------------------
# Device
# ----------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ----------------------------------
# Config
# ----------------------------------
class Config:
    img_size = 28
    n_in = img_size * img_size
    n_hidden = 512
    n_out = 10

    dt = 1e-3
    T = 30

    tau_m = 2e-2
    tau_b = 1e-2
    tau_a = 2e-2

    v_th = 0.2
    v_reset = 0.0

    # weights
    w_in_scale = 0.25
    w_rec_b_scale = 0.05
    w_rec_a_scale = 0.04
    w_out_scale = 0.2

    # delays
    delay_in = 1
    delay_rec_b = 2
    delay_rec_a = 3
    max_delay = 3

    batch_size = 64
    epochs = 6
    lr_out = 1e-3

    poisson_max_rate = 200.0  # Hz


cfg = Config()

class TwoCompRecurrentSNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.W_in = nn.Parameter(
            cfg.w_in_scale * torch.randn(cfg.n_in, cfg.n_hidden),
            requires_grad=False
        )
        self.W_rec_b = nn.Parameter(
            cfg.w_rec_b_scale * torch.randn(cfg.n_hidden, cfg.n_hidden),
            requires_grad=False
        )
        self.W_rec_a = nn.Parameter(
            cfg.w_rec_a_scale * torch.randn(cfg.n_hidden, cfg.n_hidden),
            requires_grad=False
        )

        self.W_out = nn.Parameter(
            cfg.w_out_scale * torch.randn(cfg.n_hidden, cfg.n_out),
            requires_grad=True
        )
        self.b_out = nn.Parameter(torch.zeros(cfg.n_out), requires_grad=True)

        # decays
        self.alpha_b = float(torch.exp(torch.tensor(-cfg.dt / cfg.tau_b)))
        self.alpha_a = float(torch.exp(torch.tensor(-cfg.dt / cfg.tau_a)))
        self.alpha_m = float(torch.exp(torch.tensor(-cfg.dt / cfg.tau_m)))

        # Senn learning hyperparameters
        self.eta_in = 5e-4
        self.pre_trace_tau = 5e-3
        self.alpha_pre = float(torch.exp(torch.tensor(-cfg.dt / self.pre_trace_tau)))

        # states
        self.u = None
        self.u_b = None
        self.u_a = None
        self.spikes = None
        self.hist_in = None
        self.hist_hid = None
        self.pre_trace = None

    def reset_state(self, B):
        c = self.cfg
        self.u = torch.zeros(B, c.n_hidden, device=device)
        self.u_b = torch.zeros(B, c.n_hidden, device=device)
        self.u_a = torch.zeros(B, c.n_hidden, device=device)
        self.spikes = torch.zeros(B, c.n_hidden, device=device)

        self.hist_in  = torch.zeros(c.max_delay+1, B, c.n_in, device=device)
        self.hist_hid = torch.zeros(c.max_delay+1, B, c.n_hidden, device=device)

        self.pre_trace = torch.zeros(B, c.n_in, device=device)

    def step(self, x_spikes):
        c = self.cfg

        # shift delay lines
        self.hist_in  = torch.roll(self.hist_in, 1, dims=0)
        self.hist_hid = torch.roll(self.hist_hid, 1, dims=0)
        self.hist_in[0] = x_spikes
        self.hist_hid[0] = self.spikes

        x_del   = self.hist_in[c.delay_in]
        h_del_b = self.hist_hid[c.delay_rec_b]
        h_del_a = self.hist_hid[c.delay_rec_a]

        # dendritic inputs
        I_b = x_del @ self.W_in + h_del_b @ self.W_rec_b
        I_a = h_del_a @ self.W_rec_a

        # dendritic integration
        self.u_b = self.alpha_b * self.u_b + (1 - self.alpha_b) * I_b
        self.u_a = self.alpha_a * self.u_a + (1 - self.alpha_a) * I_a

        # somatic
        u_hat = self.u_b + self.u_a
        self.u = self.alpha_m * self.u + (1 - self.alpha_m) * u_hat

        # spikes
        s = (self.u >= c.v_th).float()
        self.u = torch.where(s > 0, torch.full_like(self.u, c.v_reset), self.u)
        self.spikes = s

        # -------------------------
        # Senn plasticity update
        # -------------------------

        # update pre-trace
        self.pre_trace = self.alpha_pre * self.pre_trace + x_del

        # dendritic mismatch (local error)
        dend_mismatch = self.u_a - self.u_b   # B × H

        # batch-averaged local error
        dend_err = dend_mismatch.mean(dim=0)   # H
        pre_avg  = self.pre_trace.mean(dim=0)  # I

        # local plasticity update for W_in
        with torch.no_grad():
            dW = torch.outer(pre_avg, dend_err)
            self.W_in += self.eta_in * dW

        # readout logits
        return self.spikes @ self.W_out + self.b_out


# ----------------------------------
# Two-Compartment SNN with Recurrence
# ----------------------------------
class TwoCompRecurrentSNN1(nn.Module):
    """
    - Basal input: Poisson input + recurrent basal (delayed)
    - Apical input: feedback recurrent (delayed)
    - Soma integrates dendritic predictions
    - Spikes emitted when threshold crossed
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Feedforward
        self.W_in = nn.Parameter(
            cfg.w_in_scale * torch.randn(cfg.n_in, cfg.n_hidden),
            requires_grad=False
        )

        # Recurrent basal (lateral)
        self.W_rec_b = nn.Parameter(
            cfg.w_rec_b_scale * torch.randn(cfg.n_hidden, cfg.n_hidden),
            requires_grad=False
        )

        # Recurrent apical (feedback)
        self.W_rec_a = nn.Parameter(
            cfg.w_rec_a_scale * torch.randn(cfg.n_hidden, cfg.n_hidden),
            requires_grad=False
        )

        # Readout (only thing trained)
        self.W_out = nn.Parameter(
            cfg.w_out_scale * torch.randn(cfg.n_hidden, cfg.n_out),
            requires_grad=True
        )
        self.b_out = nn.Parameter(torch.zeros(cfg.n_out), requires_grad=True)

        # Decay factors
        self.alpha_b = float(torch.exp(torch.tensor(-cfg.dt / cfg.tau_b)))
        self.alpha_a = float(torch.exp(torch.tensor(-cfg.dt / cfg.tau_a)))
        self.alpha_m = float(torch.exp(torch.tensor(-cfg.dt / cfg.tau_m)))

        # State variables
        self.u = None
        self.u_b = None
        self.u_a = None
        self.spikes = None

        # Delay lines for input + recurrent
        self.hist_in = None
        self.hist_hid = None

    def reset_state(self, B: int):
        c = self.cfg
        self.u      = torch.zeros(B, c.n_hidden, device=device)
        self.u_b    = torch.zeros(B, c.n_hidden, device=device)
        self.u_a    = torch.zeros(B, c.n_hidden, device=device)
        self.spikes = torch.zeros(B, c.n_hidden, device=device)

        # history buffers (steps × batch × neurons)
        self.hist_in  = torch.zeros(c.max_delay + 1, B, c.n_in, device=device)
        self.hist_hid = torch.zeros(c.max_delay + 1, B, c.n_hidden, device=device)

    def step(self, x_spikes):
        c = self.cfg

        # update delay buffers
        self.hist_in  = torch.roll(self.hist_in,  shifts=1, dims=0)
        self.hist_hid = torch.roll(self.hist_hid, shifts=1, dims=0)
        self.hist_in[0]  = x_spikes
        self.hist_hid[0] = self.spikes

        # delayed inputs
        x_del     = self.hist_in[c.delay_in]      # B × n_in
        h_del_b   = self.hist_hid[c.delay_rec_b]  # B × n_hidden
        h_del_a   = self.hist_hid[c.delay_rec_a]  # B × n_hidden

        # dendritic integration
        I_b = x_del @ self.W_in     + h_del_b @ self.W_rec_b
        I_a =                0.0    + h_del_a @ self.W_rec_a

        self.u_b = self.alpha_b * self.u_b + (1 - self.alpha_b) * I_b
        self.u_a = self.alpha_a * self.u_a + (1 - self.alpha_a) * I_a

        u_hat = self.u_b + self.u_a

        # soma update
        self.u = self.alpha_m * self.u + (1 - self.alpha_m) * u_hat

        # spike generation
        s = (self.u >= c.v_th).float()
        self.u = torch.where(s > 0, torch.full_like(self.u, c.v_reset), self.u)
        self.spikes = s

        # readout activation
        return self.spikes @ self.W_out + self.b_out


# ----------------------------------
# Poisson encoding
# ----------------------------------
def poisson_encode(images, cfg: Config):
    B = images.size(0)
    x = images.view(B, -1)
    rates = x * cfg.poisson_max_rate
    p = rates * cfg.dt
    return torch.clamp(p, 0.0, 1.0)


# ----------------------------------
# Train & Eval
# ----------------------------------
def train_epoch(model, loader, optimizer, cfg):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_spikes = 0
    total_neurons = 0

    for img, lbl in loader:
        img = img.to(device)
        lbl = lbl.to(device)
        B = img.size(0)

        model.reset_state(B)
        p = poisson_encode(img, cfg)
        logits_acc = torch.zeros(B, cfg.n_out, device=device)

        for _ in range(cfg.T):
            x_spk = torch.bernoulli(p)
            logits_t = model.step(x_spk)
            logits_acc += logits_t

            total_spikes += model.spikes.sum().item()
            total_neurons += model.spikes.numel()

        logits = logits_acc / cfg.T
        loss = F.cross_entropy(logits, lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == lbl).sum().item()
        total_samples += B

    mean_fr = total_spikes / (total_neurons + 1e-9)
    return total_loss / total_samples, total_correct / total_samples, mean_fr


def eval_epoch(model, loader, cfg):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(device)
            lbl = lbl.to(device)
            B = img.size(0)

            model.reset_state(B)
            p = poisson_encode(img, cfg)
            logits_acc = torch.zeros(B, cfg.n_out, device=device)

            for _ in range(cfg.T):
                x_spk = torch.bernoulli(p)
                logits_t = model.step(x_spk)
                logits_acc += logits_t

            logits = logits_acc / cfg.T
            loss = F.cross_entropy(logits, lbl)

            total_loss += loss.item() * B
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == lbl).sum().item()
            total_samples += B

    return total_loss / total_samples, total_correct / total_samples


# ----------------------------------
# Main
# ----------------------------------
def main():
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    model = TwoCompRecurrentSNN(cfg).to(device)
    optimizer = torch.optim.Adam([model.W_out, model.b_out], lr=cfg.lr_out)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, fr = train_epoch(model, train_loader, optimizer, cfg)
        te_loss, te_acc = eval_epoch(model, test_loader, cfg)
        print(f"Epoch {epoch}: train_acc={tr_acc*100:.2f}%, test_acc={te_acc*100:.2f}%, spike_prob={fr:.3f}")


if __name__ == "__main__":
    main()
