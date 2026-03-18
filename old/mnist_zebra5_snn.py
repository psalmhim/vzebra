import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ======================================================
# DEVICE
# ======================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ======================================================
# CONFIG
# ======================================================
class Config:
    img_size = 28
    n_in = 28 * 28           # retina → 784

    # module sizes
    n_teo = 400              # tectum
    n_prt = 200              # pretectum
    n_pal = 200              # pallium
    n_mot = 100              # motor
    n_out = 10

    # simulation timing
    dt = 1e-3
    T = 30

    # firing
    v_th_default = 0.10      # default threshold
    mot_v_th = 0.05          # LOWER threshold for MOTOR neurons

    v_reset = 0.0

    # decay constants
    tau_b = 1e-2
    tau_a = 2e-2
    tau_m = 2e-2

    # weight scales
    w_ff_scale = 0.15
    w_rec_scale = 0.05
    w_fb_scale = 0.08
    mot_ff_boost = 2.0       # BOOST MOTOR feedforward weights

    # delays
    delay_ff = 1
    delay_rec = 2
    delay_fb = 2
    max_delay = 3

    # plasticity (off)
    plastic = False
    eta_in = 5e-4
    pre_trace_tau = 5e-3

    # Poisson encoder
    poisson_max_rate = 2000.0  # FIX: higher rate → more spikes

    # training
    batch_size = 64
    lr_out = 1e-3
    epochs = 6


cfg = Config()


# ======================================================
# TWO-COMPARTMENT SNN MODULE
# ======================================================
class TwoCompModule(nn.Module):
    def __init__(self, pre_ff, pre_fb, n_post,
                 has_recurrent=True,
                 has_feedback=False,
                 plastic_ff=False,
                 v_th=None):

        super().__init__()

        self.pre_ff = pre_ff
        self.pre_fb = pre_fb
        self.n_post = n_post
        self.has_recurrent = has_recurrent
        self.has_feedback = has_feedback
        self.plastic_ff = plastic_ff

        # per-module threshold
        self.v_th = v_th if v_th is not None else cfg.v_th_default

        # weights
        self.W_ff = nn.Parameter(cfg.w_ff_scale *
                                 torch.randn(pre_ff, n_post),
                                 requires_grad=False)

        if has_recurrent:
            self.W_rec = nn.Parameter(cfg.w_rec_scale *
                                      torch.randn(n_post, n_post),
                                      requires_grad=False)

        if has_feedback:
            self.W_fb = nn.Parameter(cfg.w_fb_scale *
                                     torch.randn(pre_fb, n_post),
                                     requires_grad=False)

        # decays
        self.alpha_b = float(torch.exp(torch.tensor(-cfg.dt / cfg.tau_b)))
        self.alpha_a = float(torch.exp(torch.tensor(-cfg.dt / cfg.tau_a)))
        self.alpha_m = float(torch.exp(torch.tensor(-cfg.dt / cfg.tau_m)))
        self.alpha_pre = float(torch.exp(torch.tensor(-cfg.dt / cfg.pre_trace_tau)))

        self.reset_state(1)

    def reset_state(self, B):
        self.u = torch.zeros(B, self.n_post, device=device)
        self.u_b = torch.zeros(B, self.n_post, device=device)
        self.u_a = torch.zeros(B, self.n_post, device=device)
        self.spikes = torch.zeros(B, self.n_post, device=device)

        # delay buffers (CORRECT DIMENSIONS)
        self.hist_ff = torch.zeros(cfg.max_delay + 1, B, self.pre_ff, device=device)
        self.hist_rec = torch.zeros(cfg.max_delay + 1, B, self.n_post, device=device)
        self.hist_fb = torch.zeros(cfg.max_delay + 1, B, self.pre_fb, device=device)

        self.pre_trace = torch.zeros(B, self.pre_ff, device=device)

    def step(self, ff_spikes, fb_spikes=None):
        B = ff_spikes.size(0)

        # roll delays
        self.hist_ff = torch.roll(self.hist_ff, 1, dims=0)
        self.hist_rec = torch.roll(self.hist_rec, 1, dims=0)
        self.hist_fb = torch.roll(self.hist_fb, 1, dims=0)

        # insert new spikes
        self.hist_ff[0] = ff_spikes
        self.hist_rec[0] = self.spikes

        if fb_spikes is None:
            self.hist_fb[0].zero_()
        else:
            self.hist_fb[0] = fb_spikes

        # delayed
        ff_del = self.hist_ff[cfg.delay_ff]
        rec_del = self.hist_rec[cfg.delay_rec]
        fb_del = self.hist_fb[cfg.delay_fb]

        # compute currents
        I_b = ff_del @ self.W_ff
        if self.has_recurrent:
            I_b += rec_del @ self.W_rec

        I_a = torch.zeros_like(I_b)
        if self.has_feedback:
            I_a += fb_del @ self.W_fb

        # integrate dendrites
        self.u_b = self.alpha_b * self.u_b + (1 - self.alpha_b) * I_b
        self.u_a = self.alpha_a * self.u_a + (1 - self.alpha_a) * I_a

        # soma
        u_hat = self.u_b + self.u_a
        self.u = self.alpha_m * self.u + (1 - self.alpha_m) * u_hat

        # spike
        s = (self.u >= self.v_th).float()
        self.spikes = s
        self.u = torch.where(s > 0,
                             torch.full_like(self.u, cfg.v_reset),
                             self.u)

        # plasticity (off)
        return s


# ======================================================
# FULL ZEBRA-5 NETWORK
# ======================================================
class Zebra5SNN(nn.Module):
    def __init__(self):
        super().__init__()

        # TECTUM (feedback from Pallium)
        self.teo = TwoCompModule(
            pre_ff=cfg.n_in,
            pre_fb=cfg.n_pal,
            n_post=cfg.n_teo,
            has_recurrent=True,
            has_feedback=True,
            plastic_ff=False)

        # PRETECTUM
        self.prt = TwoCompModule(
            pre_ff=cfg.n_teo,
            pre_fb=0,
            n_post=cfg.n_prt,
            has_recurrent=True,
            has_feedback=False,
            plastic_ff=False)

        # PALLIUM
        self.pal = TwoCompModule(
            pre_ff=cfg.n_prt,
            pre_fb=0,
            n_post=cfg.n_pal,
            has_recurrent=True,
            has_feedback=False,
            plastic_ff=False)

        # MOTOR — LOWER THRESHOLD + STRONGER INPUT
        self.mot = TwoCompModule(
            pre_ff=cfg.n_pal,
            pre_fb=0,
            n_post=cfg.n_mot,
            has_recurrent=True,
            has_feedback=False,
            v_th=cfg.mot_v_th,     # key fix
            plastic_ff=False)

        # Stronger motor weights
        with torch.no_grad():
            self.mot.W_ff *= cfg.mot_ff_boost

        # Readout
        self.W_out = nn.Parameter(0.2 * torch.randn(cfg.n_mot, cfg.n_out))
        self.b_out = nn.Parameter(torch.zeros(cfg.n_out))

    def reset_state(self, B):
        self.teo.reset_state(B)
        self.prt.reset_state(B)
        self.pal.reset_state(B)
        self.mot.reset_state(B)

    def forward(self, img):
        B = img.size(0)
        self.reset_state(B)

        p = poisson_encode(img)
        logits_acc = torch.zeros(B, cfg.n_out, device=device)

        for t in range(cfg.T):
            x = torch.bernoulli(p)

            teo_spk = self.teo.step(x, fb_spikes=self.pal.spikes)
            prt_spk = self.prt.step(teo_spk)
            pal_spk = self.pal.step(prt_spk)
            mot_spk = self.mot.step(pal_spk)

            logits_acc += mot_spk @ self.W_out + self.b_out

        # spike diagnostics
        #print("TeO:", float(self.teo.spikes.mean()),
        #      "PAL:", float(self.pal.spikes.mean()),
        #      "MOT:", float(self.mot.spikes.mean()))

        return logits_acc / cfg.T


# ======================================================
# POISSON ENCODER
# ======================================================
def poisson_encode(images):
    B = images.size(0)
    x = images.view(B, -1)
    rates = x * cfg.poisson_max_rate
    return torch.clamp(rates * cfg.dt, 0, 1)


# ======================================================
# TRAINING
# ======================================================
def train_epoch(model, loader, optim):
    tot_loss, tot_corr, tot_n = 0, 0, 0
    model.train()

    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)

        logits = model(img)
        loss = F.cross_entropy(logits, lbl)

        optim.zero_grad()
        loss.backward()
        optim.step()

        tot_loss += loss.item() * lbl.size(0)
        tot_corr += (logits.argmax(1) == lbl).sum().item()
        tot_n += lbl.size(0)

    return tot_loss / tot_n, tot_corr / tot_n


def test_epoch(model, loader):
    tot_loss, tot_corr, tot_n = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)

            logits = model(img)
            loss = F.cross_entropy(logits, lbl)

            tot_loss += loss.item() * lbl.size(0)
            tot_corr += (logits.argmax(1) == lbl).sum().item()
            tot_n += lbl.size(0)

    return tot_loss / tot_n, tot_corr / tot_n


# ======================================================
# MAIN
# ======================================================
def main():
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = Zebra5SNN().to(device)
    optim = torch.optim.Adam([model.W_out, model.b_out], lr=cfg.lr_out)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim)
        te_loss, te_acc = test_epoch(model, test_loader)

        print(f"Epoch {epoch}: train_acc={tr_acc*100:.2f}%, test_acc={te_acc*100:.2f}%")


if __name__ == "__main__":
    main()
