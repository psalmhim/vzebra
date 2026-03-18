import argparse, os, math, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===============================================================
# Surrogate spike
# ===============================================================
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        out = (v > 0).float(); ctx.save_for_backward(v)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        grad = 0.3 * torch.clamp(1 - v.abs(), min=0)
        return grad_output * grad
spike_fn = SpikeFn.apply

# ===============================================================
# Two-compartment neuron
# ===============================================================
class TwoCompLayer(nn.Module):
    def __init__(self, n_in, n_out, tau=0.8, fb_dim=None, use_feedback=False,
                 device="cpu", hebb_lr=1e-5, alpha=0.05, mu_anchor=1e-4):
        super().__init__()
        self.device = device
        self.tau = tau
        self.use_feedback = use_feedback
        self.W_ff = nn.Parameter(0.1 * torch.randn(n_in, n_out, device=device))
        self.W_fb = nn.Parameter(0.1 * torch.randn(fb_dim, n_out, device=device)) \
                    if use_feedback and fb_dim is not None else None
        self.hebb_lr, self.alpha, self.mu_anchor = hebb_lr, alpha, mu_anchor
        self.W_ff_anchor = None; self.plastic = False
        self.reset_state(1)

    def reset_state(self, B):
        dev = self.device; out = self.W_ff.shape[1]
        self.v_b = torch.zeros(B, out, device=dev)
        self.v_a = torch.zeros(B, out, device=dev)
        self.v_s = torch.zeros(B, out, device=dev)
        self.spikes = torch.zeros(B, out, device=dev)
        self.rbar_pre  = torch.zeros(B, self.W_ff.shape[0], device=dev)
        self.ebar_post = torch.zeros(B, out, device=dev)
        self.spike_sum = torch.zeros(B, out, device=dev)

    def set_plastic(self, flag: bool): self.plastic = bool(flag)

    def step(self, x, fb=None):
        pred_b = x @ self.W_ff
        self.v_b = self.tau * self.v_b + (1 - self.tau) * pred_b
        if self.use_feedback and fb is not None:
            pred_a = fb @ self.W_fb if self.W_fb is not None else fb
        else:
            pred_a = torch.zeros_like(self.v_a)
        self.v_a = self.tau * self.v_a + (1 - self.tau) * pred_a
        self.v_s = self.tau * self.v_s + (1 - self.tau) * (self.v_b - self.v_a)
        self.spikes = spike_fn(self.v_s)
        self.spike_sum += self.spikes
        if self.plastic:
            mismatch = self.v_s - self.v_b
            self.rbar_pre  = (1 - self.alpha) * self.rbar_pre  + self.alpha * x
            self.ebar_post = (1 - self.alpha) * self.ebar_post + self.alpha * mismatch
            dW = self.rbar_pre.transpose(1, 0) @ self.ebar_post / x.size(0)
            with torch.no_grad():
                self.W_ff += self.hebb_lr * dW
                if self.W_ff_anchor is not None:
                    self.W_ff -= self.mu_anchor * (self.W_ff - self.W_ff_anchor)
        return self.spikes

# ===============================================================
# Zebrafish predictive SNN
# ===============================================================
class ZebrafishSNN(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        # perception
        self.teo = TwoCompLayer(784, 400, tau=0.8, device=device, use_feedback=True, fb_dim=200)
        self.prt = TwoCompLayer(400, 300, tau=0.8, device=device)
        self.pal = TwoCompLayer(300, 200, tau=0.8, device=device, use_feedback=True)
        # motor
        self.mot = TwoCompLayer(200, 50, tau=0.8, device=device)
        # heads
        self.readout_digit = nn.Linear(200, 10)
        self.readout_action = nn.Linear(50, 3)
        # fixed random feedbacks
        self.register_buffer("R_fb_digit",  torch.randn(10, 200) * 0.1)
        self.register_buffer("R_fb_action", torch.randn(3,  50)  * 0.1)

    def reset_state(self, B):
        for L in [self.teo, self.prt, self.pal, self.mot]:
            L.reset_state(B)
        self.prev_pal = torch.zeros(B, 200, device=self.device)

    def set_plastic(self, flag: bool):
        for L in [self.teo, self.prt, self.pal, self.mot]:
            L.set_plastic(flag)

    def cache_anchors(self):
        for L in [self.teo, self.prt, self.pal, self.mot]:
            L.W_ff_anchor = L.W_ff.detach().clone()

    def forward_steps(self, x, T=10, fb_top_pal=None, fb_top_mot=None):
        B = x.size(0)
        self.reset_state(B)
        for t in range(T):
            fb_teo = self.prev_pal
            teo = self.teo.step(x, fb=fb_teo)
            prt = self.prt.step(teo)
            pal = self.pal.step(prt, fb=fb_top_pal)
            self.prev_pal = pal.detach()
            _ = self.mot.step(pal, fb=fb_top_mot)

    def logits_from_states(self, use_rates=False, T_for_rates=20):
        if use_rates:
            pal_feat = self.pal.spike_sum / T_for_rates
            mot_feat = self.mot.spike_sum / T_for_rates
        else:
            pal_feat = self.pal.v_s; mot_feat = self.mot.v_s
        return self.readout_digit(pal_feat), self.readout_action(mot_feat)

# ===============================================================
# Utilities
# ===============================================================
def mnist_to_action(y):
    a = torch.zeros_like(y)
    a[(y >= 0) & (y <= 3)] = 0
    a[(y >= 4) & (y <= 7)] = 1
    a[(y >= 8)] = 2
    return a

# ===============================================================
# Global loss
# ===============================================================
def compute_global_loss(model, labels, actions, mode="zhang", train_motor=False, beta=0.3):
    logit_digit, logit_action = model.logits_from_states(use_rates=False)
    task_digit = F.cross_entropy(logit_digit, labels)
    if train_motor:
        task_action = F.cross_entropy(logit_action, actions)
        task = task_digit + beta * task_action
    else:
        task_action = torch.tensor(0.0, device=labels.device); task = task_digit
    if mode == "surrogate":
        return task, task_digit.item(), task_action.item(), 0.0
    F_pred = 0.0
    for L in [model.teo, model.prt, model.pal]:
        F_pred += 0.5 * ((L.v_s - L.v_b)**2).mean()
        if L.use_feedback: F_pred += 0.5 * ((L.v_s - L.v_a)**2).mean()
    return task + F_pred, task_digit.item(), task_action.item(), F_pred.item()

# ===============================================================
# Phases
# ===============================================================
def global_phase(model, loader, device, epochs=5, lr=1e-3, mode="zhang", train_motor=False, beta=0.3, T=10):
    model.set_plastic(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\n=== Phase 1: Global training ({mode}) ===")
    for ep in range(epochs):
        acc_d, acc_a, total = 0, 0, 0; Ld, La, Lp = 0.0, 0.0, 0.0
        for imgs, labels in loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            labels = labels.to(device)
            actions = mnist_to_action(labels).to(device)
            model.forward_steps(imgs, T=T)
            loss, tD, tA, pred = compute_global_loss(model, labels, actions, mode, train_motor, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                logit_digit, logit_action = model.logits_from_states(use_rates=False)
                acc_d += (logit_digit.argmax(1)==labels).sum().item()
                if train_motor: acc_a += (logit_action.argmax(1)==actions).sum().item()
                total += labels.size(0); Ld+=tD; La+=tA; Lp+=pred
        print(f"[Global {mode}] Ep{ep+1}: digit_acc={(acc_d/total)*100:.2f}%, task={Ld/len(loader):.4f}, pred={Lp/len(loader):.4f}")
    model.cache_anchors()
    torch.save(model.state_dict(), "model_phase1_global.pt")
    print("Saved: model_phase1_global.pt")

def local_phase_supervised(model, loader, device, epochs=5, T=20, beta=0.3, teach_motor=False):
    model.set_plastic(True)
    print("\n=== Phase 2: Local supervised refinement ===")
    for ep in range(epochs):
        for imgs, labels in loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            labels = labels.to(device)
            actions = mnist_to_action(labels).to(device)
            B = imgs.size(0); model.reset_state(B)
            for t in range(T):
                fb_teo = model.prev_pal
                teo = model.teo.step(imgs, fb=fb_teo)
                prt = model.prt.step(teo)
                pal_rate = model.pal.spike_sum/(t+1)
                logits_d = model.readout_digit(pal_rate)
                err_d = F.one_hot(labels,10).float() - F.softmax(logits_d,dim=1)
                fb_pal = err_d @ model.R_fb_digit
                pal = model.pal.step(prt, fb=fb_pal)
                model.prev_pal = pal.detach()
                if teach_motor:
                    mot_rate = model.mot.spike_sum/(t+1)
                    logits_a = model.readout_action(mot_rate)
                    err_a = F.one_hot(actions,3).float() - F.softmax(logits_a,dim=1)
                    fb_mot = err_a @ model.R_fb_action
                else: fb_mot = None
                _ = model.mot.step(pal, fb=fb_mot)
        with torch.no_grad():
            rate=(model.pal.spike_sum/T).mean().item()
            mismatch=((model.pal.v_s-model.pal.v_b)**2).mean().item()
        print(f"[Local supervised] Ep{ep+1}: pal_rate={rate:.3f}, mismatch={mismatch:.4f}")
    torch.save(model.state_dict(), "model_phase2_local.pt")
    print("Saved: model_phase2_local.pt")

def readout_phase(model, loader, device, epochs=5, lr=1e-3, T=20, train_motor=False, beta=0.3):
    model.set_plastic(False)
    opt = torch.optim.Adam(
        list(model.readout_digit.parameters()) +
        (list(model.readout_action.parameters()) if train_motor else []), lr=lr)
    print("\n=== Phase 3: Readout training ===")
    for ep in range(epochs):
        acc_d, acc_a, total = 0, 0, 0
        for imgs, labels in loader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            labels = labels.to(device)
            actions = mnist_to_action(labels).to(device)
            model.forward_steps(imgs, T=T)
            logit_d, logit_a = model.logits_from_states(use_rates=True, T_for_rates=T)
            loss = F.cross_entropy(logit_d, labels)
            if train_motor:
                loss = loss + beta * F.cross_entropy(logit_a, actions)
            opt.zero_grad(); loss.backward(); opt.step()
            acc_d += (logit_d.argmax(1)==labels).sum().item()
            if train_motor: acc_a += (logit_a.argmax(1)==actions).sum().item()
            total += labels.size(0)
        msg=f"[Readout] Ep{ep+1}: digit_acc={(acc_d/total)*100:.2f}%"
        if train_motor: msg+=f", action_acc={(acc_a/total)*100:.2f}%"
        print(msg)
    torch.save(model.state_dict(), "model_phase3_readout.pt")
    print("Saved: model_phase3_readout.pt")

# ===============================================================
# Evaluation
# ===============================================================
@torch.no_grad()
def test(model, loader, device, T=20):
    model.eval(); acc_d, acc_a, total = 0,0,0
    for imgs, labels in loader:
        imgs=imgs.view(imgs.size(0),-1).to(device); labels=labels.to(device)
        actions=mnist_to_action(labels).to(device)
        model.forward_steps(imgs, T=T)
        logit_d,logit_a=model.logits_from_states(use_rates=True,T_for_rates=T)
        acc_d += (logit_d.argmax(1)==labels).sum().item()
        acc_a += (logit_a.argmax(1)==actions).sum().item()
        total += labels.size(0)
    print(f"Test: digit_acc={(acc_d/total)*100:.2f}%, action_acc={(acc_a/total)*100:.2f}%")

# ===============================================================
# Main
# ===============================================================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--global_mode",default="zhang",choices=["zhang","surrogate"])
    ap.add_argument("--global_epochs",type=int,default=5)
    ap.add_argument("--local_epochs",type=int,default=5)
    ap.add_argument("--readout_epochs",type=int,default=5)
    ap.add_argument("--train_motor",action="store_true")
    ap.add_argument("--teach_motor_locally",action="store_true")
    ap.add_argument("--batch_size",type=int,default=64)
    ap.add_argument("--lr_global",type=float,default=1e-3)
    ap.add_argument("--lr_readout",type=float,default=1e-3)
    ap.add_argument("--beta_action",type=float,default=0.3)
    ap.add_argument("--T_global",type=int,default=10)
    ap.add_argument("--T_local",type=int,default=20)
    ap.add_argument("--T_readout",type=int,default=20)
    ap.add_argument("--load_phase",type=str,default=None,
                    help="Optionally load model_phase1_global.pt or model_phase2_local.pt to resume.")
    args=ap.parse_args()

    device="cuda" if torch.cuda.is_available() else "mps"
    print("Using device:",device)
    transform=transforms.ToTensor()
    train_set=datasets.MNIST("./data",train=True,download=True,transform=transform)
    test_set =datasets.MNIST("./data",train=False,download=True,transform=transform)
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader =DataLoader(test_set,batch_size=256)
    model=ZebrafishSNN(device=device).to(device)

    # optional load
    if args.load_phase and os.path.exists(args.load_phase):
        model.load_state_dict(torch.load(args.load_phase, map_location=device))
        print(f"Loaded checkpoint: {args.load_phase}")

    global_phase(model,train_loader,device,
                 epochs=args.global_epochs,lr=args.lr_global,
                 mode=args.global_mode,train_motor=args.train_motor,
                 beta=args.beta_action,T=args.T_global)
    local_phase_supervised(model,train_loader,device,
                           epochs=args.local_epochs,T=args.T_local,
                           teach_motor=args.teach_motor_locally)
    readout_phase(model,train_loader,device,
                  epochs=args.readout_epochs,lr=args.lr_readout,
                  T=args.T_readout,train_motor=args.train_motor,beta=args.beta_action)
    test(model,test_loader,device,T=args.T_readout)
    torch.save(model.state_dict(),"model_final.pt")
    print("Saved: model_final.pt")

if __name__=="__main__":
    main()
