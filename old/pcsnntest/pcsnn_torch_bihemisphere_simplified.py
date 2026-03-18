# pcsnn_torch_bihemisphere_simplified.py
# ------------------------------------------------------------
# Simplified bihemispheric predictive-coding SNN (~2.8k neurons)
# with closed-loop perception–action (vision → motor → feedback)
# ------------------------------------------------------------

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# Surrogate spike
# ============================================================
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

# ============================================================
# Two-compartment neuron
# ============================================================
class TwoCompLayer(nn.Module):
    def __init__(self, n_in, n_out, tau=0.8, fb_dim=None, use_feedback=False,
                 device="cpu", hebb_lr=1e-5, alpha=0.05, mu_anchor=1e-4):
        super().__init__()
        self.device, self.tau, self.use_feedback = device, tau, use_feedback
        self.W_ff = nn.Parameter(0.1 * torch.randn(n_in, n_out, device=device))
        self.W_fb = nn.Parameter(0.1 * torch.randn(fb_dim, n_out, device=device)) \
                    if use_feedback and fb_dim is not None else None
        self.hebb_lr, self.alpha, self.mu_anchor = hebb_lr, alpha, mu_anchor
        self.W_ff_anchor = None; self.plastic = True
        self.reset_state(1)

    def reset_state(self, B):
        d, out = self.device, self.W_ff.shape[1]
        self.v_b = torch.zeros(B, out, device=d)
        self.v_a = torch.zeros(B, out, device=d)
        self.v_s = torch.zeros(B, out, device=d)
        self.spikes = torch.zeros(B, out, device=d)
        self.rbar_pre  = torch.zeros(B, self.W_ff.shape[0], device=d)
        self.ebar_post = torch.zeros(B, out, device=d)
        self.spike_sum = torch.zeros(B, out, device=d)

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


# ============================================================
# Simplified bihemispheric predictive-coding SNN
# ============================================================
class PCSNNTorchBiHemiSimplified(nn.Module):
    def __init__(self, n_per_region=200, device="cpu"):
        super().__init__()
        self.device = device
        self.n = n_per_region
        self.tau = 0.8
        self.k_eye, self.k_tail = 0.1, 0.05
        self.sigma_visual = 8.0
        self.regions = {}
        self._build_network()

        # Eye & body state
        self.theta_L = 0.0; self.theta_R = 0.0
        self.body_x = 0.0; self.stim_x = 10.0
        self.energy_log, self.eyeL_log, self.eyeR_log = [], [], []
        self.body_log, self.stim_log = [], []

    # --------------------------------------------------------
    def _add_region(self, name, n_in, n_out, use_fb=False, fb_dim=None):
        self.regions[name] = TwoCompLayer(n_in, n_out,
                                          tau=self.tau, fb_dim=fb_dim,
                                          use_feedback=use_fb,
                                          device=self.device)

    # --------------------------------------------------------
    def _build_network(self):
        n = self.n; d = self.device
        # Retina → TeO → EOM → Cb → NX
        self._add_region("L_R",   n, n)
        self._add_region("R_R",   n, n)
        self._add_region("L_TeO", n, n, use_fb=True, fb_dim=n)
        self._add_region("R_TeO", n, n, use_fb=True, fb_dim=n)
        self._add_region("L_EOM", n, n, use_fb=True, fb_dim=n)
        self._add_region("R_EOM", n, n, use_fb=True, fb_dim=n)
        self._add_region("L_Cb",  n, n)
        self._add_region("R_Cb",  n, n)
        self._add_region("L_NX",  n, n)
        self._add_region("R_NX",  n, n)
        self._add_region("L_Th",  n, n)
        self._add_region("R_Th",  n, n)
        self._add_region("L_P",   n, n)
        self._add_region("R_P",   n, n)

        self.names = list(self.regions.keys())
        print(f"Built simplified bihemispheric PC-SNN with {len(self.names)} regions, {len(self.names)*n} neurons")

    # --------------------------------------------------------
    def reset_state(self):
        for L in self.regions.values(): L.reset_state(1)

    # --------------------------------------------------------
    def forward_step(self):
        # 1. Sensory input (retina)
        I_L = np.exp(-((self.stim_x - self.theta_L)**2) / (2*self.sigma_visual**2))
        I_R = np.exp(-((self.stim_x + self.theta_R)**2) / (2*self.sigma_visual**2))
        I_L = torch.full((1, self.n), I_L, device=self.device)
        I_R = torch.full((1, self.n), I_R, device=self.device)

        # 2. Feedforward
        r_LR = self.regions["L_R"].step(I_L)
        r_RR = self.regions["R_R"].step(I_R)
        r_LTeO = self.regions["L_TeO"].step(r_LR, fb=self.regions["L_P"].v_s)
        r_RTeO = self.regions["R_TeO"].step(r_RR, fb=self.regions["R_P"].v_s)

        # Cross-hemispheric optic coupling
        cross_L = 0.2 * r_RR + 0.2 * r_RTeO
        cross_R = 0.2 * r_LR + 0.2 * r_LTeO
        r_LTeO += cross_L; r_RTeO += cross_R

        # 3. Eye motor control
        r_LEOM = self.regions["L_EOM"].step(r_LTeO, fb=self.regions["L_Cb"].v_s)
        r_REOM = self.regions["R_EOM"].step(r_RTeO, fb=self.regions["R_Cb"].v_s)

        # 4. Cerebellum (predictive feedback)
        r_LCb = self.regions["L_Cb"].step(r_LEOM)
        r_RCb = self.regions["R_Cb"].step(r_REOM)

        # 5. Motor neurons
        r_LNX = self.regions["L_NX"].step(r_LCb)
        r_RNX = self.regions["R_NX"].step(r_RCb)

        # 6. Thalamic modulation
        r_LTh = self.regions["L_Th"].step(r_LNX)
        r_RTh = self.regions["R_Th"].step(r_RNX)
        r_LP  = self.regions["L_P"].step(r_LTh)
        r_RP  = self.regions["R_P"].step(r_RTh)

        # 7. Motor → movement (closed loop)
        mean_EOM_L = r_LEOM.mean().item(); mean_EOM_R = r_REOM.mean().item()
        mean_NX_L  = r_LNX.mean().item();  mean_NX_R  = r_RNX.mean().item()
        self.theta_L += self.k_eye * (mean_EOM_L - mean_EOM_R)
        self.theta_R += self.k_eye * (mean_EOM_R - mean_EOM_L)
        self.body_x  += self.k_tail * (mean_NX_R - mean_NX_L)
        self.stim_x  -= self.body_x * 0.05  # move world relative to body

        # 8. Free energy (prediction mismatch)
        F = sum(((L.v_s - L.v_b)**2).mean().item() for L in self.regions.values()) / len(self.regions)
        self.energy_log.append(F)
        self.eyeL_log.append(self.theta_L)
        self.eyeR_log.append(self.theta_R)
        self.body_log.append(self.body_x)
        self.stim_log.append(self.stim_x)
        return F

    # --------------------------------------------------------
    def run_simulation(self, steps=200):
        self.reset_state()
        for t in range(steps):
            F = self.forward_step()
            if t % 10 == 0:
                print(f"t={t:03d}  F={F:.5f}  stim={self.stim_x:.2f}  eyeL={self.theta_L:.2f}  eyeR={self.theta_R:.2f}")

    # --------------------------------------------------------
    def plot_results(self):
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        ax[0].plot(self.energy_log)
        ax[0].set_ylabel("Free Energy (∑ε²)")
        ax[1].plot(self.eyeL_log, label="θ_L"); ax[1].plot(self.eyeR_log, label="θ_R")
        ax[1].legend(); ax[1].set_ylabel("Eye angles")
        ax[2].plot(self.stim_log, label="Stimulus"); ax[2].plot(self.body_log, label="Body pos")
        ax[2].legend(); ax[2].set_ylabel("Position")
        ax[2].set_xlabel("Time step")
        plt.tight_layout(); plt.show()

    # --------------------------------------------------------
    def animate(self, interval=50):
        fig, ax = plt.subplots(figsize=(6,3))
        ax.set_xlim(-20, 20); ax.set_ylim(-1, 1)
        stim_dot, = ax.plot([], [], 'ro', markersize=8)
        eye_dot,  = ax.plot([], [], 'bo', markersize=8)
        tail_dot, = ax.plot([], [], 'go', markersize=6)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_title("Zebrafish visual–motor tracking")

        def init():
            stim_dot.set_data([], []); eye_dot.set_data([], []); tail_dot.set_data([], [])
            return stim_dot, eye_dot, tail_dot

        def update(i):
            if i >= len(self.stim_log): i = len(self.stim_log)-1
            stim_dot.set_data(self.stim_log[i], 0)
            eye_dot.set_data((self.theta_L + self.theta_R)/2, 0)
            tail_dot.set_data(self.body_log[i], -0.5)
            return stim_dot, eye_dot, tail_dot

        ani = FuncAnimation(fig, update, frames=len(self.stim_log),
                            init_func=init, blit=True, interval=interval)
        plt.show()
        return ani


# ============================================================
# Run as standalone
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = PCSNNTorchBiHemiSimplified(n_per_region=200, device=device)
    model.run_simulation(steps=200)
    model.plot_results()
    model.animate()
