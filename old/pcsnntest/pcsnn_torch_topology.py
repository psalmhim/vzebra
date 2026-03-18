# pcsnn_torch_topology.py
# -------------------------------------------------------------
# Anatomically grounded zebrafish predictive-coding SNN
# Builds 72-region bihemispheric network from atlas + centroids
# -------------------------------------------------------------

import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np, math, matplotlib.pyplot as plt
from pathlib import Path

# =============================================================
# Surrogate spike
# =============================================================
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

# =============================================================
# Two-compartment neuron (same as before)
# =============================================================
class TwoCompLayer(nn.Module):
    def __init__(self, n_in, n_out, tau=0.8, fb_dim=None, use_feedback=False,
                 device="cpu", hebb_lr=1e-5, alpha=0.05, mu_anchor=1e-4):
        super().__init__()
        self.device, self.tau, self.use_feedback = device, tau, use_feedback
        self.W_ff = nn.Parameter(0.1 * torch.randn(n_in, n_out, device=device))
        self.W_fb = nn.Parameter(0.1 * torch.randn(fb_dim, n_out, device=device)) \
                    if use_feedback and fb_dim is not None else None
        self.hebb_lr, self.alpha, self.mu_anchor = hebb_lr, alpha, mu_anchor
        self.plastic = True
        self.reset_state(1)

    def reset_state(self, B):
        d, out = self.device, self.W_ff.shape[1]
        self.v_b = torch.zeros(B, out, device=d)
        self.v_a = torch.zeros(B, out, device=d)
        self.v_s = torch.zeros(B, out, device=d)
        self.spikes = torch.zeros(B, out, device=d)
        self.rbar_pre  = torch.zeros(B, self.W_ff.shape[0], device=d)
        self.ebar_post = torch.zeros(B, out, device=d)

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
        if self.plastic:
            mismatch = self.v_s - self.v_b
            self.rbar_pre  = (1 - self.alpha) * self.rbar_pre  + self.alpha * x
            self.ebar_post = (1 - self.alpha) * self.ebar_post + self.alpha * mismatch
            dW = self.rbar_pre.transpose(1, 0) @ self.ebar_post / x.size(0)
            with torch.no_grad():
                self.W_ff += self.hebb_lr * dW
        return self.spikes


# =============================================================
# Build topology from atlas and centroids
# =============================================================
def build_adjacency(atlas_csv, centroid_csv, lam=200.0, cross_weight=0.5):
    atlas = pd.read_csv(atlas_csv)
    coords = np.loadtxt(centroid_csv, delimiter=",")
    assert len(atlas) == len(coords), f"Mismatch: {len(atlas)} regions vs {len(coords)} centroids"

    n = len(atlas)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    A = np.exp(-D / lam)

    # zero self-connections
    np.fill_diagonal(A, 0.0)

    # commissural links
    for i, row_i in atlas.iterrows():
        for j, row_j in atlas.iterrows():
            if (row_i["Abbr"][2:] == row_j["Abbr"][2:]) and (row_i["Hemisphere"] != row_j["Hemisphere"]):
                A[i, j] = cross_weight

    # normalize rows
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    return A, atlas, coords


# =============================================================
# Predictive-coding zebrafish network
# =============================================================
class PCSNNTorchTopology(nn.Module):
    def __init__(self, atlas_csv, centroid_csv, n_per_region=80, device="cpu"):
        super().__init__()
        self.device = device
        self.n = n_per_region
        self.A, self.atlas, self.coords = build_adjacency(atlas_csv, centroid_csv)
        self.Nr = len(self.atlas)
        self.k_eye, self.k_tail = 0.05, 0.02
        self.sigma_visual = 8.0
        self.regions = {}
        self._build_regions()
        self.reset_state()

        # world variables
        self.theta_L = 0.0; self.theta_R = 0.0
        self.body_x = 0.0; self.stim_x = 10.0
        self.energy_log, self.eyeL_log, self.eyeR_log, self.body_log, self.stim_log = [], [], [], [], []

    # ---------------------------------------------------------
    def _build_regions(self):
        for _, row in self.atlas.iterrows():
            self.regions[row.Abbr] = TwoCompLayer(self.n, self.n, device=self.device)
        print(f"Constructed {len(self.regions)} regions ({self.n*len(self.regions)} neurons total)")

    # ---------------------------------------------------------
    def reset_state(self):
        for L in self.regions.values(): L.reset_state(1)

    # ---------------------------------------------------------
    def forward_step(self):
        # retina input (L_R, R_R)
        I_L = np.exp(-((self.stim_x - self.theta_L)**2) / (2*self.sigma_visual**2))
        I_R = np.exp(-((self.stim_x + self.theta_R)**2) / (2*self.sigma_visual**2))
        inp = {r: torch.zeros(1, self.n, device=self.device) for r in self.regions}
        inp["L_R"] = torch.full((1, self.n), I_L, device=self.device)
        inp["R_R"] = torch.full((1, self.n), I_R, device=self.device)

        # region index map
        abbr_list = list(self.atlas.Abbr)
        out = {abbr: None for abbr in abbr_list}

        # propagate through topology
        for i, abbr_i in enumerate(abbr_list):
            pre_sum = torch.zeros_like(inp[abbr_i])
            for j, abbr_j in enumerate(abbr_list):
                if self.A[j, i] > 0:
                    pre_sum += self.A[j, i] * self.regions[abbr_j].v_s
            out[abbr_i] = self.regions[abbr_i].step(pre_sum + inp[abbr_i])

        # eye and motor control from NX and EOM
        mean_EOM_L = out.get("L_EOM", torch.zeros(1,self.n,device=self.device)).mean().item()
        mean_EOM_R = out.get("R_EOM", torch.zeros(1,self.n,device=self.device)).mean().item()
        mean_NX_L  = out.get("L_NX",  torch.zeros(1,self.n,device=self.device)).mean().item()
        mean_NX_R  = out.get("R_NX",  torch.zeros(1,self.n,device=self.device)).mean().item()

        self.theta_L += self.k_eye * (mean_EOM_L - mean_EOM_R)
        self.theta_R += self.k_eye * (mean_EOM_R - mean_EOM_L)
        self.body_x  += self.k_tail * (mean_NX_R - mean_NX_L)
        self.stim_x  -= 0.05 * self.body_x

        F = sum(((L.v_s - L.v_b)**2).mean().item() for L in self.regions.values()) / len(self.regions)
        self.energy_log.append(F)
        self.eyeL_log.append(self.theta_L); self.eyeR_log.append(self.theta_R)
        self.body_log.append(self.body_x);  self.stim_log.append(self.stim_x)
        return F

    # ---------------------------------------------------------
    def run_simulation(self, steps=200):
        for t in range(steps):
            F = self.forward_step()
            if t % 10 == 0:
                print(f"t={t:03d}  F={F:.5f}  stim={self.stim_x:.2f}  eyeL={self.theta_L:.2f}  eyeR={self.theta_R:.2f}")

    # ---------------------------------------------------------
    def plot_results(self):
        fig, ax = plt.subplots(3,1,figsize=(8,8))
        ax[0].plot(self.energy_log); ax[0].set_ylabel("Free Energy")
        ax[1].plot(self.eyeL_log,label="θ_L"); ax[1].plot(self.eyeR_log,label="θ_R")
        ax[1].legend(); ax[1].set_ylabel("Eye angles")
        ax[2].plot(self.stim_log,label="Stim"); ax[2].plot(self.body_log,label="Body")
        ax[2].legend(); ax[2].set_xlabel("Time step")
        plt.tight_layout(); plt.show()

    # ---------------------------------------------------------
    def plot_brain_3d(self):
        from mpl_toolkits.mplot3d import Axes3D
        c = np.array(self.coords)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(c[:,0], c[:,1], c[:,2], c='gray', s=20)
        for i,row in self.atlas.iterrows():
            ax.text(c[i,0], c[i,1], c[i,2], row.Abbr, fontsize=6)
        ax.set_title("Zebrafish brain topology (72 regions)")
        plt.show()


# =============================================================
# Main
# =============================================================
if __name__ == "__main__":
    atlas_path = Path("./atlas/atlas_bilateral.csv")
    centroid_path = Path("./atlas/centroids.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = PCSNNTorchTopology(atlas_path, centroid_path, n_per_region=80, device=device)
    model.plot_brain_3d()
    model.run_simulation(steps=200)
    model.plot_results()
