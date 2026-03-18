# pcsnn_torch_topology_autosave_active.py
# ---------------------------------------------------------------
# Zebrafish bihemispheric PC-SNN with sensory noise and binocular drive
# ---------------------------------------------------------------

import torch, torch.nn as nn
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

PLOT_DIR = Path("plots"); PLOT_DIR.mkdir(exist_ok=True)

# =============================================================
# Spike surrogate
# =============================================================
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        out = (v > 0).float(); ctx.save_for_backward(v); return out
    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        grad = 0.3 * torch.clamp(1 - v.abs(), min=0)
        return grad_output * grad
spike_fn = SpikeFn.apply

# =============================================================
# Two-compartment neuron
# =============================================================
class TwoCompLayer(nn.Module):
    def __init__(self, n_in, n_out, tau=0.8, fb_dim=None, use_feedback=False,
                 device="cpu", hebb_lr=2e-5, alpha=0.05, mu_anchor=1e-4):
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
            self.rbar_pre  = (1 - self.alpha)*self.rbar_pre  + self.alpha*x
            self.ebar_post = (1 - self.alpha)*self.ebar_post + self.alpha*mismatch
            dW = self.rbar_pre.transpose(1,0) @ self.ebar_post / x.size(0)
            with torch.no_grad(): self.W_ff += self.hebb_lr * dW
        return self.spikes

# =============================================================
# Connectivity + subsystem setup
# =============================================================
def build_adjacency(atlas_csv, centroid_csv, lam=200.0, cross_weight=0.5):
    atlas = pd.read_csv(atlas_csv)
    coords = np.loadtxt(centroid_csv, delimiter=",")
    assert len(atlas) == len(coords)
    n = len(atlas)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    A = 2.0 * np.exp(-D / lam)  # stronger coupling
    np.fill_diagonal(A, 0.0)
    for i, ri in atlas.iterrows():
        for j, rj in atlas.iterrows():
            if (ri.Abbr[2:] == rj.Abbr[2:]) and (ri.Hemisphere != rj.Hemisphere):
                A[i, j] = cross_weight
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    return A, atlas, coords

def assign_subsystems(atlas):
    visual = ["R","TeO","Th","P","PT","PrT","TL"]
    motor  = ["EOM","NX","Cb","TS","TG"]
    internal = ["Hi","Hc","Hr","PO","Pi","SP","MON","MOS1","MOS2","MOS3","MOS4","MOS5"]
    tags=[]
    for ab in atlas.Abbr:
        core=ab.split("_")[-1]
        if any(core.startswith(v) for v in visual): tags.append("visual")
        elif any(core.startswith(m) for m in motor): tags.append("motor")
        elif any(core.startswith(i) for i in internal): tags.append("internal")
        else: tags.append("other")
    atlas["Subsystem"]=tags
    return atlas

# =============================================================
# Predictive-coding zebrafish SNN
# =============================================================
class PCSNNTorchActive(nn.Module):
    def __init__(self, atlas_csv, centroid_csv, n_per_region=80, device="cpu"):
        super().__init__()
        self.device=device; self.n=n_per_region
        self.A,self.atlas,self.coords=build_adjacency(atlas_csv,centroid_csv)
        self.atlas=assign_subsystems(self.atlas)
        self.Nr=len(self.atlas)
        self.regions={r.Abbr:TwoCompLayer(self.n,self.n,device=self.device)
                      for _,r in self.atlas.iterrows()}
        print(f"Constructed {self.Nr} regions ({self.Nr*self.n} neurons)")
        self.k_eye=0.1; self.k_tail=0.05; self.sigma_visual=8.0
        self.reset_state()
        self.theta_L=0.0; self.theta_R=0.0
        self.body_x=0.0; self.stim_x=10.0; self.timestep=0
        self.energy_log=[]; self.eyeL_log=[]; self.eyeR_log=[]
        self.body_log=[]; self.stim_log=[]; self.activity_trace=np.zeros((self.Nr,0))

    def reset_state(self):
        for L in self.regions.values(): L.reset_state(1)

    def forward_step(self):
        # ---- small random internal noise to prevent quiescence ----
        noise_scale = 0.02
        for L in self.regions.values():
            L.v_s += noise_scale * torch.randn_like(L.v_s)

        # ---- binocular drive alternating between L/R ----
        I_L=np.exp(-((self.stim_x-self.theta_L)**2)/(2*self.sigma_visual**2))
        I_R=np.exp(-((self.stim_x+self.theta_R)**2)/(2*self.sigma_visual**2))
        if self.timestep < 50:
            I_L *= 1.5; I_R *= 1.0
        elif self.timestep < 100:
            I_L *= 1.0; I_R *= 1.5
        else:
            I_L *= 1.2; I_R *= 1.2

        inp={r:torch.zeros(1,self.n,device=self.device) for r in self.regions}
        if "L_R" in inp:
            inp["L_R"]=torch.full((1,self.n),I_L,device=self.device)
            inp["L_R"] += 0.05*torch.rand_like(inp["L_R"])
        if "R_R" in inp:
            inp["R_R"]=torch.full((1,self.n),I_R,device=self.device)
            inp["R_R"] += 0.05*torch.rand_like(inp["R_R"])

        abbr_list=list(self.atlas.Abbr); out={}; act=[]
        for i,ai in enumerate(abbr_list):
            pre_sum=torch.zeros_like(inp[ai])
            for j,aj in enumerate(abbr_list):
                if self.A[j,i]>0:
                    pre_sum+=self.A[j,i]*self.regions[aj].v_s
            out[ai]=self.regions[ai].step(pre_sum+inp[ai])
            act.append(out[ai].mean().item())
        self.activity_trace=np.column_stack([self.activity_trace,np.array(act)])

        mean_EOM_L=out.get("L_EOM",torch.zeros(1,self.n,device=self.device)).mean().item()
        mean_EOM_R=out.get("R_EOM",torch.zeros(1,self.n,device=self.device)).mean().item()
        mean_NX_L=out.get("L_NX",torch.zeros(1,self.n,device=self.device)).mean().item()
        mean_NX_R=out.get("R_NX",torch.zeros(1,self.n,device=self.device)).mean().item()
        self.theta_L+=self.k_eye*(mean_EOM_L-mean_EOM_R)
        self.theta_R+=self.k_eye*(mean_EOM_R-mean_EOM_L)
        self.body_x+=self.k_tail*(mean_NX_R-mean_NX_L)
        self.stim_x-=0.05*self.body_x

        F=sum(((L.v_s-L.v_b)**2).mean().item() for L in self.regions.values())/len(self.regions)
        self.energy_log.append(F)
        self.eyeL_log.append(self.theta_L); self.eyeR_log.append(self.theta_R)
        self.body_log.append(self.body_x);  self.stim_log.append(self.stim_x)
        self.timestep += 1
        return F

    # ---------------------------------------------------------
    def run(self,steps=300):
        for t in range(steps):
            F=self.forward_step()
            if t%10==0:
                print(f"t={t:03d} F={F:.5f} stim={self.stim_x:.2f} eyeL={self.theta_L:.2f} eyeR={self.theta_R:.2f}")
        self.save_activity_heatmaps()

    # ---------------------------------------------------------
    def save_activity_heatmaps(self):
        time=np.arange(self.activity_trace.shape[1])
        atlas=self.atlas; subsystems=atlas.Subsystem.unique()
        for sub in subsystems:
            idx=np.where(atlas.Subsystem==sub)[0]
            if len(idx)==0: continue
            sub_act=self.activity_trace[idx,:]
            fig,ax=plt.subplots(figsize=(8,4))
            im=ax.imshow(sub_act,cmap="viridis",aspect="auto")
            ax.set_title(f"{sub.capitalize()} subsystem activity")
            ax.set_xlabel("Time step"); ax.set_ylabel("Region index")
            plt.colorbar(im,ax=ax,label="Mean spike")
            plt.tight_layout()
            stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            outfile=PLOT_DIR/f"activity_{sub}_{stamp}.png"
            plt.savefig(outfile,dpi=300); plt.close(fig)
            print(f"[Saved] {outfile}")

    # ---------------------------------------------------------
    def plot_brain(self):
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        c=np.array(self.coords); subs=self.atlas.Subsystem
        color_map={'visual':'blue','motor':'red','internal':'green','other':'gray'}
        colors=[color_map.get(s,'gray') for s in subs]
        fig=plt.figure(figsize=(6,6))
        ax=fig.add_subplot(111,projection='3d')
        ax.scatter(c[:,0],c[:,1],c[:,2],c=colors,s=25)
        ax.set_title("Zebrafish Brain (Subsystem color)")
        plt.tight_layout()
        outfile=PLOT_DIR/"brain_topology.png"
        plt.savefig(outfile,dpi=300); plt.close(fig)
        print(f"[Saved] {outfile}")

    # ---------------------------------------------------------
    def plot_summary(self):
        fig,ax=plt.subplots(4,1,figsize=(8,9))
        ax[0].plot(self.energy_log); ax[0].set_ylabel("Free Energy")
        ax[1].plot(self.eyeL_log,label="θ_L"); ax[1].plot(self.eyeR_log,label="θ_R")
        ax[1].legend(); ax[1].set_ylabel("Eye Angles")
        ax[2].plot(self.stim_log,label="Stimulus"); ax[2].plot(self.body_log,label="Body")
        ax[2].legend(); ax[2].set_ylabel("Position")
        eye_diff=np.array(self.eyeL_log)-np.array(self.eyeR_log)
        ax[3].plot(eye_diff); ax[3].set_ylabel("Vergence (θ_L−θ_R)")
        ax[3].set_xlabel("Time step")
        plt.tight_layout()
        outfile=PLOT_DIR/"activity_summary.png"
        plt.savefig(outfile,dpi=300); plt.close(fig)
        print(f"[Saved] {outfile}")

# =============================================================
# Main
# =============================================================
if __name__=="__main__":
    atlas_path=Path("./atlas/atlas_bilateral.csv")
    centroid_path=Path("./atlas/centroids.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:",device)
    model=PCSNNTorchActive(atlas_path,centroid_path,n_per_region=80,device=device)
    model.plot_brain()
    model.run(steps=300)
    model.plot_summary()
