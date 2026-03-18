# ===============================================================
# pcsnn_dopamine_valence_motorloop.py
# ===============================================================
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt, os

# ===============================================================
# Retina: predictive coding + dopaminergic plasticity
# ===============================================================
class Retina(nn.Module):
    def __init__(self, n_in=784, n_lat=64, tau=0.9, eta=0.01,
                 pi_min=0.01, pi_max=2.0, beta_val=0.5,
                 lr_dopa=0.001, device="cpu"):
        super().__init__()
        self.device = device
        self.W_enc = nn.Parameter(torch.randn(n_in, n_lat, device=device)*0.05)
        self.W_dec = nn.Parameter(torch.randn(n_lat, n_in, device=device)*0.05)
        self.v_lat = torch.zeros(n_lat, device=device)
        self.Pi = torch.ones(n_in, device=device)*0.1
        self.tau, self.eta = tau, eta
        self.pi_min, self.pi_max = pi_min, pi_max
        self.beta_val, self.lr_dopa = beta_val, lr_dopa

    def step(self, x, valence=0.0, rpe=0.0):
        if not isinstance(rpe, torch.Tensor):
            rpe_t = torch.tensor(rpe, device=self.device, dtype=torch.float32)
        else:
            rpe_t = rpe.to(self.device)

        z = torch.tanh(x @ self.W_enc)
        pred = z @ self.W_dec
        err = (x - pred).view(-1)
        abs_err = err.abs(); rel = abs_err/(abs_err.mean()+1e-6)
        target = (1+self.beta_val*valence)*rel
        delta = self.eta*(target-self.Pi)
        delta *= torch.exp(-torch.abs(rpe_t))
        self.Pi = torch.clamp(self.Pi+delta, self.pi_min, self.pi_max)
        F_val = (self.Pi*err**2).mean().item()
        self.v_lat = self.tau*self.v_lat + (1-self.tau)*z.squeeze()

        if abs(rpe) > 1e-5:
            dW_enc = self.lr_dopa*rpe*(x.T@z)
            dW_dec = self.lr_dopa*rpe*(z.T@err.view(1,-1))
            with torch.no_grad():
                self.W_enc.add_(dW_enc.clamp(-0.05,0.05))
                self.W_dec.add_(dW_dec.clamp(-0.05,0.05))

        return self.v_lat.unsqueeze(0), F_val, self.Pi.mean().item()

# ===============================================================
# Dopaminergic system
# ===============================================================
class DopamineSystem:
    def __init__(self, n_cat=10, lr_val=0.01):
        self.valence = torch.zeros(n_cat)
        self.rpe = 0.0; self.lr_val = lr_val

    def update(self, pred, reward):
        val = self.valence[pred].item()
        rpe = reward - val
        self.valence[pred] += self.lr_val * rpe
        self.rpe = rpe
        return val, rpe

# ===============================================================
# Recognizer (MNIST surrogate)
# ===============================================================
class Recognizer(nn.Module):
    def __init__(self, n_lat=64, n_out=10):
        super().__init__()
        self.fc = nn.Linear(n_lat, n_out)
    def forward(self,x): return self.fc(x)

# ===============================================================
# Motor system
# ===============================================================
class MotorSystem:
    def __init__(self, k_eye=0.05):
        self.k_eye=k_eye; self.eyeL=0.0; self.eyeR=0.0
    def update(self,val):
        d=self.k_eye*val
        self.eyeL+=d; self.eyeR-=d
        self.eyeL=torch.clamp(torch.tensor(self.eyeL),-1,1).item()
        self.eyeR=torch.clamp(torch.tensor(self.eyeR),-1,1).item()
        return self.eyeL,self.eyeR

# ===============================================================
# Agent
# ===============================================================
class ZebrafishAgent:
    def __init__(self, device="cpu"):
        self.device=device
        self.retina=Retina(device=device)
        self.recognizer=Recognizer().to(device)
        self.dopa=DopamineSystem()
        self.motor=MotorSystem()

    def perceive_and_act(self,img,label):
        x=img.view(1,-1).to(self.device)
        v_cat=float(self.dopa.valence[label])
        retinal_act,F,Pi=self.retina.step(x,valence=v_cat,rpe=self.dopa.rpe)
        logits=self.recognizer(retinal_act)
        pred=logits.argmax().item()

        if pred<=2: reward=0.5
        elif pred>=7: reward=-0.5
        else: reward=0.0

        val,rpe=self.dopa.update(pred,reward)
        eL,eR=self.motor.update(val)
        return F,Pi,val,rpe,pred,reward,eL,eR

# ===============================================================
# Run simulation
# ===============================================================
def main():
    device="cuda" if torch.cuda.is_available() else \
           "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:",device)
    transform=transforms.ToTensor()
    loader=DataLoader(datasets.MNIST("./data",train=True,download=True,
                                     transform=transform),batch_size=1,shuffle=True)
    agent=ZebrafishAgent(device=device)
    os.makedirs("plots",exist_ok=True)

    Fs,Pis,Vals,RPEs=[],[],[],[]
    for step,(img,label) in enumerate(loader):
        F,Pi,val,rpe,pred,rew,eL,eR=agent.perceive_and_act(img,label.item())
        if step%50==0:
            print(f"Step {step:03d}: F={F:.3f} Pi={Pi:.3f} Val={val:.2f} RPE={rpe:.3f} "
                  f"Pred={pred} Rew={rew:+.1f} eyeL={eL:.2f} eyeR={eR:.2f}")
        Fs.append(F); Pis.append(Pi); Vals.append(val); RPEs.append(rpe)
        if step>=300: break

    # -----------------------------------------------------------
    plt.figure(figsize=(8,6))
    plt.subplot(3,1,1); plt.plot(Fs,label="Free Energy"); plt.legend()
    plt.subplot(3,1,2); plt.plot(Pis,label="Precision (Pi)"); plt.legend()
    plt.subplot(3,1,3); plt.plot(Vals,label="Valence"); plt.plot(RPEs,label="RPE"); plt.legend()
    plt.tight_layout()
    plt.savefig("plots/dopamine_motorloop_dynamics.png")
    print("[Saved] plots/dopamine_motorloop_dynamics.png")

if __name__=="__main__": main()
