# ============================================================
# debug_features.py
# Quick script to check if retinal features differ for prey/predator
# ============================================================

import torch
import numpy as np
from prey_predator_env import PreyPredatorEnv
from zebrafish_agent import ZebrafishAgent


def debug_features():
    device = "cpu"
    env = PreyPredatorEnv(T=300)
    agent = ZebrafishAgent(device=device)
    
    prey_features = []
    pred_features = []
    
    # Collect 20 samples of each
    for _ in range(20):
        # Prey
        obs = env.reset(obj_type=1)
        retL = agent.renderer.render(obs["dx"], obs["dy"], obs["size"])
        onL = retL["ON"].squeeze()
        offL = retL["OFF"].squeeze()
        feat = torch.cat([onL, offL], dim=0)
        prey_features.append(feat)
        
        # Predator
        obs = env.reset(obj_type=-1)
        retL = agent.renderer.render(obs["dx"], obs["dy"], obs["size"])
        onL = retL["ON"].squeeze()
        offL = retL["OFF"].squeeze()
        feat = torch.cat([onL, offL], dim=0)
        pred_features.append(feat)
    
    prey_features = torch.stack(prey_features)
    pred_features = torch.stack(pred_features)
    
    print("Feature statistics:")
    print(f"Prey features: mean={prey_features.mean():.4f}, std={prey_features.std():.4f}")
    print(f"Pred features: mean={pred_features.mean():.4f}, std={pred_features.std():.4f}")
    print()
    
    # Channel-specific stats
    n = prey_features.shape[1] // 2
    prey_on = prey_features[:, :n].mean()
    prey_off = prey_features[:, n:].mean()
    pred_on = pred_features[:, :n].mean()
    pred_off = pred_features[:, n:].mean()
    
    print("Channel statistics:")
    print(f"Prey:     ON={prey_on:.4f}, OFF={prey_off:.4f}, ratio={prey_on/max(prey_off,1e-6):.2f}")
    print(f"Predator: ON={pred_on:.4f}, OFF={pred_off:.4f}, ratio={pred_on/max(pred_off,1e-6):.2f}")
    print()
    
    if abs(prey_on - pred_on) < 0.01 and abs(prey_off - pred_off) < 0.01:
        print("⚠ WARNING: Features are nearly identical!")
        print("  The retinal encoding may not capture size differences.")
    else:
        print("✓ Features differ between prey and predator")
        print(f"  ON difference: {abs(prey_on - pred_on):.4f}")
        print(f"  OFF difference: {abs(prey_off - pred_off):.4f}")


if __name__ == "__main__":
    debug_features()
