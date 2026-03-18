# ============================================================
# dataset_builder.py
# PURPOSE: Build supervised dataset for prey vs predator training.
# ============================================================

import torch
from prey_predator_env import PreyPredatorEnv
from zebrafish_agent import ZebrafishAgent


def build_dataset(N=2000, device="cpu"):
    """
    Build supervised dataset for prey vs predator classification.
    
    Args:
        N: number of samples to generate
        device: torch device
        
    Returns:
        X: tensor [N, latent_dim] - feature vectors from V1
        y: tensor [N] - integer labels (0=prey, 1=predator)
    """
    env = PreyPredatorEnv(T=300)
    agent = ZebrafishAgent(device=device)

    X = []
    y = []
    
    print(f"Generating {N} samples...")

    for i in range(N):
        obs = env.reset()

        dx = obs["dx"]
        dy = obs["dy"]
        size = obs["size"]
        obj_type = obs["type"]
        
        # Map object type to binary label
        # type: 1 (prey) → label 0
        # type: -1 (predator) → label 1
        # type: 0 (neutral) → skip for binary classification
        if obj_type == 0:
            continue  # Skip neutral objects for binary classifier
            
        label = 0 if obj_type == 1 else 1   # prey=0, predator=1

        # Retina render
        retL = agent.renderer.render(dx, dy, size)
        retR = agent.renderer.render(dx, dy, size)

        # Use raw retinal features directly (ON + OFF channels)
        # These encode size information: prey=small→ON, predator=large→OFF
        onL = retL["ON"].squeeze()   # [n_features]
        offL = retL["OFF"].squeeze()
        onR = retR["ON"].squeeze()
        offR = retR["OFF"].squeeze()
        
        # Concatenate all channels: [ON_L, OFF_L, ON_R, OFF_R]
        z = torch.cat([onL, offL, onR, offR], dim=0)  # [4*n_features]

        X.append(z.detach())
        y.append(label)
        
        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1}/{N} samples...")

    # Stack into tensors
    X = torch.stack(X).squeeze()      # [N, latent_dim]
    y = torch.tensor(y, dtype=torch.long)  # [N], explicitly long for CrossEntropyLoss

    print(f"✓ Dataset built:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    print(f"  Label distribution: prey={(y==0).sum().item()}, predator={(y==1).sum().item()}")
    print()

    return X, y

