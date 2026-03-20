"""
Step 42b: Reticulospinal Motor Training

Trains the OT_L→motor_R, OT_R→motor_L crossed projection using
the retinal L/R turn signal as supervision target. The retinal
turn signal (retR_sum - retL_sum) / total is a reliable motor
command — we teach the SNN motor neurons to reproduce it.

Run: python -m zebrav1.tests.step42b_reticulospinal_training
"""
import os, sys, math, torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.brain.zebrafish_snn import ZebrafishSNN
from zebrav1.brain.device_util import get_device
from zebrav1.world.world_env import WorldEnv


def run_reticulospinal_training(n_epochs=200, lr=0.003):
    print("=" * 60)
    print("Step 42b: Reticulospinal Motor Training")
    print("=" * 60)

    device = get_device()
    model = ZebrafishSNN(device=str(device))

    # Load current weights
    weights_path = "zebrav1/weights/classifier.pt"
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_saveable_state(state)
        print(f"Loaded weights from {weights_path}")

    # Add reticulospinal layers back (for training)
    import torch.nn as nn
    reticulo_L = nn.Linear(model.OTL, 100, bias=False).to(device)
    reticulo_R = nn.Linear(model.OTR, 100, bias=False).to(device)
    nn.init.normal_(reticulo_L.weight, 0, 0.01)
    nn.init.normal_(reticulo_R.weight, 0, 0.01)

    # Only train reticulospinal weights
    optimizer = torch.optim.Adam(
        list(reticulo_L.parameters()) + list(reticulo_R.parameters()),
        lr=lr)

    model.eval()  # freeze SNN layers

    # Training: run SNN forward, compute retinal turn target,
    # train reticulospinal to match
    world = WorldEnv()
    loss_hist = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_samples = 50

        for _ in range(n_samples):
            # Random scene
            world = WorldEnv()
            heading = np.random.uniform(-math.pi, math.pi)
            pos = np.array([np.random.uniform(-200, 200),
                            np.random.uniform(-200, 200)])

            # Random entity
            entity_type = np.random.choice(["food", "enemy", "colleague"])
            angle = heading + np.random.uniform(-math.pi/2, math.pi/2)
            dist = np.random.uniform(30, 150)
            ex = pos[0] + dist * math.cos(angle)
            ey = pos[1] + dist * math.sin(angle)

            if entity_type == "food":
                world.foods.append((ex, ey))
            elif entity_type == "enemy":
                world.predator = (ex, ey, angle + math.pi, 1.0)
            else:
                world.colleagues.append((ex, ey))

            goal_probs = torch.zeros(1, 4, device=device)
            if entity_type == "food":
                goal_probs[0, 0] = 1.0
            elif entity_type == "enemy":
                goal_probs[0, 1] = 1.0

            with torch.no_grad():
                out = model.forward(pos, heading, world,
                                    goal_probs=goal_probs)

            # Retinal turn signal (supervision target)
            retL_sum = float(out["retL"].sum())
            retR_sum = float(out["retR"].sum())
            total = retL_sum + retR_sum + 1e-8
            retinal_turn = (retR_sum - retL_sum) / total
            target = torch.tensor([retinal_turn], device=device)

            # Reticulospinal forward pass (trainable)
            oL = out["oL"].detach()
            oR = out["oR"].detach()
            retic_L = reticulo_L(oL)  # → right motor
            retic_R = reticulo_R(oR)  # → left motor
            # Motor readout: sigmoid(R) - sigmoid(L)
            mot_L = retic_R.sigmoid().mean()
            mot_R = retic_L.sigmoid().mean()
            pred_turn = mot_R - mot_L

            loss = F.mse_loss(pred_turn.unsqueeze(0), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_samples
        loss_hist.append(avg_loss)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}")

    # Validate
    print("\n--- Validation ---")
    correct = 0
    total_v = 20
    for _ in range(total_v):
        world.clear()
        heading = np.random.uniform(-math.pi, math.pi)
        pos = np.array([0.0, 0.0])
        angle = heading + np.random.uniform(-math.pi/3, math.pi/3)
        world.foods.append((80 * math.cos(angle), 80 * math.sin(angle)))

        with torch.no_grad():
            out = model.forward(pos, heading, world,
                                goal_probs=torch.tensor([[1,0,0,0]],
                                dtype=torch.float32, device=device))
            oL, oR = out["oL"], out["oR"]
            rL = reticulo_L(oL)
            rR = reticulo_R(oR)
            pred = rR.sigmoid().mean() - rL.sigmoid().mean()
            retL_s = float(out["retL"].sum())
            retR_s = float(out["retR"].sum())
            target_v = (retR_s - retL_s) / (retL_s + retR_s + 1e-8)

        if (pred > 0) == (target_v > 0) or abs(target_v) < 0.05:
            correct += 1

    print(f"  Direction accuracy: {correct}/{total_v} ({correct/total_v:.0%})")

    # Save reticulospinal weights
    save_path = "zebrav1/weights/reticulospinal.pt"
    torch.save({
        "reticulo_L": reticulo_L.state_dict(),
        "reticulo_R": reticulo_R.state_dict(),
        "loss_hist": loss_hist,
    }, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    run_reticulospinal_training()
