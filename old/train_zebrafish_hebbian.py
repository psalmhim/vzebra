import os
import sys
import torch
import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebra_v60.brain.zebrafish_snn_v60 import ZebrafishSNN_v60
from zebra_v60.world.world_env import WorldEnv
from hebbian_learning import HebbianLearning, ZebrafishTrainingDataset

def train_zebrafish_with_hebbian(model, dataset, hebbian_learner, epochs=20, device="cpu"):
    """
    Train the zebrafish model using Hebbian learning on multiple layers.
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for food_pos, target_turn in dataset:
            world = WorldEnv()
            world.foods = [food_pos]

            model.reset()
            pos = np.array([0., 0.])
            heading = 0.0

            # Forward pass
            out = model.forward(pos, heading, world)
            motor = out["motor"]
            pred_turn = (motor[0, :100].sigmoid().mean() - motor[0, 100:].sigmoid().mean()).item()

            # Simple accuracy check
            if (target_turn < 0 and pred_turn < 0) or (target_turn > 0 and pred_turn > 0) or (target_turn == 0 and abs(pred_turn) < 0.1):
                correct += 1
            total += 1

            # Hebbian updates on multiple layers
            with torch.no_grad():
                # OT_F to PT_L
                pre_act = model.OT_F.v.detach()
                post_act = model.PT_L.v.detach()
                hebbian_learner.update(model.PT_L.W, pre_act, post_act)

                # PT_L to PC_per
                pre_act = model.PT_L.v.detach()
                post_act = model.PC_per.v.detach()
                hebbian_learner.update(model.PC_per.W, pre_act, post_act)

                # PC_per to PC_int
                pre_act = model.PC_per.v.detach()
                post_act = model.PC_int.v.detach()
                hebbian_learner.update(model.PC_int.W, pre_act, post_act)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}: Accuracy {accuracy:.2f}")

    print("Training completed.")

if __name__ == "__main__":
    # Initialize model
    model = ZebrafishSNN_v60(device="cpu")

    # Create dataset
    dataset = ZebrafishTrainingDataset(n_samples=500)

    # Hebbian learner
    hebb = HebbianLearning(learning_rate=0.001)

    # Train
    train_zebrafish_with_hebbian(model, dataset, hebb, epochs=5)

    # Test on the original test set
    print("\nTesting on original vision-behavior test:")
    world = WorldEnv(xmin=-200, xmax=200, ymin=-150, ymax=150, n_food=0)
    headings = [
        ("Left_far", (-60, -20)),
        ("Left_mid", (-40, -5)),
        ("Left_near", (-20, 0)),
        ("Front", (40, 0)),
        ("Right_far", (60, 20)),
        ("Right_mid", (40, 5)),
        ("Right_near", (20, 0))
    ]

    pos = np.array([0., 0.])
    heading = 0.0

    passed = 0
    for name, food in headings:
        world.foods = [tuple(food)]
        model.reset()
        out = model.forward(pos, heading, world)
        turn = float(out["motor"][0,:100].sigmoid().mean() - out["motor"][0,100:].sigmoid().mean())

        print(f"{name:12s} food={food}  turn={turn:+.3f}")

        if ("Left" in name and turn < 0) or ("Right" in name and turn > 0) or ("Front" in name and abs(turn) < 0.05):
            passed += 1

    print(f"Passed {passed}/7")