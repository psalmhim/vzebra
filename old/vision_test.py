import torch
import matplotlib.pyplot as plt
import numpy as np

from zebra_v55.brain.retina_sampling import sample_retina_binocular
from zebra_v55.brain.zebrafish_snn_5k import ZebrafishSNN_5k
from zebra_v55.world.world_env import WorldEnv


def show_heatmap(M, title):
    plt.figure(figsize=(4,4))
    plt.title(title)
    plt.imshow(M, cmap="hot")
    plt.colorbar()
    plt.show()


def test_retina():
    print("[1] Retina sampling test")
    world = WorldEnv(n_food=1)
    world.foods = [(50, 0)]  # front food

    # FIXED
    retL, retR = sample_retina_binocular((0,0), 0.0, world, "cpu")

    show_heatmap(retL.view(20,20), "Left Retina")
    show_heatmap(retR.view(20,20), "Right Retina")


def test_OT():
    print("[2] Retina → OT mapping test")
    model = ZebrafishSNN_5k(device="cpu")
    model.eval()

    world = WorldEnv(n_food=1)
    world.foods = [(-50, 20)]

    # FIXED
    retL, retR = sample_retina_binocular((0,0), 0.0, world, "cpu")
    stim = torch.cat([retL, retR], dim=1)

    with torch.no_grad():
        model.reset_state(1)
        model.brain_forward(stim)

    OT = model.OT_fused.v_s[0].view(30, 40).cpu()
    show_heatmap(OT, "OT fused activation")


def test_heading_changes():
    print("[3] Heading change test")
    world = WorldEnv(n_food=1)
    world.foods = [(70, 0)]

    headings = [-1.0, -0.5, 0.0, 0.5, 1.0]

    for h in headings:
        # FIXED
        retL, retR = sample_retina_binocular((0,0), h, world, "cpu")

        plt.figure(figsize=(6,3))
        plt.suptitle(f"Heading={h}")

        plt.subplot(1,2,1)
        plt.imshow(retL.view(20,20), cmap="hot")
        plt.title("Left Retina")

        plt.subplot(1,2,2)
        plt.imshow(retR.view(20,20), cmap="hot")
        plt.title("Right Retina")

        plt.show()


if __name__ == "__main__":
    test_retina()
    test_OT()
    test_heading_changes()
