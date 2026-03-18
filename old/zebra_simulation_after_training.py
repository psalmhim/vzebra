# ============================================================
# zebra_simulation_after_training.py
# ============================================================

import torch
import matplotlib.pyplot as plt
from prey_predator_env import PreyPredatorEnv
from zebrafish_agent import ZebrafishAgent

def run_sim(device="cpu", T=300):

    agent = ZebrafishAgent(device=device)

    # Load trained vision classifier and RL policy
    agent.prey_pred.load_state_dict(torch.load("models/prey_pred.pt"))
    agent.policy.goal_prior = torch.load("models/goal_prior.pt")

    env = PreyPredatorEnv(T=T)

    # logs
    prey_hist = []
    pred_hist = []
    policy_hist = []

    obs = env.reset()

    for t in range(T):
        dx, dy, size = obs["dx"], obs["dy"], obs["size"]
        retL = agent.renderer.render(dx, dy, size)
        retR = agent.renderer.render(dx, dy, size)

        out = agent.step(retL, retR)

        prey_hist.append(out["prey_prob"])
        pred_hist.append(out["pred_prob"])
        policy_hist.append(out["policy"])

        obs = env.step()

    # plot
    plt.figure(figsize=(10,7))

    plt.subplot(3,1,1)
    plt.plot(prey_hist, label="Prey")
    plt.plot(pred_hist, label="Predator")
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(policy_hist, label="Policy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/sim_after_training.png")
    print("✓ saved: plots/sim_after_training.png")


if __name__ == "__main__":
    run_sim()
