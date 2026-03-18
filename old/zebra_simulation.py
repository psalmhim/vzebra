# ============================================================
# ZEBRAFISH v55 SIMULATION SCRIPT
# Retina → Cortex → Memory → Dopamine → Policy → Motor
# Terminal-safe (Agg backend), auto-saving plots
# ============================================================

import matplotlib
matplotlib.use("Agg")     # disable GUI for terminal use

import torch
import matplotlib.pyplot as plt
import os

from prey_predator_env import PreyPredatorEnv
from zebrafish_agent import ZebrafishAgent


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_sim(T=300, device="cpu"):

    # Environment & agent
    env = PreyPredatorEnv(T=T, device=device)
    agent = ZebrafishAgent(device=device)

    # Logs for plotting
    Fv_hist = []
    fast_hist = []
    slow_hist = []
    efe_hist = []
    prey_hist = []
    pred_hist = []
    policy_hist = []
    eye_hist = []

    # Reset env
    obs = env.reset()

    for t in range(T):

        # ------------- 1. Extract object state -------------
        dx = obs["dx"]
        dy = obs["dy"]
        size = obs["size"]

        # ------------- 2. Retina rendering -----------------
        retL = agent.renderer.render(dx, dy, size)
        retR = agent.renderer.render(dx, dy, size)

        # ------------- 3. Agent step -----------------------
        out = agent.step(retL, retR)

        # ------------- 4. Logging --------------------------
        Fv_hist.append(out["Fv"])
        fast_hist.append(out["da_fast"])
        slow_hist.append(out["da_slow"])
        efe_hist.append(out["efe_slow"])
        prey_hist.append(out["prey_prob"])
        pred_hist.append(out["pred_prob"])
        policy_hist.append(out["policy"])
        eye_hist.append(out["eye"])

        # optional: terminal logging
        if t % 50 == 0:
            print(f"[t={t}] Fv={out['Fv']:.3f}  prey={out['prey_prob']:.3f} "
                  f"pred={out['pred_prob']:.3f}  policy={out['policy']}")

        # ------------- 5. Next frame ------------------------
        obs = env.step()

    # ============================================================
    # Save plots (not show)
    # ============================================================
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12,10))

    plt.subplot(4,1,1)
    plt.plot(Fv_hist, label="Visual Free Energy")
    plt.legend()

    plt.subplot(4,1,2)
    plt.plot(fast_hist, label="Fast Dopamine")
    plt.plot(slow_hist, label="Slow Dopamine")
    plt.legend()

    plt.subplot(4,1,3)
    plt.plot(prey_hist, label="Prey Probability")
    plt.plot(pred_hist, label="Predator Probability")
    plt.legend()

    plt.subplot(4,1,4)
    plt.plot(policy_hist, label="Policy (0=approach,1=flee,2=neutral)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/simulation_summary.png", dpi=200)
    plt.close()

    print("✓ Saved plots to plots/simulation_summary.png")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_sim()
