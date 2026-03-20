"""
Step 44: Spiking Module Distillation

Trains spiking modules (dopamine, BG, amygdala) to match the outputs
of their numpy counterparts by running the numpy agent and recording
input→output pairs, then training the spiking modules to reproduce them.

Phase A: Distillation (match numpy outputs)
Phase B: End-to-end RL refinement (improve beyond numpy)

Run: python -m zebrav1.tests.step44_spiking_distillation
"""
import os, sys, math, torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav1.gym_env.brain_agent import BrainAgent
from zebrav1.brain.spiking_dopamine import SpikingDopamine
from zebrav1.brain.spiking_basal_ganglia import SpikingBasalGanglia
from zebrav1.brain.spiking_amygdala import SpikingAmygdala
from zebrav1.brain.device_util import get_device


def phase_a_collect_data(n_episodes=10, max_steps=300):
    """Run numpy agent, collect input→output pairs for each module."""
    print("Phase A: Collecting distillation data from numpy agent...")

    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15,
                                    max_steps=max_steps, side_panels=False)
    agent = BrainAgent(use_allostasis=True, use_spiking=False)

    dopa_data = []  # (F_vis, oL, oR, eaten, cls_probs) → (dopa, rpe)
    bg_data = []    # (valL, valR, dopa, rpe) → gate
    amyg_data = []  # (enemy_px, pred_dist, stress, gaze) → arousal

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 11)
        agent.reset()

        for t in range(max_steps):
            action = agent.act(obs, env)
            obs, rew, term, trunc, info = env.step(action)
            agent.update_post_step(info, reward=rew, done=term, env=env)

            d = agent.last_diagnostics

            # Dopamine input→output
            dopa_data.append({
                "F_vis": d.get("F_visual", 0.0),
                "oL": d.get("oL_mean", 0.3),
                "oR": d.get("oR_mean", 0.3),
                "eaten": info.get("food_eaten_this_step", 0),
                "cls_probs": list(d.get("cls_probs", [0.2]*5)),
                "dopa_target": d.get("dopa", 0.5),
                "rpe_target": d.get("rpe", 0.0),
            })

            # BG input→output
            bg_data.append({
                "valL": d.get("valL", 0.3),
                "valR": d.get("valR", 0.3),
                "dopa": d.get("dopa", 0.5),
                "rpe": d.get("rpe", 0.0),
                "gate_target": d.get("bg_gate", 0.0),
            })

            # Amygdala input→output
            amyg = d.get("amygdala", {})
            rf = agent._retinal_features if agent._retinal_features else {}
            amyg_data.append({
                "enemy_px": agent._enemy_pixels_total,
                "pred_dist": d.get("pred_dist_px", 200.0),
                "stress": d.get("stress", 0.0),
                "gaze": d.get("pred_facing_score", 0.0),
                "arousal_target": amyg.get("threat_arousal", 0.0),
            })

            if term:
                break

        print(f"  Episode {ep}: {t+1} steps, {env.total_eaten} food")

    env.close()
    print(f"  Collected {len(dopa_data)} samples")
    return dopa_data, bg_data, amyg_data


def phase_a_train_spiking(dopa_data, bg_data, amyg_data, n_epochs=50):
    """Train spiking modules to match numpy outputs."""
    print("\nPhase A: Training spiking modules (distillation)...")
    device = get_device()

    # --- Spiking Dopamine ---
    sd = SpikingDopamine(device=str(device))
    opt_d = torch.optim.Adam(sd.parameters(), lr=0.003)

    for epoch in range(n_epochs):
        total_loss = 0
        for sample in dopa_data:
            sd.reset()
            dopa, rpe, _, _ = sd.step(
                sample["F_vis"], sample["oL"], sample["oR"],
                eaten=sample["eaten"],
                cls_probs=sample["cls_probs"])
            # Can't backprop through spiking step, use direct weight update
            # Match output via MSE on dopa_level
            target = sample["dopa_target"]
            error = dopa - target
            total_loss += error ** 2
            # Manual gradient: adjust tonic drive
            with torch.no_grad():
                sd.tonic_drive.data -= 0.001 * error
        if epoch % 10 == 0:
            print(f"  Dopamine epoch {epoch}: MSE={total_loss/len(dopa_data):.4f}")

    # --- Spiking BG ---
    sbg = SpikingBasalGanglia(device=str(device))
    for epoch in range(n_epochs):
        total_loss = 0
        for sample in bg_data:
            sbg.reset()
            gate = sbg.step(sample["valL"], sample["valR"],
                            sample["dopa"], sample["rpe"])
            target = sample["gate_target"]
            error = gate - target
            total_loss += error ** 2
            with torch.no_grad():
                # Adjust D1/D2 input weights based on error
                x = torch.tensor([sample["valL"], sample["valR"]],
                                  device=str(device))
                sbg.W_input_d1.weight.data -= 0.001 * error * x.unsqueeze(0)
                sbg.W_input_d2.weight.data += 0.001 * error * x.unsqueeze(0)
        if epoch % 10 == 0:
            print(f"  BG epoch {epoch}: MSE={total_loss/len(bg_data):.4f}")

    # --- Spiking Amygdala ---
    sa = SpikingAmygdala(device=str(device))
    for epoch in range(n_epochs):
        total_loss = 0
        for sample in amyg_data:
            sa.reset()
            arousal = sa.step(sample["enemy_px"], sample["pred_dist"],
                              sample["stress"], sample["gaze"])
            target = sample["arousal_target"]
            error = arousal - target
            total_loss += error ** 2
            with torch.no_grad():
                x = torch.tensor([
                    min(1, sample["enemy_px"] * 0.08),
                    max(0, 1 - sample["pred_dist"] / 200),
                    sample["stress"], sample["gaze"]],
                    device=str(device))
                sa.W_sensory.weight.data -= 0.002 * error * x.unsqueeze(0)
        if epoch % 10 == 0:
            print(f"  Amygdala epoch {epoch}: MSE={total_loss/len(amyg_data):.4f}")

    # Save
    save_path = "zebrav1/weights/spiking_distilled.pt"
    torch.save({
        "dopamine": sd.state_dict(),
        "bg": sbg.state_dict(),
        "amygdala": sa.state_dict(),
    }, save_path)
    print(f"\n  Distilled weights saved to {save_path}")
    return sd, sbg, sa


def phase_b_rl_refinement(n_episodes=30, max_steps=500):
    """End-to-end RL with spiking modules."""
    print("\nPhase B: End-to-end RL refinement with spiking modules...")

    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15,
                                    max_steps=max_steps, side_panels=False)
    agent = BrainAgent(use_allostasis=True, use_spiking=True)

    # Load distilled weights if available
    distilled_path = "zebrav1/weights/spiking_distilled.pt"
    if os.path.exists(distilled_path):
        state = torch.load(distilled_path, map_location=agent.device,
                           weights_only=False)
        agent.dopa_sys.load_state_dict(state["dopamine"])
        agent.bg.load_state_dict(state["bg"])
        if agent.amygdala is not None:
            agent.amygdala.load_state_dict(state["amygdala"])
        print("  Loaded distilled weights")

    stats = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 13 + 7)
        agent.reset()

        for t in range(max_steps):
            action = agent.act(obs, env)
            obs, rew, term, trunc, info = env.step(action)

            food_eaten = info.get("food_eaten_this_step", 0)
            step_reward = food_eaten * 10.0 + 0.1
            if term:
                step_reward -= 50.0

            agent.update_post_step(info, reward=step_reward, done=term, env=env)
            if term or trunc:
                break

        eaten = env.total_eaten
        caught = "CAUGHT" if term else "survived"
        stats.append({"steps": t+1, "eaten": eaten, "caught": term})
        print(f"  Ep {ep:2d}: {t+1:4d} steps, {eaten:2d} food [{caught}]")

    env.close()

    # Summary
    half = n_episodes // 2
    first = stats[:half]
    second = stats[half:]
    print(f"\n  First half:  {np.mean([s['eaten'] for s in first]):.1f} food, "
          f"{np.mean([s['steps'] for s in first]):.0f} steps")
    print(f"  Second half: {np.mean([s['eaten'] for s in second]):.1f} food, "
          f"{np.mean([s['steps'] for s in second]):.0f} steps")

    # Save checkpoint
    ckpt = "zebrav1/weights/brain_checkpoint_spiking.pt"
    agent.save_checkpoint(ckpt)
    print(f"  Spiking checkpoint saved to {ckpt}")


def run_step44():
    print("=" * 60)
    print("Step 44: Spiking Module Training")
    print("=" * 60)

    # Phase A: Distillation
    dopa_data, bg_data, amyg_data = phase_a_collect_data(
        n_episodes=10, max_steps=300)
    phase_a_train_spiking(dopa_data, bg_data, amyg_data, n_epochs=50)

    # Phase B: End-to-end RL
    phase_b_rl_refinement(n_episodes=30, max_steps=500)

    print("\nDone!")


if __name__ == "__main__":
    run_step44()
