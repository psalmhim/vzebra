"""Step 43: Supervised reticulospinal weight training.

The reticulospinal shortcut (OT_L→motor_R, OT_R→motor_L) is initialized to
zeros and trained here with a supervised signal from the ground-truth flee
behavior. This bootstraps the pathway so online STDP can then refine it.

Architecture reminder:
  OT_L (left tectum, sees RIGHT visual field) → reticulo_L → motor_R
  OT_R (right tectum, sees LEFT visual field) → reticulo_R → motor_L

Training signal:
  GT flee_turn = clip(-esc_diff * 2, -1, 1)
    > 0 → CCW (left) turn → right motor should activate
    < 0 → CW  (right) turn → left  motor should activate

  Target reticulo_L output = max(0, flee_turn)   [right motor driver]
  Target reticulo_R output = max(0, -flee_turn)  [left  motor driver]

Conditions for collecting samples:
  - pred_state == 'HUNT' (active pursuit, GT flee signal is reliable)
  - predator visible in scene (has pred_x/y attributes)
"""

import sys
import os
import math

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav1.gym_env.brain_agent import BrainAgent
from zebrav1.brain.device_util import get_device

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, 'zebrav1', 'weights')
CHECKPOINT   = os.path.join(WEIGHTS_DIR, 'brain_checkpoint.pt')


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_episodes(agent, env, n_episodes=300, steps_per_ep=150, seed=0):
    """Run flee-heavy episodes and collect (oL, oR, flee_turn) tuples."""
    rng = np.random.default_rng(seed)
    samples = []   # list of (oL_np float32, oR_np float32, flee_turn float)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        agent.reset()

        # Force predator into HUNT mode from a random angle
        angle = rng.uniform(0, 2 * math.pi)
        dist  = rng.uniform(100, 200)
        env.pred_x      = env.fish_x + dist * math.cos(angle)
        env.pred_y      = env.fish_y + dist * math.sin(angle)
        env.pred_heading = (angle + math.pi) % (2 * math.pi)
        env.pred_state   = 'HUNT'
        env.pred_stamina = 1.0

        for _ in range(steps_per_ep):
            action = agent.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update_post_step(info, reward=reward,
                                   done=terminated or truncated, env=env)

            # Only collect when predator is actively hunting
            if getattr(env, 'pred_state', 'PATROL') != 'HUNT':
                if terminated or truncated:
                    break
                continue

            snn_out = getattr(agent, '_last_snn_out', None)
            if snn_out is None:
                if terminated or truncated:
                    break
                continue

            oL = snn_out.get('oL')
            oR = snn_out.get('oR')
            if oL is None or oR is None:
                if terminated or truncated:
                    break
                continue

            # GT flee direction
            _esc_ang  = math.atan2(env.fish_y - env.pred_y,
                                   env.fish_x - env.pred_x)
            _esc_diff = math.atan2(
                math.sin(_esc_ang - env.fish_heading),
                math.cos(_esc_ang - env.fish_heading))
            gt_flee_turn = float(np.clip(-_esc_diff * 2.0, -1.0, 1.0))

            samples.append((
                oL.detach().cpu().numpy().astype(np.float32),
                oR.detach().cpu().numpy().astype(np.float32),
                gt_flee_turn,
            ))

            if terminated or truncated:
                break

        if (ep + 1) % 50 == 0:
            print(f"  collected {len(samples)} samples after {ep+1} episodes")

    return samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reticulospinal(model, samples, n_epochs=10, lr=1e-3, batch_size=64):
    """Supervised training of reticulo_L and reticulo_R weights."""
    dev = model.reticulo_L.weight.device  # follow model device (cpu/cuda/mps)

    opt = torch.optim.Adam(
        list(model.reticulo_L.parameters()) +
        list(model.reticulo_R.parameters()),
        lr=lr,
    )

    oL_all = torch.tensor(
        np.concatenate([s[0] for s in samples], axis=0),
        dtype=torch.float32, device=dev)
    oR_all = torch.tensor(
        np.concatenate([s[1] for s in samples], axis=0),
        dtype=torch.float32, device=dev)
    gt_all = torch.tensor(
        [s[2] for s in samples], dtype=torch.float32,
        device=dev).unsqueeze(1)  # (N,1)

    n = len(samples)
    losses = []

    for epoch in range(n_epochs):
        perm  = torch.randperm(n)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            oL_b  = oL_all[idx]    # (B, 600)
            oR_b  = oR_all[idx]    # (B, 600)
            gt_b  = gt_all[idx]    # (B, 1)

            # Targets broadcast over 100 post-synaptic neurons
            target_right = gt_b.clamp(min=0).expand(-1, 100)   # for reticulo_L
            target_left  = (-gt_b).clamp(min=0).expand(-1, 100)  # for reticulo_R

            pred_right = model.reticulo_L(oL_b)  # (B, 100)
            pred_left  = model.reticulo_R(oR_b)  # (B, 100)

            loss = (
                (pred_right - target_right).pow(2).mean()
                + (pred_left  - target_left ).pow(2).mean()
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.reticulo_L.parameters()) +
                list(model.reticulo_R.parameters()),
                max_norm=1.0,
            )
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / max(1, n_batches)
        losses.append(avg)
        print(f"  epoch {epoch+1}/{n_epochs}  loss={avg:.5f}")

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_step43(n_episodes=300, n_epochs=10, lr=1e-3, seed=42):
    print("=" * 60)
    print("Step 43: Reticulospinal supervised training")
    print("=" * 60)

    device = get_device()
    env    = ZebrafishPreyPredatorEnv(
        render_mode=None, n_food=5, max_steps=150, side_panels=False)
    agent  = BrainAgent(device=device, use_allostasis=True,
                        world_model="place_cell")

    # Load existing checkpoint (keeps all other trained weights intact)
    if os.path.exists(CHECKPOINT):
        agent.load_checkpoint(CHECKPOINT)
    else:
        print(f"  [warn] no checkpoint at {CHECKPOINT}, starting fresh")

    model = agent.model

    print(f"\nBefore training:")
    print(f"  reticulo_L weight norm = {model.reticulo_L.weight.norm():.4f}")
    print(f"  reticulo_R weight norm = {model.reticulo_R.weight.norm():.4f}")

    # --- Phase 1: collect data ---
    print(f"\nPhase 1: collecting samples ({n_episodes} flee episodes)...")
    samples = collect_episodes(agent, env, n_episodes=n_episodes,
                               steps_per_ep=150, seed=seed)
    print(f"  total samples: {len(samples)}")

    if len(samples) < 64:
        print("  [error] too few samples, aborting")
        return

    # --- Phase 2: train ---
    print(f"\nPhase 2: supervised training ({n_epochs} epochs)...")
    losses = train_reticulospinal(model, samples, n_epochs=n_epochs, lr=lr)

    print(f"\nAfter training:")
    print(f"  reticulo_L weight norm = {model.reticulo_L.weight.norm():.4f}")
    print(f"  reticulo_R weight norm = {model.reticulo_R.weight.norm():.4f}")

    # --- Save ---
    agent.save_checkpoint(CHECKPOINT)
    print(f"\n[step43] Checkpoint saved → {CHECKPOINT}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, marker='o')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Step 43: Reticulospinal supervised training loss")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plot_path = os.path.join(PROJECT_ROOT, 'plots', 'v1_step43_reticulospinal.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"Plot saved: {plot_path}")


if __name__ == '__main__':
    run_step43()
