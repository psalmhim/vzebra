"""
Step: Three-Factor STDP Training for ZebrafishBrainV2

Trains v2 brain via reward-modulated STDP with eligibility traces.
Run N_EPISODES episodes, consolidate weights at end of each episode.

Curriculum structure (50 episodes each, 200 total):
  Stage 1 - FORAGE_EASY:   no predator, many food, 300 steps
  Stage 2 - FORAGE_HARD:   slow predator (STALK only), medium food, 400 steps
  Stage 3 - SURVIVE_EASY:  normal predator, many food, 500 steps
  Stage 4 - SURVIVE_HARD:  normal predator, normal food, 600 steps

Run: python -m zebrav2.tests.step_train_stdp
"""

import os
import sys
import math
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.sensory_bridge import inject_sensory as _inject_sensory
from zebrav2.spec import DEVICE

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
N_EPISODES = 200          # total episodes
EPISODES_PER_STAGE = 50   # 4 stages × 50 = 200
STDP_ETA = 0.0005         # consolidation learning rate (conservative)
REWARD_SPIKE_ETA = 0.002  # boosted eta for food-eating events

GOAL_FORAGE  = 0
GOAL_FLEE    = 1
GOAL_EXPLORE = 2
GOAL_SOCIAL  = 3

# ---------------------------------------------------------------------------
# Curriculum stage definitions
# ---------------------------------------------------------------------------
STAGES = [
    {
        "name": "FORAGE_EASY",
        "n_food": 25,
        "max_steps": 300,
        "pred_allowed": [],          # empty → predator locked in PATROL only
        "pred_speed_mult": 0.0,      # predator does not move
        "description": "No predator — pure foraging practice",
    },
    {
        "name": "FORAGE_HARD",
        "n_food": 15,
        "max_steps": 400,
        "pred_allowed": ["PATROL", "STALK"],   # no HUNT
        "pred_speed_mult": 0.5,                # half-speed predator
        "description": "Slow predator — forage with mild threat",
    },
    {
        "name": "SURVIVE_EASY",
        "n_food": 20,
        "max_steps": 500,
        "pred_allowed": ["PATROL", "STALK", "HUNT", "AMBUSH"],
        "pred_speed_mult": 1.0,
        "description": "Normal predator, many food — balanced survival",
    },
    {
        "name": "SURVIVE_HARD",
        "n_food": 12,
        "max_steps": 600,
        "pred_allowed": ["PATROL", "STALK", "HUNT", "AMBUSH"],
        "pred_speed_mult": 1.0,
        "description": "Normal predator, sparse food — full challenge",
    },
]


# ---------------------------------------------------------------------------
# Stage setup helpers
# ---------------------------------------------------------------------------

def _configure_stage(env, stage: dict, episode_seed: int):
    """Apply curriculum stage configuration to the env after reset."""
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)

    # Predator state restriction
    if len(stage["pred_allowed"]) == 0:
        # Lock predator to PATROL and freeze it
        env._pred_allowed_states = ["PATROL"]
        env.pred_state = "PATROL"
        env.pred_speed_current = 0.0
    else:
        env._pred_allowed_states = stage["pred_allowed"]
        env.pred_state = stage["pred_allowed"][0]
        # Scale predator base speed
        env.pred_speed = 2.7 * stage["pred_speed_mult"]

    # Respawn food at target count (add up to stage count)
    env.n_food_small_init = max(1, stage["n_food"] - max(1, stage["n_food"] // 5))
    env.n_food_large_init = max(1, stage["n_food"] // 5)

    # Replenish current food list to stage target
    current_food = len(getattr(env, 'foods', []))
    target_food = stage["n_food"]
    while current_food < target_food:
        try:
            fx, fy = env._spawn_food_open()
        except Exception:
            arena_w = getattr(env, 'arena_w', 800)
            arena_h = getattr(env, 'arena_h', 600)
            fx = float(np.random.uniform(40, arena_w - 40))
            fy = float(np.random.uniform(40, arena_h - 40))
        env.foods.append([fx, fy, "small"])
        current_food += 1


# ---------------------------------------------------------------------------
# STDP consolidation helpers
# ---------------------------------------------------------------------------

def _collect_learnable_modules(brain: ZebrafishBrainV2):
    """
    Return a list of (name, EligibilitySTDP-like object) for all modules
    that expose a consolidate() method, along with pre/post spike accessors.

    The tectum's SFGS-b rates serve as pre-spikes (sensory representation).
    The pallium's rate_S serves as post-spikes (cortical representation).
    We update traces once per step then consolidate with current DA/ACh.
    """
    learnable = []

    # Plasticity object attached directly to the brain
    plasticity = getattr(brain, 'plasticity', None)
    if plasticity is not None and hasattr(plasticity, 'consolidate'):
        learnable.append(('brain.plasticity', plasticity))

    return learnable


def _stdp_step(brain: ZebrafishBrainV2, eta: float = STDP_ETA):
    """
    Manually update eligibility traces and consolidate after each brain step.

    Uses:
      pre_spikes  ← tectum SFGS-b excitatory rates (375 neurons)
      post_spikes ← pallium superficial excitatory rates (500 neurons → 375 subsampled)

    Both are already computed inside brain.step() and cached in module buffers.
    """
    DA = brain.neuromod.DA.item()
    ACh = brain.neuromod.ACh.item()

    # Retrieve cached spike rates from the tectum and pallium
    tect_sfgs_b = brain.tectum.sfgs_b.get_rate_e().detach()  # (300,) E neurons
    pal_s_rate  = brain.pallium.pal_s.get_rate_e().detach()   # (400,) E neurons

    # --- Feedback PE-learner already runs inside brain.step() ---
    # Nothing extra needed for fb_learner.

    # --- If an EligibilitySTDP plasticity module exists, drive it ---
    plasticity = getattr(brain, 'plasticity', None)
    if plasticity is not None and hasattr(plasticity, 'update_traces'):
        # Align sizes: use tectum output as pre, pallium-S as post
        pre_n  = plasticity.pre_trace.shape[0]
        post_n = plasticity.post_trace.shape[0]
        # Sub-sample or zero-pad to match expected sizes
        pre_spikes  = _resize_vector(tect_sfgs_b, pre_n)
        post_spikes = _resize_vector(pal_s_rate,  post_n)
        plasticity.update_traces(pre_spikes, post_spikes)
        plasticity.consolidate(DA, ACh, eta=eta)


def _resize_vector(v: torch.Tensor, target_n: int) -> torch.Tensor:
    """Resize a 1-D tensor to target_n by truncation or zero-padding."""
    n = v.shape[0]
    if n == target_n:
        return v
    if n > target_n:
        return v[:target_n]
    pad = torch.zeros(target_n - n, device=v.device, dtype=v.dtype)
    return torch.cat([v, pad])


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(env, brain: ZebrafishBrainV2, stage: dict,
                 episode_seed: int) -> dict:
    """
    Run a single training episode.
    Returns a stats dict.
    """
    obs, info = env.reset(seed=episode_seed)
    brain.reset()
    _configure_stage(env, stage, episode_seed)

    max_steps   = stage["max_steps"]
    steps_done  = 0
    food_eaten  = 0
    total_reward = 0.0
    goal_counts = [0, 0, 0, 0]   # forage, flee, explore, social
    prev_total_eaten = 0

    for step_idx in range(max_steps):
        # --- Signal flee state to env before step ---
        is_flee = (brain.current_goal == GOAL_FLEE)
        if hasattr(env, 'set_flee_active'):
            env.set_flee_active(is_flee, panic_intensity=0.8 if is_flee else 0.0)

        # --- Inject geometric sensory ---
        _inject_sensory(env)

        # --- Brain step ---
        out = brain.step(obs, env)
        goal = out['goal']
        goal_counts[goal] += 1

        # --- Env step ---
        action = np.array([out['turn'], out['speed']], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Track food eaten this step ---
        eaten_this_step = info.get('food_eaten_this_step', 0)
        # Sync to env._eaten_now so brain.step sees it next call
        env._eaten_now = eaten_this_step

        total_reward += float(reward)
        food_eaten += int(eaten_this_step)
        steps_done += 1

        # --- STDP consolidation ---
        # Boost eta on food-eating steps (reward spike)
        step_eta = REWARD_SPIKE_ETA if eaten_this_step > 0 else STDP_ETA
        _stdp_step(brain, eta=step_eta)

        if terminated or truncated:
            break

    return {
        "steps": steps_done,
        "food_eaten": food_eaten,
        "total_reward": total_reward,
        "goal_counts": goal_counts,
        "survived": not terminated,   # terminated = died; truncated = time limit
    }


# ---------------------------------------------------------------------------
# Weight saving
# ---------------------------------------------------------------------------

def _save_weights(brain: ZebrafishBrainV2, total_episodes: int,
                  out_dir: str = None):
    """Save plasticity weights and full brain state dict."""
    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, "zebrav2", "weights")
    os.makedirs(out_dir, exist_ok=True)

    # --- Plasticity-specific checkpoint ---
    plasticity = getattr(brain, 'plasticity', None)
    plasticity_W = None
    if plasticity is not None and hasattr(plasticity, 'W'):
        plasticity_W = plasticity.W.detach().cpu()

    stdp_path = os.path.join(out_dir, "stdp_trained.pt")
    torch.save({
        'plasticity_W': plasticity_W,
        'neuromod': {
            'DA':  brain.neuromod.DA.item(),
            'HT5': brain.neuromod.HT5.item(),
            'NA':  brain.neuromod.NA.item(),
            'ACh': brain.neuromod.ACh.item(),
        },
        'episodes': total_episodes,
    }, stdp_path)
    print(f"  Plasticity checkpoint → {stdp_path}")

    # --- Full learnable state dict ---
    # Collect all nn.Linear modules by name
    learnable_sd = {}
    linear_modules = [
        ("tectum.W_on_sfgsb",    brain.tectum.W_on_sfgsb),
        ("tectum.W_off_sfgsd",   brain.tectum.W_off_sfgsd),
        ("tectum.W_loom_sgc",    brain.tectum.W_loom_sgc),
        ("tectum.W_ds_so",       brain.tectum.W_ds_so),
        ("pallium.W_tc_pals",    brain.pallium.W_tc_pals),
        ("pallium.W_pals_pald",  brain.pallium.W_pals_pald),
        ("pallium.W_FB",         brain.pallium.W_FB),
        ("pallium.W_goal_att",   brain.pallium.W_goal_att),
    ]
    for name, module in linear_modules:
        if module is not None:
            for k, v in module.state_dict().items():
                learnable_sd[f"{name}.{k}"] = v.detach().cpu()

    # fb_learner.W_FB is an nn.Parameter (not nn.Linear), save directly
    fb_W = brain.fb_learner.W_FB
    if fb_W is not None:
        learnable_sd["fb_learner.W_FB"] = fb_W.detach().cpu()

    full_path = os.path.join(out_dir, "stdp_brain_weights.pt")
    torch.save({
        'state_dict': learnable_sd,
        'episodes': total_episodes,
        'device_str': str(DEVICE),
    }, full_path)
    print(f"  Full weight checkpoint → {full_path}")


# ---------------------------------------------------------------------------
# Per-stage summary printer
# ---------------------------------------------------------------------------

def _print_summary(stage_name: str, ep_stats: list, stage_idx: int):
    """Print aggregate stats for a completed stage."""
    n = len(ep_stats)
    if n == 0:
        return
    avg_steps   = np.mean([s["steps"] for s in ep_stats])
    avg_food    = np.mean([s["food_eaten"] for s in ep_stats])
    avg_reward  = np.mean([s["total_reward"] for s in ep_stats])
    survive_pct = 100.0 * np.mean([float(s["survived"]) for s in ep_stats])
    # Goal distribution
    all_goals = np.array([s["goal_counts"] for s in ep_stats])
    total_steps_all = np.sum([s["steps"] for s in ep_stats])
    if total_steps_all > 0:
        goal_pct = all_goals.sum(axis=0) / total_steps_all * 100
    else:
        goal_pct = np.zeros(4)

    print(f"\n  Stage {stage_idx + 1} [{stage_name}] summary ({n} episodes):")
    print(f"    Avg steps:   {avg_steps:.0f}")
    print(f"    Avg food:    {avg_food:.2f}")
    print(f"    Avg reward:  {avg_reward:.2f}")
    print(f"    Survival:    {survive_pct:.0f}%")
    print(f"    Goal %:      forage={goal_pct[0]:.0f}% flee={goal_pct[1]:.0f}%"
          f" explore={goal_pct[2]:.0f}% social={goal_pct[3]:.0f}%")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    print("=" * 64)
    print("ZebrafishBrainV2 — Three-Factor STDP Curriculum Training")
    print(f"Device: {DEVICE}")
    print(f"N_EPISODES={N_EPISODES}  EPISODES_PER_STAGE={EPISODES_PER_STAGE}")
    print(f"STDP eta={STDP_ETA}  reward-spike eta={REWARD_SPIKE_ETA}")
    print("=" * 64)

    # Build environment (use generous food count; stages override at runtime)
    env = ZebrafishPreyPredatorEnv(
        render_mode=None,
        n_food=20,
        max_steps=600,
        side_panels=False,
    )

    brain = ZebrafishBrainV2(device=DEVICE)

    all_stats = []       # flat list over all episodes
    stage_stats = []     # per-stage list of lists

    global_episode = 0

    for stage_idx, stage in enumerate(STAGES):
        stage_name   = stage["name"]
        stage_desc   = stage["description"]
        ep_stats_stage = []

        print(f"\n{'─' * 64}")
        print(f"  Stage {stage_idx + 1}/4: {stage_name}")
        print(f"  {stage_desc}")
        print(f"  max_steps={stage['max_steps']}  n_food={stage['n_food']}")
        print(f"{'─' * 64}")

        for local_ep in range(EPISODES_PER_STAGE):
            episode_seed = global_episode   # deterministic seed per episode

            stats = _run_episode(env, brain, stage, episode_seed)
            ep_stats_stage.append(stats)
            all_stats.append(stats)
            global_episode += 1

            # Print progress every 10 episodes
            if (local_ep + 1) % 10 == 0:
                recent = ep_stats_stage[-10:]
                avg_food   = np.mean([s["food_eaten"] for s in recent])
                avg_steps  = np.mean([s["steps"] for s in recent])
                surv_pct   = 100.0 * np.mean([float(s["survived"]) for s in recent])
                print(f"  Ep {global_episode:3d} [{stage_name}] "
                      f"food={avg_food:.1f}  steps={avg_steps:.0f}  "
                      f"survival={surv_pct:.0f}%  "
                      f"DA={brain.neuromod.DA.item():.3f}  "
                      f"ACh={brain.neuromod.ACh.item():.3f}")

        _print_summary(stage_name, ep_stats_stage, stage_idx)
        stage_stats.append(ep_stats_stage)

    # -----------------------------------------------------------------------
    # Overall training summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 64}")
    print("TRAINING COMPLETE — Overall Statistics")
    print(f"{'=' * 64}")
    total_food   = sum(s["food_eaten"] for s in all_stats)
    total_steps  = sum(s["steps"] for s in all_stats)
    survive_pct  = 100.0 * np.mean([float(s["survived"]) for s in all_stats])
    all_goals    = np.array([s["goal_counts"] for s in all_stats])
    goal_totals  = all_goals.sum(axis=0)
    goal_frac    = goal_totals / (total_steps + 1e-8) * 100

    print(f"  Total episodes:  {N_EPISODES}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Total food:      {total_food}")
    print(f"  Overall survival:{survive_pct:.0f}%")
    print(f"  Goal distribution:")
    print(f"    forage={goal_frac[0]:.0f}%  flee={goal_frac[1]:.0f}%"
          f"  explore={goal_frac[2]:.0f}%  social={goal_frac[3]:.0f}%")

    # Stage-by-stage food progression (proxy for learning)
    print("\n  Stage food progression (expected improvement with STDP):")
    for i, (stage, stats_list) in enumerate(zip(STAGES, stage_stats)):
        avg_f = np.mean([s["food_eaten"] for s in stats_list])
        avg_s = 100.0 * np.mean([float(s["survived"]) for s in stats_list])
        print(f"    Stage {i+1} [{stage['name']:15s}]: "
              f"food={avg_f:.2f}  survival={avg_s:.0f}%")

    print("\n  Expected effect on decision quality:")
    print("    - Reward-modulated STDP strengthens food-approach circuits")
    print("    - DA-gated consolidation improves flee-vs-forage trade-off")
    print("    - ACh-gated plasticity enhances attentional focus on threats")
    print("    - Curriculum progression expected to lift v2 score above v1 89/100")

    # -----------------------------------------------------------------------
    # Save weights
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 64}")
    print("Saving weights...")
    _save_weights(brain, total_episodes=N_EPISODES)

    env.close()
    print(f"\nDone. Run python -m zebrav2.tests.step29b_v1_vs_v2 to evaluate.")
    return brain, all_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    train()
