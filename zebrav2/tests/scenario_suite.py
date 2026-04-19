"""
Scenario-based learning observatory for ZebrafishBrainV2.

Each scenario isolates one aspect of learning by starting from a distinct
knowledge state, then runs N rounds and tracks the relevant learning curves.

Scenarios
---------
1. TABULA_RASA     — fresh brain, no checkpoint; world model emerges from scratch
2. WORLD_AMNESIC   — round-50 weights but geo/place erased; measures relearning speed
3. FEAR_LEARNER    — round-50 weights + fresh amygdala + aggressive predator
4. SOCIAL_LEARNER  — round-50 weights in multi-agent env; tracks social weight evolution
5. EFE_ADAPTER     — round-50 weights + biased food layout; tracks goal_bias drift

Usage
-----
    python -m zebrav2.tests.scenario_suite                          # all scenarios
    python -m zebrav2.tests.scenario_suite --scenario fear_learner  # one scenario
    python -m zebrav2.tests.scenario_suite --rounds 20 --ckpt zebrav2/checkpoints/ckpt_round_0005.pt
"""
import os
import sys
import math
import argparse
import json
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.spec import DEVICE
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.brain.personality import get_personality, assign_personalities
from zebrav2.engine.trainer import TrainingEngine
from zebrav2.engine.config import TrainingConfig
from zebrav2.engine.checkpoint import CheckpointManager

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

GOAL_NAMES = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']


def _make_engine(predator_ai='none', max_steps=300, n_food=20, food_respawn=True):
    cfg = TrainingConfig()
    cfg['env.max_steps'] = max_steps
    cfg['env.n_food'] = n_food
    cfg['env.food_respawn'] = food_respawn
    cfg['env.predator_ai'] = predator_ai
    cfg['training.autosave_world'] = False
    engine = TrainingEngine(config=cfg)
    engine.running = True
    return engine


def _load_checkpoint(brain, ckpt_path: str, ckpt_mgr: CheckpointManager):
    """Load checkpoint into brain. Returns (round_num, metrics)."""
    if ckpt_path and os.path.exists(ckpt_path):
        rnd, metrics = ckpt_mgr.load(brain, ckpt_path)
        print(f"    Loaded checkpoint: round={rnd}")
        return rnd, metrics
    print("    No checkpoint — starting from scratch")
    brain.reset()
    return 0, {}


def _geo_coverage(brain):
    return float(np.mean(brain.geo_model.visit_count > 0))


def _place_coverage(brain):
    return float(np.mean(brain.place.food_rate.cpu().numpy() > 0))


def _vae_nodes(brain):
    return int(brain.vae.memory.n_allocated)


def _fear_norm(brain):
    return float(np.linalg.norm(brain.amygdala.W_la_cea.cpu().numpy()))


def _print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _print_row(ep, label_vals: dict):
    parts = [f"ep={ep:>3}"] + [f"{k}={v}" for k, v in label_vals.items()]
    print("  " + "  |  ".join(parts))


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 1 — TABULA_RASA
# ──────────────────────────────────────────────────────────────────────────────

def run_tabula_rasa(n_rounds=15, n_food=20, max_steps=300):
    """
    A naive fish with no prior knowledge enters the world.
    Watch geo coverage, VAE episodic memory, and food intake grow over time.
    """
    _print_header("SCENARIO 1 · TABULA RASA — world model emerging from scratch")

    engine = _make_engine(predator_ai='simple', max_steps=max_steps, n_food=n_food)
    brain = ZebrafishBrainV2(device=DEVICE)
    brain.reset()
    engine.brain = brain

    history = []
    for ep in range(1, n_rounds + 1):
        metrics = engine.run_round(ep)
        brain.on_episode_end(metrics['fitness'])
        row = {
            'food':       metrics['food_eaten'],
            'geo_cov':    f"{_geo_coverage(brain):.2f}",
            'vae_nodes':  _vae_nodes(brain),
            'place_cov':  f"{_place_coverage(brain):.2f}",
            'w_alarm':    f"{brain.social_mem.w_alarm:.2f}",
            'goal_bias':  f"{brain.meta_goal.goal_bias.tolist()}",
        }
        _print_row(ep, row)
        history.append({'ep': ep, **metrics,
                        'geo_coverage': _geo_coverage(brain),
                        'vae_nodes': _vae_nodes(brain),
                        'place_coverage': _place_coverage(brain)})

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 2 — WORLD_AMNESIC
# ──────────────────────────────────────────────────────────────────────────────

def run_world_amnesic(ckpt_path=None, n_rounds=15, max_steps=300):
    """
    Load pretrained classifier/SNN weights but erase spatial world knowledge.
    Measures: how fast does the fish re-learn the map vs a tabula rasa baseline?
    """
    _print_header("SCENARIO 2 · WORLD AMNESIC — spatial memory erased, motor skills intact")

    ckpt_mgr = CheckpointManager(os.path.join(PROJECT_ROOT, 'zebrav2', 'checkpoints'))
    engine = _make_engine(predator_ai='simple', max_steps=max_steps)
    brain = ZebrafishBrainV2(device=DEVICE)
    _load_checkpoint(brain, ckpt_path, ckpt_mgr)
    engine.brain = brain

    # Surgical amnesia: erase only spatial world knowledge
    brain.geo_model.reset()
    brain.place.food_rate.zero_()
    brain.place.risk_rate.zero_()
    brain.place.visit_count.zero_()
    n = brain.vae.memory.n_allocated
    if n > 0:
        brain.vae.memory.centroids[:n] = 0.0
        brain.vae.memory.food_rate[:n] = 0.0
        brain.vae.memory.risk[:n] = 0.0
        brain.vae.memory.n_allocated = 0
    print("    [Amnesia applied] geo, place-cells, VAE episodic memory → zeroed")

    history = []
    for ep in range(1, n_rounds + 1):
        metrics = engine.run_round(ep)
        brain.on_episode_end(metrics['fitness'])
        row = {
            'food':      metrics['food_eaten'],
            'geo_cov':   f"{_geo_coverage(brain):.2f}",
            'vae_nodes': _vae_nodes(brain),
            'place_cov': f"{_place_coverage(brain):.2f}",
        }
        _print_row(ep, row)
        history.append({'ep': ep, **metrics,
                        'geo_coverage': _geo_coverage(brain),
                        'vae_nodes': _vae_nodes(brain),
                        'place_coverage': _place_coverage(brain)})

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 3 — FEAR_LEARNER
# ──────────────────────────────────────────────────────────────────────────────

def run_fear_learner(ckpt_path=None, n_rounds=15, max_steps=400):
    """
    Load pretrained weights, wipe amygdala fear conditioning, then expose fish
    to an aggressive predator. Tracks W_la_cea norm and fear_baseline growing.
    """
    _print_header("SCENARIO 3 · FEAR LEARNER — naive amygdala meets aggressive predator")

    ckpt_mgr = CheckpointManager(os.path.join(PROJECT_ROOT, 'zebrav2', 'checkpoints'))
    engine = _make_engine(predator_ai='intelligent', max_steps=max_steps)
    brain = ZebrafishBrainV2(device=DEVICE, personality=get_personality('shy'))
    _load_checkpoint(brain, ckpt_path, ckpt_mgr)
    engine.brain = brain

    # Reset fear conditioning (not neuron firing state)
    brain.amygdala.reset_full()
    print("    [Fear reset] amygdala W_la_cea → 0.5, fear_baseline → 0")

    history = []
    for ep in range(1, n_rounds + 1):
        metrics = engine.run_round(ep)
        brain.on_episode_end(metrics['fitness'])
        fear_n = _fear_norm(brain)
        row = {
            'food':       metrics['food_eaten'],
            'caught':     int(metrics['caught']),
            'flee_frac':  f"{metrics['goal_distribution'].get('FLEE', 0) / max(1, metrics['survived']):.2f}",
            'fear_norm':  f"{fear_n:.3f}",
            'fear_base':  f"{brain.amygdala.fear_baseline:.3f}",
        }
        _print_row(ep, row)
        history.append({'ep': ep, **metrics,
                        'fear_norm': fear_n,
                        'fear_baseline': float(brain.amygdala.fear_baseline)})

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 4 — SOCIAL_LEARNER
# ──────────────────────────────────────────────────────────────────────────────

def run_social_learner(ckpt_path=None, n_rounds=15, max_steps=400):
    """
    Load pretrained weights, reset social inference weights to neutral (1.0),
    then run in multi-agent env with mixed-personality shoal.
    Tracks w_alarm, w_food_cue, w_competition per episode.
    """
    _print_header("SCENARIO 4 · SOCIAL LEARNER — adapting to a mixed-personality shoal")

    ckpt_mgr = CheckpointManager(os.path.join(PROJECT_ROOT, 'zebrav2', 'checkpoints'))
    engine = _make_engine(predator_ai='intelligent', max_steps=max_steps)

    # 5 fish: 1 focal + 4 conspecifics
    personalities = assign_personalities(5, mode='mixed')
    brains = [ZebrafishBrainV2(device=DEVICE, personality=p) for p in personalities]

    for i, brain in enumerate(brains):
        _load_checkpoint(brain, ckpt_path, ckpt_mgr)
        # Reset social weights for focal fish to observe learning
        brain.social_mem.w_alarm = 1.0
        brain.social_mem.w_food_cue = 1.0
        brain.social_mem.w_competition = 1.0
        brain.social_mem._alarm_precision_ema = 0.5
        brain.social_mem._food_precision_ema = 0.5

    focal = brains[0]
    engine.brain = focal

    # Attach multi-brain list so run_round_multi can be used
    # We run run_round_multi with all brains (returns focal metrics)
    history = []
    for ep in range(1, n_rounds + 1):
        metrics = engine.run_round_multi(ep, brains, n_fish=len(brains))
        focal.on_episode_end(metrics['fitness'])
        row = {
            'food':        metrics['food_eaten'],
            'w_alarm':     f"{focal.social_mem.w_alarm:.3f}",
            'w_food_cue':  f"{focal.social_mem.w_food_cue:.3f}",
            'w_comp':      f"{focal.social_mem.w_competition:.3f}",
            'alarm_prec':  f"{focal.social_mem._alarm_precision_ema:.2f}",
        }
        _print_row(ep, row)
        history.append({'ep': ep, **metrics,
                        'w_alarm': focal.social_mem.w_alarm,
                        'w_food_cue': focal.social_mem.w_food_cue,
                        'w_competition': focal.social_mem.w_competition})

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 5 — EFE_ADAPTER
# ──────────────────────────────────────────────────────────────────────────────

def _bias_food_toward_corners(engine, env, rng, corner='top_right'):
    """Override env food spawn to cluster 80% of food in one corner."""
    aw = getattr(env, 'arena_w', 800)
    ah = getattr(env, 'arena_h', 600)
    corners = {
        'top_right':    (aw * 0.75, ah * 0.20),
        'bottom_left':  (aw * 0.20, ah * 0.80),
    }
    cx, cy = corners.get(corner, corners['top_right'])
    env.foods = []
    for _ in range(16):  # 80% in corner patch
        fx = float(np.clip(cx + rng.normal(0, 40), 30, aw - 30))
        fy = float(np.clip(cy + rng.normal(0, 40), 30, ah - 30))
        env.foods.append([fx, fy, 'small'])
    for _ in range(4):  # 20% scattered
        env.foods.append([
            float(rng.uniform(50, aw - 50)),
            float(rng.uniform(50, ah - 50)),
            'small'])


def run_efe_adapter(ckpt_path=None, n_rounds=20, max_steps=400):
    """
    Load pretrained weights, then shift food distribution so FORAGE in top-right
    is consistently rewarded. Tracks goal_bias[FORAGE] drift and food intake.
    """
    _print_header("SCENARIO 5 · EFE ADAPTER — food biased to top-right, watch goal_bias shift")

    ckpt_mgr = CheckpointManager(os.path.join(PROJECT_ROOT, 'zebrav2', 'checkpoints'))
    engine = _make_engine(predator_ai='none', max_steps=max_steps)
    brain = ZebrafishBrainV2(device=DEVICE, personality=get_personality('explorer'))
    _load_checkpoint(brain, ckpt_path, ckpt_mgr)

    # Reset MetaGoalWeights to neutral so we observe fresh learning
    import torch
    with torch.no_grad():
        brain.meta_goal.goal_bias.zero_()
        brain.meta_goal.mod_w.fill_(1.0)
    brain.meta_goal._fitness_ema = None  # bootstrap on first episode
    print("    [EFE reset] goal_bias → 0, mod_w → 1, fitness_ema → bootstrap")

    engine.brain = brain

    rng = np.random.RandomState(42)
    history = []

    # Monkey-patch _create_env to force biased food layout
    _orig_spawn = engine._spawn_food_patches
    def _biased_spawn(env, n_total=20, seed=None):
        _bias_food_toward_corners(engine, env, rng, corner='top_right')
    engine._spawn_food_patches = _biased_spawn

    for ep in range(1, n_rounds + 1):
        metrics = engine.run_round(ep)
        brain.on_episode_end(metrics['fitness'])
        gb = brain.meta_goal.goal_bias.tolist()
        row = {
            'food':     metrics['food_eaten'],
            'fitness':  f"{metrics['fitness']:.1f}",
            'gb_for':   f"{gb[0]:.3f}",
            'gb_flee':  f"{gb[1]:.3f}",
            'gb_expl':  f"{gb[2]:.3f}",
            'gb_soc':   f"{gb[3]:.3f}",
        }
        _print_row(ep, row)
        history.append({'ep': ep, **metrics,
                        'goal_bias': gb,
                        'mod_w': brain.meta_goal.mod_w.tolist()})

    engine._spawn_food_patches = _orig_spawn
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(name: str, history: list):
    if not history:
        return
    first = history[0]
    last = history[-1]
    print(f"\n  Summary [{name}]")
    print(f"    Episodes:   {len(history)}")
    print(f"    Food ep1:   {first.get('food_eaten', '?')}  →  ep{len(history)}: {last.get('food_eaten', '?')}")
    print(f"    Fitness ep1: {first.get('fitness', 0):.1f}  →  ep{len(history)}: {last.get('fitness', 0):.1f}")
    if 'geo_coverage' in last:
        print(f"    GeoCoverage ep1: {first.get('geo_coverage', 0):.2f}  →  ep{len(history)}: {last.get('geo_coverage', 0):.2f}")
    if 'vae_nodes' in last:
        print(f"    VAE nodes ep1: {first.get('vae_nodes', 0)}  →  ep{len(history)}: {last.get('vae_nodes', 0)}")
    if 'fear_norm' in last:
        print(f"    Fear norm ep1: {first.get('fear_norm', 0):.3f}  →  ep{len(history)}: {last.get('fear_norm', 0):.3f}")
    if 'w_alarm' in last:
        print(f"    w_alarm ep1: {first.get('w_alarm', 1):.3f}  →  ep{len(history)}: {last.get('w_alarm', 1):.3f}")
    if 'goal_bias' in last:
        print(f"    goal_bias ep1: {[f'{x:.3f}' for x in first.get('goal_bias', [])]}")
        print(f"    goal_bias ep{len(history)}: {[f'{x:.3f}' for x in last.get('goal_bias', [])]}")


# ──────────────────────────────────────────────────────────────────────────────
# Save JSON results
# ──────────────────────────────────────────────────────────────────────────────

def _save_results(results: dict, out_dir='zebrav2/tests/scenario_results'):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'scenario_results_{ts}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    'tabula_rasa':   run_tabula_rasa,
    'world_amnesic': run_world_amnesic,
    'fear_learner':  run_fear_learner,
    'social_learner': run_social_learner,
    'efe_adapter':   run_efe_adapter,
}

SCENARIO_NEEDS_CKPT = {
    'tabula_rasa':   False,
    'world_amnesic': True,
    'fear_learner':  True,
    'social_learner': True,
    'efe_adapter':   True,
}


def main():
    parser = argparse.ArgumentParser(description='Scenario-based learning observatory')
    parser.add_argument('--scenario', default='all',
                        help='Scenario name or "all". Options: ' + ', '.join(SCENARIOS))
    parser.add_argument('--rounds', type=int, default=10,
                        help='Episodes per scenario (default: 10)')
    parser.add_argument('--ckpt', default=None,
                        help='Checkpoint path for scenarios that need pretrained weights')
    parser.add_argument('--no_save', action='store_true',
                        help='Skip saving JSON results')
    args = parser.parse_args()

    # Auto-find latest checkpoint if not specified
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_dir = os.path.join(PROJECT_ROOT, 'zebrav2', 'checkpoints')
        if os.path.isdir(ckpt_dir):
            pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt')
                         and f.startswith('ckpt_round_'))
            if pts:
                ckpt_path = os.path.join(ckpt_dir, pts[-1])
                print(f"  Auto-selected checkpoint: {ckpt_path}")

    to_run = list(SCENARIOS.items()) if args.scenario == 'all' else [
        (args.scenario, SCENARIOS[args.scenario])]

    all_results = {}
    t0 = time.time()

    for name, fn in to_run:
        needs_ckpt = SCENARIO_NEEDS_CKPT[name]
        if needs_ckpt and ckpt_path is None:
            print(f"\n  [SKIP] {name} requires a checkpoint (use --ckpt or train first)")
            continue
        try:
            if needs_ckpt:
                history = fn(ckpt_path=ckpt_path, n_rounds=args.rounds)
            else:
                history = fn(n_rounds=args.rounds)
            _print_summary(name, history)
            all_results[name] = history
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {name}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  All scenarios complete in {elapsed:.1f}s")

    if not args.no_save and all_results:
        _save_results(all_results)


if __name__ == '__main__':
    main()
