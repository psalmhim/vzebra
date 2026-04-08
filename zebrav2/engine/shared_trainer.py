"""
Shared-weight multi-agent trainer (A3C-style).

N workers run episodes with different seeds. All workers start from the
same shared weights. After each round, weight deltas are averaged and
applied to the master model.

Benefits over single-agent training:
- Diverse experience: each worker encounters different food patches,
  predator positions, and decision contexts
- Reduced variance: averaging N worker deltas cancels out seed-specific noise
- Better generalization: policy improves across a distribution of environments
- Predator naturally hunts different fish across workers (realistic pressure)

Usage:
    trainer = SharedWeightTrainer(config, n_workers=5)
    trainer.train_shared(n_rounds=20)
"""

import os
import sys
import time
import math
import json
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebrav2.engine.trainer import TrainingEngine
from zebrav2.brain.personality import assign_personalities
from zebrav2.brain.brain_v2 import ZebrafishBrainV2
from zebrav2.spec import DEVICE

GOAL_NAMES = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']


# ---------------------------------------------------------------------------
# Weight snapshot helpers (parameters only — no episode-state buffers)
# ---------------------------------------------------------------------------

def _snap_module_params(module):
    """Capture nn.Parameter tensors only (skips registered buffers)."""
    return {name: param.data.clone().cpu()
            for name, param in module.named_parameters()}


def _snap_tensor(t):
    """Clone a plain torch tensor (not nn.Parameter)."""
    return t.clone().cpu()


def _snap_numpy(a):
    """Copy a numpy array."""
    return a.copy()


def snapshot_weights(brain):
    """
    Deep-copy all learnable state from brain.
    Returns a flat dict of tensors/arrays.
    """
    return {
        # nn.Module parameter dicts
        'critic':          _snap_module_params(brain.critic),
        'classifier':      _snap_module_params(brain.classifier),
        'pallium':         _snap_module_params(brain.pallium),
        'habit':           _snap_module_params(brain.habit),
        'thalamus':        _snap_module_params(brain.thalamus),
        'vae_pool':        _snap_module_params(brain.vae.pool),
        'vae_encoder':     _snap_module_params(brain.vae.encoder),
        'vae_decoder':     _snap_module_params(brain.vae.decoder),
        'vae_transition':  _snap_module_params(brain.vae.transition),
        # Plain tensor weights (STDP / RL arrays)
        'cerebellum_W_pf':   _snap_tensor(brain.cerebellum.W_pf.data),
        'amygdala_W_la_cea': _snap_tensor(brain.amygdala.W_la_cea),
        'place_food_rate':   _snap_tensor(brain.place.food_rate),
        'place_risk_rate':   _snap_tensor(brain.place.risk_rate),
        'place_visit_count': _snap_tensor(brain.place.visit_count),
        # Numpy arrays
        'geo_food':  _snap_numpy(brain.geo_model.food_score),
        'geo_risk':  _snap_numpy(brain.geo_model.risk_score),
        'geo_visits': _snap_numpy(brain.geo_model.visit_count),
        'habenula_frustration': _snap_numpy(brain.habenula.frustration),
        # Scalars / floats
        'amygdala_fear_baseline': float(brain.amygdala.fear_baseline),
        'neuromod_V': float(brain.neuromod.V.item()),
    }


def restore_weights(brain, snap):
    """Set brain's learnable weights to the values in snap."""
    dev = brain.device

    # nn.Module parameters
    for mod_name, param_dict in [
        ('critic', snap['critic']),
        ('classifier', snap['classifier']),
        ('pallium', snap['pallium']),
        ('habit', snap['habit']),
        ('thalamus', snap['thalamus']),
    ]:
        module = getattr(brain, mod_name)
        with torch.no_grad():
            for name, param in module.named_parameters():
                param.data.copy_(param_dict[name].to(dev))

    for vae_name, param_dict in [
        ('pool', snap['vae_pool']),
        ('encoder', snap['vae_encoder']),
        ('decoder', snap['vae_decoder']),
        ('transition', snap['vae_transition']),
    ]:
        module = getattr(brain.vae, vae_name)
        with torch.no_grad():
            for name, param in module.named_parameters():
                param.data.copy_(param_dict[name].to(dev))

    # Plain tensors
    brain.cerebellum.W_pf.data.copy_(snap['cerebellum_W_pf'].to(dev))
    brain.amygdala.W_la_cea.copy_(snap['amygdala_W_la_cea'].to(dev))
    brain.place.food_rate.copy_(snap['place_food_rate'].to(dev))
    brain.place.risk_rate.copy_(snap['place_risk_rate'].to(dev))
    brain.place.visit_count.copy_(snap['place_visit_count'].to(dev))

    # Numpy arrays
    brain.geo_model.food_score[:] = snap['geo_food']
    brain.geo_model.risk_score[:] = snap['geo_risk']
    brain.geo_model.visit_count[:] = snap['geo_visits']
    brain.habenula.frustration[:] = snap['habenula_frustration']

    # Scalars
    brain.amygdala.fear_baseline = snap['amygdala_fear_baseline']
    brain.neuromod.V.fill_(snap['neuromod_V'])


def _compute_delta(before, after):
    """Compute delta = after - before for each weight."""
    delta = {}
    for key in before:
        b, a = before[key], after[key]
        if isinstance(b, dict):        # param_dict
            delta[key] = {k: a[k] - b[k] for k in b}
        elif isinstance(b, torch.Tensor):
            delta[key] = a - b
        elif isinstance(b, np.ndarray):
            delta[key] = a - b
        else:                          # float / scalar
            delta[key] = a - b
    return delta


def _average_deltas(deltas, visit_count_keys):
    """
    Average N weight deltas. Visit counts are SUMMED (exploration union)
    rather than averaged, so the master accumulates all workers' visits.
    """
    n = len(deltas)
    avg = {}
    for key in deltas[0]:
        b = deltas[0][key]
        if key in visit_count_keys:
            # Sum: master gets the union of all workers' exploration
            if isinstance(b, dict):
                avg[key] = {k: sum(d[key][k] for d in deltas) for k in b}
            elif isinstance(b, (torch.Tensor, np.ndarray)):
                avg[key] = sum(d[key] for d in deltas)
            else:
                avg[key] = sum(d[key] for d in deltas)
        else:
            # Average
            if isinstance(b, dict):
                avg[key] = {k: sum(d[key][k] for d in deltas) / n for k in b}
            elif isinstance(b, (torch.Tensor, np.ndarray)):
                avg[key] = sum(d[key] for d in deltas) / n
            else:
                avg[key] = sum(d[key] for d in deltas) / n
    return avg


def apply_avg_delta(brain, initial, avg_delta):
    """Apply averaged delta: master_new = initial + avg_delta."""
    dev = brain.device

    # nn.Module parameters
    for mod_name, mod_key in [
        ('critic', 'critic'), ('classifier', 'classifier'),
        ('pallium', 'pallium'), ('habit', 'habit'), ('thalamus', 'thalamus'),
    ]:
        module = getattr(brain, mod_name)
        param_init = initial[mod_key]
        param_delta = avg_delta[mod_key]
        with torch.no_grad():
            for name, param in module.named_parameters():
                new_val = (param_init[name] + param_delta[name]).to(dev)
                param.data.copy_(new_val)

    for vae_name, key in [
        ('pool', 'vae_pool'), ('encoder', 'vae_encoder'),
        ('decoder', 'vae_decoder'), ('transition', 'vae_transition'),
    ]:
        module = getattr(brain.vae, vae_name)
        param_init = initial[key]
        param_delta = avg_delta[key]
        with torch.no_grad():
            for name, param in module.named_parameters():
                new_val = (param_init[name] + param_delta[name]).to(dev)
                param.data.copy_(new_val)

    # Plain tensors
    for attr, key in [
        (brain.cerebellum.W_pf.data, 'cerebellum_W_pf'),
        (brain.amygdala.W_la_cea, 'amygdala_W_la_cea'),
        (brain.place.food_rate, 'place_food_rate'),
        (brain.place.risk_rate, 'place_risk_rate'),
        (brain.place.visit_count, 'place_visit_count'),
    ]:
        new_val = (initial[key] + avg_delta[key]).to(dev)
        attr.copy_(new_val)

    # Numpy arrays — food/risk averaged, visits summed (handled by avg_delta)
    brain.geo_model.food_score[:] = initial['geo_food'] + avg_delta['geo_food']
    brain.geo_model.risk_score[:] = initial['geo_risk'] + avg_delta['geo_risk']
    brain.geo_model.visit_count[:] = initial['geo_visits'] + avg_delta['geo_visits']
    brain.habenula.frustration[:] = (initial['habenula_frustration']
                                     + avg_delta['habenula_frustration'])

    # Scalars
    brain.amygdala.fear_baseline = (initial['amygdala_fear_baseline']
                                    + avg_delta['amygdala_fear_baseline'])
    brain.neuromod.V.fill_(initial['neuromod_V'] + avg_delta['neuromod_V'])


# ---------------------------------------------------------------------------
# SharedWeightTrainer
# ---------------------------------------------------------------------------

class SharedWeightTrainer(TrainingEngine):
    """
    Multi-agent trainer with a single shared model.

    Each round:
      1. Snapshot master weights as initial state.
      2. For each worker (different seed / personality):
           a. Restore brain to initial state.
           b. Run one episode (brain.step() modifies weights in-place).
           c. Snapshot final weights → compute delta.
      3. Average all N deltas (sum visit counts).
      4. Apply averaged delta to master.
      5. Checkpoint if needed.

    This gives each round N × as many gradient signals as single-agent
    training, while averaging out seed-specific noise.
    """

    def __init__(self, config=None, n_workers=5):
        super().__init__(config)
        self.n_workers = n_workers
        # Keys whose deltas are summed (exploration counts), not averaged
        self._visit_keys = {'place_visit_count', 'geo_visits'}

    def train_shared(self, n_rounds=None):
        """Run n_rounds of shared-weight multi-agent training."""
        if n_rounds is None:
            n_rounds = self.config.get('training.n_rounds', 20)
        save_every = self.config.get('training.save_every', 5)

        self.running = True
        self.brain = self._create_brain()   # loads checkpoint → sets total_rounds_done

        personalities = assign_personalities(self.n_workers, mode='mixed')

        print(f"\n{'=' * 60}")
        print(f"  Shared-Weight Training: {n_rounds} rounds, "
              f"{self.n_workers} workers")
        print(f"  Starting from round {self.total_rounds_done}")
        print(f"  Worker personalities: "
              f"{[p.get('description', '?') for p in personalities]}")
        print(f"{'=' * 60}")

        for r in range(1, n_rounds + 1):
            if not self.running:
                break
            round_num = self.total_rounds_done + r
            t0 = time.time()

            worker_metrics, avg_delta, initial = self._run_shared_round(
                round_num, personalities)

            # Apply averaged delta to master
            apply_avg_delta(self.brain, initial, avg_delta)

            elapsed = time.time() - t0
            metrics = _aggregate_metrics(worker_metrics, round_num, elapsed)
            self.round_history.append(metrics)

            # Console summary
            survived_list = [m['survived'] for m in worker_metrics]
            food_list = [m['food_eaten'] for m in worker_metrics]
            fitness_list = [m['fitness'] for m in worker_metrics]
            flee_pcts = [
                round(100 * m['goal_distribution'].get('FLEE', 0)
                      / max(1, m['survived']), 0)
                for m in worker_metrics]
            print(
                f"  Round {round_num}: "
                f"survived=[{','.join(str(s) for s in survived_list)}] "
                f"food=[{','.join(str(f) for f in food_list)}] "
                f"fitness={metrics['fitness']:.0f} "
                f"flee%=[{','.join(str(int(p)) for p in flee_pcts)}] "
                f"({elapsed:.0f}s)"
            )

            if self.on_round_end:
                self.on_round_end(metrics)

            if r % save_every == 0 or r == n_rounds:
                path = self.checkpoint_mgr.save(
                    self.brain, round_num, metrics, self.config)
                print(f"    Checkpoint saved: {path}", flush=True)

        self.total_rounds_done += n_rounds
        self.running = False

        if self.on_training_done:
            self.on_training_done(self.round_history)

        return self.round_history

    def train_shared_async(self, n_rounds=None):
        """Run shared-weight training in a background thread."""
        import threading
        t = threading.Thread(
            target=self.train_shared, args=(n_rounds,), daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------

    def _run_shared_round(self, round_num, personalities):
        """
        Run n_workers episodes. Returns (worker_metrics, avg_delta, initial_snap).
        """
        # Snapshot master weights before any worker modifies them
        initial = snapshot_weights(self.brain)

        deltas = []
        all_metrics = []

        for w in range(self.n_workers):
            # Restore master weights to this worker's starting state
            restore_weights(self.brain, initial)
            self.brain.reset()                # reset episode state (not weights)
            self.brain.personality = personalities[w]
            self.brain._apply_personality()

            # Use a different seed per worker so each sees different food/predator layout
            worker_seed = round_num * self.n_workers + w

            # Temporarily override seed
            orig_seed = self.config.data.get('env', {}).get('seed')
            self.config.data.setdefault('env', {})['seed'] = worker_seed

            # Progress indicator every 100 steps (flushed immediately)
            _step_counter = [0]
            def _progress_cb(step_data, _w=w, _n=self.n_workers, _ctr=_step_counter):
                _ctr[0] += 1
                if _ctr[0] % 100 == 0:
                    import sys
                    print(f"      worker {_w+1}/{_n} step {step_data['step']} "
                          f"energy={step_data['energy']:.0f} "
                          f"food={step_data['food_total']} "
                          f"goal={step_data['goal']}", flush=True)
            self.on_step = _progress_cb

            # Run one episode — modifies brain weights in-place
            snap_before = snapshot_weights(self.brain)
            metrics = self.run_round(round_num)
            snap_after = snapshot_weights(self.brain)
            self.on_step = None

            # Restore seed setting
            if orig_seed is None:
                self.config.data['env'].pop('seed', None)
            else:
                self.config.data['env']['seed'] = orig_seed

            delta = _compute_delta(snap_before, snap_after)
            deltas.append(delta)
            all_metrics.append(metrics)

            goal_dist = metrics['goal_distribution']
            flee_pct = round(
                100 * goal_dist.get('FLEE', 0) / max(1, metrics['survived']), 0)
            print(f"    worker {w+1}/{self.n_workers} "
                  f"[{personalities[w].get('description', '?')[:8]}]: "
                  f"survived={metrics['survived']}, "
                  f"food={metrics['food_eaten']}, "
                  f"fitness={metrics['fitness']:.0f}, "
                  f"FLEE={int(flee_pct)}%", flush=True)

        avg_delta = _average_deltas(deltas, self._visit_keys)
        return all_metrics, avg_delta, initial


def _aggregate_metrics(worker_metrics, round_num, elapsed=0.0):
    """Aggregate per-worker metrics into a single round summary."""
    n = len(worker_metrics)
    from collections import Counter
    goal_totals = Counter()
    for m in worker_metrics:
        for g, cnt in m['goal_distribution'].items():
            goal_totals[g] += cnt

    # Use best-worker fitness as round fitness (reflects ceiling, not average)
    best_idx = max(range(n), key=lambda i: worker_metrics[i]['fitness'])
    best = worker_metrics[best_idx]

    return {
        'round': round_num,
        'mode': 'shared-weight',
        'n_workers': n,
        'survived': sum(m['survived'] for m in worker_metrics) // n,
        'survived_per_worker': [m['survived'] for m in worker_metrics],
        'food_eaten': sum(m['food_eaten'] for m in worker_metrics),
        'food_per_worker': [m['food_eaten'] for m in worker_metrics],
        'fitness': best['fitness'],               # best worker (ceiling)
        'fitness_mean': sum(m['fitness'] for m in worker_metrics) / n,
        'fitness_per_worker': [m['fitness'] for m in worker_metrics],
        'mean_efe': sum(m['mean_efe'] for m in worker_metrics) / n,
        'energy_final': sum(m['energy_final'] for m in worker_metrics) / n,
        'caught': sum(1 for m in worker_metrics if m['caught']),
        'geo_coverage': sum(m['geo_coverage'] for m in worker_metrics) / n,
        'goal_distribution': dict(goal_totals),
        'vae_memory_nodes': best['vae_memory_nodes'],
        'critic_mean_value': sum(m['critic_mean_value'] for m in worker_metrics) / n,
        'personality': 'mixed',
        'elapsed_sec': elapsed,
    }
