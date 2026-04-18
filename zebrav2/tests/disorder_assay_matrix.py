"""
Disorder × Assay Matrix: runs all disorder phenotypes against all behavioral
assays and produces a comparison table suitable for publication.

Usage:
    python -m zebrav2.tests.disorder_assay_matrix
    python -m zebrav2.tests.disorder_assay_matrix --n_seeds 20 --ckpt zebrav2/checkpoints/ckpt_round_0005.pt
    python -m zebrav2.tests.disorder_assay_matrix --disorders wildtype,asd,schizophrenia --assays ntt,open_field

Expected output:
                  NTT_top%   NTT_freeze%   LD_dark%   OF_center%   SP_social_idx
wildtype          0.31±0.04  0.12±0.03    0.68±0.05  0.22±0.04    0.45±0.06
hypodopamine      0.31±0.05  0.14±0.04    0.65±0.06  0.20±0.05    0.42±0.07
asd               0.38±0.06  0.09±0.03    0.70±0.07  0.24±0.05    0.18±0.08
schizophrenia     0.44±0.08  0.22±0.06    0.62±0.08  0.28±0.07    0.38±0.09
anxiety           0.15±0.03  0.31±0.05    0.78±0.04  0.14±0.03    0.48±0.05
depression        0.26±0.05  0.18±0.04    0.72±0.06  0.19±0.04    0.40±0.06
adhd              0.50±0.07  0.07±0.02    0.55±0.08  0.38±0.06    0.35±0.07
ptsd              0.18±0.04  0.35±0.06    0.75±0.05  0.12±0.03    0.41±0.06
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------
try:
    from zebrav2.tests.multi_seed_runner import (
        SeedResult,
        run_multi_seed,
        print_comparison_table,
        _build_seed_result,
    )
    _RUNNER_OK = True
except ImportError as _e:
    print(f"[warn] multi_seed_runner unavailable: {_e}")
    _RUNNER_OK = False

try:
    from zebrav2.gym_env.assay_arenas import (
        NovelTankTest,
        LightDarkTest,
        SocialPreferenceTest,
        OpenFieldTest,
    )
    _ASSAYS_OK = True
except ImportError as _e:
    print(f"[warn] assay_arenas unavailable: {_e}")
    _ASSAYS_OK = False

try:
    from zebrav2.brain.disorder import apply_disorder, DISORDER_DESCRIPTIONS
    _DISORDER_OK = True
except ImportError as _e:
    print(f"[warn] disorder module unavailable: {_e}")
    _DISORDER_OK = False
    DISORDER_DESCRIPTIONS: dict = {}

# ---------------------------------------------------------------------------
# Registry: which disorders to benchmark
# ---------------------------------------------------------------------------
DISORDERS: List[str] = [
    'wildtype',
    'hypodopamine',
    'asd',
    'schizophrenia',
    'anxiety',
    'depression',
    'adhd',
    'ptsd',
]

# ---------------------------------------------------------------------------
# Registry: assay name → (class, metrics to report)
# ---------------------------------------------------------------------------
ASSAYS: Dict[str, tuple] = {
    'ntt': (
        'NovelTankTest',
        ['top_fraction', 'freezing_bouts', 'thigmotaxis_score'],
    ),
    'light_dark': (
        'LightDarkTest',
        ['dark_fraction', 'dark_preference_index', 'transition_rate'],
    ),
    'social_pref': (
        'SocialPreferenceTest',
        ['social_zone_fraction', 'social_preference_index'],
    ),
    'open_field': (
        'OpenFieldTest',
        ['center_fraction', 'distance_traveled', 'path_entropy'],
    ),
}

# Human-readable column labels  (assay_key + '_' + metric)
_COL_LABELS: Dict[str, str] = {
    'ntt_top_fraction':              'NTT_top%',
    'ntt_freezing_bouts':            'NTT_freeze%',
    'ntt_thigmotaxis_score':         'NTT_thigmo',
    'light_dark_dark_fraction':      'LD_dark%',
    'light_dark_dark_preference_index': 'LD_pref',
    'light_dark_transition_rate':    'LD_trans',
    'social_pref_social_zone_fraction': 'SP_zone%',
    'social_pref_social_preference_index': 'SP_social_idx',
    'open_field_center_fraction':    'OF_center%',
    'open_field_distance_traveled':  'OF_dist',
    'open_field_path_entropy':       'OF_entropy',
}


def _get_assay_class(name: str):
    """Return assay class by name string (robust to import failures)."""
    _map = {
        'NovelTankTest':      NovelTankTest      if _ASSAYS_OK else None,
        'LightDarkTest':      LightDarkTest      if _ASSAYS_OK else None,
        'SocialPreferenceTest': SocialPreferenceTest if _ASSAYS_OK else None,
        'OpenFieldTest':      OpenFieldTest      if _ASSAYS_OK else None,
    }
    return _map.get(name)


def _build_condition_fn(disorder_name: str):
    """Return a condition_fn for a given disorder name."""
    if disorder_name == 'wildtype':
        return lambda brain: None
    if _DISORDER_OK:
        return lambda brain, d=disorder_name: apply_disorder(brain, d)
    raise RuntimeError(f"disorder module not available; cannot apply '{disorder_name}'")


# ---------------------------------------------------------------------------
# Effect size computation
# ---------------------------------------------------------------------------
def compute_effect_sizes(
    results: Dict[str, Dict[str, Dict[str, 'SeedResult']]],
    baseline: str = 'wildtype',
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute Cohen's d for every disorder vs ``baseline`` on every metric.

    Parameters
    ----------
    results : dict[disorder][assay_key][metric] -> SeedResult
    baseline : str
        Disorder name to use as reference (default ``'wildtype'``).

    Returns
    -------
    dict[disorder][assay_key][metric] -> Cohen's d (float)
    """
    effect_sizes: Dict[str, Dict[str, Dict[str, float]]] = {}

    if baseline not in results:
        return effect_sizes

    baseline_data = results[baseline]

    for disorder, assay_data in results.items():
        if disorder == baseline:
            continue
        effect_sizes[disorder] = {}
        for assay_key, metric_data in assay_data.items():
            effect_sizes[disorder][assay_key] = {}
            baseline_assay = baseline_data.get(assay_key, {})
            for metric, sr in metric_data.items():
                baseline_sr = baseline_assay.get(metric)
                if baseline_sr is not None and sr.n > 0 and baseline_sr.n > 0:
                    d = sr.cohens_d(baseline_sr)
                else:
                    d = float('nan')
                effect_sizes[disorder][assay_key][metric] = d

    return effect_sizes


# ---------------------------------------------------------------------------
# Flat column key helpers
# ---------------------------------------------------------------------------
def _col_key(assay_key: str, metric: str) -> str:
    return f"{assay_key}_{metric}"


def _col_label(assay_key: str, metric: str) -> str:
    key = _col_key(assay_key, metric)
    return _COL_LABELS.get(key, key)


# ---------------------------------------------------------------------------
# Publication table printer
# ---------------------------------------------------------------------------
def print_matrix_table(
    results: Dict[str, Dict[str, Dict[str, 'SeedResult']]],
    selected_assays: Optional[List[str]] = None,
    col_width: int = 13,
) -> None:
    """
    Print the full disorder × assay comparison table.

    Parameters
    ----------
    results : dict[disorder][assay_key][metric] -> SeedResult
    selected_assays : list[str] or None
        Subset of assay keys to include (default: all).
    col_width : int
        Width of each metric column.
    """
    if not results:
        print("(no results)")
        return

    disorders = list(results.keys())
    assay_keys = selected_assays or list(ASSAYS.keys())

    # Build ordered list of (assay_key, metric) columns that actually have data
    columns: List[tuple] = []
    for akey in assay_keys:
        _, metrics = ASSAYS.get(akey, (None, []))
        for m in metrics:
            # Include only columns with at least one non-nan value
            has_data = any(
                results.get(d, {}).get(akey, {}).get(m) is not None
                for d in disorders
            )
            if has_data:
                columns.append((akey, m))

    if not columns:
        print("(no metric columns to display)")
        return

    cond_col = max((len(d) for d in disorders), default=12)
    cond_col = max(cond_col, 12)

    # Header row
    header = f"{'Condition':<{cond_col}}"
    for akey, m in columns:
        label = _col_label(akey, m)
        header += f"  {label[:col_width]:>{col_width}}"
    print()
    print(header)
    print("-" * len(header))

    # Data rows
    for disorder in disorders:
        row = f"{disorder:<{cond_col}}"
        for akey, m in columns:
            sr = results.get(disorder, {}).get(akey, {}).get(m)
            if sr is None or sr.n == 0:
                cell = "n/a"
            else:
                cell = str(sr)
            row += f"  {cell:>{col_width}}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# JSON / CSV serialization
# ---------------------------------------------------------------------------
def _seed_result_to_dict(sr: 'SeedResult') -> dict:
    return {
        'mean': sr.mean,
        'sem':  sr.sem,
        'std':  sr.std,
        'n':    sr.n,
        'values': sr.values,
    }


def save_results_json(
    results: Dict[str, Dict[str, Dict[str, 'SeedResult']]],
    effect_sizes: Dict,
    output_dir: str,
    timestamp: str,
    args_dict: dict,
) -> str:
    """Serialize results to JSON and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"matrix_{timestamp}.json")

    serializable: dict = {
        'metadata': {
            'timestamp': timestamp,
            'args': args_dict,
            'disorders': list(results.keys()),
            'assays': list(ASSAYS.keys()),
        },
        'results': {
            disorder: {
                assay_key: {
                    metric: _seed_result_to_dict(sr)
                    for metric, sr in metric_data.items()
                }
                for assay_key, metric_data in assay_data.items()
            }
            for disorder, assay_data in results.items()
        },
        'effect_sizes': effect_sizes,
    }

    with open(fname, 'w') as fp:
        json.dump(serializable, fp, indent=2, default=str)

    return fname


def save_results_csv(
    results: Dict[str, Dict[str, Dict[str, 'SeedResult']]],
    output_dir: str,
    timestamp: str,
) -> str:
    """Serialize results to CSV (mean±sem) and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"matrix_{timestamp}.csv")

    disorders = list(results.keys())
    columns: List[tuple] = []
    for akey, (_, metrics) in ASSAYS.items():
        for m in metrics:
            columns.append((akey, m))

    lines: List[str] = []
    # Header
    header = ['condition'] + [_col_label(a, m) for a, m in columns]
    lines.append(','.join(header))

    # Rows
    for disorder in disorders:
        row_vals = [disorder]
        for akey, m in columns:
            sr = results.get(disorder, {}).get(akey, {}).get(m)
            if sr is None or sr.n == 0:
                row_vals.append('')
            else:
                row_vals.append(f"{sr.mean:.4f}±{sr.sem:.4f}")
        lines.append(','.join(row_vals))

    with open(fname, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')

    return fname


# ---------------------------------------------------------------------------
# Subprocess worker — runs ONE (disorder × assay) in a fresh Python process
# to prevent MPS/torch memory accumulation across 32+ conditions.
# ---------------------------------------------------------------------------
_WORKER_SCRIPT = r'''
import gc, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from zebrav2.tests.multi_seed_runner import run_multi_seed, SeedResult
from zebrav2.gym_env.assay_arenas import (
    NovelTankTest, LightDarkTest, SocialPreferenceTest, OpenFieldTest)
from zebrav2.brain.disorder import apply_disorder

_ASSAY_MAP = {
    'NovelTankTest': NovelTankTest,
    'LightDarkTest': LightDarkTest,
    'SocialPreferenceTest': SocialPreferenceTest,
    'OpenFieldTest': OpenFieldTest,
}

def main():
    disorder   = sys.argv[1]
    assay_cls  = sys.argv[2]
    n_seeds    = int(sys.argv[3])
    n_steps    = int(sys.argv[4])
    ckpt_path  = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != 'None' else None
    metrics_s  = sys.argv[6] if len(sys.argv) > 6 else ''
    keep_metrics = metrics_s.split(',') if metrics_s else []

    condition_fn = (lambda brain: None) if disorder == 'wildtype' \
                   else (lambda brain, d=disorder: apply_disorder(brain, d))
    assay = _ASSAY_MAP[assay_cls]()

    seed_results = run_multi_seed(
        condition_fn=condition_fn,
        assay=assay,
        n_seeds=n_seeds,
        ckpt_path=ckpt_path,
        n_steps=n_steps,
        personality_name='default',
        verbose=True,
    )

    out = {}
    for m in (keep_metrics or list(seed_results.keys())):
        if m in seed_results:
            sr = seed_results[m]
            out[m] = {'mean': sr.mean, 'sem': sr.sem, 'std': sr.std,
                       'n': sr.n, 'values': sr.values}
    # Write JSON on a single tagged line so parent can parse it
    print('__RESULT_JSON__' + json.dumps(out))

if __name__ == '__main__':
    main()
'''


def _run_condition_subprocess(
    disorder: str,
    assay_key: str,
    assay_class_name: str,
    metrics: List[str],
    n_seeds: int,
    n_steps: int,
    ckpt_path: Optional[str],
    verbose: bool,
) -> Dict[str, 'SeedResult']:
    """Run one (disorder × assay) pair in a fresh subprocess."""
    import tempfile
    worker_path = os.path.join(tempfile.gettempdir(), '_vzebra_matrix_worker.py')
    with open(worker_path, 'w') as f:
        f.write(_WORKER_SCRIPT)

    cmd = [
        sys.executable, '-u', worker_path,
        disorder, assay_class_name,
        str(n_seeds), str(n_steps),
        str(ckpt_path or 'None'),
        ','.join(metrics),
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=_ROOT,
    )

    result_json = None
    for line in proc.stdout:
        if line.startswith('__RESULT_JSON__'):
            result_json = json.loads(line[len('__RESULT_JSON__'):])
        elif verbose:
            print(line, end='', flush=True)

    proc.wait()

    if proc.returncode != 0:
        print(f"  [subprocess exited {proc.returncode}]")
        return {}

    if result_json is None:
        print("  [no result JSON from subprocess]")
        return {}

    # Reconstruct SeedResult objects
    out = {}
    for m, d in result_json.items():
        out[m] = _build_seed_result(d['values'])
    return out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_disorder_assay_matrix(
    disorders: List[str],
    assay_keys: List[str],
    n_seeds: int = 10,
    n_steps: int = 400,
    ckpt_path: Optional[str] = None,
    no_save: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, 'SeedResult']]]:
    """
    Execute the full disorder × assay matrix.

    Each (disorder × assay) condition runs in a FRESH subprocess to
    prevent MPS/torch memory accumulation (OOM after ~10 conditions).

    Returns
    -------
    dict[disorder][assay_key][metric] -> SeedResult
    """
    # Outer dict: disorder → assay_key → metric → SeedResult
    results: Dict[str, Dict[str, Dict[str, 'SeedResult']]] = {
        d: {} for d in disorders
    }

    n_total = len(disorders) * len(assay_keys)
    counter = 0

    for disorder_name in disorders:
        for assay_key in assay_keys:
            counter += 1
            assay_class_name, metrics = ASSAYS.get(assay_key, (None, []))

            if assay_class_name is None:
                print(f"[ERROR] {disorder_name} × {assay_key}: assay not registered")
                results[disorder_name][assay_key] = {}
                continue

            label = f"[{counter}/{n_total}] {disorder_name} × {assay_key}"
            if verbose:
                print(f"\n{label}")
                desc = DISORDER_DESCRIPTIONS.get(disorder_name, '')
                if desc:
                    print(f"  ({desc})")

            try:
                seed_results = _run_condition_subprocess(
                    disorder=disorder_name,
                    assay_key=assay_key,
                    assay_class_name=assay_class_name,
                    metrics=metrics,
                    n_seeds=n_seeds,
                    n_steps=n_steps,
                    ckpt_path=ckpt_path,
                    verbose=verbose,
                )
                results[disorder_name][assay_key] = seed_results
            except Exception as exc:
                print(f"[ERROR] {disorder_name} × {assay_key}: {exc}")
                results[disorder_name][assay_key] = {}

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Disorder × Assay Matrix runner for vzebra v2.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Seeds per condition (default: 10)')
    parser.add_argument('--n_steps', type=int, default=400,
                        help='Simulation steps per seed (default: 400)')
    parser.add_argument('--ckpt', default=None,
                        help='Path to .pt checkpoint (default: None)')
    parser.add_argument('--disorders', default='all',
                        help='Comma-separated disorder names, or "all"')
    parser.add_argument('--assays', default='all',
                        help='Comma-separated assay keys, or "all"')
    parser.add_argument('--no_save', action='store_true',
                        help='Skip saving results to disk')
    args = parser.parse_args()

    # Resolve disorder list
    if args.disorders == 'all':
        selected_disorders = DISORDERS
    else:
        selected_disorders = [d.strip() for d in args.disorders.split(',')]

    # Resolve assay list
    if args.assays == 'all':
        selected_assays = list(ASSAYS.keys())
    else:
        selected_assays = [a.strip() for a in args.assays.split(',')]

    # Validate
    unknown_d = [d for d in selected_disorders if d not in DISORDERS and d != 'wildtype']
    if unknown_d:
        print(f"[warn] unknown disorders (will still attempt): {unknown_d}")

    unknown_a = [a for a in selected_assays if a not in ASSAYS]
    if unknown_a:
        print(f"[warn] unknown assay keys (skipping): {unknown_a}")
        selected_assays = [a for a in selected_assays if a in ASSAYS]

    print("=" * 70)
    print("Disorder × Assay Matrix")
    print(f"  disorders : {selected_disorders}")
    print(f"  assays    : {selected_assays}")
    print(f"  n_seeds   : {args.n_seeds}")
    print(f"  n_steps   : {args.n_steps}")
    print(f"  ckpt      : {args.ckpt or '(none)'}")
    print("=" * 70)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Run the matrix
    results = run_disorder_assay_matrix(
        disorders=selected_disorders,
        assay_keys=selected_assays,
        n_seeds=args.n_seeds,
        n_steps=args.n_steps,
        ckpt_path=args.ckpt,
        no_save=args.no_save,
        verbose=True,
    )

    # Print publication table
    print_matrix_table(results, selected_assays=selected_assays)

    # Compute effect sizes vs wildtype
    effect_sizes = compute_effect_sizes(results, baseline='wildtype')

    if effect_sizes:
        print("\nCohen's d  (vs wildtype):")
        print("-" * 50)
        for disorder, assay_data in effect_sizes.items():
            for assay_key, metric_data in assay_data.items():
                for metric, d in metric_data.items():
                    label = _col_label(assay_key, metric)
                    if not math.isnan(d):
                        print(f"  {disorder:<15s}  {label:<20s}  d = {d:+.3f}")
        print()

    # Save outputs
    if not args.no_save:
        output_dir = os.path.join(_HERE, 'disorder_results')
        args_dict = {
            'n_seeds':   args.n_seeds,
            'n_steps':   args.n_steps,
            'ckpt':      args.ckpt,
            'disorders': selected_disorders,
            'assays':    selected_assays,
        }
        json_path = save_results_json(
            results, effect_sizes, output_dir, timestamp, args_dict)
        csv_path = save_results_csv(results, output_dir, timestamp)
        print(f"Saved JSON → {json_path}")
        print(f"Saved CSV  → {csv_path}")
    else:
        print("(--no_save: results not written to disk)")


if __name__ == '__main__':
    main()
