"""
Master test runner for all V2 step tests.

Runs every step_XX test in order and reports a unified pass/fail summary.
Use this after any code change to quickly verify nothing regressed.

Run: .venv/bin/python -m zebrav2.tests.run_all_steps
"""
import os, sys, time, importlib, traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ordered list of (display_name, module_path, main_fn_or_None)
# None means the module defines __main__ via a direct if-block — we import
# the test functions individually.
STEPS = [
    ('Phase 1: Izhikevich Neurons',   'zebrav2.tests.test_phase1_neuron',     'test_phase1'),
    ('Phase 2: E/I Layer',            'zebrav2.tests.test_phase2_ei',          'test_phase2'),
    ('All Phases (import+func)',       'zebrav2.tests.test_all_phases',         'run_all'),
    ('Neuromod Cross-Talk',            'zebrav2.tests.step05_neuromod_crosstalk', None),
    ('Temporal Dynamics (STDP)',       'zebrav2.tests.step06_temporal_dynamics',  None),
    ('Untested Modules',               'zebrav2.tests.step07_untested_modules',   None),
    ('Regression Suite',               'zebrav2.tests.step08_regression',         None),
]

# Steps that expose individual test_* functions rather than a single entry fn
MULTI_FN_STEPS = {
    'zebrav2.tests.step05_neuromod_crosstalk': [
        'test_da_reward', 'test_ht5_flee', 'test_na_arousal',
        'test_ach_attention', 'test_reset_clears_state',
    ],
    'zebrav2.tests.step06_temporal_dynamics': [
        'test_causal_ltp', 'test_anticausal_ltd', 'test_da_scales_update',
        'test_eligibility_decays', 'test_homeostatic_scaling',
        'test_ach_gates_plasticity', 'test_stdp_reset',
    ],
    'zebrav2.tests.step07_untested_modules': [
        'test_working_memory', 'test_vestibular', 'test_proprioception',
        'test_binocular_depth', 'test_shoaling', 'test_circadian',
        'test_color_vision', 'test_olfaction', 'test_personality',
    ],
    'zebrav2.tests.step08_regression': [
        'test_imports', 'test_brain_instantiation',
        'test_survival_regression', 'test_classifier_regression',
        'test_neuromod_regression',
    ],
}


def run_step(display_name, module_path, entry_fn):
    """Import module and run its tests. Returns (passed, failed, elapsed_s)."""
    t0 = time.time()
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        print(f'\n  IMPORT ERROR: {e}')
        traceback.print_exc()
        return 0, 1, time.time() - t0

    # Reset global pass/fail counters if module exposes them
    if hasattr(mod, 'passes'):
        mod.passes = 0
    if hasattr(mod, 'fails'):
        mod.fails = 0

    if entry_fn is not None:
        # Single entry-point function (e.g. test_phase1, run_all)
        fn = getattr(mod, entry_fn, None)
        if fn is None:
            print(f'  MISSING fn {entry_fn} in {module_path}')
            return 0, 1, time.time() - t0
        try:
            result = fn()
            ok = (result is True) or (result is None)  # None = didn't return bool
        except Exception as e:
            print(f'\n  RUNTIME ERROR: {e}')
            traceback.print_exc()
            ok = False
        elapsed = time.time() - t0
        p = getattr(mod, 'passes', 1 if ok else 0)
        f = getattr(mod, 'fails', 0 if ok else 1)
        return p, f, elapsed
    else:
        # Multi-function module
        fns = MULTI_FN_STEPS.get(module_path, [])
        for fn_name in fns:
            fn = getattr(mod, fn_name, None)
            if fn is None:
                continue
            try:
                fn()
            except Exception as e:
                print(f'\n  ERROR in {fn_name}: {e}')
                traceback.print_exc()
                if hasattr(mod, 'fails'):
                    mod.fails += 1
        elapsed = time.time() - t0
        p = getattr(mod, 'passes', 0)
        f = getattr(mod, 'fails', 0)
        return p, f, elapsed


def main():
    print('=' * 70)
    print('V2 Master Test Runner')
    print('=' * 70)

    summary = []
    total_p, total_f = 0, 0

    for display_name, module_path, entry_fn in STEPS:
        sep = '-' * 70
        print(f'\n{sep}')
        print(f'>>> {display_name}')
        print(sep)
        p, f, elapsed = run_step(display_name, module_path, entry_fn)
        total_p += p
        total_f += f
        status = 'PASS' if f == 0 else 'FAIL'
        summary.append((display_name, p, f, elapsed, status))
        print(f'  [{status}]  {p}/{p+f} passed  ({elapsed:.1f}s)')

    print('\n' + '=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)
    for name, p, f, elapsed, status in summary:
        mark = 'OK' if status == 'PASS' else 'XX'
        print(f'  [{mark}]  {name:<40}  {p}/{p+f}  ({elapsed:.1f}s)')

    print(f'\nTotal: {total_p}/{total_p+total_f} assertions passed')
    if total_f == 0:
        print('ALL TESTS PASSED')
    else:
        print(f'{total_f} FAILURES — see output above')

    sys.exit(0 if total_f == 0 else 1)


if __name__ == '__main__':
    main()
