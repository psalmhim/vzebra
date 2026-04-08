"""Run all phase tests and print summary."""
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_all():
    results = {}

    # Phase 1
    try:
        from zebrav2.tests.test_phase1_neuron import test_phase1
        results['Phase 1: Izhikevich Neuron'] = test_phase1()
    except Exception as e:
        print(f"Phase 1 ERROR: {e}")
        results['Phase 1: Izhikevich Neuron'] = False

    # Phase 2
    try:
        from zebrav2.tests.test_phase2_ei import test_phase2
        results['Phase 2: E/I Layer'] = test_phase2()
    except Exception as e:
        print(f"Phase 2 ERROR: {e}")
        results['Phase 2: E/I Layer'] = False

    # Phases 3-9: import test
    modules_to_test = [
        ('Phase 3: Retina', 'zebrav2.brain.retina', 'RetinaV2'),
        ('Phase 3: Tectum', 'zebrav2.brain.tectum', 'Tectum'),
        ('Phase 4: Basal Ganglia', 'zebrav2.brain.basal_ganglia', 'BasalGanglia'),
        ('Phase 4: Reticulospinal', 'zebrav2.brain.reticulospinal', 'ReticulospinalSystem'),
        ('Phase 5: Neuromod', 'zebrav2.brain.neuromod', 'NeuromodSystem'),
        ('Phase 6: Thalamus', 'zebrav2.brain.thalamus', 'Thalamus'),
        ('Phase 6: Pallium', 'zebrav2.brain.pallium', 'Pallium'),
        ('Phase 7: Plasticity', 'zebrav2.brain.plasticity', 'EligibilitySTDP'),
        ('Phase 9: Place Cells', 'zebrav2.brain.place_cells', 'ThetaPlaceCells'),
        ('Amygdala', 'zebrav2.brain.amygdala', 'SpikingAmygdalaV2'),
        ('Classifier', 'zebrav2.brain.classifier', 'ClassifierV2'),
        ('Predator Model', 'zebrav2.brain.predator_model', 'PredatorModel'),
        ('Allostasis', 'zebrav2.brain.allostasis', 'AllostaticRegulator'),
        ('World Model', 'zebrav2.brain.internal_model', 'InternalWorldModel'),
        ('Phase 8+: Brain V2', 'zebrav2.brain.brain_v2', 'ZebrafishBrainV2'),
    ]

    for name, module_path, class_name in modules_to_test:
        try:
            import importlib
            import torch as _torch
            from zebrav2.spec import DEVICE as _DEVICE
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            # Some classes require arguments
            if class_name == 'EligibilitySTDP':
                dummy_W = _torch.nn.Parameter(_torch.zeros(10, 10, device=_DEVICE))
                instance = cls(dummy_W, device=_DEVICE)
            elif class_name == 'FeedbackPELearning':
                dummy_W = _torch.nn.Parameter(_torch.zeros(10, 10, device=_DEVICE))
                instance = cls(dummy_W, device=_DEVICE)
            else:
                instance = cls()
            results[name] = True
            print(f"  PASS  {name}: {class_name} instantiated OK")
        except Exception as e:
            results[name] = False
            print(f"  FAIL  {name}: {e}")

    # Functional forward-pass tests for key modules
    import torch as _torch
    from zebrav2.spec import DEVICE as _DEVICE

    try:
        from zebrav2.brain.retina import RetinaV2
        ret = RetinaV2()
        L = _torch.rand(800, device=_DEVICE)
        R = _torch.rand(800, device=_DEVICE)
        out = ret(L, R)
        ok = ('on_fused' in out or 'on' in out) and out.get('on_fused', out.get('on')).shape[0] > 0
        results['Retina forward pass'] = ok
        print(f"  {'PASS' if ok else 'FAIL'}  Retina forward pass: keys={list(out.keys())[:4]}")
    except Exception as e:
        results['Retina forward pass'] = False
        print(f"  FAIL  Retina forward pass: {e}")

    try:
        from zebrav2.brain.neuromod import NeuromodSystem
        nm = NeuromodSystem()
        nm_out = nm.update(reward=1.0, amygdala_alpha=0.5, cms=0.2,
                           flee_active=False, fatigue=0.2, circadian=0.8, current_goal=0)
        da_high = nm_out['DA'] > 0.5   # positive reward → DA > 0.5
        results['Neuromod reward→DA'] = da_high
        print(f"  {'PASS' if da_high else 'FAIL'}  Neuromod reward→DA: DA={nm_out['DA']:.3f} (expect > 0.5)")
    except Exception as e:
        results['Neuromod reward→DA'] = False
        print(f"  FAIL  Neuromod reward→DA: {e}")

    try:
        from zebrav2.brain.neuromod import NeuromodSystem
        nm2 = NeuromodSystem()
        # Run flee for 30 steps — 5-HT should fall
        for _ in range(30):
            nm2.update(reward=0.0, amygdala_alpha=0.9, cms=0.5,
                       flee_active=True, fatigue=0.3, circadian=0.8, current_goal=1)
        ht5_low = nm2.HT5.item() < 0.5
        results['Neuromod flee→5HT falls'] = ht5_low
        print(f"  {'PASS' if ht5_low else 'FAIL'}  Neuromod flee→5HT falls: 5HT={nm2.HT5.item():.3f} (expect < 0.5)")
    except Exception as e:
        results['Neuromod flee→5HT falls'] = False
        print(f"  FAIL  Neuromod flee→5HT falls: {e}")

    try:
        from zebrav2.brain.place_cells import ThetaPlaceCells
        pc = ThetaPlaceCells()
        out1 = pc(100.0, 200.0, food_eaten=False, predator_near=False)
        out2 = pc(110.0, 205.0, food_eaten=True, predator_near=False)
        rate = out2['rate']
        ok = rate is not None and rate.shape[0] > 0 and rate.max().item() >= 0
        results['PlaceCells forward pass'] = ok
        print(f"  {'PASS' if ok else 'FAIL'}  PlaceCells forward pass: max_rate={rate.max():.3f}, food_value={out2['food_value']:.4f}")
    except Exception as e:
        results['PlaceCells forward pass'] = False
        print(f"  FAIL  PlaceCells forward pass: {e}")

    # Integration test: brain_v2 step
    try:
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        brain = ZebrafishBrainV2()

        class MockEnv:
            brain_L = [0.0]*800
            brain_R = [0.0]*800
            pred_state = 'PATROL'
            pred_x = 400; pred_y = 600
            fish_x = 400; fish_y = 300
            fish_heading = 0.0
            fish_energy = 80.0
            _enemy_pixels_total = 0
            _eaten_now = 0

        env = MockEnv()
        out = brain.step(None, env)
        ok = ('turn' in out and 'speed' in out and 'goal' in out)
        results['Integration: brain_v2.step()'] = ok
        print(f"  {'PASS' if ok else 'FAIL'}  Integration: brain_v2.step() -> turn={out.get('turn',0):.3f}, speed={out.get('speed',0):.3f}, goal={out.get('goal',0)}")
    except Exception as e:
        results['Integration: brain_v2.step()'] = False
        print(f"  FAIL  Integration: {e}")
        import traceback; traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        mark = 'OK' if ok else 'XX'
        print(f"  [{mark}]  {name}")
    print(f"\n{passed}/{total} phases passed")
    return passed == total

if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)
