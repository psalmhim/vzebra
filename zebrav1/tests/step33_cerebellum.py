"""
Step 33: Cerebellum Forward Model Tests.
Run: python -m zebrav1.tests.step33_cerebellum
"""
import os, sys, numpy as np
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from zebrav1.brain.cerebellum import CerebellumForwardModel

def run_step33():
    print("=" * 60)
    print("Step 33: Cerebellum Forward Model")
    print("=" * 60)
    results = {}

    # A. Prediction error decreases with training
    cb = CerebellumForwardModel()
    actual = np.array([0.05, -0.01, 0.1, 0.1, 0.0, -0.04])
    errors_early, errors_late = [], []
    for i in range(200):
        cb.step(np.array([0.1, 0.5, 0]), np.array([0, 0.5, 0.9, 0.1, 0, 0, 0, 0.5]))
        cb.update(actual)
        pe = cb.get_prediction_error_magnitude()
        if i < 20: errors_early.append(pe)
        if i >= 180: errors_late.append(pe)
    results["A"] = np.mean(errors_late) < np.mean(errors_early)
    print(f"  A. PE decreases: early={np.mean(errors_early):.3f} > late={np.mean(errors_late):.3f} {'PASS' if results['A'] else 'FAIL'}")

    # B. Motor correction bounded
    turn_c, speed_c = cb.get_motor_correction()
    results["B"] = abs(turn_c) <= 0.1 and 0.85 <= speed_c <= 1.15
    print(f"  B. Correction bounded: turn={turn_c:.4f} speed={speed_c:.4f} {'PASS' if results['B'] else 'FAIL'}")

    # C. Granule sparsity
    cb2 = CerebellumForwardModel()
    cb2.step(np.array([0.2, 0.7, 1]), np.array([1, 0.7, 0.5, 0.3, 0.2, 5, 3, 0.6]))
    active = (cb2._granule_activity > 0).sum()
    total = len(cb2._granule_activity)
    ratio = active / total
    results["C"] = 0.1 < ratio < 0.8
    print(f"  C. Granule sparsity: {ratio:.2f} active {'PASS' if results['C'] else 'FAIL'}")

    # D. Reafference prediction exists
    fl, fr = cb.get_reafference_prediction()
    results["D"] = isinstance(fl, float) and isinstance(fr, float)
    print(f"  D. Reafference: L={fl:.4f} R={fr:.4f} {'PASS' if results['D'] else 'FAIL'}")

    # E. Full integration
    from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
    from zebrav1.gym_env.brain_agent import BrainAgent
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=10, max_steps=100, side_panels=False)
    agent = BrainAgent(use_allostasis=True)
    obs, info = env.reset(seed=42); agent.reset()
    for t in range(50):
        obs, rew, term, trunc, info = env.step(agent.act(obs, env))
        agent.update_post_step(info, reward=rew, done=term, env=env)
        if term: break
    results["E"] = "cerebellum" in agent.last_diagnostics
    print(f"  E. Integration: {'PASS' if results['E'] else 'FAIL'}")
    env.close()

    n = sum(results.values())
    print(f"\nRESULTS: {n}/{len(results)} PASS")
    print(f"VERDICT: {'PASS' if n >= 4 else 'FAIL'}")

if __name__ == "__main__": run_step33()
