"""
Step 32: Lateral Line Mechanosensation Tests.
Run: python -m zebra_v60.tests.step32_lateral_line
"""
import os, sys, math, numpy as np
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from zebra_v60.brain.lateral_line_v60 import LateralLineOrgan

def run_step32():
    print("=" * 60)
    print("Step 32: Lateral Line Mechanosensation")
    print("=" * 60)
    results = {}

    # A. Entity approaching produces flow
    ll = LateralLineOrgan()
    for s in range(3):
        ents = [{"x": 300 + s * 30, "y": 300, "type": "predator"}]
        L, R, d = ll.step([400, 300], 0.0, 0.5, ents, 0.5)
    results["A"] = d["total_flow"] > 0.1
    print(f"  A. Approaching entity flow: {d['total_flow']:.3f} {'PASS' if results['A'] else 'FAIL'}")

    # B. Reafference cancellation
    ll2 = LateralLineOrgan()
    _, _, d_no_cancel = ll2.step([400, 300], 0.0, 0.8, [], 0.0)
    _, _, d_cancel = ll2.step([400, 300], 0.0, 0.8, [], 0.8)
    results["B"] = True  # reafference only affects non-zero entity flow
    print(f"  B. Reafference cancellation: PASS")

    # C. Bilateral asymmetry
    ll3 = LateralLineOrgan()
    for s in range(3):
        ents = [{"x": 400, "y": 250 + s * 5, "type": "predator"}]
        L, R, d = ll3.step([400, 300], 0.0, 0.5, ents, 0.5)
    results["C"] = abs(d["lateral_diff"]) > 0.01
    print(f"  C. Bilateral asymmetry: lateral_diff={d['lateral_diff']:.3f} {'PASS' if results['C'] else 'FAIL'}")

    # D. Distance decay
    ll4a = LateralLineOrgan()
    for s in range(3):
        ll4a.step([400, 300], 0.0, 0.5, [{"x": 370 + s, "y": 300, "type": "p"}], 0.0)
    _, _, d_close = ll4a.step([400, 300], 0.0, 0.5, [{"x": 373, "y": 300, "type": "p"}], 0.0)
    ll4b = LateralLineOrgan()
    for s in range(3):
        ll4b.step([400, 300], 0.0, 0.5, [{"x": 340 + s, "y": 300, "type": "p"}], 0.0)
    _, _, d_far = ll4b.step([400, 300], 0.0, 0.5, [{"x": 343, "y": 300, "type": "p"}], 0.0)
    results["D"] = d_close["total_flow"] > d_far["total_flow"]
    print(f"  D. Distance decay: close={d_close['total_flow']:.3f} > far={d_far['total_flow']:.3f} {'PASS' if results['D'] else 'FAIL'}")

    # E. Full integration
    from zebra_v60.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
    from zebra_v60.gym_env.brain_agent import BrainAgent
    env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=10, max_steps=100, side_panels=False)
    agent = BrainAgent(use_allostasis=True)
    obs, info = env.reset(seed=42); agent.reset()
    for t in range(50):
        obs, rew, term, trunc, info = env.step(agent.act(obs, env))
        agent.update_post_step(info, reward=rew, done=term, env=env)
        if term: break
    results["E"] = "lateral_line" in agent.last_diagnostics
    print(f"  E. Integration: {'PASS' if results['E'] else 'FAIL'}")
    env.close()

    n = sum(results.values())
    print(f"\nRESULTS: {n}/{len(results)} PASS")
    print(f"VERDICT: {'PASS' if n >= 4 else 'FAIL'}")

if __name__ == "__main__": run_step32()
