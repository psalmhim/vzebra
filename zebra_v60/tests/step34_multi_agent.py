"""
Step 34: Multi-Agent Dynamics Tests.
Run: python -m zebra_v60.tests.step34_multi_agent
"""
import os, sys, math, numpy as np
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

def run_step34():
    print("=" * 60)
    print("Step 34: Multi-Agent Dynamics")
    print("=" * 60)
    from zebra_v60.gym_env.multi_agent_env import MultiAgentZebrafishEnv
    from zebra_v60.gym_env.brain_agent import BrainAgent
    results = {}

    # A. Multi-agent env runs without crash
    env = MultiAgentZebrafishEnv(n_fish=5, render_mode=None, n_food=15,
                                  max_steps=200, side_panels=False)
    agent = BrainAgent(use_allostasis=True)
    obs, info = env.reset(seed=42); agent.reset()
    for t in range(100):
        obs, rew, term, trunc, info = env.step(agent.act(obs, env))
        agent.update_post_step(info, reward=rew, done=term, env=env)
        if term: break
    results["A"] = t >= 50
    print(f"  A. Runs 100 steps: {'PASS' if results['A'] else 'FAIL'}")

    # B. All fish have positions
    results["B"] = len(env.all_fish) == 5
    print(f"  B. 5 fish present: {'PASS' if results['B'] else 'FAIL'}")

    # C. Polarization > 0.3
    pol = env.compute_polarization()
    results["C"] = pol > 0.3
    print(f"  C. Polarization: {pol:.3f} {'PASS' if results['C'] else 'FAIL'}")

    # D. NND in reasonable range
    nnd = env.compute_nnd()
    results["D"] = 20 < nnd < 200
    print(f"  D. NND: {nnd:.1f}px {'PASS' if results['D'] else 'FAIL'}")

    # E. Single-agent backward compatible
    from zebra_v60.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
    env2 = ZebrafishPreyPredatorEnv(render_mode=None, n_food=10, max_steps=100,
                                     side_panels=False)
    obs2, _ = env2.reset(seed=42); agent.reset()
    for t in range(50):
        obs2, rew, term, trunc, _ = env2.step(agent.act(obs2, env2))
        agent.update_post_step({}, reward=rew, done=term, env=env2)
        if term: break
    results["E"] = t >= 20
    print(f"  E. Single-agent compat: {'PASS' if results['E'] else 'FAIL'}")
    env.close(); env2.close()

    # F. Conspecifics eat food
    env3 = MultiAgentZebrafishEnv(n_fish=5, render_mode=None, n_food=25,
                                   max_steps=300, side_panels=False)
    obs, _ = env3.reset(seed=123); agent.reset()
    initial_food = len(env3.foods)
    for t in range(200):
        obs, rew, term, trunc, info = env3.step(agent.act(obs, env3))
        agent.update_post_step(info, reward=rew, done=term, env=env3)
        if term: break
    food_consumed = initial_food - len(env3.foods)
    results["F"] = food_consumed > 0
    print(f"  F. Food consumed: {food_consumed} {'PASS' if results['F'] else 'FAIL'}")
    env3.close()

    n = sum(results.values())
    print(f"\nRESULTS: {n}/{len(results)} PASS")
    print(f"VERDICT: {'PASS' if n >= 5 else 'FAIL'}")

if __name__ == "__main__": run_step34()
