"""
Tier 4: Food collection efficiency under safe conditions (predator far away).

Reference: Gahtan 2005 — Zebrafish foraging behavior
"""
from __future__ import annotations
import numpy as np

from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register

N_EPISODES = 3
MAX_STEPS  = 200
N_FOOD     = 20
# Predator is placed far away by seeding: we use a reset seed that keeps predator distant,
# and rely on default reset behavior (predator spawns at fixed offset from fish).


@register
class T4Foraging(EvalPlugin):
    tier = 4
    name = 't4_foraging'
    tags = ['foraging', 'behavior', 'full_brain']
    requires = ['t4_survival']
    purpose = 'Food collection efficiency under safe conditions (predator absent/far).'
    method = (
        '3 episodes × 200 steps, predator kept distant (pred_x set to -9999). '
        'Measure food/steps ratio.'
    )
    reference = 'Gahtan 2005 — Zebrafish foraging behavior'

    def run(self, ctx: EvalContext) -> EvalResult:
        if ctx.brain is None:
            return EvalResult.skipped_result(self, 'ctx.brain is None — skip full-brain test')

        from zebrav2.brain.sensory_bridge import inject_sensory
        from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

        brain = ctx.brain

        food_per_ep    = []
        steps_per_ep   = []
        ep_with_food   = []

        for ep in range(N_EPISODES):
            env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=N_FOOD, max_steps=MAX_STEPS)
            obs, _ = env.reset(seed=ep + 42)
            # Move predator far outside the arena so it poses no threat
            env.pred_x = -9999.0
            env.pred_y = -9999.0
            brain.reset()

            food_this  = 0
            steps_this = 0

            for t in range(MAX_STEPS):
                if hasattr(env, 'set_flee_active'):
                    env.set_flee_active(False, 0.0)
                inject_sensory(env)
                out = brain.step(obs, env)
                action = np.array([out['turn'], out['speed']], dtype=np.float32)
                obs, _, term, trunc, info = env.step(action)
                env._eaten_now = info.get('food_eaten_this_step', 0)
                food_this  += env._eaten_now
                steps_this  = t + 1
                # Re-displace predator each step (env.step() may move it)
                env.pred_x = -9999.0
                env.pred_y = -9999.0
                if term or trunc:
                    break

            food_per_ep.append(food_this)
            steps_per_ep.append(steps_this)
            ep_with_food.append(1.0 if food_this > 0 else 0.0)

        total_food  = sum(food_per_ep)
        total_steps = sum(steps_per_ep)
        food_per_step = float(total_food / (total_steps + 1e-8))
        episodes_with_food = float(np.mean(ep_with_food))

        metrics = {
            'food_per_step':      food_per_step,
            'episodes_with_food': episodes_with_food,
        }
        pass_criteria = {
            'food_per_step':      ('>', 0.01),
            'episodes_with_food': ('>', 0.0),
        }
        return self._result(metrics, pass_criteria)
