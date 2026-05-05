"""
Tier 4: Whole-animal survival across 3 random seeds × 300 steps.

Reference: Internal benchmark (vzebra v2 9-seed protocol)
"""
from __future__ import annotations
import numpy as np

from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register

SEEDS    = [1, 2, 3]
MAX_STEPS = 300
N_FOOD   = 15


@register
class T4Survival(EvalPlugin):
    tier = 4
    name = 't4_survival'
    tags = ['survival', 'behavior', 'full_brain']
    requires = []
    purpose = 'Whole-animal survival across 3 random seeds, 300 steps each.'
    method = (
        '3 seeds × 300 steps. '
        'Measure: survival rate, food collected, mean steps survived.'
    )
    reference = 'Internal benchmark (vzebra v2 9-seed protocol)'

    def run(self, ctx: EvalContext) -> EvalResult:
        if ctx.brain is None:
            return EvalResult.skipped_result(self, 'ctx.brain is None — skip full-brain test')

        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        from zebrav2.brain.sensory_bridge import inject_sensory
        from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

        brain = ctx.brain

        survived_all = []
        food_all = []
        steps_survived_all = []

        for seed in SEEDS:
            env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=N_FOOD, max_steps=MAX_STEPS)
            obs, _ = env.reset(seed=seed)
            brain.reset()

            food_this = 0
            steps_this = 0
            survived_episode = True

            for t in range(MAX_STEPS):
                if hasattr(env, 'set_flee_active'):
                    env.set_flee_active(
                        brain.current_goal == 1,
                        0.8 if brain.current_goal == 1 else 0.0
                    )
                inject_sensory(env)
                out = brain.step(obs, env)
                action = np.array([out['turn'], out['speed']], dtype=np.float32)
                obs, _, term, trunc, info = env.step(action)
                env._eaten_now = info.get('food_eaten_this_step', 0)
                food_this += env._eaten_now
                steps_this = t + 1
                if term:
                    survived_episode = False
                    break

            # trunc means max_steps reached without dying — that counts as survival
            survived_all.append(1.0 if survived_episode else 0.0)
            food_all.append(food_this)
            steps_survived_all.append(steps_this)

        survival_rate       = float(np.mean(survived_all))
        mean_survival_steps = float(np.mean(steps_survived_all))
        mean_food           = float(np.mean(food_all))

        # Store for downstream plugins
        ctx.artifacts['t4_survival_results'] = {
            'survival_rate': survival_rate,
            'mean_steps': mean_survival_steps,
            'mean_food': mean_food,
            'per_seed': list(zip(SEEDS, steps_survived_all, food_all)),
        }

        metrics = {
            'survival_rate':       survival_rate,
            'mean_survival_steps': mean_survival_steps,
            'mean_food':           mean_food,
        }
        pass_criteria = {
            'survival_rate':       ('>', 0.5),
            'mean_survival_steps': ('>', 150.0),
            'mean_food':           ('>', 0.0),
        }
        return self._result(metrics, pass_criteria)
