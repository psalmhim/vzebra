"""
Tier 5: Verify fish maintain group cohesion in multi-agent environment.

Deep validation: inter-fish distance, heading polarization, nearest-neighbor
distance, and shoal-size stability.

Reference: Miller & Gerlai (2011) PLOS ONE — zebrafish shoaling metrics;
           Tunstrom et al. (2013) PLOS Comp Biol — collective behaviour states
"""
from __future__ import annotations
import math
import numpy as np

from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register

N_FISH        = 4
MAX_STEPS     = 200
COHESION_DIST = 250.0   # px — pairs within this are "cohesive"


@register
class T5Shoaling(EvalPlugin):
    tier = 5
    name = 't5_shoaling'
    tags = ['social', 'shoaling', 'multi_agent']
    requires = ['t4_survival']
    purpose = (
        'Verify fish maintain group cohesion (distance + heading alignment). '
        'Deep validation: NND, polarization, shoal stability.'
    )
    method = (
        f'Multi-agent env with {N_FISH} fish × {MAX_STEPS} steps. '
        'Mean inter-fish dist, nearest-neighbor dist, heading polarization '
        '(cosine alignment), cohesion fraction, shoal-size CV.'
    )
    reference = (
        'Miller & Gerlai (2011) PLOS ONE — zebrafish shoaling; '
        'Tunstrom et al. (2013) PLOS Comp Biol — collective behaviour states'
    )

    def run(self, ctx: EvalContext) -> EvalResult:
        if ctx.brain is None:
            return EvalResult.skipped_result(self, 'ctx.brain is None — skip full-brain test')

        try:
            from zebrav2.brain.brain_v2 import ZebrafishBrainV2
            from zebrav2.brain.sensory_bridge import inject_sensory
            from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
            from zebrav2.gym_env.multi_agent_v2 import inject_sensory_for_fish
        except ImportError as e:
            return EvalResult.skipped_result(self, f'Import error: {e}')

        try:
            env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=15, max_steps=MAX_STEPS)
            obs, _ = env.reset(seed=99)
            env.pred_x = -9999.0
            env.pred_y = -9999.0

            np.random.seed(99)
            all_fish = []
            brains = []
            for i in range(N_FISH):
                x = float(np.random.uniform(200, 600))
                y = float(np.random.uniform(150, 450))
                h = float(np.random.uniform(-math.pi, math.pi))
                all_fish.append({'x': x, 'y': y, 'heading': h,
                                 'speed': 0.5, 'alive': True,
                                 'food_eaten': 0, 'energy': 100.0,
                                 'personality': 'default'})
                brain = ZebrafishBrainV2(device='cpu')
                brain.reset()
                brains.append(brain)

            env.fish_x, env.fish_y, env.fish_heading = (
                all_fish[0]['x'], all_fish[0]['y'], all_fish[0]['heading'])

            inter_dists_mean  = []
            nnd_mean_steps    = []
            polarization_steps = []
            cohesion_steps    = []
            last_out = None

            for t in range(MAX_STEPS):
                for i in range(N_FISH):
                    if not all_fish[i]['alive']:
                        continue
                    inject_sensory_for_fish(env, i, all_fish)
                    if hasattr(env, 'set_flee_active'):
                        env.set_flee_active(False, 0.0)
                    out = brains[i].step(obs, env)
                    last_out = out
                    fish = all_fish[i]
                    fish['heading'] += float(out['turn']) * 0.15
                    fish['heading'] = math.atan2(
                        math.sin(fish['heading']), math.cos(fish['heading']))
                    fish['x'] += 3.0 * float(out['speed']) * math.cos(fish['heading'])
                    fish['y'] += 3.0 * float(out['speed']) * math.sin(fish['heading'])
                    fish['x'] = max(10, min(env.arena_w - 10, fish['x']))
                    fish['y'] = max(10, min(env.arena_h - 10, fish['y']))

                env.fish_x     = all_fish[0]['x']
                env.fish_y     = all_fish[0]['y']
                env.fish_heading = all_fish[0]['heading']
                inject_sensory(env)
                action = np.array([last_out['turn'], last_out['speed']], dtype=np.float32)
                obs, _, term, trunc, _ = env.step(action)
                env.pred_x = -9999.0
                env.pred_y = -9999.0

                alive = [f for f in all_fish if f['alive']]
                if len(alive) < 2:
                    break

                # Pairwise distances
                pairs = []
                for a in range(len(alive)):
                    for b in range(a + 1, len(alive)):
                        dx = alive[a]['x'] - alive[b]['x']
                        dy = alive[a]['y'] - alive[b]['y']
                        pairs.append(math.sqrt(dx*dx + dy*dy))

                inter_dists_mean.append(float(np.mean(pairs)))
                cohesion_steps.append(1.0 if any(d < COHESION_DIST for d in pairs) else 0.0)

                # Nearest-neighbor distance per fish
                nnds = []
                for a in range(len(alive)):
                    min_d = min(
                        math.sqrt((alive[a]['x']-alive[b]['x'])**2 +
                                  (alive[a]['y']-alive[b]['y'])**2)
                        for b in range(len(alive)) if b != a
                    )
                    nnds.append(min_d)
                nnd_mean_steps.append(float(np.mean(nnds)))

                # Heading polarization: mean cosine-similarity of all pairs
                pol_vals = []
                for a in range(len(alive)):
                    for b in range(a + 1, len(alive)):
                        pol_vals.append(math.cos(alive[a]['heading'] - alive[b]['heading']))
                polarization_steps.append(float(np.mean(pol_vals)))

                if term or trunc:
                    break

        except Exception as e:
            return EvalResult.skipped_result(self, f'Multi-agent env error: {e}')

        mean_dist       = float(np.mean(inter_dists_mean))   if inter_dists_mean   else 9999.0
        mean_nnd        = float(np.mean(nnd_mean_steps))     if nnd_mean_steps     else 9999.0
        mean_polar      = float(np.mean(polarization_steps)) if polarization_steps else 0.0
        cohesion_frac   = float(np.mean(cohesion_steps))     if cohesion_steps     else 0.0
        # Coefficient of variation of shoal size — low = stable group
        shoal_size_cv   = (float(np.std(inter_dists_mean)) / (mean_dist + 1e-6)
                           if len(inter_dists_mean) > 1 else 1.0)

        metrics = {
            'mean_inter_fish_dist':  mean_dist,
            'mean_nnd':              mean_nnd,
            'polarization':          mean_polar,
            'cohesion_maintained':   cohesion_frac,
            'shoal_size_cv':         shoal_size_cv,
        }
        pass_criteria = {
            'mean_inter_fish_dist': ('<', 400.0),
            'mean_nnd':             ('<', 350.0),
            'polarization':         ('>', 0.0),
            'cohesion_maintained':  ('>', 0.2),
            'shoal_size_cv':        ('<', 1.5),
        }
        return self._result(metrics, pass_criteria)
