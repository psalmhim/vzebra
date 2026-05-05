"""
Tier 4: Verify agent explores the arena (does not stay in one spot).

Reference: Gahtan 2005 — Zebrafish novel environment exploration
"""
from __future__ import annotations
import math
import numpy as np

from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register

MAX_STEPS  = 200
GRID_ROWS  = 4
GRID_COLS  = 4
N_CELLS    = GRID_ROWS * GRID_COLS   # 16


@register
class T4Exploration(EvalPlugin):
    tier = 4
    name = 't4_exploration'
    tags = ['exploration', 'behavior', 'full_brain']
    requires = ['t4_survival']
    purpose = 'Verify the agent explores the arena (does not stay in one spot).'
    method = (
        '1 episode × 200 steps. '
        'Divide arena into 4×4 grid = 16 cells. '
        'Count unique cells visited.'
    )
    reference = 'Gahtan 2005 — Zebrafish novel environment exploration'

    def run(self, ctx: EvalContext) -> EvalResult:
        if ctx.brain is None:
            return EvalResult.skipped_result(self, 'ctx.brain is None — skip full-brain test')

        from zebrav2.brain.sensory_bridge import inject_sensory
        from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv

        brain = ctx.brain

        env = ZebrafishPreyPredatorEnv(render_mode=None, n_food=10, max_steps=MAX_STEPS)
        obs, _ = env.reset(seed=7)
        # Move predator away to avoid confounding exploration with flight
        env.pred_x = -9999.0
        env.pred_y = -9999.0
        brain.reset()

        arena_w = getattr(env, 'arena_w', 800)
        arena_h = getattr(env, 'arena_h', 600)
        cell_w  = arena_w / GRID_COLS
        cell_h  = arena_h / GRID_ROWS

        visited_cells = set()
        positions = []

        for t in range(MAX_STEPS):
            if hasattr(env, 'set_flee_active'):
                env.set_flee_active(False, 0.0)
            inject_sensory(env)
            out = brain.step(obs, env)
            action = np.array([out['turn'], out['speed']], dtype=np.float32)
            obs, _, term, trunc, info = env.step(action)
            env._eaten_now = info.get('food_eaten_this_step', 0)
            # Re-displace predator
            env.pred_x = -9999.0
            env.pred_y = -9999.0

            fx = float(getattr(env, 'fish_x', arena_w / 2))
            fy = float(getattr(env, 'fish_y', arena_h / 2))
            positions.append((fx, fy))

            col = min(GRID_COLS - 1, int(fx / cell_w))
            row = min(GRID_ROWS - 1, int(fy / cell_h))
            visited_cells.add((row, col))

            if term or trunc:
                break

        arena_coverage = len(visited_cells) / N_CELLS

        # Mean speed: consecutive position distances
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            speeds.append(math.sqrt(dx*dx + dy*dy))
        mean_speed = float(np.mean(speeds)) if speeds else 0.0

        metrics = {
            'arena_coverage': arena_coverage,
            'mean_speed':     mean_speed,
        }
        pass_criteria = {
            'arena_coverage': ('>', 0.15),
            'mean_speed':     ('>', 0.5),
        }
        return self._result(metrics, pass_criteria)
