"""
Tier 3: Neuromodulatory system responds correctly to reward/threat signals.

Reference: Tay 2011 — DA projectome (no VTA); Yokogawa 2007 — Zebrafish arousal
"""
from __future__ import annotations
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register

DEVICE = 'cpu'


@register
class T3Neuromod(EvalPlugin):
    tier = 3
    name = 't3_neuromod'
    tags = ['neuromod', 'dopamine', 'reward']
    requires = []
    purpose = 'Verify neuromodulatory system responds correctly to reward/threat signals.'
    method = (
        '(1) Reward=1.0 → DA increases. '
        '(2) Threat → NA increases. '
        '(3) Baseline DA > 0.'
    )
    reference = 'Tay 2011 — DA projectome (no VTA); Yokogawa 2007 — Zebrafish arousal'

    def run(self, ctx: EvalContext) -> EvalResult:
        from zebrav2.brain.neuromod import NeuromodSystem

        # --- Baseline DA (no reward) ---
        nm_base = NeuromodSystem(device=DEVICE)
        nm_base.reset()
        out_no_reward = nm_base.update(
            reward=0.0, amygdala_alpha=0.0, cms=0.0,
            flee_active=False, fatigue=0.0, circadian=0.5, current_goal=0
        )
        da_no_reward = float(out_no_reward['DA'])

        # --- DA with reward ---
        nm_reward = NeuromodSystem(device=DEVICE)
        nm_reward.reset()
        out_reward = nm_reward.update(
            reward=1.0, amygdala_alpha=0.0, cms=0.0,
            flee_active=False, fatigue=0.0, circadian=0.5, current_goal=0
        )
        da_reward = float(out_reward['DA'])

        # Ratio: da_reward / da_no_reward (expect > 1)
        da_ratio = da_reward / (da_no_reward + 1e-8)

        # --- NA with threat ---
        nm_threat = NeuromodSystem(device=DEVICE)
        nm_threat.reset()
        out_threat = nm_threat.update(
            reward=0.0, amygdala_alpha=0.8, cms=0.0,
            flee_active=False, fatigue=0.0, circadian=0.5, current_goal=0
        )
        na_threat = float(out_threat['NA'])

        nm_calm = NeuromodSystem(device=DEVICE)
        nm_calm.reset()
        out_calm = nm_calm.update(
            reward=0.0, amygdala_alpha=0.0, cms=0.0,
            flee_active=False, fatigue=0.0, circadian=0.5, current_goal=0
        )
        na_calm = float(out_calm['NA'])

        na_ratio = na_threat / (na_calm + 1e-8)

        metrics = {
            'da_increases_with_reward': float(da_ratio),
            'na_increases_with_threat': float(na_ratio),
            'baseline_da':             float(da_no_reward),
        }
        pass_criteria = {
            'da_increases_with_reward': ('>', 1.0),
            'na_increases_with_threat': ('>', 1.0),
            'baseline_da':             ('>', 0.0),
        }
        return self._result(metrics, pass_criteria)
