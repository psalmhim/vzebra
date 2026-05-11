"""
Tier 5: Behavioral change in response to alarm substance signal.

Reference: Masuda 2024 — Daniol sulfate alarm substance identification
"""
from __future__ import annotations
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register

DEVICE = 'cpu'


@register
class T5AlarmSubstance(EvalPlugin):
    tier = 5
    name = 't5_alarm_substance'
    tags = ['social', 'alarm', 'olfaction']
    requires = ['t3_neuromod']
    purpose = 'Verify behavioral change (flee/freeze) in response to alarm substance signal.'
    method = (
        'Inject alarm substance signal via olfaction module. '
        'Pass conspecific_injured=True vs False. '
        'Measure output difference in alarm_level.'
    )
    reference = 'Masuda 2024 — Daniol sulfate alarm substance identification'

    def run(self, ctx: EvalContext) -> EvalResult:
        from zebrav2.brain.olfaction import SpikingOlfaction

        try:
            olf = SpikingOlfaction(device=DEVICE)
            olfaction_alive = 1.0
        except Exception:
            metrics = {'alarm_response': 0.0, 'olfaction_alive': 0.0}
            pass_criteria = {'olfaction_alive': ('>', 0.5)}
            return self._result(metrics, pass_criteria)

        # Baseline: no injury, no predator — conspecific is at 30 px
        # We need dummy food list and fish position
        foods      = []   # no food odor
        fish_x     = 400.0
        fish_y     = 300.0
        fish_heading = 0.0

        out_no_alarm = olf.forward(
            fish_x=fish_x, fish_y=fish_y, fish_heading=fish_heading,
            foods=foods,
            conspecific_injured=False,
            pred_dist=999.0,
            conspc_dist=30.0,
        )
        alarm_baseline = float(out_no_alarm['alarm_level'])

        # Reset and measure with injured conspecific
        olf.reset()
        out_alarm = olf.forward(
            fish_x=fish_x, fish_y=fish_y, fish_heading=fish_heading,
            foods=foods,
            conspecific_injured=True,
            pred_dist=999.0,
            conspc_dist=30.0,
        )
        alarm_injured = float(out_alarm['alarm_level'])

        # alarm_response: 1.0 if injured > baseline, 0.0 otherwise
        alarm_response = 1.0 if alarm_injured > alarm_baseline else 0.0

        metrics = {
            'alarm_response':  alarm_response,
            'olfaction_alive': olfaction_alive,
        }
        pass_criteria = {
            'alarm_response':  ('>', 0.5),
            'olfaction_alive': ('>', 0.5),
        }
        notes = f'alarm_baseline={alarm_baseline:.4f}, alarm_injured={alarm_injured:.4f}'
        return self._result(metrics, pass_criteria, notes=notes)
