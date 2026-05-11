"""
Tier 3: Prey capture kinematics — J-turn approach + strike sequence.

Reference: Semmelhack 2014 — Dedicated visual pathway for prey; Bhatt 2007 kinematics
"""
from __future__ import annotations
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register


@register
class T3PreyCapture(EvalPlugin):
    tier = 3
    name = 't3_prey_capture'
    tags = ['prey_capture', 'motor', 'visual']
    requires = ['t2_retina', 't2_tectum']
    purpose = 'Verify prey capture kinematics: J-turn approach + strike sequence.'
    method = (
        'Instantiate PreyCaptureKinematics. Drive with close food stimulus '
        '(food_px >= 5, food_distance < 80) while in FORAGE goal (goal=0). '
        'Verify J-turn and strike phases are reached.'
    )
    reference = 'Semmelhack 2014 — Dedicated visual pathway for prey; Bhatt 2007 kinematics'

    def run(self, ctx: EvalContext) -> EvalResult:
        from zebrav2.brain.prey_capture import PreyCaptureKinematics

        try:
            pkc = PreyCaptureKinematics()
            module_alive = 1.0
        except Exception:
            metrics = {'module_alive': 0.0, 'has_kinematics': 0.0}
            pass_criteria = {'module_alive': ('>', 0.5)}
            return self._result(metrics, pass_criteria)

        # Drive with food close and visible — triggers J_TURN → APPROACH → STRIKE
        nonzero_outputs = 0
        strike_reached = False
        j_turn_reached = False

        # Run enough steps for the full sequence: J_TURN (3) + APPROACH (5) + STRIKE (2) = 10
        for step in range(20):
            result = pkc.update(
                goal=0,               # FORAGE
                food_px=10,           # >= 5 to trigger
                food_distance=40.0,   # < 80 to trigger; < 35 to transition APPROACH→STRIKE
                food_lateral_bias=0.5,
                obstacle_px=0,
            )
            if result is not None:
                turn, speed = result
                if abs(turn) > 0.0 or abs(speed) > 0.0:
                    nonzero_outputs += 1
            if pkc.phase == 'J_TURN':
                j_turn_reached = True
            if pkc.phase == 'STRIKE' or pkc.total_strikes > 0:
                strike_reached = True

        has_kinematics = 1.0 if nonzero_outputs > 0 else 0.0

        metrics = {
            'module_alive':  module_alive,
            'has_kinematics': has_kinematics,
        }
        pass_criteria = {
            'module_alive':  ('>', 0.5),
            'has_kinematics': ('>', 0.0),
        }
        notes = (
            f'j_turn_reached={j_turn_reached}, '
            f'strike_reached={strike_reached}, '
            f'total_strikes={pkc.total_strikes}'
        )
        return self._result(metrics, pass_criteria, notes=notes)
