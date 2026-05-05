"""
Tier-2 plugin: Spinal CPG L-R alternation and speed modulation validation.

Verifies left-right alternation in CPG and speed modulation by descending drive.
"""
import torch
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register
from zebrav2.brain.spinal_cpg import SpinalCPG


@register
class CPGPlugin(EvalPlugin):
    tier = 2
    name = 't2_cpg'
    tags = ['cpg', 'motor', 'locomotion']
    requires = []
    purpose = (
        'Verify L-R alternation in CPG and speed modulation by descending drive. '
        'Anti-phase left-right motor output and frequency scaling with drive.'
    )
    method = (
        'Drive CPG at 0.5 for 100 steps → measure left-right anti-phase '
        'via sign changes in (motor_L - motor_R). '
        'Compare mean speed at drive=0.8 vs drive=0.2.'
    )
    reference = (
        'McLean 2008 — Frequency tuning in zebrafish spinal CPG; Bhatt 2007'
    )

    def run(self, ctx: EvalContext) -> EvalResult:
        cpg = SpinalCPG(device='cpu')

        # L-R alternation: 100 steps at drive=0.5
        motor_L_vals = []
        motor_R_vals = []
        for _ in range(100):
            mL, mR, speed, turn_out, diag = cpg.step(0.5, turn=0.0)
            motor_L_vals.append(mL)
            motor_R_vals.append(mR)

        # Count sign changes in (motor_L - motor_R), skipping zero-crossings.
        # The CPG has glide phases (zeros) between bouts; real alternations occur
        # when consecutive non-zero values have opposite sign.
        diff_seq = [motor_L_vals[i] - motor_R_vals[i] for i in range(100)]
        nonzero_diffs = [(i, d) for i, d in enumerate(diff_seq) if abs(d) > 1e-6]
        lr_alternations = 0
        for j in range(1, len(nonzero_diffs)):
            if nonzero_diffs[j][1] * nonzero_diffs[j - 1][1] < 0:
                lr_alternations += 1

        motor_active = float(sum(abs(motor_L_vals[i]) + abs(motor_R_vals[i])
                                  for i in range(100)) / 100.0)

        # Speed comparison: high drive (0.8) vs low drive (0.2)
        cpg.reset()
        speeds_high = []
        for _ in range(50):
            _, _, speed, _, _ = cpg.step(0.8, turn=0.0)
            speeds_high.append(speed)
        speed_high = float(sum(speeds_high) / len(speeds_high))

        cpg.reset()
        speeds_low = []
        for _ in range(50):
            _, _, speed, _, _ = cpg.step(0.2, turn=0.0)
            speeds_low.append(speed)
        speed_low = float(sum(speeds_low) / len(speeds_low))

        speed_high_vs_low = speed_high / (speed_low + 1e-6)

        metrics = {
            'lr_alternations':  float(lr_alternations),
            'speed_high_vs_low': speed_high_vs_low,
            'motor_active':      motor_active,
        }
        pass_criteria = {
            'lr_alternations':   ('>', 1.0),
            'speed_high_vs_low': ('>', 1.0),
            'motor_active':      ('>', 0.0),
        }
        return self._result(metrics, pass_criteria, artifacts={
            'speed_high': speed_high,
            'speed_low':  speed_low,
            'diff_seq_sample': diff_seq[:20],
        })
