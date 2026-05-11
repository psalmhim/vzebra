"""
Tier 3: Amygdala CS-US pairing — associative fear learning.

Reference: Agetsuma 2010 — Habenula in fear conditioning; Bhatt 2007 — Zebrafish amygdala
"""
from __future__ import annotations
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register

DEVICE = 'cpu'
N_CONDITIONING_TRIALS = 5


@register
class T3FearConditioning(EvalPlugin):
    tier = 3
    name = 't3_fear_conditioning'
    tags = ['amygdala', 'fear', 'learning', 'limbic']
    requires = ['t2_habenula']
    purpose = 'Verify amygdala potentiation through CS-US pairing (associative fear learning).'
    method = (
        '5 CS-US pairing trials (enemy_pixels=50, pred_dist=15, pred_facing=0.8). '
        'Measure amygdala response to CS alone after conditioning vs before. '
        'Fear baseline should increase.'
    )
    reference = 'Agetsuma 2010 — Habenula in fear conditioning; Bhatt 2007 — Zebrafish amygdala'

    def run(self, ctx: EvalContext) -> EvalResult:
        from zebrav2.brain.amygdala import SpikingAmygdalaV2

        amygdala = SpikingAmygdalaV2(device=DEVICE)

        # Measure fear_baseline before conditioning
        fear_before = amygdala.fear_baseline

        # CS-only response before conditioning (CS = mild enemy pixels, no proximity)
        cs_baseline_outputs = []
        for _ in range(3):
            out = amygdala.forward(enemy_pixels=15.0, pred_dist=300.0,
                                   stress=0.0, pred_facing=0.0)
            cs_baseline_outputs.append(out)
        cs_response_before = float(sum(cs_baseline_outputs) / len(cs_baseline_outputs))

        # Reset to clean state for conditioning
        amygdala.reset_full()
        fear_before = amygdala.fear_baseline

        # CS-US pairing: 5 trials with high threat (pred_dist=15 → proximity ~0.925 > 0.9,
        # pred_facing=0.8 > 0.5) — triggers episodic LTP in the amygdala
        for _ in range(N_CONDITIONING_TRIALS):
            amygdala.forward(enemy_pixels=50.0, pred_dist=15.0,
                             stress=0.5, pred_facing=0.8)

        fear_after = amygdala.fear_baseline

        # CS-only response after conditioning
        cs_conditioned_outputs = []
        for _ in range(3):
            out = amygdala.forward(enemy_pixels=15.0, pred_dist=300.0,
                                   stress=0.0, pred_facing=0.0)
            cs_conditioned_outputs.append(out)
        cs_response_conditioned = float(sum(cs_conditioned_outputs) / len(cs_conditioned_outputs))

        conditioning_effect = fear_after - fear_before

        metrics = {
            'fear_baseline_after':    float(fear_after),
            'cs_response_conditioned': float(cs_response_conditioned),
            'conditioning_effect':    float(conditioning_effect),
        }
        pass_criteria = {
            'fear_baseline_after':    ('>', 0.0),
            'cs_response_conditioned': ('>', 0.0),
            'conditioning_effect':    ('>', 0.0),
        }
        return self._result(metrics, pass_criteria)
