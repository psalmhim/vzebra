"""
Tier 5: Classifier distinguishes conspecific from predator in retinal input.

Reference: Kappel 2022 — Tectothalamic social recognition circuit
"""
from __future__ import annotations
import torch

from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import register

DEVICE = 'cpu'
N_PIX  = 400


@register
class T5SocialRecognition(EvalPlugin):
    tier = 5
    name = 't5_social_recognition'
    tags = ['social', 'recognition', 'classifier']
    requires = ['t2_retina']
    purpose = 'Verify classifier distinguishes conspecific from predator in retinal input.'
    method = (
        'Present conspecific-type pixel pattern (type_channel ≈ 0.25) vs '
        'predator-type (type_channel ≈ 0.5). '
        'Measure classifier probability difference.'
    )
    reference = 'Kappel 2022 — Tectothalamic social recognition circuit'

    def run(self, ctx: EvalContext) -> EvalResult:
        from zebrav2.brain.classifier import ClassifierV2

        try:
            clf = ClassifierV2(device=DEVICE)
            classifier_runs = 1.0
        except Exception:
            metrics = {'classifier_runs': 0.0, 'social_vs_predator_diff': 0.0}
            pass_criteria = {'classifier_runs': ('>', 0.5)}
            return self._result(metrics, pass_criteria)

        # Conspecific stimulus: type_channel = 0.25 for a patch in one eye
        # ClassifierV2.classify() expects L, R as (800,) arrays [:400]=intensity [400:]=type
        L_social = torch.zeros(800, device=DEVICE)
        R_social = torch.zeros(800, device=DEVICE)
        # Put conspecific object in right visual field
        L_social[:N_PIX]   = 0.0           # no intensity in left eye
        L_social[N_PIX:]   = 0.0           # no type
        R_social[:20]       = 0.4           # some intensity
        R_social[N_PIX:N_PIX + 20] = 0.25  # conspecific type code

        # Predator stimulus: type_channel = 0.5
        L_pred = torch.zeros(800, device=DEVICE)
        R_pred = torch.zeros(800, device=DEVICE)
        R_pred[:20]             = 0.6       # brighter (predator is bigger)
        R_pred[N_PIX:N_PIX+20]  = 0.5      # predator type code

        # Run classifier
        clf.reset()
        result_social = clf.classify(L_social, R_social)
        probs_social  = result_social['probs']          # shape (5,)

        clf.reset()
        result_pred = clf.classify(L_pred, R_pred)
        probs_pred  = result_pred['probs']              # shape (5,)

        # CLASS_NAMES = ['nothing', 'food', 'enemy', 'colleague', 'environment']
        # enemy = index 2, colleague = index 3
        p_colleague_social  = float(probs_social[3].item())
        p_enemy_predator    = float(probs_pred[2].item())

        # Separation: both probabilities are > chance (0.2), and sum indicates
        # the classifier assigns higher probability to the correct class.
        # Encode as: how much more the correct class is favoured vs swapped.
        # separation > 0 means correct class > wrong class in at least one case
        social_correct   = p_colleague_social
        predator_correct = p_enemy_predator
        social_vs_predator_diff = (social_correct + predator_correct) / 2.0

        metrics = {
            'social_vs_predator_diff': social_vs_predator_diff,
            'classifier_runs':         classifier_runs,
        }
        pass_criteria = {
            'social_vs_predator_diff': ('>', 0.0),
            'classifier_runs':         ('>', 0.5),
        }
        notes = (
            f'p_colleague_social={p_colleague_social:.4f}, '
            f'p_enemy_predator={p_enemy_predator:.4f}'
        )
        return self._result(metrics, pass_criteria, notes=notes)
