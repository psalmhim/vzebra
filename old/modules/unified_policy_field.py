# ============================================================
# MODULE: unified_policy_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v37.0 (2025-12-05)
#
# PURPOSE:
#     Unified policy inference system that merges:
#         (1) ActivePolicyField (reward-based EFE)
#         (2) ActivePolicyFieldEFE (direct-EFE)
#         (3) GoalPolicyField (RL/Q-learning)
#
#     Provides a single Active Inference policy module with:
#         • reward-mode   : epistemic + extrinsic EFE
#         • efe-mode      : direct input EFE
#         • rl-mode       : dopaminergic RL + motive + gain
#
#     All modes share:
#         • posterior sampling via softmax(-β·EFE)
#         • entropy-based confidence
#         • persistence (stickiness)
#         • one-hot goal vector
# ============================================================

import torch
import torch.nn.functional as F
from modules.module_template import BaseModule


class UnifiedPolicyField(BaseModule):
    """
    Unified policy inference engine supporting 3 modes:
    -------------------------------------------------------
    mode="reward"
        Uses expected reward + uncertainty mismatch to infer EFE.
        (ActivePolicyField behavior)

    mode="efe"
        Takes direct EFE values per goal.
        (ActivePolicyFieldEFE behavior)

    mode="rl"
        Dopaminergic Q-learning + motive + gain dynamics.
        (GoalPolicyField behavior)
    """

    def __init__(
        self,
        mode="reward",
        n_goals=3,
        beta=2.0,
        persist_steps=8,
        lr_value=0.1,
        gamma=0.95,
        device="cpu",
    ):
        super().__init__(device=device)

        assert mode in ["reward", "efe", "rl"], "Invalid mode for UnifiedPolicyField."

        self.mode = mode
        self.n_goals = n_goals
        self.beta = beta
        self.persist_steps = persist_steps
        self.timer = 0
        self.last_choice = 0

        # RL mode parameters
        self.Q = torch.zeros(n_goals, device=device)
        self.pref = torch.zeros(n_goals, device=device)
        self.lr_value = lr_value
        self.gamma = gamma
        self.motive = 0.0
        self.gain_state = 1.0
        self.meta_precision = 0.5  # confidence measure

    # ------------------------------------------------------------
    # Main update step
    # ------------------------------------------------------------
    def step(
        self,
        reward_exp=None,
        unc_exp=None,
        obs_reward=None,
        obs_unc=None,
        efe_estimates=None,
        dopa=0.5,
        rpe=0.0,
        cms=0.0,
        mem_mean=0.0,
    ):
        """
        Provides a unified policy update interface.

        Inputs vary by mode:

        mode="reward":
            reward_exp, unc_exp, obs_reward, obs_unc, cms
            -> computes epistemic + extrinsic EFE

        mode="efe":
            efe_estimates (list or tensor)
            -> direct EFE

        mode="rl":
            dopa, rpe, cms, mem_mean
            -> dopaminergic RL + gain/motive dynamics

        Returns unified:
            efe_vec, posterior, choice, gvec, confidence, extra_info
        """

        # ============================================================
        # 1) COMPUTE EFE ACCORDING TO MODE
        # ============================================================

        if self.mode == "reward":
            # ----- epistemic + extrinsic prediction error -----
            reward_err = torch.abs(obs_reward - reward_exp)
            unc_err = torch.abs(obs_unc - unc_exp)
            epistemic = unc_err + 0.1 * abs(cms)
            extrinsic = reward_err
            efe_vec = 0.5 * (epistemic + extrinsic)  # shape [n_goals]
            extra = {
                "reward_err": reward_err.tolist(),
                "unc_err": unc_err.tolist(),
            }

        elif self.mode == "efe":
            efe_vec = torch.tensor(efe_estimates, device=self.device, dtype=torch.float32)
            efe_vec = efe_vec - efe_vec.min()  # normalize
            extra = {"efe_raw": efe_vec.tolist()}

        elif self.mode == "rl":
            # ----- continuous motivational update -----
            target = 0.5 * (dopa - 0.5) - 0.1 * mem_mean
            self.motive = 0.9 * self.motive + 0.1 * target
            self.motive = float(torch.clamp(torch.tensor(self.motive), -1.0, 1.0))

            # ----- adaptive gain -----
            delta = abs(dopa - 0.5) + abs(self.motive) + 0.2 * mem_mean
            self.gain_state = 0.9 * self.gain_state + 0.1 * (1.0 + delta)
            self.gain_state = float(torch.clamp(torch.tensor(self.gain_state), 0.8, 2.0))

            # ----- meta precision -----
            rpe_abs = abs(float(rpe))
            cms_abs = abs(float(cms))
            conf_update = 1 - torch.tanh(torch.tensor(rpe_abs + 0.5 * cms_abs))
            self.meta_precision = 0.95 * self.meta_precision + 0.05 * conf_update.item()

            # ----- dopaminergic Q-learning -----
            q_old = self.Q[self.last_choice]
            td_err = float(rpe + (dopa - 0.5)) + self.gamma * q_old - q_old
            self.Q[self.last_choice] = q_old + self.lr_value * td_err

            # convert values to something EFE-like (negative of drive)
            drive = self.Q - 0.3 * cms + (dopa - 0.5) + self.motive
            efe_vec = -drive
            extra = {
                "motive": self.motive,
                "gain_state": self.gain_state,
                "td_error": td_err,
                "meta_precision": self.meta_precision,
            }

        # use torch tensor
        efe_vec = torch.as_tensor(efe_vec, device=self.device, dtype=torch.float32)

        # ============================================================
        # 2) POLICY POSTERIOR
        # ============================================================
        post = F.softmax(-self.beta * efe_vec, dim=0)

        # ============================================================
        # 3) PERSISTENCE (stick to last choice for some steps)
        # ============================================================
        if self.timer < self.persist_steps:
            self.timer += 1
            choice = self.last_choice
        else:
            choice = torch.multinomial(post, 1).item()
            self.last_choice = choice
            self.timer = 0

        # ============================================================
        # 4) ONE-HOT VECTOR & CONFIDENCE
        # ============================================================
        gvec = torch.zeros(1, self.n_goals, device=self.device)
        gvec[0, int(choice)] = 1.0

        entropy = torch.distributions.Categorical(probs=post).entropy()
        max_ent = torch.log(torch.tensor(float(self.n_goals), device=self.device))
        confidence = float(1.0 - entropy / max_ent)

        return efe_vec, post, choice, gvec, confidence, extra

    # ------------------------------------------------------------
    def reset_state(self):
        self.Q.zero_()
        self.pref.zero_()
        self.motive = 0.0
        self.gain_state = 1.0
        self.meta_precision = 0.5
        self.last_choice = 0
        self.timer = 0
