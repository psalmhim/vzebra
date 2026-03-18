# ============================================================
# MODULE: policy_field.py
# AUTHOR: H.J. Park & GPT-5 (PC-SNN Project)
# VERSION: v15.3 (2025-11-15)
#
# PURPOSE:
#     Adds meta-gain regulation via confidence tracking.
#     Confidence (meta-precision) modulates motivational gain.
#
# UPDATE SUMMARY:
#     • Introduces self.meta_precision (confidence variable)
#     • Updates with running average of |RPE| and CMS
#     • Scales gain_state accordingly
# ============================================================

# ============================================================
# MODULE: policy_field.py (extension)
# VERSION: v16.0 (2025-11-16)
#
# PURPOSE:
#     Active Inference version of GoalPolicyField.
#     Uses expected free-energy (EFE) to infer policy posteriors.
# ============================================================

import torch
from modules.module_template import BaseModule

# ============================================================
# MODULE: policy_field.py (v16.1)
# AUTHOR: H.J. Park & GPT-5
# PURPOSE:
#     Expected Outcome–based Active Policy Inference.
# ============================================================

class ActivePolicyField(BaseModule):
    """Policy inference based on decomposed Expected Free Energy."""

    def __init__(self, mode, n_goals=3, beta=2.0, persist_steps=8, device="cpu"):
        super().__init__(device=device)
        self.mode = mode
        self.n_goals = n_goals
        self.beta = beta
        self.persist_steps = persist_steps
        self.timer = 0
        self.last_choice = 1

    # --------------------------------------------------------
    def step(self, reward_exp, unc_exp, obs_reward, obs_unc, cms=0.0):
        """
        Compute EFE for each goal using expected vs. observed mismatch.
        reward_exp, unc_exp: expected outcomes (reward, uncertainty)
        obs_reward, obs_unc: observed sensory evidence
        """
        reward_pred_err = torch.abs(obs_reward - reward_exp)
        unc_pred_err = torch.abs(obs_unc - unc_exp)
        cms = torch.as_tensor(cms, device=self.device, dtype=torch.float32)
        epistemic = unc_pred_err + 0.1 * torch.abs(cms)
        extrinsic = reward_pred_err
        efe_vec = 0.5 * (epistemic + extrinsic)

        post = torch.softmax(-self.beta * efe_vec, dim=0)

        # Ensure post is a 1D vector
        post = torch.as_tensor(post, device=self.device, dtype=torch.float32)
        if post.ndim == 0:
            post = post.unsqueeze(0)
        elif post.ndim > 1:
            post = post.squeeze()

        if self.timer < self.persist_steps:
            self.timer += 1
            choice = self.last_choice
        else:
            choice = torch.multinomial(post, 1).item()
            self.last_choice = choice
            self.timer = 0


        gvec = torch.zeros(1, self.n_goals, device=self.device)
        gvec[0, int(choice)] = 1.0

        post_1d = post.squeeze() if post.dim() > 1 else post

        # Compute confidence (entropy-based)
        post_1d = torch.as_tensor(post_1d, device=self.device, dtype=torch.float32)
        if post_1d.ndim == 0:
            post_1d = post_1d.unsqueeze(0)

        entropy = torch.distributions.Categorical(probs=post_1d).entropy()
        max_entropy = torch.log(torch.tensor(float(self.n_goals), device=self.device))
        conf = float(1.0 - entropy / max_entropy)

        return efe_vec, post, choice, gvec, conf



class ActivePolicyFieldEFE(BaseModule):
    """
    Policy inference via expected free-energy minimization.
    Each policy π_i corresponds to a goal.
    The agent samples actions with probability ∝ exp(−EFE_i).
    """

    def __init__(self, n_goals=3, beta=2.0, persist_steps=8, device="cpu"):
        super().__init__(device=device)
        self.n_goals = n_goals
        self.beta = beta
        self.persist_steps = persist_steps
        self.timer = 0
        self.last_choice = 1
        self.efe_hist = torch.zeros(n_goals, device=device)

    # --------------------------------------------------------
    def step(self, efe_estimates, cms=0.0):
        """
        efe_estimates: list or tensor of expected free-energy per goal
        cms: novelty signal modulating precision
        """
        efe_vec = torch.tensor(efe_estimates, device=self.device, dtype=torch.float32)
        efe_vec = efe_vec - efe_vec.min()  # relative EFE

        # softmax over −EFE to obtain policy posterior
        policy_post = torch.softmax(-self.beta * efe_vec, dim=0)

        if self.timer < self.persist_steps:
            self.timer += 1
            choice = self.last_choice
        else:
            choice = torch.multinomial(policy_post, 1).item()
            self.last_choice = choice
            self.timer = 0

        # one-hot goal vector
        gvec = torch.zeros(1, self.n_goals, device=self.device)
        gvec[0, int(choice)] = 1.0

        # meta confidence (inverse entropy)
        conf = float(1.0 - torch.distributions.Categorical(policy_post).entropy().item() / torch.log(torch.tensor(self.n_goals)))
        self.efe_hist = 0.9 * self.efe_hist + 0.1 * efe_vec

        return choice, gvec, policy_post.cpu().numpy(), conf


class GoalPolicyField(BaseModule):
    """
    Unified motivational–policy system with dopaminergic goal-value learning
    and meta-precision regulation.
    """

    def __init__(self, mode, n_goals=3, tau=0.2, gain=1.5, persist_steps=10,
                 lr_value=0.1, gamma=0.95, device="cpu"):
        super().__init__(device=device)
        self.mode = mode
        self.n_goals = n_goals
        self.tau = tau
        self.gain = gain
        self.persist_steps = persist_steps
        self.lr_value = lr_value
        self.gamma = gamma

        # Internal state
        self.pref = torch.zeros(n_goals, device=device)
        self.Q = torch.zeros(n_goals, device=device)
        self.last_choice = 1
        self.timer = 0

        # Continuous fields
        self.motive = 0.0
        self.gain_state = 1.0

        # Meta-precision (confidence tracker)
        self.meta_precision = 0.5  # 0.0 = uncertain, 1.0 = confident

    # ------------------------------------------------------------
    def step(self, dopa, rpe, F_total, mem_mean, cms=0.0):
        # === continuous motivational update ===
        target = 0.5 * (dopa - 0.5) - 0.1 * F_total
        self.motive = 0.9 * self.motive + 0.1 * target
        self.motive = float(torch.clamp(torch.tensor(self.motive), -1.0, 1.0))

        # === adaptive gain ===
        delta = abs(dopa - 0.5) + abs(self.motive) + 0.2 * mem_mean
        self.gain_state = 0.9 * self.gain_state + 0.1 * (1.0 + delta)

        # === meta-precision update (confidence tracking) ===
        rpe_abs = abs(float(rpe))
        cms_abs = abs(float(cms))
        conf_update = 1.0 - torch.tanh(torch.tensor(rpe_abs + 0.5 * cms_abs))
        self.meta_precision = 0.95 * self.meta_precision + 0.05 * conf_update.item()

        # modulate gain_state by meta-precision (confidence-weighted control)
        self.gain_state *= (1.0 + 0.5 * (self.meta_precision - 0.5))
        self.gain_state = float(torch.clamp(torch.tensor(self.gain_state), 0.8, 2.0))

        # === dopaminergic Q-value update ===
        q_old = self.Q[self.last_choice]
        td_error = float(rpe + (dopa - 0.5)) + self.gamma * q_old - q_old
        self.Q[self.last_choice] = q_old + self.lr_value * td_error

        # === persistence and decision ===
        if self.timer < self.persist_steps:
            self.timer += 1
            prob = torch.softmax(self.pref, dim=0)
            gvec = torch.zeros(1, self.n_goals, device=self.device)
            gvec[0, int(self.last_choice)] = 1.0
            return self.motive, self.gain_state, self.last_choice, gvec, td_error, self.meta_precision

        # re-evaluate after persistence
        self.timer = 0
        drive = self.gain * (dopa - 0.5) + self.motive - 0.3 * cms + self.Q
        noise = 0.05 * torch.randn(self.n_goals, device=self.device)
        self.pref = self.tau * self.pref + (1 - self.tau) * (drive + noise)

        prob = torch.softmax(self.pref, dim=0)
        choice = torch.multinomial(prob, 1).item()
        self.last_choice = choice

        gvec = torch.zeros(1, self.n_goals, device=self.device)
        gvec[0, int(choice)] = 1.0

        return self.motive, self.gain_state, choice, gvec, td_error, self.meta_precision

    # ------------------------------------------------------------
    def reset_state(self):
        self.pref.zero_()
        self.Q.zero_()
        self.motive = 0.0
        self.gain_state = 1.0
        self.meta_precision = 0.5
        self.timer = 0
        self.last_choice = 1


# ============================================================
# BACKWARD-COMPATIBLE SIMPLE GoalPolicy (for one-hot vectors)
# ============================================================
class GoalPolicy(BaseModule):
    """Lightweight one-hot goal vector generator (for legacy scripts)."""
    def __init__(self, mode, n_goals=3, device="cpu"):
        super().__init__(device=device)
        self.mode = mode
        self.n_goals = n_goals

    def one_hot(self, idx):
        g = torch.zeros(1, self.n_goals, device=self.device)
        g[0, int(idx) % self.n_goals] = 1.0
        return g


# ============================================================
# ALIAS FOR BACKWARD IMPORTS
# ============================================================
PolicyField = GoalPolicyField
