"""
Active Inference Policy Selection via Expected Free Energy (EFE).

Maintains four goal states (FORAGE, FLEE, EXPLORE, SOCIAL) and selects
between them by minimizing EFE. Persistence timer prevents goal flickering.
Emergency override for high-confidence enemy detection.

Includes SpikingGoalSelector (Level 3 SNN-up): winner-take-all attractor
network with lateral inhibition for biologically-grounded goal selection,
replacing hand-crafted persistence timer with emergent dynamics.
"""
import numpy as np
import torch
import torch.nn as nn

GOAL_FORAGE = 0
GOAL_FLEE = 1
GOAL_EXPLORE = 2
GOAL_SOCIAL = 3
GOAL_NAMES = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]


class GoalPolicy_v60:
    """EFE-based policy selection with persistence and emergency override."""

    def __init__(self, n_goals=4, beta=2.0, persist_steps=8,
                 weight_adapter=None, efe_engine=None):
        self.n_goals = n_goals
        self.beta = beta
        self.persist_steps = persist_steps
        self.weight_adapter = weight_adapter
        self.efe_engine = efe_engine

        self.timer = 0
        self.last_choice = GOAL_EXPLORE
        self.efe_smooth = np.zeros(n_goals)  # exponential moving average
        self._alpha = 0.7  # EFE smoothing factor (higher = faster adaptation)
        self._external_efe_bonus = np.zeros(n_goals)
        self._efe_engine_result = None  # precomputed EFE from EFEEngine

    def set_plan_bonus(self, bonus):
        """Set external EFE bonus from VAE planner (Step 16).

        Automatically pads shorter arrays to n_goals with zeros.
        """
        bonus = np.asarray(bonus, dtype=np.float64)
        if len(bonus) < self.n_goals:
            bonus = np.pad(bonus, (0, self.n_goals - len(bonus)))
        self._external_efe_bonus = bonus

    def set_efe_engine_result(self, efe_result):
        """Set precomputed EFE from EFEEngine (Step 25).

        When set, compute_efe() returns this instead of hand-crafted EFE.
        Cleared after each step() call.

        Args:
            efe_result: numpy [n_goals] — EFE per goal from EFEEngine
        """
        self._efe_engine_result = np.asarray(efe_result, dtype=np.float64)

    def compute_efe(self, cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual,
                    mem_mean, pred_facing_score=0.0, ttc=999.0,
                    energy_ratio=1.0):
        """Compute Expected Free Energy for each goal.

        Args:
            cls_probs: array [5] — softmax classifier output
                       [nothing, food, enemy, colleague, environment]
            pi_OT: float — optic tectum precision (0–1)
            pi_PC: float — predictive coding precision (0–1)
            dopa: float — dopamine level (0–1)
            rpe: float — reward prediction error
            cms: float — cross-modal surprise
            F_visual: float — visual free energy
            mem_mean: float — working memory mean activation
            pred_facing_score: float [0, 1] — predator gaze alignment
            ttc: float — time-to-contact in steps (999 if retreating)
            energy_ratio: float [0, 1] — fish energy / max energy

        Returns:
            efe: array [n_goals] — EFE per goal (lower = preferred)
        """
        # Delegate to EFE engine if result precomputed (Step 25)
        if self._efe_engine_result is not None:
            return self._efe_engine_result

        # Delegate to weight adapter if present (Step 15 RL critic)
        if self.weight_adapter is not None:
            efe_3 = self.weight_adapter.compute_efe(
                cls_probs, pi_OT, pi_PC, dopa, cms,
                rpe=rpe, F_visual=F_visual, mem_mean=mem_mean)
            # Pad to n_goals if adapter returns fewer dims
            if len(efe_3) < self.n_goals:
                efe_3 = np.pad(efe_3, (0, self.n_goals - len(efe_3)),
                               constant_values=0.3)
            return efe_3

        p_nothing = cls_probs[0]
        p_food = cls_probs[1]
        p_enemy = cls_probs[2]
        p_colleague = cls_probs[3] if len(cls_probs) > 3 else 0.0
        p_environ = cls_probs[4] if len(cls_probs) > 4 else 0.0

        uncertainty = 1.0 - 0.5 * (pi_OT + pi_PC)

        # --- Bayesian survival trade-off ---
        # Starvation risk: ramps from 0 at 50% energy to 1.0 at 0%
        starvation_risk = max(0.0, (0.50 - energy_ratio) / 0.50)
        # Critical zone (<30%): quadratic urgency ramp
        if energy_ratio < 0.30:
            starvation_risk = starvation_risk ** 0.5  # sharper curve

        # FORAGE: attractive when food visible OR energy is low
        #   - starvation_risk lowers EFE (makes foraging urgent)
        #   - predator proximity adds foraging cost (risk of being caught)
        forage_predator_cost = 0.15 * p_enemy  # foraging near predator is risky
        g_forage = (0.2 * uncertainty
                    - 0.8 * p_food
                    + 0.2 * (0.5 - dopa)
                    + 0.15
                    - 0.8 * starvation_risk     # starving → forage urgency
                    + forage_predator_cost)      # predator → foraging cost

        # FLEE: attractive when enemy is detected
        #   - starvation_risk raises EFE (fleeing costs energy → death risk)
        #   - flee speed drain = 2.5-3.5x base → burns reserves fast
        gaze_boost = -0.3 * pred_facing_score * p_enemy
        ttc_boost = -0.2 * max(0, 1.0 - ttc / 50.0) if ttc < 50 else 0.0
        flee_energy_cost = 0.5 * starvation_risk  # fleeing when starving → death
        g_flee = (0.1 * cms
                  - 0.8 * p_enemy
                  + 0.2
                  + flee_energy_cost             # energy cost of fleeing
                  + gaze_boost + ttc_boost)

        # EXPLORE: attractive when seeing nothing/environment
        g_explore = (0.3 * uncertainty
                     - 0.5 * (p_nothing + p_environ)
                     + 0.1 * cms
                     + 0.2
                     + 0.2 * starvation_risk)    # don't explore when starving

        # SOCIAL: attractive when colleagues visible, aversive when enemy near
        g_social = (0.2 * uncertainty
                    - 0.6 * p_colleague
                    + 0.15 * p_enemy
                    + 0.25
                    + 0.15 * starvation_risk)    # don't socialize when starving

        return np.array([g_forage, g_flee, g_explore, g_social])

    def step(self, cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual,
             mem_mean, hunger=0.0, hunger_error=0.0,
             pred_facing_score=0.0, ttc=999.0, energy_ratio=1.0):
        """Select goal by minimizing smoothed EFE.

        Returns:
            choice: int — selected goal index
            goal_vec: array [n_goals] — one-hot goal vector
            posterior: array [n_goals] — policy posterior probabilities
            confidence: float — entropy-based meta-precision (0–1)
            efe_vec: array [n_goals] — raw EFE values
        """
        efe_raw = self.compute_efe(cls_probs, pi_OT, pi_PC, dopa, rpe, cms,
                                   F_visual, mem_mean,
                                   pred_facing_score=pred_facing_score,
                                   ttc=ttc, energy_ratio=energy_ratio)
        # Clear engine result after use (one-shot per step)
        self._efe_engine_result = None

        # Ensure efe_raw matches n_goals (pad if adapter returned fewer)
        if len(efe_raw) < self.n_goals:
            efe_raw = np.pad(efe_raw, (0, self.n_goals - len(efe_raw)),
                             constant_values=0.3)

        # Apply external planning bonus (Step 16 VAE planner)
        efe_raw = efe_raw + self._external_efe_bonus

        # Light smoothing for EFE history (used for diagnostics/plots)
        self.efe_smooth = ((1 - self._alpha) * self.efe_smooth
                           + self._alpha * efe_raw)

        # Use raw EFE for selection — the persistence timer provides stability.
        # Smoothing the selection signal causes excessive goal stickiness.
        efe_rel = efe_raw - efe_raw.min()

        # Policy posterior: softmax(-beta * G)
        logits = -self.beta * efe_rel
        logits = logits - logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        posterior = exp_logits / (exp_logits.sum() + 1e-12)

        # Survival trade-off: Bayesian model comparison
        # Both predator and starvation are death risks — compare urgencies
        p_enemy = cls_probs[2]
        p_food = cls_probs[1]
        p_colleague = cls_probs[3] if len(cls_probs) > 3 else 0.0
        starvation_urgency = max(0.0, (0.50 - energy_ratio) / 0.50)

        if p_enemy > 0.25 and starvation_urgency < 0.6:
            # Predator threat + adequate energy → hard FLEE
            choice = GOAL_FLEE
            self.last_choice = choice
            self.timer = 0
        elif p_enemy > 0.25 and starvation_urgency >= 0.6:
            # BOTH threats active — Bayesian model comparison
            # Let the EFE softmax decide: flee (risk starvation) vs forage
            # (risk predator). The compute_efe() already weighted both risks.
            choice = int(np.argmax(posterior))
            self.last_choice = choice
            self.timer = 0
        elif starvation_urgency > 0.7:
            # Critical starvation, no immediate predator → force FORAGE
            choice = GOAL_FORAGE
            self.last_choice = choice
            self.timer = 0
        elif hunger > 0.6 and hunger_error > 0.2:
            # Moderate hunger with declining trend → FORAGE
            choice = GOAL_FORAGE
            self.last_choice = choice
            self.timer = 0
        elif (self.last_choice == GOAL_FLEE and p_enemy < 0.20
              and self.timer >= 3):
            # Early exit from FLEE when no enemy detected
            choice = int(np.argmax(posterior))
            self.last_choice = choice
            self.timer = 0
        elif (self.last_choice == GOAL_FORAGE and p_food < 0.15
              and self.timer >= 4):
            # Early exit from FORAGE when no food detected
            choice = int(np.argmax(posterior))
            self.last_choice = choice
            self.timer = 0
        elif (self.last_choice == GOAL_SOCIAL and p_colleague < 0.1
              and self.timer >= 3):
            # Early exit from SOCIAL when no colleagues detected
            choice = int(np.argmax(posterior))
            self.last_choice = choice
            self.timer = 0
        elif self.timer < self.persist_steps:
            # Persist current goal
            self.timer += 1
            choice = self.last_choice
        else:
            # Re-select: pick goal with highest posterior (lowest EFE)
            choice = int(np.argmax(posterior))
            self.last_choice = choice
            self.timer = 0

        # Goal vector (one-hot)
        goal_vec = np.zeros(self.n_goals)
        goal_vec[choice] = 1.0

        # Confidence: entropy-based meta-precision
        entropy = -np.sum(posterior * np.log(posterior + 1e-12))
        max_entropy = np.log(self.n_goals)
        confidence = 1.0 - entropy / max_entropy

        return choice, goal_vec, posterior, confidence, self.efe_smooth.copy()

    def reset(self):
        self.timer = 0
        self.last_choice = GOAL_EXPLORE
        self.efe_smooth = np.zeros(self.n_goals)
        self._external_efe_bonus = np.zeros(self.n_goals)
        self._efe_engine_result = None


class SpikingGoalSelector(nn.Module):
    """Winner-take-all spiking attractor for goal selection (Level 3 SNN-up).

    Models basal ganglia action selection via mutual inhibition.
    Each goal accumulates evidence (excitation from -EFE), and lateral
    inhibition suppresses competitors. The winner emerges through
    temporal dynamics — no explicit persistence timer needed.

    Emergency overrides (threat, hunger) inject strong excitatory current
    that rapidly overwhelms accumulated evidence for other goals.
    """

    def __init__(self, n_goals=4, beta=2.0, device="cpu",
                 weight_adapter=None, efe_engine=None):
        super().__init__()
        self.n_goals = n_goals
        self.beta = beta
        self.device = device
        self.weight_adapter = weight_adapter
        self.efe_engine = efe_engine

        # Membrane state for each goal unit
        self.v = torch.zeros(1, n_goals, device=device)
        self.tau = 0.75  # membrane time constant

        # Lateral inhibition: off-diagonal negative weights
        self.W_lat = nn.Parameter(
            -0.4 * (torch.ones(n_goals, n_goals, device=device)
                     - torch.eye(n_goals, device=device)))

        # Self-excitation: diagonal positive weights (persistence)
        self.W_self = nn.Parameter(
            0.3 * torch.eye(n_goals, device=device))

        # EFE tracking (same interface as GoalPolicy_v60)
        self.efe_smooth = np.zeros(n_goals)
        self._alpha = 0.7
        self._external_efe_bonus = np.zeros(n_goals)
        self._efe_engine_result = None

        # Track last choice for diagnostics
        self.last_choice = GOAL_EXPLORE

        # RL-modulated learning parameters
        self.eta_lat = 0.001    # lateral inhibition learning rate
        self.eta_self = 0.0008  # self-excitation learning rate
        self.decay = 0.9999     # weight decay
        self._prev_choice = GOAL_EXPLORE
        self._prev_reward = 0.0

        self.to(device)

    def set_plan_bonus(self, bonus):
        """Set external EFE bonus from VAE planner (Step 16)."""
        bonus = np.asarray(bonus, dtype=np.float64)
        if len(bonus) < self.n_goals:
            bonus = np.pad(bonus, (0, self.n_goals - len(bonus)))
        self._external_efe_bonus = bonus

    def set_efe_engine_result(self, efe_result):
        """Set precomputed EFE from EFEEngine (Step 25)."""
        self._efe_engine_result = np.asarray(efe_result, dtype=np.float64)

    def compute_efe(self, cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual,
                    mem_mean, pred_facing_score=0.0, ttc=999.0,
                    energy_ratio=1.0):
        """Compute EFE — delegates same as GoalPolicy_v60."""
        if self._efe_engine_result is not None:
            return self._efe_engine_result

        if self.weight_adapter is not None:
            efe = self.weight_adapter.compute_efe(
                cls_probs, pi_OT, pi_PC, dopa, cms,
                rpe=rpe, F_visual=F_visual, mem_mean=mem_mean)
            if len(efe) < self.n_goals:
                efe = np.pad(efe, (0, self.n_goals - len(efe)),
                             constant_values=0.3)
            return efe

        p_food = cls_probs[1]
        p_enemy = cls_probs[2]
        p_colleague = cls_probs[3] if len(cls_probs) > 3 else 0.0
        p_environ = cls_probs[4] if len(cls_probs) > 4 else 0.0
        p_nothing = cls_probs[0]
        uncertainty = 1.0 - 0.5 * (pi_OT + pi_PC)

        # Bayesian survival trade-off (same as GoalPolicy_v60)
        starvation_risk = max(0.0, (0.50 - energy_ratio) / 0.50)
        if energy_ratio < 0.30:
            starvation_risk = starvation_risk ** 0.5

        forage_predator_cost = 0.15 * p_enemy
        g_forage = (0.2 * uncertainty - 0.8 * p_food
                    + 0.2 * (0.5 - dopa) + 0.15
                    - 0.8 * starvation_risk + forage_predator_cost)
        gaze_boost = -0.3 * pred_facing_score * p_enemy
        ttc_boost = -0.2 * max(0, 1.0 - ttc / 50.0) if ttc < 50 else 0.0
        flee_energy_cost = 0.5 * starvation_risk
        g_flee = (0.1 * cms - 0.8 * p_enemy + 0.2
                  + flee_energy_cost + gaze_boost + ttc_boost)
        g_explore = (0.3 * uncertainty - 0.5 * (p_nothing + p_environ)
                     + 0.1 * cms + 0.2 + 0.2 * starvation_risk)
        g_social = (0.2 * uncertainty - 0.6 * p_colleague
                    + 0.15 * p_enemy + 0.25 + 0.15 * starvation_risk)

        return np.array([g_forage, g_flee, g_explore, g_social])

    def step(self, cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual,
             mem_mean, hunger=0.0, hunger_error=0.0,
             pred_facing_score=0.0, ttc=999.0, energy_ratio=1.0):
        """Select goal via WTA spiking dynamics.

        Returns same interface as GoalPolicy_v60.step().
        """
        efe_raw = self.compute_efe(
            cls_probs, pi_OT, pi_PC, dopa, rpe, cms, F_visual, mem_mean,
            pred_facing_score=pred_facing_score, ttc=ttc,
            energy_ratio=energy_ratio)
        self._efe_engine_result = None

        if len(efe_raw) < self.n_goals:
            efe_raw = np.pad(efe_raw, (0, self.n_goals - len(efe_raw)),
                             constant_values=0.3)

        efe_raw = efe_raw + self._external_efe_bonus

        # EFE smoothing for diagnostics
        self.efe_smooth = ((1 - self._alpha) * self.efe_smooth
                           + self._alpha * efe_raw)

        # Convert EFE to excitatory drive: lower EFE = stronger excitation
        efe_t = torch.tensor(efe_raw, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        excitation = -self.beta * efe_t

        # Survival override: inject current proportional to urgency
        # Both threats compete through WTA dynamics (Bayesian model comparison)
        p_enemy = cls_probs[2]
        starvation_urgency = max(0.0, (0.50 - energy_ratio) / 0.50)

        if p_enemy > 0.25:
            # Scale FLEE injection by how safe energy is
            # Full boost (5.0) when energy adequate, reduced when starving
            flee_boost = 5.0 * max(0.3, 1.0 - starvation_urgency * 0.7)
            excitation[0, GOAL_FLEE] += flee_boost
        if starvation_urgency > 0.4:
            # Starvation current: ramps from 2.0 at 30% to 5.0 at 0%
            forage_boost = 2.0 + 3.0 * starvation_urgency
            excitation[0, GOAL_FORAGE] += forage_boost
        elif hunger > 0.6 and hunger_error > 0.2:
            excitation[0, GOAL_FORAGE] += 4.0

        # WTA dynamics: v = tau*v + (1-tau)*(excitation + lateral + self)
        lateral = self.v @ self.W_lat      # mutual inhibition
        self_exc = self.v @ self.W_self     # self-excitation (persistence)
        self.v = (self.tau * self.v
                  + (1 - self.tau) * (excitation + lateral + self_exc))

        # Rectify (neurons can't have negative firing rate)
        self.v = torch.clamp(self.v, 0.0, 5.0)

        # Winner = highest accumulated membrane potential
        v_np = self.v[0].detach().cpu().numpy()
        choice = int(np.argmax(v_np))
        self.last_choice = choice

        # Posterior from softmax of membrane potentials (for interface compat)
        v_rel = v_np - v_np.max()
        exp_v = np.exp(v_rel)
        posterior = exp_v / (exp_v.sum() + 1e-12)

        # Goal vector (one-hot)
        goal_vec = np.zeros(self.n_goals)
        goal_vec[choice] = 1.0

        # Confidence: entropy-based meta-precision
        entropy = -np.sum(posterior * np.log(posterior + 1e-12))
        max_entropy = np.log(self.n_goals)
        confidence = 1.0 - entropy / max_entropy

        return choice, goal_vec, posterior, confidence, self.efe_smooth.copy()

    @torch.no_grad()
    def learn(self, rpe, dopa, reward):
        """RL-modulated learning for lateral inhibition and self-excitation.

        W_lat learns which goal pairs conflict (should inhibit each other
        more strongly) vs which are compatible (weaker inhibition).
        W_self learns how persistent each goal should be.

        Positive RPE after goal switch → previous goal was wrong,
        strengthen inhibition of previous by current.
        Negative RPE → current goal is bad, reduce its self-excitation.

        Args:
            rpe: float — reward prediction error
            dopa: float — dopamine level [0, 1]
            reward: float — immediate reward signal
        """
        if abs(rpe) < 0.05:
            self._prev_choice = self.last_choice
            self._prev_reward = reward
            return

        choice = self.last_choice
        dopa_mod = 2.0 * (dopa - 0.5)  # [-1, 1]

        # W_lat: strengthen inhibition between conflicting goals
        # If current goal yields positive RPE → it was right to suppress
        # the previous goal → strengthen current→previous inhibition
        if choice != self._prev_choice:
            delta_inh = -self.eta_lat * rpe * dopa_mod
            self.W_lat.data[choice, self._prev_choice] += max(-0.02,
                                                               min(0.02, delta_inh))
            self.W_lat.data[self._prev_choice, choice] += max(-0.02,
                                                               min(0.02, delta_inh))

        # W_self: adjust persistence based on reward outcome
        # Positive RPE → current goal is working → increase persistence
        # Negative RPE → current goal is failing → decrease persistence
        delta_self = self.eta_self * rpe * dopa_mod
        self.W_self.data[choice, choice] += max(-0.01, min(0.01, delta_self))

        # Clamp to maintain biological constraints
        # Lateral: must stay negative (inhibitory)
        mask_off = ~torch.eye(self.n_goals, dtype=torch.bool,
                              device=self.device)
        self.W_lat.data[mask_off] = torch.clamp(
            self.W_lat.data[mask_off], -1.0, -0.05)
        # Self: must stay positive (excitatory)
        mask_diag = torch.eye(self.n_goals, dtype=torch.bool,
                              device=self.device)
        self.W_self.data[mask_diag] = torch.clamp(
            self.W_self.data[mask_diag], 0.05, 0.8)
        # Zero out off-diagonal of W_self (self-excitation only)
        self.W_self.data[mask_off] = 0.0

        # Weight decay
        self.W_lat.data *= self.decay
        self.W_self.data *= self.decay

        self._prev_choice = choice
        self._prev_reward = reward

    def get_saveable_state(self):
        """Return learned weights for checkpoint persistence."""
        return {
            "W_lat": self.W_lat.data.cpu().clone(),
            "W_self": self.W_self.data.cpu().clone(),
        }

    def load_saveable_state(self, state):
        """Restore learned weights from checkpoint."""
        self.W_lat.data = state["W_lat"].to(self.device)
        self.W_self.data = state["W_self"].to(self.device)

    def reset(self):
        self.v = torch.zeros(1, self.n_goals, device=self.device)
        self.last_choice = GOAL_EXPLORE
        self._prev_choice = GOAL_EXPLORE
        self._prev_reward = 0.0
        self.efe_smooth = np.zeros(self.n_goals)
        self._external_efe_bonus = np.zeros(self.n_goals)
        self._efe_engine_result = None


def goal_to_behavior(active_goal, cls_probs, posterior, confidence,
                     pred_facing_score=0.0):
    """Convert active goal to behavioral modulation parameters.

    Args:
        active_goal: int — GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, or GOAL_SOCIAL
        cls_probs: array [5] — classifier probabilities
        posterior: array [n_goals] — goal posterior
        confidence: float — meta-precision (0–1)
        pred_facing_score: float [0, 1] — predator gaze alignment

    Returns:
        approach_gain: float — positive=approach, negative=flee
        speed_mod: float — speed multiplier
        explore_mod: float — BG exploration multiplier
        turn_strategy: str — description for logging
    """
    p_food = cls_probs[1]
    p_enemy = cls_probs[2]

    if active_goal == GOAL_FORAGE:
        approach_gain = 1.0 + 0.5 * p_food * confidence
        speed_mod = 1.0 + 0.3 * p_food
        explore_mod = 0.5
        turn_strategy = "forage-approach"

    elif active_goal == GOAL_FLEE:
        # Strong reversal — must turn AWAY from predator decisively
        approach_gain = -(1.2 + 0.8 * p_enemy * confidence)
        speed_mod = 1.5 + 0.5 * p_enemy
        # Adrenaline burst when predator is facing
        if pred_facing_score > 0.3:
            speed_mod += 0.5 * pred_facing_score
        explore_mod = 0.2
        turn_strategy = "flee-avoid"

    elif active_goal == GOAL_SOCIAL:
        approach_gain = 0.6
        speed_mod = 0.8
        explore_mod = 0.4
        turn_strategy = "social-shoal"

    else:  # GOAL_EXPLORE
        approach_gain = 0.3
        speed_mod = 0.9
        explore_mod = 1.5 + 0.5 * (1 - confidence)
        turn_strategy = "explore-search"

    return approach_gain, speed_mod, explore_mod, turn_strategy
