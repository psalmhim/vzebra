"""
Proper Active Inference EFE Engine (Step 25).

Computes Expected Free Energy (EFE) over a generative model with three
ecological drives of a larval zebrafish:

  1. Energy Economy — risk from deviation of predicted energy from setpoint
  2. Predator Risk  — risk from predicted threat proximity
  3. Exploration     — epistemic value from novelty × decoder uncertainty

Allostasis modulates the *precision* (inverse variance) of preferred outcome
distributions — hungry fish have tight energy precision, stressed fish have
tight safety precision. This replaces additive allostatic bias with a proper
Bayesian mechanism.

EFE per policy π over horizon H:
  G(π) = Σ γ^τ [ λ_E·Risk_E + λ_S·Risk_S + λ_F·Risk_F − λ_I·Epistemic + Ambiguity ]

Pure numpy for core math; torch only for VAE rollout calls.
"""
import numpy as np


class PreferredOutcomes:
    """Allostasis-modulated preferred outcome distribution C.

    Maintains preferred setpoints and precision (inverse σ²) for three
    ecological drives. Precision tightens under allostatic stress:
      - Hungry → tight energy precision
      - Stressed → tight safety precision
      - Fatigued → tight rest precision
    """

    def __init__(self, energy_star=70.0, sigma_E_base=20.0,
                 sigma_S_base=0.5, sigma_F_base=0.3):
        self.energy_star = energy_star
        self.sigma_E_base = sigma_E_base
        self.sigma_S_base = sigma_S_base
        self.sigma_F_base = sigma_F_base

        # Current precisions (updated each step)
        self.sigma_E = sigma_E_base
        self.sigma_S = sigma_S_base
        self.sigma_F = sigma_F_base

    def update(self, hunger, fatigue, stress):
        """Tighten σ based on allostatic state.

        Args:
            hunger: float [0, 1] — 0=full, 1=starving
            fatigue: float [0, 1]
            stress: float [0, 1]
        """
        # Hungry → tight energy precision (sharp preference for food)
        self.sigma_E = self.sigma_E_base / (
            1.0 + 3.0 * max(0.0, hunger - 0.5))
        # Stressed → tight safety precision (strong flee drive)
        self.sigma_S = self.sigma_S_base / (1.0 + 2.0 * stress)
        # Fatigued → tight rest precision (prefer slow/rest)
        self.sigma_F = self.sigma_F_base / (1.0 + 2.0 * fatigue)

    def energy_risk(self, predicted_energy):
        """Risk from energy deviation: (Ê − E*)² / (2σ²_E).

        Args:
            predicted_energy: float — predicted energy at future step

        Returns:
            float — energy risk (higher = worse)
        """
        diff = predicted_energy - self.energy_star
        return diff * diff / (2.0 * self.sigma_E * self.sigma_E + 1e-8)

    def safety_risk(self, threat_proximity):
        """Risk from threat: threat² / (2σ²_S).

        Args:
            threat_proximity: float [0, 1] — inverse distance to predator

        Returns:
            float — safety risk (higher = worse)
        """
        return (threat_proximity * threat_proximity
                / (2.0 * self.sigma_S * self.sigma_S + 1e-8))

    def fatigue_risk(self, predicted_speed, fatigue):
        """Risk from exertion: speed² · fatigue / (2σ²_F).

        Args:
            predicted_speed: float [0, 1] — speed under policy
            fatigue: float [0, 1] — current fatigue level

        Returns:
            float — fatigue risk
        """
        return (predicted_speed * predicted_speed * fatigue
                / (2.0 * self.sigma_F * self.sigma_F + 1e-8))

    def reset(self):
        self.sigma_E = self.sigma_E_base
        self.sigma_S = self.sigma_S_base
        self.sigma_F = self.sigma_F_base


class EFEEngine:
    """Proper EFE computation with three ecological drives.

    For each of 4 goals (FORAGE, FLEE, EXPLORE, SOCIAL), rolls out the
    VAE transition model over horizon H steps and scores each trajectory
    using energy risk, safety risk, fatigue risk, epistemic value, and
    ambiguity.

    Args:
        vae_world: VAEWorldModel instance (for rollout + memory)
        preferred: PreferredOutcomes instance
        horizon: int — planning horizon (default 5)
        gamma: float — temporal discount factor (default 0.9)
    """

    # Stereotyped speed per goal (normalized 0–1)
    GOAL_SPEEDS = [0.6, 1.0, 0.5, 0.4]  # FORAGE, FLEE, EXPLORE, SOCIAL
    # Energy drain multipliers per goal
    DRAIN_MULT = [1.0, 2.5, 0.8, 0.5]   # FLEE is expensive

    def __init__(self, vae_world, preferred, horizon=5, gamma=0.9):
        self.vae = vae_world
        self.preferred = preferred
        self.horizon = horizon
        self.gamma = gamma
        self._last_efe = np.zeros(4, dtype=np.float32)
        self._last_components = {}

    def compute_efe(self, z, retinal_features, allo_state, energy_belief,
                    lambdas=None):
        """Compute EFE for all 4 goals via VAE rollout.

        Args:
            z: numpy [16] — current latent state
            retinal_features: dict — from _extract_retinal_features()
            allo_state: dict — from allostasis.step() (or None)
            energy_belief: float — current energy belief [0, 100]
            lambdas: numpy [4] or None — [λ_E, λ_S, λ_F, λ_I] weights

        Returns:
            efe: numpy [4] — EFE per goal (lower = preferred)
        """
        if lambdas is None:
            lambdas = np.ones(4, dtype=np.float32)
        lam_E, lam_S, lam_F, lam_I = lambdas

        # Update preferred outcome precisions from allostatic state
        if allo_state is not None:
            self.preferred.update(
                allo_state["hunger"], allo_state["fatigue"],
                allo_state["stress"])
            fatigue = allo_state["fatigue"]
        else:
            fatigue = 0.0

        H = self.horizon
        n_goals = 4
        efe = np.zeros(n_goals, dtype=np.float32)
        components = {}

        for gi in range(n_goals):
            # Rollout trajectory through VAE transition model
            z_traj = self._rollout_policy(z, gi, H)

            risk_e_sum = 0.0
            risk_s_sum = 0.0
            risk_f_sum = 0.0
            epistemic_sum = 0.0
            ambiguity_sum = 0.0
            e_pred = energy_belief

            speed = self.GOAL_SPEEDS[gi]
            drain = self.DRAIN_MULT[gi]
            energy_gain_per_step = (0.3 if gi == 0 else 0.0)  # FORAGE gains

            for tau in range(H):
                discount = self.gamma ** (tau + 1)
                z_tau = z_traj[tau]

                # Starvation pressure: metabolic cost rises at low energy
                e_ratio = e_pred / 100.0
                if e_ratio < 0.30:
                    starvation_mult = 2.0
                elif e_ratio < 0.50:
                    starvation_mult = 1.5
                else:
                    starvation_mult = 1.0
                energy_drain_per_step = (0.08 * speed * drain
                                         * starvation_mult)

                # Predict energy at step τ
                e_pred = e_pred - energy_drain_per_step + energy_gain_per_step
                e_pred = max(0.0, min(100.0, e_pred))

                # Score this step
                (r_e, r_s, r_f, epist, ambig) = self._score_step(
                    z_tau, self.preferred, e_pred, speed, fatigue)

                risk_e_sum += discount * r_e
                risk_s_sum += discount * r_s
                risk_f_sum += discount * r_f
                epistemic_sum += discount * epist
                ambiguity_sum += discount * ambig

            # Final EFE: weighted sum of drives
            g = (lam_E * risk_e_sum
                 + lam_S * risk_s_sum
                 + lam_F * risk_f_sum
                 - lam_I * epistemic_sum
                 + ambiguity_sum)
            efe[gi] = g

            components[gi] = {
                "risk_e": risk_e_sum,
                "risk_s": risk_s_sum,
                "risk_f": risk_f_sum,
                "epistemic": epistemic_sum,
                "ambiguity": ambiguity_sum,
            }

        self._last_efe = efe.copy()
        self._last_components = components
        return efe

    def _rollout_policy(self, z, goal_idx, H):
        """Roll out VAE transition model for H steps under goal policy.

        Args:
            z: numpy [16] — starting latent state
            goal_idx: int — goal index (0–3)
            H: int — horizon

        Returns:
            list of numpy [16] — predicted latent states z_1..z_H
        """
        # Build stereotyped action contexts for this goal
        n_goals = 4
        speed = self.GOAL_SPEEDS[goal_idx]

        # Turn patterns per goal
        if goal_idx == 0:    # FORAGE: gentle forward
            turns = [0.1 * ((-1) ** s) for s in range(H)]
        elif goal_idx == 1:  # FLEE: sharp turn
            turns = [0.7] * H
        elif goal_idx == 2:  # EXPLORE: gentle weaving
            turns = [0.3 * ((-1) ** s) for s in range(H)]
        else:                # SOCIAL: slow approach
            turns = [0.05 * ((-1) ** s) for s in range(H)]

        action_contexts = []
        for s in range(H):
            goal_oh = np.zeros(n_goals, dtype=np.float32)
            goal_oh[goal_idx] = 1.0
            act = np.array([turns[s], speed, *goal_oh], dtype=np.float32)
            action_contexts.append(act)

        return self.vae.rollout_trajectory(z, action_contexts, H)

    def _score_step(self, z_tau, preferred, energy_tau, speed, fatigue):
        """Score a single predicted latent state.

        Args:
            z_tau: numpy [16] — predicted latent state
            preferred: PreferredOutcomes instance
            energy_tau: float — predicted energy at this step
            speed: float — speed under policy
            fatigue: float — current fatigue

        Returns:
            (risk_e, risk_s, risk_f, epistemic, ambiguity)
        """
        # Energy risk: deviation from setpoint
        risk_e = preferred.energy_risk(energy_tau)

        # Safety risk: threat from VAE threat decoder
        threat = self.vae.encode_threat(z_tau)
        threat_prox = float(threat[0])  # proximity channel
        risk_s = preferred.safety_risk(threat_prox)

        # Fatigue risk
        risk_f = preferred.fatigue_risk(speed, fatigue)

        # Epistemic value: novelty × decoder uncertainty
        novelty, _ = self.vae.memory.query_epistemic(z_tau)
        decoder_unc = self.vae.decode_uncertainty(z_tau)
        epistemic = novelty * decoder_unc

        # Ambiguity: reconstruction uncertainty (proxy for observation noise)
        ambiguity = decoder_unc * 0.3  # scaled down

        return risk_e, risk_s, risk_f, epistemic, ambiguity

    def get_diagnostics(self):
        return {
            "efe": self._last_efe.copy(),
            "components": {
                k: {ck: float(cv) for ck, cv in v.items()}
                for k, v in self._last_components.items()
            },
        }

    def reset(self):
        self._last_efe = np.zeros(4, dtype=np.float32)
        self._last_components = {}
