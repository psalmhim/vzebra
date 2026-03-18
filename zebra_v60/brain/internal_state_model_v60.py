"""
Internal State Model — interoceptive forward model (Step 31).

Extends InteroceptiveEnergyModel with multi-step energy trajectory
simulation under different behavioral policies.  The agent can now
answer: "if I flee for 20 steps, will I starve?"

Components:
  - MetabolicCostModel: learned drain/gain rates per behavioral mode
  - InternalStateModel: Bayesian energy tracker + trajectory simulator

Neuroscience: anterior insula maintains forward models of bodily states
(Craig 2009; Seth & Friston 2016).  Zebrafish hypothalamic circuits
regulate energy homeostasis with predictive components (Yokogawa et al.
2012).  The trajectory comparison implements *counterfactual active
inference* — evaluating energy futures before committing to a policy.

Pure numpy — no torch.
"""
import numpy as np


# Goal indices (must match goal_policy_v60)
GOAL_FORAGE = 0
GOAL_FLEE = 1
GOAL_EXPLORE = 2
GOAL_SOCIAL = 3

# Stereotyped speed and drain per goal (calibrated to motor controller)
# FLEE speed 1.4 matches burst speed (1.5x base * speed_mod)
_DEFAULT_SPEEDS = [0.6, 1.4, 0.5, 0.4]
_DEFAULT_DRAINS = [1.0, 3.0, 0.8, 0.5]


class MetabolicCostModel:
    """Learned metabolic cost model, updated from prediction errors.

    Tracks energy drain per behavioural mode.  Priors come from
    EFE engine constants; corrections are learned online.
    """

    def __init__(self):
        self.base_drain = 0.08        # idle metabolic rate per step
        self.speed_factor = 1.0       # multiplier per unit speed
        self.drain_mult = list(_DEFAULT_DRAINS)  # per-goal multiplier
        self.food_gain = 2.0          # energy per small food item

        # EMA corrections
        self._drain_corrections = [0.0, 0.0, 0.0, 0.0]
        self._gain_correction = 0.0
        self._n_updates = 0
        self._alpha = 0.02  # learning rate

    def predict_drain(self, speed, goal_idx, energy_ratio):
        """Predict energy drain for one step.

        Args:
            speed: float [0, 1]
            goal_idx: int
            energy_ratio: float [0, 1] — current energy fraction

        Returns:
            drain: float — predicted energy loss this step
        """
        # Starvation multiplier (from env model)
        if energy_ratio < 0.20:
            starv_mult = 1.3
        elif energy_ratio < 0.50:
            starv_mult = 1.15
        else:
            starv_mult = 1.0

        mult = self.drain_mult[goal_idx] + self._drain_corrections[goal_idx]
        drain = self.base_drain * speed * max(0.5, mult) * starv_mult
        return drain

    def predict_gain(self, is_foraging, food_density=0.0):
        """Predict expected energy gain for one step.

        Args:
            is_foraging: bool
            food_density: float [0, 1] — from geographic model

        Returns:
            gain: float — expected energy gain
        """
        if not is_foraging:
            return 0.0
        # Probability of eating ≈ food_density × base_capture_rate
        base_rate = 0.02  # ~1 food per 50 steps when foraging
        p_eat = min(0.1, base_rate * (1.0 + 3.0 * food_density))
        return p_eat * (self.food_gain + self._gain_correction)

    def update(self, predicted_drain, actual_drain, goal_idx,
               predicted_gain, actual_gain):
        """Update model from prediction errors."""
        self._n_updates += 1
        alpha = self._alpha

        # Drain correction
        drain_pe = actual_drain - predicted_drain
        self._drain_corrections[goal_idx] += alpha * drain_pe

        # Gain correction
        gain_pe = actual_gain - predicted_gain
        self._gain_correction += alpha * gain_pe

        # Keep corrections bounded
        for i in range(4):
            self._drain_corrections[i] = np.clip(
                self._drain_corrections[i], -0.5, 0.5)
        self._gain_correction = np.clip(self._gain_correction, -5.0, 5.0)


class InternalStateModel:
    """Interoceptive forward model: energy tracking + trajectory simulation.

    Drop-in replacement for InteroceptiveEnergyModel (same API for
    predict / observe / get_energy / reset).  Adds trajectory
    simulation and policy comparison.

    Args:
        initial_energy: float
        noise_std: float — interoceptive observation noise
        observation_weight: float — Kalman-like gain
        simulation_horizon: int — look-ahead steps
    """

    def __init__(self, initial_energy=100.0, noise_std=3.0,
                 observation_weight=0.2, simulation_horizon=30):
        self.noise_std = noise_std
        self.observation_weight = observation_weight
        self._belief = initial_energy
        self._prediction = initial_energy
        self.prediction_error = 0.0

        self.metabolic = MetabolicCostModel()
        self.simulation_horizon = simulation_horizon

        # Cached trajectories (updated when compare_policies is called)
        self._trajectories = {}
        self._time_to_starvation = {}
        self._cached_risk = None
        self._cached_tts = None
        self._cache_step = -1
        self._cache_interval = 5  # recompute every 5 steps

        # History for metabolic learning
        self._prev_energy = initial_energy
        self._prev_speed = 0.0
        self._prev_goal = GOAL_EXPLORE
        self._prev_eaten = 0

    # ------------------------------------------------------------------
    # InteroceptiveEnergyModel API (backward compatible)
    # ------------------------------------------------------------------

    def predict(self, speed, eaten):
        """Predict energy from motor efference copy."""
        drain = self.metabolic.predict_drain(
            speed, self._prev_goal, self._belief / 100.0)
        gain = self.metabolic.food_gain * eaten
        self._prediction = max(0.0, min(100.0, self._belief - drain + gain))

        # Store for learning
        self._prev_speed = speed
        self._prev_eaten = eaten

    def observe(self, raw_energy):
        """Bayesian update with noisy interoceptive signal."""
        noisy = raw_energy + np.random.normal(0, self.noise_std)
        noisy = max(0.0, min(100.0, noisy))
        w = self.observation_weight
        new_belief = (1 - w) * self._prediction + w * noisy
        new_belief = max(0.0, min(100.0, new_belief))
        self.prediction_error = abs(self._prediction - noisy)

        # Metabolic learning: use raw observation (not filtered belief)
        # to avoid self-fulfilling prediction confound
        actual_change = noisy - self._prev_energy
        predicted_change = self._prediction - self._prev_energy
        predicted_drain = max(0.0, -predicted_change)
        actual_drain = max(0.0, -actual_change)
        predicted_gain = max(0.0, predicted_change)
        actual_gain = max(0.0, actual_change)

        self.metabolic.update(
            predicted_drain, actual_drain, self._prev_goal,
            predicted_gain, actual_gain)

        self._prev_energy = new_belief
        self._belief = new_belief

    def get_energy(self):
        """Return current energy belief."""
        return self._belief

    # ------------------------------------------------------------------
    # Trajectory simulation
    # ------------------------------------------------------------------

    def simulate_trajectory(self, goal_idx, food_density=0.0):
        """Simulate energy trajectory under a goal policy for N steps.

        Args:
            goal_idx: int — 0=FORAGE, 1=FLEE, 2=EXPLORE, 3=SOCIAL
            food_density: float — expected food availability

        Returns:
            trajectory: numpy [horizon] — predicted energy at each step
            time_to_starvation: int — steps until energy ≤ 0 (or horizon)
        """
        H = self.simulation_horizon
        speed = _DEFAULT_SPEEDS[goal_idx]
        traj = np.zeros(H, dtype=np.float32)
        e = self._belief

        starvation_step = H
        for t in range(H):
            e_ratio = e / 100.0
            drain = self.metabolic.predict_drain(speed, goal_idx, e_ratio)
            gain = self.metabolic.predict_gain(
                goal_idx == GOAL_FORAGE, food_density)
            e = max(0.0, min(100.0, e - drain + gain))
            traj[t] = e
            if e <= 0 and starvation_step == H:
                starvation_step = t

        return traj, starvation_step

    def compare_policies(self, food_density=0.0):
        """Compare all 4 goal policies by energy trajectory.

        Returns:
            energy_risk: numpy [4] — risk per goal (higher = worse)
            time_to_starvation: numpy [4] — steps until energy=0
        """
        risk = np.zeros(4, dtype=np.float32)
        tts = np.zeros(4, dtype=np.float32)

        for gi in range(4):
            traj, starvation_step = self.simulate_trajectory(
                gi, food_density)
            self._trajectories[gi] = traj
            self._time_to_starvation[gi] = starvation_step
            tts[gi] = starvation_step

            # Risk: how much energy is lost + starvation penalty
            min_e = float(traj.min())
            mean_e = float(traj.mean())
            # Below 20 → high risk, below 0 → critical
            risk[gi] = max(0.0, (50.0 - mean_e) / 50.0)
            if starvation_step < self.simulation_horizon:
                risk[gi] += 1.0  # starvation penalty

        return risk, tts

    def get_energy_efe_bias(self, food_density=0.0):
        """Compute EFE bias from energy trajectory comparison.

        Caches policy comparison — only recomputes every _cache_interval steps.

        Returns:
            bias: numpy [4] — lower = more attractive
        """
        self._cache_step += 1
        if (self._cached_risk is None
                or self._cache_step % self._cache_interval == 0):
            risk, tts = self.compare_policies(food_density)
            self._cached_risk = risk
            self._cached_tts = tts
        else:
            risk = self._cached_risk
            tts = self._cached_tts

        # Normalise: shift so best policy has bias=0
        bias = risk - risk.min()

        # Scale: stronger effect when energy is low
        energy_urgency = max(0.0, (50.0 - self._belief) / 50.0)
        bias *= 0.3 * (1.0 + 2.0 * energy_urgency)

        # FORAGE always gets a bonus when energy < 70%
        if self._belief < 70:
            forage_bonus = 0.15 * (70.0 - self._belief) / 70.0
            bias[GOAL_FORAGE] -= forage_bonus

        return bias

    def get_starvation_urgency(self):
        """How close is the fish to starvation?

        Returns:
            urgency: float [0, 1]
            estimated_steps: int — steps until death at current rate
        """
        # Quick estimate: at current drain rate
        if self._prev_speed > 0.01:
            drain = self.metabolic.predict_drain(
                self._prev_speed, self._prev_goal, self._belief / 100.0)
            if drain > 0.001:
                steps = self._belief / drain
            else:
                steps = 9999
        else:
            steps = 9999

        urgency = max(0.0, min(1.0, (50.0 - self._belief) / 50.0))
        return urgency, int(min(9999, steps))

    def set_goal(self, goal_idx):
        """Track current goal for metabolic learning."""
        self._prev_goal = goal_idx

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def get_saveable_state(self):
        return {
            "drain_corrections": list(self.metabolic._drain_corrections),
            "gain_correction": self.metabolic._gain_correction,
            "n_updates": self.metabolic._n_updates,
        }

    def load_saveable_state(self, state):
        self.metabolic._drain_corrections = list(state["drain_corrections"])
        self.metabolic._gain_correction = state["gain_correction"]
        self.metabolic._n_updates = state["n_updates"]

    def reset(self, initial_energy=100.0):
        self._belief = initial_energy
        self._prediction = initial_energy
        self.prediction_error = 0.0
        self._prev_energy = initial_energy
        self._prev_speed = 0.0
        self._prev_goal = GOAL_EXPLORE
        self._prev_eaten = 0
        self._trajectories = {}
        self._time_to_starvation = {}

    def get_diagnostics(self):
        urgency, est_steps = self.get_starvation_urgency()
        return {
            "energy_belief": self._belief,
            "prediction_error": self.prediction_error,
            "starvation_urgency": urgency,
            "estimated_steps_to_starvation": est_steps,
            "metabolic_updates": self.metabolic._n_updates,
            "drain_corrections": list(self.metabolic._drain_corrections),
        }
