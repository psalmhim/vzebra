"""
Cerebellum Forward Model — motor learning and prediction (Step 33).

Predicts sensory consequences of motor commands using a Marr-Albus-Ito
architecture: mossy fiber input → granule cell sparse expansion →
parallel fiber–Purkinje cell linear readout.  Climbing fiber error
(actual − predicted outcome) drives LTD at parallel fiber synapses.

Outputs:
  - Motor correction (adaptive gain for turn and speed)
  - Reafference prediction (expected visual/flow change from self-motion)
  - Prediction error (novelty signal for unexpected outcomes)

Neuroscience: zebrafish cerebellum (corpus cerebelli) is critical for
adaptive gain control of VOR/OKR (Ahrens et al. 2012), calibrating
burst-glide swim timing (Sengupta & Bhatt 2019), and precise prey
capture (Matsui et al. 2014).

Pure numpy.
"""
import numpy as np


class CerebellumForwardModel:
    """Cerebellar forward model with LTD learning.

    Args:
        dim_motor: int — motor command dimensions [turn, speed, goal]
        dim_context: int — sensory context dimensions
        dim_predict: int — predicted output dimensions
        n_granule_ratio: int — granule:mossy expansion factor
        sparsity: float — fraction of granule cells active
        lr: float — parallel fiber–Purkinje learning rate
        eligibility_decay: float — trace decay
        warmup: int — steps before corrections activate
    """

    def __init__(self, dim_motor=3, dim_context=8, dim_predict=6,
                 n_granule_ratio=4, sparsity=0.5, lr=0.005,
                 eligibility_decay=0.95, warmup=50):
        self.dim_input = dim_motor + dim_context
        self.dim_predict = dim_predict
        self.n_granule = self.dim_input * n_granule_ratio
        self.sparsity = sparsity
        self.lr = lr
        self.eligibility_decay = eligibility_decay
        self.warmup = warmup

        # Mossy fiber → granule cell random projection (frozen)
        rng = np.random.RandomState(42)
        self.W_mossy = rng.randn(
            self.dim_input, self.n_granule).astype(np.float32) * 0.3

        # Parallel fiber → Purkinje cell weights (learnable via LTD)
        self.W_pf = np.zeros(
            (self.n_granule, dim_predict), dtype=np.float32)

        # Eligibility trace (credit assignment across time)
        self._eligibility = np.zeros(self.n_granule, dtype=np.float32)

        # State
        self._granule_activity = np.zeros(self.n_granule, dtype=np.float32)
        self._prediction = np.zeros(dim_predict, dtype=np.float32)
        self._step_count = 0

        # Running motor correction (EMA of recent errors)
        self._turn_correction = 0.0
        self._speed_correction = 0.0
        self._error_ema = np.zeros(dim_predict, dtype=np.float32)

    def step(self, motor_command, sensory_context):
        """Forward pass: predict sensory consequences.

        Args:
            motor_command: np.array[3] — [turn_rate, speed, goal_idx]
            sensory_context: np.array[8] — [heading, speed, energy_ratio,
                p_food, p_enemy, obs_L, obs_R, dopa]

        Returns:
            prediction: np.array[6] — predicted next-step:
                [delta_heading, delta_speed, visual_change_L, visual_change_R,
                 food_eaten_prob, delta_energy]
        """
        self._step_count += 1

        # Mossy fiber input
        mossy = np.concatenate([
            np.asarray(motor_command, dtype=np.float32),
            np.asarray(sensory_context, dtype=np.float32)])

        # Granule cell expansion (sparse ReLU)
        granule_raw = mossy @ self.W_mossy
        granule = np.maximum(0, granule_raw)

        # Sparsify: keep top fraction
        if granule.sum() > 0:
            threshold = np.percentile(
                granule[granule > 0], (1 - self.sparsity) * 100)
            granule[granule < threshold] = 0.0

        self._granule_activity = granule

        # Update eligibility trace
        self._eligibility = (self.eligibility_decay * self._eligibility
                             + granule)

        # Purkinje readout → prediction
        self._prediction = granule @ self.W_pf
        return self._prediction.copy()

    def update(self, actual_outcome):
        """Climbing fiber error: update Purkinje weights via LTD.

        Args:
            actual_outcome: np.array[6] — actual sensory outcome
                (same format as prediction)
        """
        actual = np.asarray(actual_outcome, dtype=np.float32)
        error = actual - self._prediction

        # Normalise error to prevent large gradients
        error_norm = np.clip(error, -1.0, 1.0)

        # LTD: decrease weights that contributed to wrong prediction
        if self._step_count > 5:
            elig_norm = self._eligibility / (
                np.linalg.norm(self._eligibility) + 1e-8)
            dW = -self.lr * np.outer(elig_norm, error_norm)
            dW = np.clip(dW, -0.005, 0.005)
            self.W_pf += dW
            self.W_pf *= 0.9999  # weight decay

        # Update error EMA for motor correction (clamped)
        self._error_ema = np.clip(
            0.95 * self._error_ema + 0.05 * error_norm, -1.0, 1.0)

    def get_motor_correction(self):
        """Additive motor correction from recent prediction errors.

        Returns:
            turn_correction: float — bias added to turn_rate
            speed_correction: float — multiplicative factor for speed
        """
        if self._step_count < self.warmup:
            return 0.0, 1.0

        # Turn correction: if we consistently over-predict left turn,
        # add a rightward correction (clamped to prevent runaway)
        turn_err = self._error_ema[0]  # delta_heading error
        self._turn_correction = np.clip(
            0.95 * self._turn_correction - 0.05 * turn_err, -0.1, 0.1)
        turn_corr = float(self._turn_correction)

        # Speed correction: multiplicative (clamped)
        speed_err = self._error_ema[1]  # delta_speed error
        self._speed_correction = np.clip(
            0.95 * self._speed_correction - 0.05 * speed_err, -0.15, 0.15)
        speed_corr = float(np.clip(1.0 + self._speed_correction, 0.85, 1.15))

        return turn_corr, speed_corr

    def get_reafference_prediction(self):
        """Predicted visual/flow change from self-motion.

        Returns:
            pred_flow_L: float — expected left visual change
            pred_flow_R: float — expected right visual change
        """
        return float(self._prediction[2]), float(self._prediction[3])

    def get_prediction_error_magnitude(self):
        """Overall prediction error magnitude (novelty signal)."""
        return float(np.sqrt(np.sum(self._error_ema ** 2)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def get_saveable_state(self):
        return {
            "W_pf": self.W_pf.copy(),
            "error_ema": self._error_ema.copy(),
            "turn_correction": self._turn_correction,
            "speed_correction": self._speed_correction,
            "step_count": self._step_count,
        }

    def load_saveable_state(self, state):
        self.W_pf = state["W_pf"].copy()
        self._error_ema = state["error_ema"].copy()
        self._turn_correction = state["turn_correction"]
        self._speed_correction = state["speed_correction"]
        self._step_count = state["step_count"]

    def reset(self):
        """Reset transient state, keep learned weights."""
        self._eligibility[:] = 0.0
        self._granule_activity[:] = 0.0
        self._prediction[:] = 0.0
        self._error_ema[:] = 0.0
        self._turn_correction = 0.0
        self._speed_correction = 0.0

    def get_diagnostics(self):
        return {
            "prediction_error": self.get_prediction_error_magnitude(),
            "turn_correction": self._turn_correction,
            "speed_correction": self._speed_correction,
            "step_count": self._step_count,
            "weight_norm": float(np.linalg.norm(self.W_pf)),
        }
