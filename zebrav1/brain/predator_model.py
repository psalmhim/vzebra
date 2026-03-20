"""
Predator World Model — tectal/pallial threat tracker (Step 31).

Maintains a Bayesian belief about predator position, velocity, and intent
using a simplified Kalman filter.  Provides *object permanence*: when the
predator leaves the visual field, the belief propagates forward with growing
uncertainty rather than dropping to zero.

Key outputs:
  - Predicted predator position (even when invisible)
  - Time-to-contact (TTC) estimate with confidence
  - Hunting-intent probability (from velocity toward fish)
  - Vigilance drive (epistemic urge to re-locate lost predator)
  - Flee-direction recommendation (away from *predicted* future position)

Neuroscience: zebrafish optic tectum maintains persistent object
representations (Bianco & Bhatt 2019).  Velocity-sensitive looming
neurons in tectum estimate TTC (Temizer et al. 2015).  Object
permanence emerges in pallium via predictive coding.

Pure numpy — no torch.
"""
import math
import numpy as np


class PredatorBelief:
    """Bayesian belief state for a single predator.

    State vector: [x, y, vx, vy, intent]
      - (x, y) position in gym pixels
      - (vx, vy) velocity (pixels/step)
      - intent: 0=patrolling, 1=hunting
    """

    __slots__ = ("x", "y", "vx", "vy", "intent",
                 "pos_var", "vel_var",
                 "visible", "steps_since_seen", "last_seen_pos",
                 "arena_w", "arena_h")

    def __init__(self, arena_w=800, arena_h=600):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.x = arena_w / 2
        self.y = arena_h / 2
        self.vx = 0.0
        self.vy = 0.0
        self.intent = 0.0

        self.pos_var = 200.0 ** 2  # high initial uncertainty
        self.vel_var = 5.0 ** 2
        self.visible = False
        self.steps_since_seen = 999
        self.last_seen_pos = None


class PredatorModel:
    """Tectal/pallial predator tracker with object permanence.

    Args:
        arena_w, arena_h: arena pixel dimensions
        process_noise_pos: position diffusion per step (px²)
        process_noise_vel: velocity diffusion per step
        obs_noise_pos: observation noise for retinal position estimate (px²)
        visibility_threshold: min enemy pixels to count as "visible"
        intent_sigmoid_scale: steepness of intent inference
        memory_horizon: steps after which belief reverts to prior
    """

    def __init__(self, arena_w=800, arena_h=600,
                 process_noise_pos=4.0, process_noise_vel=1.0,
                 obs_noise_pos=900.0,
                 visibility_threshold=3,
                 intent_sigmoid_scale=0.3,
                 memory_horizon=100):
        self.belief = PredatorBelief(arena_w, arena_h)
        self.arena_w = arena_w
        self.arena_h = arena_h

        self.Q_pos = process_noise_pos
        self.Q_vel = process_noise_vel
        self.R_pos = obs_noise_pos
        self.vis_thresh = visibility_threshold
        self.intent_scale = intent_sigmoid_scale
        self.memory_horizon = memory_horizon

        # Short history for velocity estimation
        self._obs_history = []  # [(step, x_est, y_est)]

    # ------------------------------------------------------------------
    # Prediction step
    # ------------------------------------------------------------------

    def predict(self):
        """Propagate belief one step forward (constant-velocity model).

        Called every step regardless of visibility.
        """
        b = self.belief
        b.x += b.vx
        b.y += b.vy

        # Clamp to arena
        b.x = float(np.clip(b.x, 0, b.arena_w))
        b.y = float(np.clip(b.y, 0, b.arena_h))

        # Uncertainty grows (capped velocity contribution)
        b.pos_var += self.Q_pos + min(10.0, abs(b.vx) + abs(b.vy))
        b.vel_var += self.Q_vel

        # Intent decays toward patrol when unseen
        if not b.visible:
            b.steps_since_seen += 1
            b.intent *= 0.97  # slow decay toward 0 (patrol)

            # After long absence, revert to arena-centre prior
            if b.steps_since_seen > self.memory_horizon:
                revert = 0.01
                b.x += revert * (b.arena_w / 2 - b.x)
                b.y += revert * (b.arena_h / 2 - b.y)
                b.vx *= 0.99
                b.vy *= 0.99

    # ------------------------------------------------------------------
    # Observation update
    # ------------------------------------------------------------------

    def update(self, retinal_features, fish_pos, fish_heading, step,
               gt_pred_pos=None):
        """Update belief from retinal observation + optional ground truth.

        Args:
            retinal_features: dict from _extract_retinal_features()
            fish_pos: array-like [2] — fish gym position
            fish_heading: float — fish heading (radians)
            step: int — current step
            gt_pred_pos: (x, y) or None — ground truth predator position
                         (used in non-AI mode for accurate tracking)
        """
        # Ground truth update (non-AI mode): direct position with small noise
        if gt_pred_pos is not None:
            b = self.belief
            # Noisy observation (simulates imperfect omniscience)
            noise = 10.0  # 10px noise
            obs_x = gt_pred_pos[0] + np.random.normal(0, noise)
            obs_y = gt_pred_pos[1] + np.random.normal(0, noise)
            # Kalman update with low noise
            K = 0.5
            b.x += K * (obs_x - b.x)
            b.y += K * (obs_y - b.y)
            b.pos_var = (1 - K) * b.pos_var + 50.0
            b.visible = True
            b.steps_since_seen = 0
            b.last_seen_pos = (obs_x, obs_y)
            # Velocity from position delta
            self._obs_history.append((step, obs_x, obs_y))
            if len(self._obs_history) > 20:
                self._obs_history = self._obs_history[-20:]
            if len(self._obs_history) >= 3:
                recent = self._obs_history[-3:]
                dt = recent[-1][0] - recent[0][0]
                if dt > 0:
                    b.vx = 0.7 * b.vx + 0.3 * (recent[-1][1] - recent[0][1]) / dt
                    b.vy = 0.7 * b.vy + 0.3 * (recent[-1][2] - recent[0][2]) / dt
            self._infer_intent(fish_pos)
            return

        # Retinal update (AI mode)
        rf = retinal_features
        enemy_px = rf.get("enemy_px_total", 0.0)

        if enemy_px >= self.vis_thresh:
            self._update_visible(rf, fish_pos, fish_heading, step)
        else:
            self.belief.visible = False

    def _update_visible(self, rf, fish_pos, fish_heading, step):
        """Kalman-like update when predator is visible."""
        b = self.belief

        # Estimate predator position from retinal features
        x_est, y_est, conf = self._estimate_position(
            rf, fish_pos, fish_heading)

        if conf < 0.1:
            b.visible = False
            return

        b.visible = True
        b.steps_since_seen = 0
        b.last_seen_pos = (x_est, y_est)

        # Kalman gain: K = P / (P + R)
        K_pos = b.pos_var / (b.pos_var + self.R_pos / (conf + 0.1))

        # Position update
        b.x += K_pos * (x_est - b.x)
        b.y += K_pos * (y_est - b.y)
        b.pos_var = (1 - K_pos) * b.pos_var

        # Velocity from observation history
        self._obs_history.append((step, x_est, y_est))
        if len(self._obs_history) > 20:
            self._obs_history = self._obs_history[-20:]

        if len(self._obs_history) >= 3:
            # Use last 3 observations for velocity
            recent = self._obs_history[-3:]
            dt = recent[-1][0] - recent[0][0]
            if dt > 0:
                vx_est = (recent[-1][1] - recent[0][1]) / dt
                vy_est = (recent[-1][2] - recent[0][2]) / dt
                K_vel = min(0.3, b.vel_var / (b.vel_var + 4.0))
                b.vx += K_vel * (vx_est - b.vx)
                b.vy += K_vel * (vy_est - b.vy)
                b.vel_var = (1 - K_vel) * b.vel_var

        # Infer intent
        self._infer_intent(fish_pos)

    def _estimate_position(self, rf, fish_pos, fish_heading):
        """Convert retinal enemy features → position estimate.

        Uses:
          - enemy_lateral_bias → bearing offset from heading
          - enemy_intensity_mean → distance proxy (intensity ∝ 1/d)
          - enemy_px_total → confidence

        Returns:
            (x_est, y_est, confidence)
        """
        lateral = rf.get("enemy_lateral_bias", 0.0)
        intensity = rf.get("enemy_intensity_mean", 0.0)
        px_total = rf.get("enemy_px_total", 0.0)

        # Bearing: lateral_bias is unreliable at extremes, compress via atan
        bearing = fish_heading + math.atan(lateral * 1.5)

        # Distance: intensity ∝ exp(-d/scale). Invert via log.
        # Scale 120 calibrated to renderer: ~0.6 at 60px, ~0.2 at 200px
        if intensity > 0.02:
            dist_est = min(400.0, -120.0 * math.log(max(0.01, intensity)))
        else:
            dist_est = 350.0

        x_est = fish_pos[0] + dist_est * math.cos(bearing)
        y_est = fish_pos[1] + dist_est * math.sin(bearing)

        # Clamp to arena
        x_est = float(np.clip(x_est, 0, self.arena_w))
        y_est = float(np.clip(y_est, 0, self.arena_h))

        # Confidence from pixel count (more pixels = more reliable)
        confidence = min(1.0, px_total / 30.0)

        return x_est, y_est, confidence

    def _infer_intent(self, fish_pos):
        """Infer hunting intent from velocity direction toward fish."""
        b = self.belief
        dx = fish_pos[0] - b.x
        dy = fish_pos[1] - b.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8

        # Velocity toward fish
        speed = math.sqrt(b.vx * b.vx + b.vy * b.vy) + 1e-8
        v_toward = (b.vx * dx + b.vy * dy) / (dist * speed + 1e-8)

        # Sigmoid: positive v_toward + high speed → hunting
        raw = v_toward * speed * self.intent_scale
        b.intent = 1.0 / (1.0 + math.exp(-raw))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def predict_future_position(self, horizon=10):
        """Predict predator position N steps ahead.

        Returns:
            pred_pos: numpy [2]
            uncertainty_radius: float (px)
        """
        b = self.belief
        px = b.x + b.vx * horizon
        py = b.y + b.vy * horizon
        px = float(np.clip(px, 0, b.arena_w))
        py = float(np.clip(py, 0, b.arena_h))

        # Uncertainty grows with horizon
        unc = math.sqrt(b.pos_var + self.Q_pos * horizon * horizon)

        return np.array([px, py], dtype=np.float32), unc

    def get_flee_direction(self, fish_pos, horizon=5):
        """Recommend flee direction: away from predicted future position.

        Returns:
            flee_angle: float (radians)
            urgency: float [0, 1]
        """
        pred_pos, unc = self.predict_future_position(horizon)
        dx = fish_pos[0] - pred_pos[0]
        dy = fish_pos[1] - pred_pos[1]
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8

        flee_angle = math.atan2(dy, dx)

        # Urgency: close + high intent + low uncertainty
        prox = max(0.0, 1.0 - dist / 300.0)
        conf = 1.0 / (1.0 + unc / 100.0)
        urgency = prox * conf * (0.3 + 0.7 * self.belief.intent)

        return flee_angle, float(urgency)

    def get_ttc(self, fish_pos):
        """Time-to-contact estimate from belief state.

        Returns:
            ttc: float (steps, 999 if retreating)
            confidence: float [0, 1]
        """
        b = self.belief
        dx = fish_pos[0] - b.x
        dy = fish_pos[1] - b.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8

        # Closing speed
        speed = math.sqrt(b.vx * b.vx + b.vy * b.vy) + 1e-8
        v_toward = (b.vx * dx + b.vy * dy) / (dist + 1e-8)

        if v_toward > 0.5:
            ttc = dist / v_toward
        else:
            ttc = 999.0

        confidence = 1.0 / (1.0 + math.sqrt(b.pos_var) / 50.0)
        if not b.visible:
            confidence *= max(0.1, 1.0 - b.steps_since_seen / 50.0)

        return float(ttc), float(confidence)

    def get_vigilance_drive(self):
        """Epistemic drive to re-locate a lost predator.

        High when: recently seen + now lost + was hunting.
        Low when: visible, or very long since seen.

        Returns:
            vigilance: float [0, 1]
        """
        b = self.belief
        if b.visible:
            return 0.0

        recency = max(0.0, 1.0 - b.steps_since_seen / 60.0)
        return float(recency * (0.3 + 0.7 * b.intent))

    def get_threat_level(self, fish_pos=None):
        """Integrated threat from belief: proximity × intent × confidence.

        Args:
            fish_pos: array-like [2] or None — fish position for distance calc

        Returns:
            threat: float [0, 1]
        """
        b = self.belief

        # Proximity from fish_pos (if available)
        if fish_pos is not None:
            dx = fish_pos[0] - b.x
            dy = fish_pos[1] - b.y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-8
            proximity = max(0.0, 1.0 - dist / 400.0)
        elif b.last_seen_pos is not None:
            proximity = 0.5  # fallback: moderate
        else:
            proximity = 0.2

        # Confidence decays with uncertainty and time unseen
        uncertainty = math.sqrt(b.pos_var)
        conf = 1.0 / (1.0 + uncertainty / 80.0)
        if not b.visible:
            conf *= max(0.1, 1.0 - b.steps_since_seen / 80.0)

        return float(proximity * conf * (0.2 + 0.8 * b.intent))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def get_saveable_state(self):
        b = self.belief
        return {
            "x": b.x, "y": b.y, "vx": b.vx, "vy": b.vy,
            "intent": b.intent, "pos_var": b.pos_var, "vel_var": b.vel_var,
            "steps_since_seen": b.steps_since_seen,
        }

    def load_saveable_state(self, state):
        b = self.belief
        b.x = state["x"]
        b.y = state["y"]
        b.vx = state["vx"]
        b.vy = state["vy"]
        b.intent = state["intent"]
        b.pos_var = state["pos_var"]
        b.vel_var = state["vel_var"]
        b.steps_since_seen = state["steps_since_seen"]

    def reset_episode(self):
        """Reset belief for new episode (predator location unknown)."""
        b = self.belief
        b.x = b.arena_w / 2
        b.y = b.arena_h / 2
        b.vx = 0.0
        b.vy = 0.0
        b.intent = 0.0
        b.pos_var = 200.0 ** 2
        b.vel_var = 5.0 ** 2
        b.visible = False
        b.steps_since_seen = 999
        b.last_seen_pos = None
        self._obs_history = []

    def reset(self):
        self.reset_episode()

    def get_diagnostics(self):
        b = self.belief
        return {
            "pred_belief_x": b.x,
            "pred_belief_y": b.y,
            "pred_belief_vx": b.vx,
            "pred_belief_vy": b.vy,
            "pred_intent": b.intent,
            "pred_pos_uncertainty": math.sqrt(b.pos_var),
            "pred_visible": b.visible,
            "pred_steps_since_seen": b.steps_since_seen,
            "vigilance": self.get_vigilance_drive(),
            "threat_level": self.get_threat_level(),
        }
