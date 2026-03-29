"""
Predator world model — Kalman-filter tracker with object permanence.

Maintains Bayesian belief about predator [x, y, vx, vy, intent].
When predator leaves visual field, belief propagates with growing uncertainty.

Two modes:
  - GT mode: reads env.pred_x/y directly (current default)
  - Retinal mode: estimates position from retinal enemy bearing + intensity

Outputs: predicted position, TTC, intent, flee direction, vigilance drive.
"""
import math
import numpy as np


class PredatorModel:
    def __init__(self, arena_w=800, arena_h=600,
                 process_noise_pos=4.0, process_noise_vel=1.0,
                 obs_noise_pos=900.0, vis_thresh=3,
                 memory_horizon=100):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.Q_pos = process_noise_pos
        self.Q_vel = process_noise_vel
        self.R_pos = obs_noise_pos
        self.vis_thresh = vis_thresh
        self.memory_horizon = memory_horizon

        # Belief state
        self.x = arena_w / 2
        self.y = arena_h / 2
        self.vx = 0.0
        self.vy = 0.0
        self.intent = 0.0
        self.pos_var = 200.0 ** 2
        self.vel_var = 5.0 ** 2
        self.visible = False
        self.steps_since_seen = 999

        self._obs_history = []

    def predict(self):
        """Propagate belief one step forward (constant-velocity)."""
        self.x = np.clip(self.x + self.vx, 0, self.arena_w)
        self.y = np.clip(self.y + self.vy, 0, self.arena_h)
        self.pos_var += self.Q_pos + min(10.0, abs(self.vx) + abs(self.vy))
        self.vel_var += self.Q_vel
        if not self.visible:
            self.steps_since_seen += 1
            self.intent *= 0.97
            if self.steps_since_seen > self.memory_horizon:
                self.x += 0.01 * (self.arena_w / 2 - self.x)
                self.y += 0.01 * (self.arena_h / 2 - self.y)
                self.vx *= 0.99
                self.vy *= 0.99

    def update_retinal(self, enemy_px, enemy_lateral_bias,
                       enemy_intensity, fish_pos, fish_heading, step):
        """Update from retinal observations (autonomous mode)."""
        if enemy_px < self.vis_thresh:
            self.visible = False
            return

        self.visible = True
        self.steps_since_seen = 0

        # Estimate position from retinal features
        bearing = fish_heading + math.atan(enemy_lateral_bias * 1.5)
        # Distance from pixel count + intensity (fused estimate)
        if enemy_px > 1:
            px_dist = min(400.0, max(30.0, 1200.0 / (enemy_px + 5.0)))
            if enemy_intensity > 0.02:
                int_dist = min(400.0, -120.0 * math.log(max(0.01, min(0.99, enemy_intensity))))
            else:
                int_dist = 350.0
            # Weighted average: pixel count reliable at close range, intensity at far
            w_px = min(1.0, enemy_px / 30.0)
            dist_est = w_px * px_dist + (1 - w_px) * int_dist
        else:
            dist_est = 350.0

        x_est = np.clip(fish_pos[0] + dist_est * math.cos(bearing), 0, self.arena_w)
        y_est = np.clip(fish_pos[1] + dist_est * math.sin(bearing), 0, self.arena_h)
        conf = min(1.0, enemy_px / 30.0)

        # Kalman update
        K = self.pos_var / (self.pos_var + self.R_pos / (conf + 0.1))
        self.x += K * (x_est - self.x)
        self.y += K * (y_est - self.y)
        self.pos_var = (1 - K) * self.pos_var

        # Velocity from history
        self._obs_history.append((step, x_est, y_est))
        if len(self._obs_history) > 20:
            self._obs_history = self._obs_history[-20:]
        if len(self._obs_history) >= 3:
            recent = self._obs_history[-3:]
            dt = recent[-1][0] - recent[0][0]
            if dt > 0:
                K_vel = min(0.3, self.vel_var / (self.vel_var + 4.0))
                self.vx += K_vel * ((recent[-1][1] - recent[0][1]) / dt - self.vx)
                self.vy += K_vel * ((recent[-1][2] - recent[0][2]) / dt - self.vy)
                self.vel_var = (1 - K_vel) * self.vel_var

        self._infer_intent(fish_pos)

    def update_gt(self, pred_x, pred_y, fish_pos, step):
        """Update from ground-truth position (GT mode)."""
        noise = 10.0
        obs_x = pred_x + np.random.normal(0, noise)
        obs_y = pred_y + np.random.normal(0, noise)
        K = 0.5
        self.x += K * (obs_x - self.x)
        self.y += K * (obs_y - self.y)
        self.pos_var = (1 - K) * self.pos_var + 50.0
        self.visible = True
        self.steps_since_seen = 0

        self._obs_history.append((step, obs_x, obs_y))
        if len(self._obs_history) > 20:
            self._obs_history = self._obs_history[-20:]
        if len(self._obs_history) >= 3:
            recent = self._obs_history[-3:]
            dt = recent[-1][0] - recent[0][0]
            if dt > 0:
                self.vx = 0.7 * self.vx + 0.3 * (recent[-1][1] - recent[0][1]) / dt
                self.vy = 0.7 * self.vy + 0.3 * (recent[-1][2] - recent[0][2]) / dt
        self._infer_intent(fish_pos)

    def _infer_intent(self, fish_pos):
        dx = fish_pos[0] - self.x
        dy = fish_pos[1] - self.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2) + 1e-8
        v_toward = (self.vx * dx + self.vy * dy) / (dist * speed + 1e-8)
        self.intent = 1.0 / (1.0 + math.exp(-v_toward * speed * 0.3))

    def get_flee_direction(self, fish_pos, horizon=5):
        """Flee angle away from predicted future position + urgency."""
        px = np.clip(self.x + self.vx * horizon, 0, self.arena_w)
        py = np.clip(self.y + self.vy * horizon, 0, self.arena_h)
        unc = math.sqrt(self.pos_var + self.Q_pos * horizon * horizon)
        dx = fish_pos[0] - px
        dy = fish_pos[1] - py
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8
        flee_angle = math.atan2(dy, dx)
        prox = max(0.0, 1.0 - dist / 300.0)
        conf = 1.0 / (1.0 + unc / 100.0)
        urgency = prox * conf * (0.3 + 0.7 * self.intent)
        return flee_angle, float(urgency)

    def get_ttc(self, fish_pos):
        """Time-to-contact estimate."""
        dx = fish_pos[0] - self.x
        dy = fish_pos[1] - self.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8
        v_toward = (self.vx * dx + self.vy * dy) / (dist + 1e-8)
        ttc = dist / v_toward if v_toward > 0.5 else 999.0
        conf = 1.0 / (1.0 + math.sqrt(self.pos_var) / 50.0)
        if not self.visible:
            conf *= max(0.1, 1.0 - self.steps_since_seen / 50.0)
        return float(ttc), float(conf)

    def get_threat_level(self, fish_pos):
        dx = fish_pos[0] - self.x
        dy = fish_pos[1] - self.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8
        prox = max(0.0, 1.0 - dist / 400.0)
        conf = 1.0 / (1.0 + math.sqrt(self.pos_var) / 80.0)
        if not self.visible:
            conf *= max(0.1, 1.0 - self.steps_since_seen / 80.0)
        return float(prox * conf * (0.2 + 0.8 * self.intent))

    def get_vigilance(self):
        if self.visible:
            return 0.0
        recency = max(0.0, 1.0 - self.steps_since_seen / 60.0)
        return float(recency * (0.3 + 0.7 * self.intent))

    def get_pred_dist(self, fish_pos):
        dx = fish_pos[0] - self.x
        dy = fish_pos[1] - self.y
        return math.sqrt(dx * dx + dy * dy)

    def reset(self):
        self.x = self.arena_w / 2
        self.y = self.arena_h / 2
        self.vx = 0.0
        self.vy = 0.0
        self.intent = 0.0
        self.pos_var = 200.0 ** 2
        self.vel_var = 5.0 ** 2
        self.visible = False
        self.steps_since_seen = 999
        self._obs_history = []
