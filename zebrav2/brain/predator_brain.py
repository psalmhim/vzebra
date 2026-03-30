"""
Intelligent predator brain: strategic hunting behavior.

Replaces simple chase with multi-strategy predator AI:
  1. PATROL: cruise the arena, scan for prey
  2. STALK: slow approach when prey detected, stay outside flee range
  3. AMBUSH: hide near food patches, wait for prey to approach
  4. HUNT: burst sprint toward selected target
  5. REST: recover stamina after failed hunt

Target selection: prioritize weakest (lowest energy) or most isolated fish.
Stamina system: burst speed depletes stamina, must rest to recover.
Learning: track which areas have more fish (spatial memory).
"""
import math
import numpy as np


class PredatorBrain:
    def __init__(self, arena_w=800, arena_h=600):
        self.arena_w = arena_w
        self.arena_h = arena_h

        # State
        self.state = 'PATROL'
        self.target_idx = -1
        self.target_x = arena_w / 2
        self.target_y = arena_h / 2

        # Stamina system
        self.stamina = 1.0          # 0-1, depletes during HUNT
        self.stamina_drain = 0.02   # per step during HUNT
        self.stamina_regen = 0.008  # per step during REST/PATROL
        self.burst_threshold = 0.3  # minimum stamina to start HUNT

        # Speed parameters
        self.patrol_speed = 1.5
        self.stalk_speed = 1.0
        self.hunt_speed = 4.0       # burst sprint
        self.rest_speed = 0.3

        # Spatial memory: where fish tend to be (10x8 grid)
        self.fish_density = np.zeros((8, 10), dtype=np.float32)

        # Hunting parameters
        self.detection_range = 250.0
        self.stalk_range = 150.0     # switch from stalk to hunt
        self.ambush_patience = 0      # countdown timer
        self.hunt_timer = 0
        self.hunt_max = 30            # max steps in HUNT before giving up
        self.rest_timer = 0

        # Patrol waypoints
        self._patrol_angle = 0.0
        self._patrol_target = (arena_w / 2, arena_h / 2)
        self._step = 0

        # Intelligence parameters
        self.distraction_chance = 0.03  # chance to lose focus
        self.nav_noise = 8.0            # navigation imprecision

    def step(self, pred_x, pred_y, fish_list):
        """
        Main predator brain step.
        pred_x, pred_y: current predator position
        fish_list: list of dicts with 'x', 'y', 'energy', 'alive', 'speed'
        Returns: (dx, dy, speed, state)
        """
        self._step += 1

        # Find alive fish
        alive_fish = [(i, f) for i, f in enumerate(fish_list)
                      if f.get('alive', True)]
        if not alive_fish:
            return 0, 0, 0, 'PATROL'

        # Update spatial memory
        for i, f in alive_fish:
            row = max(0, min(7, int(f['y'] / (self.arena_h / 8))))
            col = max(0, min(9, int(f['x'] / (self.arena_w / 10))))
            self.fish_density[row, col] += 0.1
        self.fish_density *= 0.999  # slow decay

        # Distance to each alive fish
        dists = []
        for i, f in alive_fish:
            d = math.sqrt((pred_x - f['x'])**2 + (pred_y - f['y'])**2)
            dists.append((d, i, f))
        dists.sort(key=lambda x: x[0])
        nearest_dist, nearest_idx, nearest_fish = dists[0]

        # State machine
        if self.state == 'PATROL':
            self._update_patrol(pred_x, pred_y, dists, alive_fish)
        elif self.state == 'STALK':
            self._update_stalk(pred_x, pred_y, dists)
        elif self.state == 'AMBUSH':
            self._update_ambush(pred_x, pred_y, dists, alive_fish)
        elif self.state == 'HUNT':
            self._update_hunt(pred_x, pred_y, dists)
        elif self.state == 'REST':
            self._update_rest(pred_x, pred_y)

        # Compute movement toward target
        dx = self.target_x - pred_x
        dy = self.target_y - pred_y
        dist_to_target = math.sqrt(dx * dx + dy * dy) + 1e-8
        dx /= dist_to_target
        dy /= dist_to_target

        # Add navigation noise
        dx += np.random.normal(0, self.nav_noise / 100)
        dy += np.random.normal(0, self.nav_noise / 100)

        # Distraction: random chance to lose focus
        if np.random.random() < self.distraction_chance and self.state != 'HUNT':
            dx += np.random.normal(0, 0.3)
            dy += np.random.normal(0, 0.3)

        # Speed based on state
        if self.state == 'HUNT':
            speed = self.hunt_speed * self.stamina
            self.stamina = max(0, self.stamina - self.stamina_drain)
        elif self.state == 'STALK':
            speed = self.stalk_speed
            self.stamina = min(1.0, self.stamina + self.stamina_regen * 0.5)
        elif self.state == 'REST':
            speed = self.rest_speed
            self.stamina = min(1.0, self.stamina + self.stamina_regen)
        elif self.state == 'AMBUSH':
            speed = 0.1  # nearly stationary
            self.stamina = min(1.0, self.stamina + self.stamina_regen)
        else:  # PATROL
            speed = self.patrol_speed
            self.stamina = min(1.0, self.stamina + self.stamina_regen * 0.5)

        return dx * speed, dy * speed, speed, self.state

    def _select_target(self, dists, alive_fish):
        """Select best target: weakest fish or most isolated."""
        if not alive_fish:
            return -1, self.arena_w / 2, self.arena_h / 2

        # Score each fish: low energy + isolation + proximity
        best_score = -999
        best_idx = 0
        best_x, best_y = alive_fish[0][1]['x'], alive_fish[0][1]['y']

        for dist, idx, fish in dists:
            if dist > self.detection_range:
                continue
            energy = fish.get('energy', 100)
            speed = fish.get('speed', 1.0)
            # Low energy = easier catch
            energy_score = (100 - energy) / 100.0
            # Slow fish = easier catch
            speed_score = max(0, 1.0 - speed)
            # Closer = better
            dist_score = max(0, 1.0 - dist / self.detection_range)
            # Isolation: far from other fish = no confusion effect
            isolation = 1.0
            for d2, i2, f2 in dists:
                if i2 != idx:
                    other_dist = math.sqrt((fish['x'] - f2['x'])**2 +
                                           (fish['y'] - f2['y'])**2)
                    if other_dist < 80:
                        isolation -= 0.2  # penalize grouped fish
            isolation = max(0, isolation)

            score = (0.3 * energy_score + 0.2 * speed_score +
                     0.3 * dist_score + 0.2 * isolation)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_x, best_y = fish['x'], fish['y']

        return best_idx, best_x, best_y

    def _update_patrol(self, px, py, dists, alive_fish):
        """Patrol: cruise arena, look for prey."""
        # Move toward high-density area or waypoint
        if self._step % 50 == 0 or math.sqrt(
                (px - self._patrol_target[0])**2 +
                (py - self._patrol_target[1])**2) < 30:
            # Pick new patrol target from density map
            best_r, best_c = np.unravel_index(
                np.argmax(self.fish_density), self.fish_density.shape)
            self._patrol_target = (
                (best_c + 0.5) * self.arena_w / 10 + np.random.normal(0, 50),
                (best_r + 0.5) * self.arena_h / 8 + np.random.normal(0, 50))
            self._patrol_target = (
                max(50, min(self.arena_w - 50, self._patrol_target[0])),
                max(50, min(self.arena_h - 50, self._patrol_target[1])))

        self.target_x, self.target_y = self._patrol_target

        # Detect prey → switch to STALK
        if dists[0][0] < self.detection_range:
            idx, tx, ty = self._select_target(dists, alive_fish)
            self.target_idx = idx
            self.target_x, self.target_y = tx, ty
            self.state = 'STALK'

            # Sometimes choose AMBUSH instead (near food patches)
            if self.stamina > 0.7 and np.random.random() < 0.3:
                self.state = 'AMBUSH'
                self.ambush_patience = 40

    def _update_stalk(self, px, py, dists):
        """Stalk: slow approach, stay outside flee range."""
        # Update target position
        for d, i, f in dists:
            if i == self.target_idx:
                self.target_x, self.target_y = f['x'], f['y']
                break

        target_dist = math.sqrt((px - self.target_x)**2 + (py - self.target_y)**2)

        # Close enough → HUNT
        if target_dist < self.stalk_range and self.stamina > self.burst_threshold:
            self.state = 'HUNT'
            self.hunt_timer = self.hunt_max
        # Lost target
        elif target_dist > self.detection_range * 1.5:
            self.state = 'PATROL'

    def _update_ambush(self, px, py, dists, alive_fish):
        """Ambush: wait near food, strike when prey approaches."""
        self.ambush_patience -= 1

        # Check if any fish came close
        for d, i, f in dists:
            if d < self.stalk_range and self.stamina > self.burst_threshold:
                self.target_idx = i
                self.target_x, self.target_y = f['x'], f['y']
                self.state = 'HUNT'
                self.hunt_timer = self.hunt_max
                return

        # Patience ran out
        if self.ambush_patience <= 0:
            self.state = 'PATROL'

    def _update_hunt(self, px, py, dists):
        """Hunt: burst sprint toward target."""
        self.hunt_timer -= 1

        # Update target position (track moving fish)
        for d, i, f in dists:
            if i == self.target_idx:
                # Predict intercept: aim ahead of target
                speed = f.get('speed', 0.5)
                heading = f.get('heading', 0)
                predict_steps = 3
                self.target_x = f['x'] + speed * 3.0 * math.cos(heading) * predict_steps
                self.target_y = f['y'] + speed * 3.0 * math.sin(heading) * predict_steps
                self.target_x = max(10, min(self.arena_w - 10, self.target_x))
                self.target_y = max(10, min(self.arena_h - 10, self.target_y))
                break

        # Exhausted or timed out → REST
        if self.stamina < 0.05 or self.hunt_timer <= 0:
            self.state = 'REST'
            self.rest_timer = 20

    def _update_rest(self, px, py):
        """Rest: recover stamina after failed hunt."""
        self.rest_timer -= 1
        # Drift slowly
        self.target_x = px + np.random.normal(0, 10)
        self.target_y = py + np.random.normal(0, 10)

        if self.rest_timer <= 0 and self.stamina > 0.5:
            self.state = 'PATROL'

    def reset(self):
        self.state = 'PATROL'
        self.target_idx = -1
        self.stamina = 1.0
        self.fish_density[:] = 0
        self.ambush_patience = 0
        self.hunt_timer = 0
        self.rest_timer = 0
        self._step = 0
