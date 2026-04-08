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
Energy system: energy drains over time, eating fish restores it.
  Low energy → higher aggression (wider detection, shorter rest, lower thresholds).
Learning: track which areas have more fish (spatial memory).
"""
import math
import numpy as np
from zebrav2.brain.predator_place_cells import PredatorPlaceCells


class PredatorBrain:
    def __init__(self, arena_w=800, arena_h=600):
        self.arena_w = arena_w
        self.arena_h = arena_h

        # State
        self.state = 'PATROL'
        self.target_idx = -1
        self.target_x = arena_w / 2
        self.target_y = arena_h / 2

        # Energy system: drains over time, eating fish restores it
        self.energy = 80.0              # 0-100
        self.energy_drain = 0.08        # per step passive drain
        self.energy_per_catch = 40.0    # restored when catching a fish

        # Stamina system (short-term burst capacity)
        self.stamina = 1.0          # 0-1, depletes during HUNT
        self.stamina_drain = 0.02   # per step during HUNT
        self.stamina_regen = 0.008  # per step during REST/PATROL
        self.burst_threshold = 0.3  # minimum stamina to start HUNT

        # Speed parameters
        self.patrol_speed = 1.5
        self.stalk_speed = 1.0
        self.hunt_speed = 4.0       # burst sprint
        self.rest_speed = 0.3

        # Spatial memory: place cells
        self.place = PredatorPlaceCells(arena_w=arena_w, arena_h=arena_h)

        # Hunting parameters (base values — modulated by hunger)
        self._base_detection_range = 250.0
        self._base_stalk_range = 150.0
        self._base_burst_threshold = 0.3
        self._base_hunt_max = 30
        self._base_rest_duration = 20
        self._base_distraction = 0.03
        self.detection_range = self._base_detection_range
        self.stalk_range = self._base_stalk_range
        self.ambush_patience = 0      # countdown timer
        self.hunt_timer = 0
        self.hunt_max = self._base_hunt_max
        self.rest_timer = 0

        # Patrol waypoints
        self._patrol_angle = 0.0
        self._patrol_target = (arena_w / 2, arena_h / 2)
        self._step = 0

        # Intelligence parameters
        self.distraction_chance = self._base_distraction
        self.nav_noise = 8.0            # navigation imprecision

    def _update_hunger_drive(self):
        """Modulate aggression parameters based on energy (hunger).

        Low energy → desperate predator: wider detection, shorter rest,
        lower burst threshold, longer hunts, less distraction.
        """
        # hunger: 0 = full (energy 100), 1 = starving (energy 0)
        hunger = max(0.0, min(1.0, 1.0 - self.energy / 100.0))

        # Detection range: 250 at full → 375 when starving (+50%)
        self.detection_range = self._base_detection_range * (1.0 + 0.5 * hunger)
        # Stalk range: 150 at full → 225 when starving (commits to hunt earlier)
        self.stalk_range = self._base_stalk_range * (1.0 + 0.5 * hunger)
        # Burst threshold: 0.3 at full → 0.1 when starving (hunts even when tired)
        self.burst_threshold = self._base_burst_threshold * (1.0 - 0.67 * hunger)
        # Hunt duration: 30 at full → 50 when starving (more persistent)
        self.hunt_max = int(self._base_hunt_max * (1.0 + 0.67 * hunger))
        # Distraction: 3% at full → 0.5% when starving (hyper-focused)
        self.distraction_chance = self._base_distraction * (1.0 - 0.83 * hunger)

    def on_catch(self):
        """Called when predator catches a fish -- restores energy."""
        self.energy = min(100.0, self.energy + self.energy_per_catch)
        # Record successful catch location in place cells
        if hasattr(self, '_last_pos'):
            px, py = self._last_pos
            self.place.update(px, py, [], hunt_success=True)

    def step(self, pred_x, pred_y, fish_list):
        """
        Main predator brain step.
        pred_x, pred_y: current predator position
        fish_list: list of dicts with 'x', 'y', 'energy', 'alive', 'speed'
        Returns: (dx, dy, speed, state)
        """
        self._step += 1
        self._last_pos = (pred_x, pred_y)

        # Energy drain and hunger-driven aggression
        self.energy = max(0.0, self.energy - self.energy_drain)
        self._update_hunger_drive()

        # Find alive fish
        alive_fish = [(i, f) for i, f in enumerate(fish_list)
                      if f.get('alive', True)]
        if not alive_fish:
            return 0, 0, 0, 'PATROL'

        # Update spatial memory via place cells
        visible_prey = [
            (f['x'], f['y']) for _, f in alive_fish
            if math.sqrt((pred_x - f['x'])**2 + (pred_y - f['y'])**2)
            < self.detection_range
        ]
        self.place.update(pred_x, pred_y, visible_prey)

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
        # Hungry predators hunt faster (4.0 at full → 5.5 when starving)
        hunger = max(0.0, min(1.0, 1.0 - self.energy / 100.0))
        effective_hunt_speed = self.hunt_speed * (1.0 + 0.4 * hunger)

        if self.state == 'HUNT':
            speed = effective_hunt_speed * self.stamina
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

        # Energy cost proportional to speed² (fast movement costs more)
        speed_cost = 0.0005 * speed * speed
        self.energy = max(0.0, self.energy - speed_cost)

        return dx * speed, dy * speed, speed, self.state

    def _select_target(self, pred_x, pred_y, dists, alive_fish):
        """Select best target: weakest fish or most isolated."""
        if not alive_fish:
            return -1, self.arena_w / 2, self.arena_h / 2

        # Place cell hunting bonus at current predator location
        bonus = self.place.get_hunting_bonus(pred_x, pred_y)

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
                     0.3 * dist_score + 0.2 * isolation +
                     bonus['hunt_bonus'] * 0.15)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_x, best_y = fish['x'], fish['y']

        return best_idx, best_x, best_y

    def _update_patrol(self, px, py, dists, alive_fish):
        """Patrol: cruise arena, look for prey."""
        # Move toward prey-rich area via place cells
        if self._step % 50 == 0 or math.sqrt(
                (px - self._patrol_target[0])**2 +
                (py - self._patrol_target[1])**2) < 30:
            hunger = max(0.0, min(1.0, 1.0 - self.energy / 100.0))
            self._patrol_target = self.place.get_patrol_target(hunger)

        self.target_x, self.target_y = self._patrol_target

        # Detect prey -> switch to STALK
        if dists[0][0] < self.detection_range:
            idx, tx, ty = self._select_target(px, py, dists, alive_fish)
            self.target_idx = idx
            self.target_x, self.target_y = tx, ty
            self.state = 'STALK'

            # Sometimes choose AMBUSH instead (hunger increases ambush chance)
            hunger = max(0.0, min(1.0, 1.0 - self.energy / 100.0))
            ambush_chance = 0.3 + 0.4 * hunger
            if self.stamina > 0.7 and np.random.random() < ambush_chance:
                self.state = 'AMBUSH'
                self.ambush_patience = 40
                ax, ay = self.place.get_ambush_site()
                self.target_x, self.target_y = ax, ay

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

        # Exhausted or timed out -> REST (failed hunt)
        if self.stamina < 0.05 or self.hunt_timer <= 0:
            self.place.update(px, py, [], hunt_failure=True)
            self.state = 'REST'
            # Hungry predators rest less (20 at full -> 8 when starving)
            hunger = max(0.0, min(1.0, 1.0 - self.energy / 100.0))
            self.rest_timer = int(self._base_rest_duration * (1.0 - 0.6 * hunger))

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
        self.energy = 80.0
        self.stamina = 1.0
        self.place.reset()
        self.ambush_patience = 0
        self.hunt_timer = 0
        self.rest_timer = 0
        self._step = 0
        self._update_hunger_drive()
