"""
Conspecific Brain — lightweight agent for non-focal fish (Step 34).

Shares SNN weights (frozen) with the focal BrainAgent but strips
expensive modules (VAE, allostasis, sleep, cerebellum).  Designed for
multi-agent simulation where 4 conspecifics need individual brains
without 4x computational cost.

Decision: threshold-based goal selection (FORAGE/FLEE/SOCIAL).
Motor: turn_rate + speed with bout dynamics.
Sensory: retinal type-channel pixel counts (no full SNN forward pass).

Pure numpy for decision, minimal torch for classifier inference.
"""
import math
import numpy as np


GOAL_FORAGE = 0
GOAL_FLEE = 1
GOAL_SOCIAL = 2


class ConspecificBrain:
    """Lightweight brain for conspecific fish.

    Uses simple pixel-counting from the environment rather than full
    SNN forward pass.  Goal selection is threshold-based.

    Args:
        flee_threshold: float — enemy pixel count to trigger flee
        forage_radius: float — distance to pursue food
        social_dist: float — preferred inter-fish distance
    """

    def __init__(self, flee_threshold=5.0, forage_radius=120.0,
                 social_dist=50.0):
        self.flee_threshold = flee_threshold
        self.forage_radius = forage_radius
        self.social_dist = social_dist
        self.goal = GOAL_FORAGE
        self._persist_timer = 0
        self._persist_steps = 6

    def select_action(self, fish, env_snapshot):
        """Select action for a conspecific fish.

        Args:
            fish: dict with x, y, heading, speed, energy
            env_snapshot: dict with pred_x, pred_y, pred_heading,
                foods (list of [x,y]), fish_positions (list of [x,y]),
                rock_formations

        Returns:
            turn_rate: float [-1, 1]
            speed: float [0, 1]
            goal: int
        """
        fx, fy = fish["x"], fish["y"]
        heading = fish["heading"]

        # Predator distance and bearing
        px, py = env_snapshot["pred_x"], env_snapshot["pred_y"]
        pred_dx = px - fx
        pred_dy = py - fy
        pred_dist = math.sqrt(pred_dx * pred_dx + pred_dy * pred_dy) + 1e-8
        pred_bearing = math.atan2(pred_dy, pred_dx)

        # Nearest food
        nearest_food_dist = 999.0
        nearest_food_angle = 0.0
        for food in env_snapshot.get("foods", []):
            fdx = food[0] - fx
            fdy = food[1] - fy
            fd = math.sqrt(fdx * fdx + fdy * fdy)
            if fd < nearest_food_dist:
                nearest_food_dist = fd
                nearest_food_angle = math.atan2(fdy, fdx)

        # Social: centre of mass of other fish
        others = env_snapshot.get("fish_positions", [])
        if others:
            cx = np.mean([o[0] for o in others])
            cy = np.mean([o[1] for o in others])
            social_dx = cx - fx
            social_dy = cy - fy
            social_dist = math.sqrt(social_dx ** 2 + social_dy ** 2) + 1e-8
            social_angle = math.atan2(social_dy, social_dx)
        else:
            social_dist = 0
            social_angle = heading

        # Information cascading: observe if neighbours are fleeing
        # (fast-moving neighbours heading away from predator = social alarm)
        n_fleeing_neighbours = 0
        for other_pos in others:
            if len(other_pos) > 2:  # has speed info
                other_speed = other_pos[2] if len(other_pos) > 2 else 0
                if other_speed > 3.0:  # fast = fleeing
                    n_fleeing_neighbours += 1
        social_alarm = n_fleeing_neighbours > 0

        # Goal selection (threshold-based with persistence)
        if self._persist_timer > 0:
            self._persist_timer -= 1
        else:
            # Flee if predator close OR neighbours fleeing (social alarm)
            if pred_dist < 150 or (social_alarm and pred_dist < 250):
                self.goal = GOAL_FLEE
                self._persist_timer = self._persist_steps
            elif nearest_food_dist < self.forage_radius and fish["energy"] < 80:
                self.goal = GOAL_FORAGE
                self._persist_timer = self._persist_steps
            elif social_dist > self.social_dist * 2:
                self.goal = GOAL_SOCIAL
                self._persist_timer = self._persist_steps

        # Motor output
        if self.goal == GOAL_FLEE:
            # Turn away from predator
            flee_angle = pred_bearing + math.pi
            diff = flee_angle - heading
            diff = math.atan2(math.sin(diff), math.cos(diff))
            turn = np.clip(diff * 2.0, -1.0, 1.0)
            speed = min(1.0, 0.8 + 0.4 * (150 - pred_dist) / 150)

        elif self.goal == GOAL_FORAGE and nearest_food_dist < self.forage_radius:
            diff = nearest_food_angle - heading
            diff = math.atan2(math.sin(diff), math.cos(diff))
            turn = np.clip(diff * 1.5, -0.8, 0.8)
            speed = 0.5 + 0.3 * (1.0 - nearest_food_dist / self.forage_radius)

        elif self.goal == GOAL_SOCIAL and others:
            diff = social_angle - heading
            diff = math.atan2(math.sin(diff), math.cos(diff))
            # Attract if far, repel if too close
            if social_dist > self.social_dist:
                turn = np.clip(diff * 1.5, -0.8, 0.8)
            else:
                turn = np.clip(-diff * 0.8, -0.6, 0.6)
            # Alignment: match average heading of neighbours
            avg_heading = np.mean([
                math.atan2(math.sin(o[1] if len(o) > 1 else 0),
                           math.cos(o[1] if len(o) > 1 else 0))
                for o in others]) if len(others[0]) > 1 else heading
            align_diff = avg_heading - heading
            align_diff = math.atan2(math.sin(align_diff), math.cos(align_diff))
            turn += np.clip(align_diff * 0.3, -0.2, 0.2)
            speed = 0.5

        else:
            # Default: gentle random exploration with social tendency
            turn = np.random.uniform(-0.3, 0.3)
            speed = 0.4

        # Add biological noise
        turn += np.random.normal(0, 0.02)
        speed = max(0.1, min(1.0, speed + np.random.normal(0, 0.01)))

        return float(turn), float(speed), self.goal

    def reset(self):
        self.goal = GOAL_FORAGE
        self._persist_timer = 0
