"""
Multi-Agent Zebrafish Environment (Step 34).

Extends the base prey-predator environment with multiple fish, each
controlled by its own brain.  Fish[0] is the focal fish (full BrainAgent),
fish[1:] are conspecifics (lightweight ConspecificBrain).

Emergent behaviors: schooling, flash expansion, collective vigilance,
predator confusion effect, competitive foraging.

Neuroscience: zebrafish shoaling emerges from local sensory rules
(Abaid et al. 2013).  Group-level patterns arise without global
planning (Miller & Gerlai 2012).
"""
import math
import numpy as np
from zebrav1.gym_env.zebrafish_env import ZebrafishPreyPredatorEnv
from zebrav1.brain.conspecific_brain import ConspecificBrain


class MultiAgentZebrafishEnv(ZebrafishPreyPredatorEnv):
    """Multi-fish environment with individual brains per fish.

    Args:
        n_fish: int — total fish count (1 focal + n-1 conspecifics)
        **kwargs: passed to base ZebrafishPreyPredatorEnv
    """

    def __init__(self, n_fish=5, **kwargs):
        self.n_fish = n_fish
        self.all_fish = []
        self.conspecific_brains = [
            ConspecificBrain() for _ in range(n_fish - 1)]
        self._pred_target_idx = 0
        self._pred_target_lock = 0
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}

        # Initialise all fish positions
        self.all_fish = []
        # Fish 0 = focal (already positioned by super)
        self.all_fish.append({
            "x": self.fish_x, "y": self.fish_y,
            "heading": self.fish_heading,
            "speed": 0.0, "energy": self.fish_energy,
        })
        # Fish 1..n-1 = conspecifics (random positions near focal)
        for i in range(1, self.n_fish):
            angle = np.random.uniform(0, 2 * math.pi)
            dist = np.random.uniform(40, 100)
            fx = self.fish_x + dist * math.cos(angle)
            fy = self.fish_y + dist * math.sin(angle)
            fx = float(np.clip(fx, 30, self.arena_w - 30))
            fy = float(np.clip(fy, 30, self.arena_h - 30))
            self.all_fish.append({
                "x": fx, "y": fy,
                "heading": np.random.uniform(-math.pi, math.pi),
                "speed": 0.0, "energy": 80.0,
            })

        for brain in self.conspecific_brains:
            brain.reset()

        # Sync colleagues for focal fish's retinal sampling
        self._sync_colleagues()
        self._pred_target_idx = 0
        self._pred_target_lock = 0

        return obs, info

    def step(self, focal_action):
        """Step all fish simultaneously.

        Args:
            focal_action: np.array[2] — [turn, speed] for focal fish

        Returns:
            Same interface as base env (obs, reward, term, trunc, info)
        """
        # 1. Step conspecifics first (they see previous state)
        snap = self._build_env_snapshot()
        for i, brain in enumerate(self.conspecific_brains):
            fish = self.all_fish[i + 1]
            # Build per-fish snapshot (other fish positions)
            other_pos = [[f["x"], f["y"]]
                         for j, f in enumerate(self.all_fish) if j != i + 1]
            snap_i = dict(snap)
            snap_i["fish_positions"] = other_pos
            turn, speed, goal = brain.select_action(fish, snap_i)

            # Apply movement
            fish["heading"] += turn * 0.3  # scale to match env
            fish["heading"] = math.atan2(
                math.sin(fish["heading"]), math.cos(fish["heading"]))
            move = speed * 3.0  # scale to match env step size
            fish["x"] += move * math.cos(fish["heading"])
            fish["y"] += move * math.sin(fish["heading"])
            fish["x"] = float(np.clip(fish["x"], 10, self.arena_w - 10))
            fish["y"] = float(np.clip(fish["y"], 10, self.arena_h - 10))
            fish["speed"] = speed

            # Conspecific food eating
            eaten_idx = None
            for fi, food in enumerate(self.foods):
                dx = fish["x"] - food[0]
                dy = fish["y"] - food[1]
                if dx * dx + dy * dy < self.eat_radius ** 2:
                    eaten_idx = fi
                    break
            if eaten_idx is not None:
                fish["energy"] = min(100, fish["energy"] + 2.0)
                self.foods.pop(eaten_idx)

            # Energy drain
            fish["energy"] -= 0.08 * max(0.3, speed)
            fish["energy"] = max(0, fish["energy"])

        # 2. Update predator targeting (nearest fish)
        self._update_predator_target()

        # 3. Sync colleagues for focal fish
        self._sync_colleagues()

        # 4. Step focal fish via base env
        obs, reward, terminated, truncated, info = super().step(focal_action)

        # 5. Sync focal fish state back
        self.all_fish[0]["x"] = self.fish_x
        self.all_fish[0]["y"] = self.fish_y
        self.all_fish[0]["heading"] = self.fish_heading
        self.all_fish[0]["speed"] = getattr(self, 'fish_speed', 0.5)
        self.all_fish[0]["energy"] = self.fish_energy

        # 6. Check if conspecifics are caught
        for fish in self.all_fish[1:]:
            dx = fish["x"] - self.pred_x
            dy = fish["y"] - self.pred_y
            if dx * dx + dy * dy < self.pred_catch_radius ** 2:
                # Conspecific caught — respawn nearby
                angle = np.random.uniform(0, 2 * math.pi)
                fish["x"] = self.arena_w / 2 + 100 * math.cos(angle)
                fish["y"] = self.arena_h / 2 + 100 * math.sin(angle)
                fish["energy"] = 60.0

        info["all_fish"] = [dict(f) for f in self.all_fish]
        return obs, reward, terminated, truncated, info

    def _sync_colleagues(self):
        """Update env.colleagues with actual conspecific positions."""
        self.colleagues = []
        for i, fish in enumerate(self.all_fish[1:]):
            self.colleagues.append({
                "x": fish["x"], "y": fish["y"],
                "heading": fish["heading"],
                "speed": fish["speed"],
            })

    def _build_env_snapshot(self):
        """Build shared env snapshot for conspecific brains."""
        return {
            "pred_x": self.pred_x,
            "pred_y": self.pred_y,
            "pred_heading": self.pred_heading,
            "foods": [[f[0], f[1]] for f in self.foods],
            "fish_positions": [[f["x"], f["y"]] for f in self.all_fish],
            "rock_formations": getattr(self, 'rock_formations', []),
        }

    def _update_predator_target(self):
        """Predator targets nearest fish with hysteresis."""
        if self._pred_target_lock > 0:
            self._pred_target_lock -= 1
            # Update target position to current fish
            target = self.all_fish[self._pred_target_idx]
            self._pred_target_pos = (target["x"], target["y"])
            return

        # Find nearest fish
        best_dist = 999999
        best_idx = 0
        for i, fish in enumerate(self.all_fish):
            if fish["energy"] <= 0:
                continue
            dx = fish["x"] - self.pred_x
            dy = fish["y"] - self.pred_y
            d2 = dx * dx + dy * dy
            if d2 < best_dist:
                best_dist = d2
                best_idx = i

        # Hysteresis: only switch if new target is much closer
        if best_idx != self._pred_target_idx:
            curr = self.all_fish[self._pred_target_idx]
            cdx = curr["x"] - self.pred_x
            cdy = curr["y"] - self.pred_y
            curr_d2 = cdx * cdx + cdy * cdy
            if best_dist < curr_d2 * 0.6:  # 40% closer to switch
                self._pred_target_idx = best_idx
                self._pred_target_lock = 10  # lock for 10 steps

        target = self.all_fish[self._pred_target_idx]
        # Override the base env's fish position for predator AI
        # (predator chases the targeted fish, not necessarily fish[0])
        self._pred_target_pos = (target["x"], target["y"])

    # ------------------------------------------------------------------
    # Schooling metrics
    # ------------------------------------------------------------------

    def compute_polarization(self):
        """School polarization P ∈ [0, 1]."""
        if len(self.all_fish) < 2:
            return 0.0
        vecs = np.array([[math.cos(f["heading"]), math.sin(f["heading"])]
                         for f in self.all_fish])
        return float(np.linalg.norm(vecs.mean(axis=0)))

    def compute_nnd(self):
        """Mean nearest-neighbor distance."""
        n = len(self.all_fish)
        if n < 2:
            return 0.0
        pos = np.array([[f["x"], f["y"]] for f in self.all_fish])
        nnds = []
        for i in range(n):
            dists = np.sqrt(np.sum((pos - pos[i]) ** 2, axis=1))
            dists[i] = 9999
            nnds.append(float(dists.min()))
        return float(np.mean(nnds))
