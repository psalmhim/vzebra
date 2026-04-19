"""
RateCodedBrain: AbstractBrain wrapper around the rate-coded pipeline from server.py.

Satisfies the AbstractBrain protocol for use with VirtualZebrafish(fidelity='rate_coded').
Runs ~100x faster than the spiking brain — suitable for batch experiments and sweeps.
"""
from __future__ import annotations

import math
from typing import Any


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))


def _ema(old: float, new: float, tau: float) -> float:
    a = 1.0 / max(1, tau)
    return old * (1 - a) + new * a


# Goal constants (matches brain_v2.py)
GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, GOAL_SOCIAL = 0, 1, 2, 3
GOAL_NAMES = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']


class RateCodedBrain:
    """Rate-coded brain satisfying the AbstractBrain protocol.

    Implements the 11-stage neurobiological pipeline from server.py
    as a standalone class with state management.
    """

    def __init__(self, brain_config=None, body_config=None):
        from zebrav2.config.brain_config import BrainConfig
        from zebrav2.config.body_config import BodyConfig
        self.cfg = brain_config or BrainConfig()
        self.body_cfg = body_config or BodyConfig()
        self._ablated: set[str] = self.cfg.get_ablated_set()

        # Observable state
        self.current_goal: int = GOAL_EXPLORE
        self.energy: float = self.body_cfg.metabolism.energy_start

        # Neural state (mirrors server.py _neural dict)
        self._n = self._default_state()

        # Step counter
        self._step = 0

    def _default_state(self) -> dict[str, Any]:
        return {
            'DA': 0.5, 'NA': 0.3, '5HT': 0.6, 'ACh': 0.7,
            'amygdala_trace': 0.0,
            'frustration': [0.0, 0.0, 0.0, 0.0],
            'V_prev': 0.0,
            'cb_pred': [0.0, 0.0],
            'place_fam': 0.0,
            'goal_lock': 0, 'locked_goal': None,
            'cstart_timer': 0, 'cstart_dir': 0.0,
            'sht_acc': 0.0,
            'energy_prev': self.body_cfg.metabolism.energy_start,
            'energy_rate': 0.0,
            'steps_since_food': 0,
            'starvation_anxiety': 0.0,
            'food_memory_xy': None, 'food_memory_age': 999,
            'dead': False,
            'prev_retina_L': 0.0, 'prev_retina_R': 0.0,
            'orient_dir': 0.0, 'orient_habituation': 0.0,
        }

    def reset(self) -> None:
        self._n = self._default_state()
        self.current_goal = GOAL_EXPLORE
        self.energy = self.body_cfg.metabolism.energy_start
        self._step = 0

    def set_region_enabled(self, region_name: str, enabled: bool) -> None:
        if enabled:
            self._ablated.discard(region_name)
        else:
            self._ablated.add(region_name)

    def step(self, obs: dict, env: Any = None) -> dict:
        """Process one timestep through the rate-coded pipeline.

        Args:
            obs: dict with retinal_L, retinal_R, fish_pos, energy, etc.
            env: optional environment reference (for food/predator positions)

        Returns:
            dict with turn, speed, goal, DA, NA, 5HT, ACh, etc.
        """
        n = self._n
        _ec = self.cfg.efe
        _tc = self.cfg.threat

        # --- 1. Sensory input ---
        import numpy as np
        ret_L = obs.get('retinal_L', np.zeros(800))
        ret_R = obs.get('retinal_R', np.zeros(800))
        fish_x, fish_y = obs.get('fish_pos', (400, 300))
        heading = obs.get('fish_heading', 0.0)
        self.energy = obs.get('energy', self.energy)
        arena_w = obs.get('arena_w', 800)
        arena_h = obs.get('arena_h', 600)

        # Retinal activation (sum of brightness)
        ret_L_sum = float(np.sum(ret_L[:400])) if len(ret_L) >= 400 else float(np.sum(ret_L))
        ret_R_sum = float(np.sum(ret_R[:400])) if len(ret_R) >= 400 else float(np.sum(ret_R))

        # --- 2. Optic chiasm (cross) ---
        tect_L_input = ret_R_sum  # right eye → left tectum
        tect_R_input = ret_L_sum  # left eye → right tectum

        # --- 3. Tectum (threat/object detection) ---
        tect_threat = max(tect_L_input, tect_R_input) * _tc.tect_threat_scale / 1000.0
        tect_threat = min(1.0, tect_threat)

        # --- 4. Thalamus gating ---
        na_gate = min(1.0, n['NA'] + 0.5)  # arousal opens gate
        thal_output = tect_threat * na_gate

        # --- 5. Amygdala ---
        if 'amygdala' not in self._ablated:
            n['amygdala_trace'] = _ema(n['amygdala_trace'], thal_output, 10)
        amygdala_out = n['amygdala_trace']

        # --- 6. Neuromodulators ---
        # Reward signal
        eaten_now = obs.get('eaten_now', 0)
        reward = eaten_now * 0.3
        n['DA'] = _ema(n['DA'], 0.5 + reward - 0.3 * amygdala_out, 20)
        n['NA'] = _ema(n['NA'], 0.3 + 0.5 * amygdala_out, 30)
        n['5HT'] = _ema(n['5HT'], 0.6 - 0.2 * amygdala_out, 50)
        n['ACh'] = _ema(n['ACh'], 0.7, 40)

        # Clamp
        for nm in ['DA', 'NA', '5HT', 'ACh']:
            n[nm] = max(0.0, min(1.0, n[nm]))

        # --- 7. Energy tracking ---
        energy_delta = self.energy - n['energy_prev']
        n['energy_rate'] = _ema(n['energy_rate'], energy_delta, 20)
        n['energy_prev'] = self.energy
        if eaten_now > 0:
            n['steps_since_food'] = 0
        else:
            n['steps_since_food'] += 1

        # Starvation anxiety
        if self.energy < _ec.starvation_threshold:
            n['starvation_anxiety'] = min(1.0,
                n['starvation_anxiety'] + 0.02)
        else:
            n['starvation_anxiety'] *= 0.95

        # --- 8. EFE goal selection ---
        G_forage = _ec.forage_offset
        G_flee = 0.0
        G_explore = _ec.explore_offset
        G_social = _ec.social_offset

        # Threat → flee
        p_enemy = amygdala_out
        if p_enemy > 0.15:
            G_flee += p_enemy * _ec.flee_enemy_weight

        # Hunger → forage
        if self.energy < _ec.starvation_threshold:
            G_forage += _ec.starvation_urgency_weight * (
                1.0 - self.energy / _ec.starvation_threshold)

        # Exploration drive
        if n['steps_since_food'] > 50:
            G_explore += 0.1

        # Goal lock (persistence)
        if n['goal_lock'] > 0:
            n['goal_lock'] -= 1
            if n['locked_goal'] is not None:
                self.current_goal = n['locked_goal']
        else:
            goals = [G_forage, G_flee, G_explore, G_social]
            self.current_goal = int(max(range(4), key=lambda i: goals[i]))
            n['goal_lock'] = 10
            n['locked_goal'] = self.current_goal

        # --- 9. Motor output ---
        # Turn: retinal balance
        turn_signal = (tect_R_input - tect_L_input) / max(1.0, tect_L_input + tect_R_input + 1.0)
        turn = max(-1.0, min(1.0, turn_signal))

        # Speed: goal-dependent
        _mc = self.body_cfg.motor
        if self.current_goal == GOAL_FLEE:
            speed = _mc.speed_flee
        elif self.current_goal == GOAL_FORAGE:
            speed = _mc.speed_forage
        elif self.current_goal == GOAL_SOCIAL:
            speed = _mc.speed_social
        else:
            speed = _mc.speed_explore

        # Energy modulation (inverted-U)
        energy_frac = self.energy / self.body_cfg.metabolism.energy_start
        motility = 1.0 - 4.0 * (energy_frac - 0.5) ** 2
        speed *= max(0.3, motility)

        # Dead fish check
        if self.energy <= 0:
            speed = 0.0
            turn = 0.0

        self._step += 1

        return {
            'turn': turn,
            'speed': speed,
            'goal': self.current_goal,
            'DA': n['DA'],
            'NA': n['NA'],
            '5HT': n['5HT'],
            'ACh': n['ACh'],
            'amygdala': amygdala_out,
            'free_energy': G_forage + G_flee + G_explore + G_social,
            'critic_value': n['V_prev'],
            'predictive_surprise': 0.0,
            'novelty': 0.0,
        }
