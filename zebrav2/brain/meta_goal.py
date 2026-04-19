"""
MetaGoalWeights: learns goal selection biases and EFE modulation weights via REINFORCE.

Two learnable components updated online at end of each episode:
  goal_bias[4]  — additive offset per goal [FORAGE, FLEE, EXPLORE, SOCIAL]
                  Updated via REINFORCE (policy gradient w/ autograd).
  mod_w[8]      — multiplicative scale for 8 EFE modulation sources
                  [wm, social, geo, novelty, vae, circadian, cerebellum, urgency]
                  Updated via fitness-correlation heuristic.

REINFORCE design notes:
  - Uses mean log-prob (not sum) to be invariant to episode length.
  - Advantage is normalized by running std to prevent large gradient steps.
  - Accumulated log_probs are always cleared in a try/finally block.
  - fitness_ema starts as None; first episode bootstraps baseline with zero update.
"""
import math
import numpy as np
import torch
import torch.nn as nn

GOAL_NAMES = ['FORAGE', 'FLEE', 'EXPLORE', 'SOCIAL']
MOD_NAMES = ['wm', 'social', 'geo', 'novelty', 'vae', 'circadian', 'cerebellum', 'urgency']


class MetaGoalWeights(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # Additive bias per goal — init 0, clamped to [-0.5, 0.5]
        self.goal_bias = nn.Parameter(torch.zeros(4, device=device))
        # Multiplicative scale for 8 modulation sources — init 1.0, clamped to [0.1, 3.0]
        self.mod_w = nn.Parameter(torch.ones(8, device=device))
        self._optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        # Running fitness baseline: None → bootstrap on first episode (no update that episode)
        self._fitness_ema: float | None = None
        # Running advantage variance for normalization
        self._adv_var: float = 1.0
        # Per-episode accumulated log probs for REINFORCE
        self._log_probs: list = []
        # Per-episode accumulated modulation magnitudes [8]
        self._mod_accum = np.zeros(8, dtype=np.float32)
        self._mod_steps: int = 0

    def scales(self) -> torch.Tensor:
        """Return clamped mod weights [0.1, 3.0] — use .detach() when applying as floats."""
        return torch.clamp(self.mod_w, 0.1, 3.0)

    def record_step(self, chosen_goal: int, goal_probs: torch.Tensor,
                    mod_contribs: np.ndarray):
        """
        Accumulate per-step data for end-of-episode update.

        chosen_goal:  index 0–3 (FORAGE/FLEE/EXPLORE/SOCIAL)
        goal_probs:   softmax probabilities WITH autograd graph through goal_bias
        mod_contribs: [8] array of unsigned contribution magnitudes (for mod_w correlation)
        """
        if self.goal_bias.requires_grad:
            log_p = torch.log(goal_probs[chosen_goal].clamp(min=1e-8))
            self._log_probs.append(log_p)
        self._mod_accum += np.abs(mod_contribs)
        self._mod_steps += 1

    def episode_update(self, fitness: float):
        """
        Update learnable parameters at end of episode.
        Always clears accumulated state even if an exception occurs.
        """
        try:
            self._do_update(fitness)
        finally:
            # Always clear — prevents stale graph from poisoning next episode
            self._log_probs = []
            self._mod_accum[:] = 0.0
            self._mod_steps = 0

    def _do_update(self, fitness: float):
        # Bootstrap baseline on first episode: record fitness but skip update
        if self._fitness_ema is None:
            self._fitness_ema = float(fitness)
            return

        advantage = fitness - self._fitness_ema
        self._fitness_ema = 0.95 * self._fitness_ema + 0.05 * fitness

        # Normalize advantage by running std (length- and scale-invariant signal)
        self._adv_var = 0.95 * self._adv_var + 0.05 * (advantage ** 2)
        norm_adv = float(np.clip(
            advantage / (math.sqrt(self._adv_var) + 1e-6), -3.0, 3.0))

        # --- 1. REINFORCE: update goal_bias via mean log-prob ---
        if self._log_probs:
            # Mean (not sum) makes learning rate independent of episode length
            log_prob_mean = torch.stack(self._log_probs).mean()
            policy_loss = -norm_adv * log_prob_mean
            self._optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self._optimizer.step()
            with torch.no_grad():
                self.goal_bias.clamp_(-0.5, 0.5)
                self.mod_w.clamp_(0.1, 3.0)

        # --- 2. Fitness-correlation update for mod_w (no autograd needed) ---
        if self._mod_steps > 0:
            mean_contrib = self._mod_accum / max(1, self._mod_steps)
            lr_mod = 2e-4
            with torch.no_grad():
                for i in range(8):
                    if mean_contrib[i] > 0.005:  # source was meaningfully active
                        self.mod_w[i] += lr_mod * norm_adv
                self.mod_w.clamp_(0.1, 3.0)

    def clear_episode(self):
        """Defensively clear episode state (also called from brain.reset())."""
        self._log_probs = []
        self._mod_accum[:] = 0.0
        self._mod_steps = 0

    def state_dict_extra(self) -> dict:
        """Serializable state for checkpoint (complements nn.Module state_dict)."""
        return {
            'goal_bias':   self.goal_bias.data.cpu().numpy().tolist(),
            'mod_w':       self.mod_w.data.cpu().numpy().tolist(),
            'fitness_ema': self._fitness_ema,  # None is preserved
            'adv_var':     float(self._adv_var),
        }

    def load_state_dict_extra(self, d: dict):
        """Restore from checkpoint extra state."""
        with torch.no_grad():
            self.goal_bias.copy_(
                torch.tensor(d['goal_bias'], device=self.device, dtype=torch.float32))
            self.mod_w.copy_(
                torch.tensor(d['mod_w'], device=self.device, dtype=torch.float32))
        ema = d.get('fitness_ema')
        self._fitness_ema = float(ema) if ema is not None else None
        self._adv_var = float(d.get('adv_var', 1.0))
