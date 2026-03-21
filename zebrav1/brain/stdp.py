"""
Reward-Modulated STDP — spike-timing dependent plasticity with dopamine gating.

Implements three-factor learning: pre × post × reward
  - Pre-before-post (causal): LTP if rewarded, LTD if punished
  - Post-before-pre (anticausal): opposite
  - Dopamine gates all plasticity: no DA → no learning

Also provides population vector decoding for motor readout:
  - 200 motor neurons encode direction via preferred angles
  - Population vector sum → turn direction + speed

Neuroscience: STDP in zebrafish tectum (Mu & Bhatt 2020).
Reward-modulated STDP as model for DA-gated learning (Izhikevich 2007).

Pure numpy for STDP, torch for integration with SNN.
"""
import math
import numpy as np
import torch


class RewardModulatedSTDP:
    """Three-factor STDP: pre × post × dopamine.

    Args:
        n_pre: int — presynaptic neuron count
        n_post: int — postsynaptic neuron count
        tau_plus: float — LTP time constant (ms equivalent in steps)
        tau_minus: float — LTD time constant
        A_plus: float — LTP amplitude
        A_minus: float — LTD amplitude
        w_max: float — maximum weight
        eligibility_decay: float — trace decay per step
    """

    def __init__(self, n_pre, n_post, tau_plus=10.0, tau_minus=10.0,
                 A_plus=0.005, A_minus=0.005, w_max=1.0,
                 eligibility_decay=0.95):
        self.n_pre = n_pre
        self.n_post = n_post
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_max = w_max
        self.elig_decay = eligibility_decay

        # Eligibility traces (accumulated STDP, waiting for reward)
        self.eligibility = np.zeros((n_pre, n_post), dtype=np.float32)

        # Spike traces (exponential decay)
        self.pre_trace = np.zeros(n_pre, dtype=np.float32)
        self.post_trace = np.zeros(n_post, dtype=np.float32)

        # Time constants as decay factors
        self.pre_decay = math.exp(-1.0 / tau_plus)
        self.post_decay = math.exp(-1.0 / tau_minus)

    def update_traces(self, pre_spikes, post_spikes):
        """Update spike traces and eligibility from spike events.

        Args:
            pre_spikes: np.array [n_pre] — binary spike vector
            post_spikes: np.array [n_post] — binary spike vector
        """
        # Decay traces
        self.pre_trace *= self.pre_decay
        self.post_trace *= self.post_decay

        # Update with new spikes
        self.pre_trace += pre_spikes
        self.post_trace += post_spikes

        # STDP eligibility: pre-before-post → LTP, post-before-pre → LTD
        # When post fires, LTP proportional to pre_trace (causal)
        if post_spikes.sum() > 0:
            self.eligibility += self.A_plus * np.outer(
                self.pre_trace, post_spikes)
        # When pre fires, LTD proportional to post_trace (anticausal)
        if pre_spikes.sum() > 0:
            self.eligibility -= self.A_minus * np.outer(
                pre_spikes, self.post_trace)

        # Decay eligibility
        self.eligibility *= self.elig_decay

    def apply_reward(self, weights, dopamine, reward_signal):
        """Apply reward-modulated weight update.

        Args:
            weights: torch.Parameter [n_pre, n_post] or [n_post, n_pre]
            dopamine: float [0, 1] — DA level gates learning
            reward_signal: float — RPE (positive=reward, negative=punishment)

        Returns:
            dW_norm: float — magnitude of weight change
        """
        if abs(reward_signal) < 0.01:
            return 0.0

        # Three-factor: eligibility × dopamine × reward
        dW = self.eligibility * dopamine * reward_signal
        dW = np.clip(dW, -0.01, 0.01)

        # Apply to weights
        with torch.no_grad():
            w = weights.data
            if w.shape == (self.n_post, self.n_pre):
                w += torch.tensor(dW.T, dtype=w.dtype, device=w.device)
            else:
                w += torch.tensor(dW, dtype=w.dtype, device=w.device)
            weights.data = torch.clamp(w, -self.w_max, self.w_max)

        return float(np.abs(dW).mean())

    def reset(self):
        self.eligibility[:] = 0
        self.pre_trace[:] = 0
        self.post_trace[:] = 0


class PopulationDecoder:
    """Decode motor command from population of neurons via vector sum.

    200 motor neurons arranged with preferred directions:
      neurons 0-99: left motor pool (preferred turn = -π/2)
      neurons 100-199: right motor pool (preferred turn = +π/2)

    Population vector: sum of (firing_rate × preferred_direction)

    Returns turn direction and speed from population activity.
    """

    def __init__(self, n_neurons=200):
        self.n = n_neurons
        # Preferred directions: left pool = negative, right pool = positive
        self.pref_dirs = np.zeros(n_neurons, dtype=np.float32)
        self.pref_dirs[:n_neurons // 2] = -1.0  # left
        self.pref_dirs[n_neurons // 2:] = +1.0   # right

    def decode(self, motor_activity):
        """Decode turn and speed from motor neuron population.

        Args:
            motor_activity: np.array [200] — firing rates

        Returns:
            turn: float [-1, 1]
            speed: float [0, 1]
        """
        activity = np.maximum(0, motor_activity)  # rectify

        # Population vector: weighted sum of preferred directions
        total = activity.sum() + 1e-8
        pop_vector = np.dot(activity, self.pref_dirs) / total

        # Turn from population vector
        turn = float(np.clip(pop_vector * 2.0, -1.0, 1.0))

        # Speed from total activity (more firing = faster)
        speed = float(np.clip(total / (self.n * 0.1), 0.0, 1.0))

        return turn, speed

    def decode_tensor(self, motor_tensor):
        """Decode from torch tensor [1, 200]."""
        activity = motor_tensor.detach().cpu().numpy().flatten()
        return self.decode(activity)
