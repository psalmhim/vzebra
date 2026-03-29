"""
Spiking TD critic: estimates state value for each goal.

Architecture:
  Input: 16 state features (energy, threat, food, classifier probs, goal one-hot, neuromod)
  Hidden: 64 Izhikevich RS neurons
  Output: 4 value neurons (one per goal), rate-coded

Learning: TD(0) with eligibility traces and DA modulation.
  delta = reward + gamma * V(s') - V(s)
  Eligibility traces accumulate Hebbian co-activation.
  DA (= RPE) consolidates traces into weight updates.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingCritic(nn.Module):
    def __init__(self, n_features=16, n_hidden=64, n_goals=4,
                 gamma=0.95, lr=1e-3, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_goals = n_goals
        self.gamma = gamma
        self.lr = lr

        # Spiking hidden layer
        self.hidden = IzhikevichLayer(n_hidden, 'RS', device)
        self.hidden.i_tonic.fill_(0.0)

        # Value readout (spiking)
        self.value_neurons = IzhikevichLayer(n_goals, 'RS', device)
        self.value_neurons.i_tonic.fill_(0.0)

        # Weights
        self.W_in = nn.Linear(n_features, n_hidden, bias=True)
        self.W_out = nn.Linear(n_hidden, n_goals, bias=True)
        nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.3)
        self.W_in.to(device)
        self.W_out.to(device)

        # Eligibility traces
        self.register_buffer('elig_in', torch.zeros(n_hidden, n_features, device=device))
        self.register_buffer('elig_out', torch.zeros(n_goals, n_hidden, device=device))

        # State
        self.register_buffer('values', torch.zeros(n_goals, device=device))
        self.register_buffer('prev_values', torch.zeros(n_goals, device=device))
        self.register_buffer('hidden_rate', torch.zeros(n_hidden, device=device))
        self.register_buffer('prev_features', torch.zeros(n_features, device=device))

        self.prev_goal = 0
        self.td_error = 0.0

    def _build_features(self, energy: float, threat: float,
                        food_visible: float, goal: int,
                        cls_probs: torch.Tensor = None,
                        DA: float = 0.5, NA: float = 0.3) -> torch.Tensor:
        """Build 16-dim feature vector."""
        f = torch.zeros(self.n_features, device=self.device)
        f[0] = energy / 100.0
        f[1] = threat
        f[2] = food_visible
        f[3] = 1.0 - energy / 100.0  # hunger
        # Goal one-hot (4 dims)
        f[4 + goal] = 1.0
        # Classifier probs (5 dims)
        if cls_probs is not None:
            n = min(5, cls_probs.shape[0])
            f[8:8+n] = cls_probs[:n].detach()
        # Neuromod (2 dims)
        f[13] = DA
        f[14] = NA
        f[15] = max(0.0, 0.5 - energy / 100.0)  # starvation urgency
        return f

    @torch.no_grad()
    def forward(self, energy: float, threat: float, food_visible: float,
                goal: int, DA: float = 0.5, NA: float = 0.3,
                cls_probs: torch.Tensor = None,
                reward: float = 0.0) -> dict:
        """
        Estimate values and update via TD learning.
        Returns: values per goal, TD error, current value
        """
        features = self._build_features(energy, threat, food_visible, goal,
                                         cls_probs, DA, NA)

        # Forward pass through spiking hidden
        I_in = self.W_in(features.unsqueeze(0)).squeeze(0).detach()
        I_in = I_in * (5.0 / (I_in.abs().mean() + 1e-8))

        h_spikes = torch.zeros(self.n_hidden, device=self.device)
        v_spikes = torch.zeros(self.n_goals, device=self.device)

        for _ in range(SUBSTEPS):
            sp_h = self.hidden(I_in + torch.randn(self.n_hidden, device=self.device) * 0.3)
            h_spikes += sp_h

            I_out = self.W_out(self.hidden.rate.unsqueeze(0)).squeeze(0).detach()
            I_out = I_out * (4.0 / (I_out.abs().mean() + 1e-8))
            sp_v = self.value_neurons(I_out)
            v_spikes += sp_v

        self.hidden_rate.copy_(self.hidden.rate)

        # Value readout: rate * weight → scalar per goal
        raw_values = self.W_out(self.hidden.rate.unsqueeze(0)).squeeze(0).detach()
        self.values.copy_(raw_values)

        # TD learning
        V_now = self.values[goal].item()
        V_prev = self.prev_values[self.prev_goal].item()
        self.td_error = reward + self.gamma * V_now - V_prev

        # Update eligibility traces
        self.elig_in = 0.9 * self.elig_in + torch.outer(self.hidden.rate, self.prev_features)
        self.elig_out = 0.9 * self.elig_out + torch.outer(self.value_neurons.rate, self.hidden.rate)

        # Consolidate with TD error (DA-modulated)
        if abs(self.td_error) > 0.001:
            self.W_in.weight.data.add_(self.lr * self.td_error * DA * self.elig_in)
            self.W_in.weight.data.clamp_(-3.0, 3.0)
            self.W_out.weight.data.add_(self.lr * self.td_error * DA * self.elig_out)
            self.W_out.weight.data.clamp_(-3.0, 3.0)

        # Store for next step
        self.prev_values.copy_(self.values)
        self.prev_features.copy_(features)
        self.prev_goal = goal

        return {
            'values': self.values.clone(),
            'td_error': self.td_error,
            'current_value': V_now,
            'hidden_active': float((h_spikes > 0).float().mean()),
        }

    def get_value(self, goal: int = None) -> float:
        """Get current value estimate."""
        if goal is not None:
            return self.values[goal].item()
        return self.values.max().item()

    def get_goal_values(self) -> torch.Tensor:
        """Return values for all 4 goals."""
        return self.values.clone()

    def reset(self):
        self.hidden.reset()
        self.value_neurons.reset()
        self.values.zero_()
        self.prev_values.zero_()
        self.hidden_rate.zero_()
        self.prev_features.zero_()
        self.elig_in.zero_()
        self.elig_out.zero_()
        self.prev_goal = 0
        self.td_error = 0.0
