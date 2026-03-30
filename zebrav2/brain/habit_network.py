"""
Spiking habit network: fast cached sensorimotor associations.

Biological basis: dorsolateral striatum habit circuit.
Learns frequent stimulus→action mappings via Hebbian LTP.
When confidence is high, bypasses slow EFE/deliberation.

Architecture:
  Input: classifier output (5) + goal (4) + retinal summary (4) = 13 dims
  Hidden: 32 Izhikevich RS neurons
  Output: 8 population-coded motor neurons (4 turn bins + 4 speed bins)

Habit strength grows with repetition, decays with prediction error.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingHabitNet(nn.Module):
    def __init__(self, n_input=13, n_hidden=32, n_output=8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Spiking layers
        self.hidden = IzhikevichLayer(n_hidden, 'RS', device)
        self.output = IzhikevichLayer(n_output, 'RS', device)
        self.hidden.i_tonic.fill_(-1.0)
        self.output.i_tonic.fill_(-2.0)

        # Weights (Hebbian-learnable)
        self.W_in = nn.Linear(n_input, n_hidden, bias=False)
        self.W_out = nn.Linear(n_hidden, n_output, bias=False)
        nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.3)
        self.W_in.to(device)
        self.W_out.to(device)

        # State
        self.register_buffer('output_rate', torch.zeros(n_output, device=device))
        self.register_buffer('hidden_rate', torch.zeros(n_hidden, device=device))

        # Habit strength per input-output association
        self.register_buffer('habit_strength', torch.zeros(n_output, device=device))
        self.confidence = 0.0
        self.n_repetitions = 0

    def _build_input(self, cls_probs: torch.Tensor, goal: int,
                     retinal_summary: torch.Tensor = None) -> torch.Tensor:
        """Build 13-dim input vector."""
        x = torch.zeros(self.n_input, device=self.device)
        # Classifier (5 dims)
        if cls_probs is not None:
            n = min(5, cls_probs.shape[0])
            x[:n] = cls_probs[:n].detach()
        # Goal one-hot (4 dims)
        x[5 + goal] = 1.0
        # Retinal summary (4 dims): L/R intensity, L/R food
        if retinal_summary is not None:
            n = min(4, retinal_summary.shape[0])
            x[9:9+n] = retinal_summary[:n].detach()
        return x

    @torch.no_grad()
    def forward(self, cls_probs: torch.Tensor, goal: int,
                turn: float, speed: float,
                retinal_summary: torch.Tensor = None) -> dict:
        """
        Run habit network. If confident, returns cached action.
        Also learns from current turn/speed (target).
        """
        x = self._build_input(cls_probs, goal, retinal_summary)

        # Forward pass
        I_in = self.W_in(x.unsqueeze(0)).squeeze(0).detach()
        I_in = I_in * (5.0 / (I_in.abs().mean() + 1e-8))

        h_spikes = torch.zeros(self.n_hidden, device=self.device)
        o_spikes = torch.zeros(self.n_output, device=self.device)

        for _ in range(15):  # reduced substeps
            sp_h = self.hidden(I_in + torch.randn(self.n_hidden, device=self.device) * 0.3)
            h_spikes += sp_h
            I_out = self.W_out(self.hidden.rate.unsqueeze(0)).squeeze(0).detach()
            I_out = I_out * (4.0 / (I_out.abs().mean() + 1e-8))
            sp_o = self.output(I_out)
            o_spikes += sp_o

        self.hidden_rate.copy_(self.hidden.rate)
        self.output_rate.copy_(self.output.rate)

        # Decode population-coded motor output
        # Turn: 4 bins [-1, -0.33, 0.33, 1.0]
        turn_bins = torch.tensor([-1.0, -0.33, 0.33, 1.0], device=self.device)
        turn_rates = self.output_rate[:4]
        if turn_rates.sum() > 0.01:
            habit_turn = float((turn_rates * turn_bins).sum() / (turn_rates.sum() + 1e-8))
        else:
            habit_turn = 0.0

        # Speed: 4 bins [0.5, 0.8, 1.0, 1.5]
        speed_bins = torch.tensor([0.5, 0.8, 1.0, 1.5], device=self.device)
        speed_rates = self.output_rate[4:]
        if speed_rates.sum() > 0.01:
            habit_speed = float((speed_rates * speed_bins).sum() / (speed_rates.sum() + 1e-8))
        else:
            habit_speed = 1.0

        # Confidence from output spike consistency
        total_spikes = o_spikes.sum().item()
        max_spikes = o_spikes.max().item()
        self.confidence = max_spikes / (total_spikes + 1e-8) if total_spikes > 2 else 0.0

        # Hebbian learning: strengthen input→output associations that match current action
        # Target: which bins match current turn/speed
        turn_target = torch.zeros(4, device=self.device)
        turn_idx = int(torch.argmin(torch.abs(turn_bins - turn)))
        turn_target[turn_idx] = 1.0
        speed_target = torch.zeros(4, device=self.device)
        speed_idx = int(torch.argmin(torch.abs(speed_bins - speed)))
        speed_target[speed_idx] = 1.0
        target = torch.cat([turn_target, speed_target])

        # Hebbian update: pre (hidden) × post (target) strengthens weights
        dW = 0.001 * torch.outer(target, self.hidden_rate)
        self.W_out.weight.data.add_(dW)
        self.W_out.weight.data.clamp_(-1.0, 1.0)

        # Update habit strength
        self.habit_strength = 0.99 * self.habit_strength + 0.01 * target
        self.n_repetitions += 1

        return {
            'turn': habit_turn,
            'speed': habit_speed,
            'confidence': self.confidence,
            'habit_strength': float(self.habit_strength.max()),
            'n_repetitions': self.n_repetitions,
        }

    def reset(self):
        self.hidden.reset()
        self.output.reset()
        self.output_rate.zero_()
        self.hidden_rate.zero_()
        self.confidence = 0.0
