"""
Spiking goal selector — WTA attractor network for action selection.

4 goal neurons (FORAGE, FLEE, EXPLORE, SOCIAL) with:
  - Excitatory self-connections (attractor dynamics)
  - Mutual inhibition (winner-take-all)
  - Input from pallium-D population code + neuromodulatory bias

Replaces hardcoded EFE softmax formula.
Active inference interpretation: each attractor basin represents a policy;
the network settles into the policy that minimizes expected free energy
given current beliefs (pallium-D) and preferences (neuromod bias).
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, N_PAL_D
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingGoalSelector(nn.Module):
    def __init__(self, n_goals=4, device=DEVICE):
        super().__init__()
        self.n_goals = n_goals
        self.device = device

        # 4 goal neurons (RS type — regular spiking attractor)
        self.neurons = IzhikevichLayer(n_goals, 'RS', device)
        self.neurons.i_tonic.fill_(-3.0)  # subthreshold without input

        # Pallium-D → goal neurons projection
        pal_d_n_e = int(0.75 * N_PAL_D)
        self.W_pald_goal = nn.Linear(pal_d_n_e, n_goals, bias=False)
        nn.init.xavier_uniform_(self.W_pald_goal.weight, gain=0.5)
        self.W_pald_goal.to(device)

        # Self-excitation (attractor) and mutual inhibition (WTA)
        # W_recurrent[i,j]: j→i; diagonal=excitatory, off-diagonal=inhibitory
        w_self = 4.0   # self-excitation
        w_inhib = -2.0  # mutual inhibition
        W_rec = torch.full((n_goals, n_goals), w_inhib, device=device)
        W_rec.fill_diagonal_(w_self)
        self.register_buffer('W_rec', W_rec)

        # Spike accumulator for rate readout
        self.register_buffer('spike_counts', torch.zeros(n_goals, device=device))
        self.register_buffer('goal_rates', torch.zeros(n_goals, device=device))

        # Neuromod bias input
        self.register_buffer('bias', torch.zeros(n_goals, device=device))

    def forward(self, pal_d_rate: torch.Tensor,
                neuromod_bias: torch.Tensor = None,
                substeps: int = 30) -> dict:
        """
        Run WTA attractor for substeps, return goal rates.

        pal_d_rate: (N_PAL_D_E,) pallium-D excitatory rates
        neuromod_bias: (4,) additive bias [forage, flee, explore, social]
        Returns: {'goal_rates': (4,), 'winner': int, 'confidence': float}
        """
        # Pallium-D drive
        I_pald = self.W_pald_goal(pal_d_rate.unsqueeze(0)).squeeze(0).detach()
        # Normalize to target range
        I_mean = I_pald.abs().mean() + 1e-8
        I_pald = I_pald * (3.0 / I_mean) if I_mean > 0.001 else I_pald * 0.0

        # Add neuromod bias
        if neuromod_bias is not None:
            I_pald = I_pald + neuromod_bias * 2.0

        # Run WTA dynamics
        self.spike_counts.zero_()
        for _ in range(substeps):
            # Recurrent input
            I_rec = self.W_rec @ self.neurons.rate
            I_total = I_pald + I_rec
            spikes = self.neurons(I_total)
            self.spike_counts += spikes

        # Rate readout
        rates = self.spike_counts / substeps
        self.goal_rates.copy_(rates)

        # Winner and confidence
        winner = int(rates.argmax().item())
        total = rates.sum() + 1e-8
        confidence = float(rates[winner] / total)

        return {
            'goal_rates': rates,
            'winner': winner,
            'confidence': confidence,
        }

    def reinforce(self, goal: int, pal_d_rate: torch.Tensor,
                  DA: float = 1.0, eta: float = 2e-4):
        """
        Reward-modulated Hebbian update for W_pald_goal.
        Strengthens the pallium-D → winning-goal connection when DA is high.
        Three-factor rule: ΔW = η × DA × pre × post
        """
        with torch.no_grad():
            post = float(self.goal_rates[goal])
            if post < 0.01:
                return
            delta = eta * DA * post * pal_d_rate  # (N_PAL_D_E,)
            self.W_pald_goal.weight.data[goal].add_(delta)
            self.W_pald_goal.weight.data.clamp_(-2.0, 2.0)

    def reset(self):
        self.neurons.reset()
        self.spike_counts.zero_()
        self.goal_rates.zero_()
        self.bias.zero_()
