"""
Pectoral fin motor neurons: slow-turn kinematics.

Zebrafish larval pectoral fins:
  Used for slow, precise turns during prey capture (J-turns) and
  gentle exploration steering. Distinct from tail-driven fast escape.

  Anatomy:
    4 motor neurons per side (8 total), innervating pectoral fin muscles.
    Alternating L/R activation creates differential thrust for steering.

  Behavior:
    - FORAGE: active, moderate gain (precise food approach)
    - EXPLORE: active, low gain (gentle turns)
    - FLEE: SUPPRESSED (tail C-start dominates, fins folded)
    - SOCIAL: active, low gain

  Output blends with existing brain_turn at 15% weight (conservative).

References:
  - Green et al. (2011) "Pectoral fin movements in larval zebrafish"
    J Exp Biol
  - Thorsen et al. (2004) "Swimming of larval zebrafish: fin-body
    coordination and implications for function" J Exp Biol
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer

GOAL_FORAGE, GOAL_FLEE, GOAL_EXPLORE, GOAL_SOCIAL = 0, 1, 2, 3


class PectoralFinMotor(nn.Module):
    def __init__(self, n_per_side=4, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_per_side = n_per_side
        self.n_total = n_per_side * 2

        # Left and right fin motor pools
        self.fin_L = IzhikevichLayer(n_per_side, 'RS', device)
        self.fin_R = IzhikevichLayer(n_per_side, 'RS', device)
        self.fin_L.i_tonic.fill_(-1.5)
        self.fin_R.i_tonic.fill_(-1.5)

        # State buffers
        self.register_buffer('rate_L', torch.zeros(n_per_side, device=device))
        self.register_buffer('rate_R', torch.zeros(n_per_side, device=device))

        # Goal-dependent gain
        self._goal_gains = {
            GOAL_FORAGE: 1.0,
            GOAL_FLEE: 0.0,     # suppressed
            GOAL_EXPLORE: 0.5,
            GOAL_SOCIAL: 0.4,
        }

    @torch.no_grad()
    def forward(self, food_bearing: float = 0.0,
                turn_request: float = 0.0,
                goal: int = GOAL_EXPLORE,
                goal_speed: float = 1.0) -> dict:
        """
        Parameters
        ----------
        food_bearing : float [-1, 1] — food direction (+ = right)
        turn_request : float [-1, 1] — brain's desired turn
        goal : int — current goal (0-3)
        goal_speed : float — desired speed

        Returns dict with fin_turn, fin_L_rate, fin_R_rate, active
        """
        gain = self._goal_gains.get(goal, 0.5)
        active = gain > 0.01

        if not active:
            self.rate_L.zero_()
            self.rate_R.zero_()
            return {'fin_turn': 0.0, 'fin_L_rate': 0.0,
                    'fin_R_rate': 0.0, 'active': False}

        # Drive: turn request + food bearing → differential L/R
        # Positive turn → more R fin activity (push left → turn right)
        combined = turn_request * 0.6 + food_bearing * 0.4
        base_drive = goal_speed * 3.0

        I_L = torch.full((self.n_per_side,),
                         base_drive + max(0.0, -combined) * 10.0 * gain,
                         device=self.device)
        I_R = torch.full((self.n_per_side,),
                         base_drive + max(0.0, combined) * 10.0 * gain,
                         device=self.device)

        for _ in range(15):
            self.fin_L(I_L + torch.randn(self.n_per_side, device=self.device) * 0.3)
            self.fin_R(I_R + torch.randn(self.n_per_side, device=self.device) * 0.3)

        self.rate_L.copy_(self.fin_L.rate)
        self.rate_R.copy_(self.fin_R.rate)
        L_mean = float(self.rate_L.mean())
        R_mean = float(self.rate_R.mean())

        # Differential thrust → turn signal
        fin_turn = (R_mean - L_mean) * 3.0 * gain
        fin_turn = max(-1.0, min(1.0, fin_turn))

        return {
            'fin_turn': fin_turn,
            'fin_L_rate': L_mean,
            'fin_R_rate': R_mean,
            'active': True,
        }

    def reset(self):
        self.fin_L.reset()
        self.fin_R.reset()
        self.rate_L.zero_()
        self.rate_R.zero_()
