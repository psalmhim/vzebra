"""
Theta-phase place cells with phase precession.
128E + 32I, theta oscillation at 8Hz, STDP-based sequence learning.
"""
import torch
import torch.nn as nn
import math
from zebrav2.spec import DEVICE, T_THETA, SUBSTEPS

class ThetaPlaceCells(nn.Module):
    def __init__(self, n_cells: int = 128, arena_w: int = 800, arena_h: int = 600,
                 sigma_px: float = 60.0, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_cells = n_cells
        self.sigma = sigma_px
        self.arena_w = arena_w
        self.arena_h = arena_h
        # Place cell centroids: random tiling
        cx = torch.rand(n_cells) * arena_w
        cy = torch.rand(n_cells) * arena_h
        self.register_buffer('cx', cx.to(device))
        self.register_buffer('cy', cy.to(device))
        # Per-cell stored values
        self.register_buffer('food_rate', torch.zeros(n_cells, device=device))
        self.register_buffer('risk_rate', torch.zeros(n_cells, device=device))
        self.register_buffer('visit_count', torch.zeros(n_cells, device=device))
        # Theta oscillator phase
        self.theta_phase = 0.0
        self.theta_period_steps = int(T_THETA / (SUBSTEPS * 0.001))  # steps per theta cycle
        self.step_count = 0
        # Sequence weights (forward STDP)
        self.W_seq = nn.Parameter(torch.zeros(n_cells, n_cells, device=device))
        # Rate output
        self.register_buffer('rate', torch.zeros(n_cells, device=device))

    def activation(self, pos_x: float, pos_y: float) -> torch.Tensor:
        """Gaussian place field activation."""
        px = torch.tensor(float(pos_x), dtype=torch.float32, device=self.device)
        py = torch.tensor(float(pos_y), dtype=torch.float32, device=self.device)
        dist2 = (self.cx - px)**2 + (self.cy - py)**2
        return torch.exp(-dist2 / (2 * self.sigma**2))

    def forward(self, pos_x: float, pos_y: float,
                food_eaten: bool = False, predator_near: bool = False) -> dict:
        """One behavioral step."""
        self.step_count += 1
        # Theta phase
        self.theta_phase = (2 * math.pi * self.step_count / self.theta_period_steps) % (2 * math.pi)
        theta = math.sin(self.theta_phase)
        # Place field activation
        base_act = self.activation(pos_x, pos_y)
        # Phase precession: spike phase advances with depth into field
        phase_offset = math.pi * base_act  # 0 at edge, π at center (tensor)
        phase_mod = torch.sin(torch.tensor(self.theta_phase, device=self.device) - phase_offset)
        # Theta-gated firing
        rate = base_act * (1.0 + 0.5 * theta) * (phase_mod > 0).float() * base_act
        rate = rate.clamp(0, 1)
        self.rate.copy_(rate)
        # Update stored values for active cells
        active = (base_act > 0.2)
        if active.any():
            alpha = 0.05
            if food_eaten:
                self.food_rate[active] = (1 - alpha) * self.food_rate[active] + alpha
            else:
                self.food_rate[active] = (1 - alpha * 0.1) * self.food_rate[active]
            if predator_near:
                self.risk_rate[active] = (1 - alpha) * self.risk_rate[active] + alpha
            self.visit_count[active] += 1
        # Planning bonus: food-rich direction
        food_val = (self.food_rate * base_act).sum()
        risk_val = (self.risk_rate * base_act).sum()
        return {
            'rate': rate,
            'food_value': food_val.item(),
            'risk_value': risk_val.item(),
            'theta_phase': self.theta_phase,
        }

    def get_efe_bonus(self) -> dict:
        """Planning bonus for EFE (from place cell memory)."""
        food_map = self.food_rate.mean().item()
        risk_map = self.risk_rate.mean().item()
        return {
            'forage_bonus': food_map * 0.3,
            'flee_bonus': risk_map * 0.2,
        }

    def reset(self):
        self.rate.zero_()
        self.theta_phase = 0.0
        self.step_count = 0
