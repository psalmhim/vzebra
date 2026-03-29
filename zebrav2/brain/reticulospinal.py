"""
Named reticulospinal neurons (21 per side).
Key cells: Mauthner (C-start), MiD2/3 (turns), RoM2 (voluntary), MeM (sustained), CaD (speed).
"""
import torch
import torch.nn as nn
import math
from zebrav2.spec import DEVICE

class ReticulospinalSystem(nn.Module):
    NEURON_NAMES = [
        'Mauthner', 'MiD2', 'MiD3',          # fast escape
        'RoM2a', 'RoM2b', 'RoM2c', 'RoM2d',  # voluntary turn
        'MeM1', 'MeM2', 'MeM3', 'MeM4',      # sustained locomotion
        'MeM5', 'MeM6', 'MeM7', 'MeM8',
        'CaD1', 'CaD2', 'CaD3', 'CaD4',      # speed modulation
        'CaD5', 'CaD6',
    ]  # 21 neurons per side

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_rs = len(self.NEURON_NAMES)  # 21
        # State per side (L and R)
        self.register_buffer('rate_L', torch.zeros(self.n_rs, device=device))
        self.register_buffer('rate_R', torch.zeros(self.n_rs, device=device))
        # Mauthner refractory counter
        self.mauthner_refrac = 0
        self.mauthner_active = 0  # C-start motor sequence step
        # Simplified motor output
        self.register_buffer('motor_turn', torch.tensor(0.0, device=device))
        self.register_buffer('motor_speed', torch.tensor(1.0, device=device))

    def forward(self, sgc_rate: torch.Tensor, bg_gate: float,
                pal_d_rate: torch.Tensor, flee_dir: float,
                goal_speed: float, looming: bool) -> dict:
        """
        sgc_rate: (N_SGC_E,) looming input from tectal SGC
        bg_gate: float (0-1) from BG, gates voluntary movement
        pal_d_rate: (N_PAL_D,) for voluntary turn computation
        flee_dir: float (-1 to 1) escape direction (from v1 logic)
        goal_speed: float (0-2) desired speed
        looming: bool from tectum
        Returns: {'turn': float, 'speed': float, 'cstart': bool}
        """
        cstart = False
        # --- Mauthner C-start ---
        if self.mauthner_refrac > 0:
            self.mauthner_refrac -= 1
        if self.mauthner_active > 0:
            # Motor sequence: steps 1-2 sharp turn, 3-4 power burst
            if self.mauthner_active > 2:
                turn = 1.5 * math.copysign(1, flee_dir) if flee_dir != 0 else 1.5
                speed = 1.0
            else:
                turn = 0.0
                speed = 1.6
            self.mauthner_active -= 1
            self.motor_turn.fill_(turn)
            self.motor_speed.fill_(speed)
            return {'turn': turn, 'speed': speed, 'cstart': True,
                    'rate_L': self.rate_L, 'rate_R': self.rate_R}
        # Check Mauthner trigger: looming + SGC activity
        sgc_mean = sgc_rate.mean().item()
        if looming and sgc_mean > 0.05 and self.mauthner_refrac == 0:
            cstart = True
            self.mauthner_active = 4
            self.mauthner_refrac = 12
            turn = 1.5 * math.copysign(1, flee_dir) if flee_dir != 0 else 1.5
            self.motor_turn.fill_(turn)
            self.motor_speed.fill_(1.0)
            return {'turn': turn, 'speed': 1.0, 'cstart': True,
                    'rate_L': self.rate_L, 'rate_R': self.rate_R}
        # --- Voluntary movement (gated by BG) ---
        # Turn from pallium-D asymmetry (L vs R halves)
        pal_L = pal_d_rate[:len(pal_d_rate)//2].mean().item()
        pal_R = pal_d_rate[len(pal_d_rate)//2:].mean().item()
        voluntary_turn = (pal_R - pal_L) * bg_gate * 2.0
        # Combine with flee direction
        turn = flee_dir * 0.7 + voluntary_turn * 0.3
        turn = max(-1.0, min(1.0, turn))
        # Speed: flee bypasses BG gate (survival reflex), others are gated
        if abs(flee_dir) > 0.1:
            speed = goal_speed  # full speed during flee
        else:
            speed = goal_speed * (0.5 + 0.5 * bg_gate)
        self.motor_turn.fill_(turn)
        self.motor_speed.fill_(speed)
        return {'turn': turn, 'speed': speed, 'cstart': False,
                'rate_L': self.rate_L, 'rate_R': self.rate_R}

    def reset(self):
        self.rate_L.zero_()
        self.rate_R.zero_()
        self.mauthner_refrac = 0
        self.mauthner_active = 0
        self.motor_turn.zero_()
        self.motor_speed.fill_(1.0)
