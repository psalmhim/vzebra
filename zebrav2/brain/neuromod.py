"""
Four-axis neuromodulation: DA, NA, 5-HT, ACh.
Each axis is a small population with projection dynamics.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE

class NeuromodSystem(nn.Module):
    """
    Maintains all four neuromodulatory axes.
    Updated once per behavioral step (not per Izhikevich ms).
    """
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        # State variables (all scalar-valued signals)
        self.register_buffer('DA',    torch.tensor(0.5, device=device))
        self.register_buffer('NA',    torch.tensor(0.3, device=device))
        self.register_buffer('HT5',   torch.tensor(0.5, device=device))  # 5-HT
        self.register_buffer('ACh',   torch.tensor(0.5, device=device))
        # Value function for DA (simple scalar)
        self.register_buffer('V',     torch.tensor(0.0, device=device))
        # Accumulators
        self.register_buffer('flee_frac_ema', torch.tensor(0.0, device=device))

    def update(self, reward: float, amygdala_alpha: float, cms: float,
               flee_active: bool, fatigue: float, circadian: float,
               current_goal: int) -> dict:
        """
        Update all neuromodulatory axes.
        Returns dict of current levels.
        """
        r = torch.tensor(reward, device=self.device)
        # --- DA (dopamine): reward prediction error ---
        RPE = r - self.V
        self.V.copy_(self.V + 0.05 * RPE)
        DA_new = torch.sigmoid(3.0 * RPE) + 0.01 * torch.randn(1, device=self.device).squeeze()
        self.DA.copy_(DA_new.clamp(0.0, 1.0))
        # --- NA (noradrenaline): arousal/threat ---
        na_drive = 0.3 + 0.5 * amygdala_alpha + 0.2 * cms
        self.NA.copy_(0.85 * self.NA + 0.15 * torch.tensor(na_drive, device=self.device))
        self.NA.clamp_(0.0, 1.0)
        # --- 5-HT (serotonin): patience/habituation ---
        # Rises when not fleeing (behavioral tolerance)
        flee_frac = 1.0 if flee_active else 0.0
        self.flee_frac_ema.copy_(0.95 * self.flee_frac_ema + 0.05 * flee_frac)
        ht5_drive = 1.0 - self.flee_frac_ema.item()
        self.HT5.copy_(0.92 * self.HT5 + 0.08 * torch.tensor(ht5_drive, device=self.device))
        self.HT5.clamp_(0.0, 1.0)
        # --- ACh (acetylcholine): attention/plasticity ---
        ach_drive = circadian * (1.0 - fatigue) * (1.0 + 0.3 * cms)
        self.ACh.copy_(0.90 * self.ACh + 0.10 * torch.tensor(ach_drive, device=self.device))
        self.ACh.clamp_(0.0, 1.0)
        return {
            'DA': self.DA.item(), 'NA': self.NA.item(),
            '5HT': self.HT5.item(), 'ACh': self.ACh.item(),
            'RPE': RPE.item(),
        }

    def get_flee_efe_bias(self) -> float:
        """5-HT-mediated flee suppression for PATROL states."""
        return 0.4 * self.HT5.item()

    def get_plasticity_gate(self) -> float:
        """ACh-gated plasticity multiplier."""
        return 1.0 + self.ACh.item()

    def get_gain_all(self) -> float:
        """NA-mediated global gain boost."""
        return 1.0 + 0.4 * self.NA.item()

    def reset(self):
        self.DA.fill_(0.5)
        self.NA.fill_(0.3)
        self.HT5.fill_(0.5)
        self.ACh.fill_(0.5)
        self.V.zero_()
        self.flee_frac_ema.zero_()
