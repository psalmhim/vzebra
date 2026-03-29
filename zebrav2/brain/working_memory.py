"""
Spiking working memory: persistent activity buffer.

Implements bump attractor dynamics for short-term state maintenance.
When input is removed, activity persists via recurrent excitation
until a new input overwrites or decay extinguishes it.

Architecture:
  32 RS neurons with strong recurrent excitation
  8 FS inhibitory neurons for WTA / selective maintenance
  Stores: last goal, last food direction, threat direction
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingWorkingMemory(nn.Module):
    def __init__(self, n_exc=32, n_inh=8, n_slots=4, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_slots = n_slots  # number of memory items
        self.n_per_slot = n_exc // n_slots  # 8 neurons per slot

        self.exc = IzhikevichLayer(n_exc, 'RS', device)
        self.inh = IzhikevichLayer(n_inh, 'FS', device)
        # Bistable: moderate tonic so recurrence can sustain
        self.exc.i_tonic.fill_(1.0)
        self.inh.i_tonic.fill_(1.0)

        # Strong recurrent excitation within slots (bump attractor)
        self.register_buffer('W_rec', torch.zeros(n_exc, n_exc, device=device))
        for s in range(n_slots):
            start = s * self.n_per_slot
            end = start + self.n_per_slot
            self.W_rec[start:end, start:end] = 0.8  # within-slot recurrence
            for i in range(start, end):
                self.W_rec[i, i] = 0.0  # no self-connection

        # Cross-slot inhibition (WTA between slots for capacity limit)
        self.register_buffer('W_inh', torch.ones(n_inh, n_exc, device=device) * 0.3)
        self.register_buffer('W_inh_back', -torch.ones(n_exc, n_inh, device=device) * 0.5)

        # State
        self.register_buffer('exc_rate', torch.zeros(n_exc, device=device))
        self.register_buffer('inh_rate', torch.zeros(n_inh, device=device))
        self.register_buffer('slot_activity', torch.zeros(n_slots, device=device))

    @torch.no_grad()
    def forward(self, input_drive: torch.Tensor = None,
                gate: float = 1.0) -> dict:
        """
        input_drive: (n_exc,) external input. If None, rely on recurrence only.
        gate: float [0,1] — ACh-gated write signal. Low gate = read-only (persistent).
        """
        if input_drive is None:
            I_ext = torch.zeros(self.n_exc, device=self.device)
        else:
            if input_drive.shape[0] != self.n_exc:
                # Pad or truncate
                I_ext = torch.zeros(self.n_exc, device=self.device)
                n = min(input_drive.shape[0], self.n_exc)
                I_ext[:n] = input_drive[:n]
            else:
                I_ext = input_drive
            I_ext = I_ext * gate * 8.0  # scale for Izhikevich

        for _ in range(SUBSTEPS):
            # Recurrent excitation
            I_rec = self.W_rec @ self.exc.rate * 10.0
            # Inhibition
            I_inh_to_exc = self.W_inh_back @ self.inh.rate * 8.0

            self.exc(I_ext + I_rec + I_inh_to_exc
                     + torch.randn(self.n_exc, device=self.device) * 0.3)

            I_exc_to_inh = self.W_inh @ self.exc.rate * 5.0
            self.inh(I_exc_to_inh + torch.randn(self.n_inh, device=self.device) * 0.3)

        self.exc_rate.copy_(self.exc.rate)
        self.inh_rate.copy_(self.inh.rate)

        # Per-slot activity
        for s in range(self.n_slots):
            start = s * self.n_per_slot
            end = start + self.n_per_slot
            self.slot_activity[s] = self.exc_rate[start:end].mean()

        return {
            'exc_rate': self.exc_rate,
            'slot_activity': self.slot_activity.clone(),
            'active_slots': int((self.slot_activity > 0.01).sum()),
            'total_activity': float(self.exc_rate.mean()),
        }

    def read_slot(self, slot_idx: int) -> torch.Tensor:
        """Read activity from a specific memory slot."""
        start = slot_idx * self.n_per_slot
        end = start + self.n_per_slot
        return self.exc_rate[start:end].clone()

    def write_slot(self, slot_idx: int, pattern: torch.Tensor):
        """Inject pattern into a specific slot (strong external drive)."""
        start = slot_idx * self.n_per_slot
        I_write = torch.zeros(self.n_exc, device=self.device)
        n = min(pattern.shape[0], self.n_per_slot)
        I_write[start:start + n] = pattern[:n] * 15.0
        self.forward(I_write, gate=1.0)

    def reset(self):
        self.exc.reset()
        self.inh.reset()
        self.exc_rate.zero_()
        self.inh_rate.zero_()
        self.slot_activity.zero_()
