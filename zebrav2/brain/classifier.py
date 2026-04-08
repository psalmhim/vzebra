"""
Entity classifier — spiking LIF readout for retinal scene classification.

Input: 804 dims (800 type pixels + 4 aggregate pixel counts)
Hidden: 128 LIF neurons with lateral inhibition
Output: 5 classes (nothing, food, enemy, colleague, environment)

Same architecture as v1 SpikingClassifier, adapted for v2 device handling.
Can load v1 weights directly.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE


CLASS_NAMES = ['nothing', 'food', 'enemy', 'colleague', 'environment']


class ClassifierV2(nn.Module):
    def __init__(self, n_input=804, n_hidden=128, n_classes=5,
                 tau=0.8, v_thresh=0.5, n_integration=5, device=DEVICE):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.tau = tau
        self.v_thresh = v_thresh
        self.n_integration = n_integration
        self.device = device

        self.W_in = nn.Linear(n_input, n_hidden, bias=True)
        self.W_out = nn.Linear(n_hidden, n_classes, bias=True)
        self.W_lat = nn.Parameter(
            -0.1 * (torch.ones(n_hidden, n_hidden) - torch.eye(n_hidden)))

        self.register_buffer('v_hidden', torch.zeros(1, n_hidden, device=device))
        self.register_buffer('v_out', torch.zeros(1, n_classes, device=device))
        self._last_hidden = None  # cached for Hebbian fine-tuning

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 804) or (804,)
        Returns: logits (1, 5)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        spike_counts = torch.zeros(1, self.n_classes, device=self.device)

        for _ in range(self.n_integration):
            I_in = self.W_in(x)
            I_lat = self.v_hidden @ self.W_lat
            self.v_hidden = self.tau * self.v_hidden + (1 - self.tau) * (I_in + I_lat)
            h_spikes = (self.v_hidden >= self.v_thresh).float()
            self.v_hidden = self.v_hidden * (1 - h_spikes)

            I_out = self.W_out(h_spikes)
            self.v_out = self.tau * self.v_out + (1 - self.tau) * I_out
            o_spikes = (self.v_out >= self.v_thresh).float()
            self.v_out = self.v_out * (1 - o_spikes)
            spike_counts += o_spikes

        self._last_hidden = h_spikes.detach()  # cache for Hebbian update
        logits = spike_counts / self.n_integration
        analog = self.W_out(torch.relu(self.W_in(x)))
        logits = logits + 0.5 * analog
        return logits

    def classify(self, L: torch.Tensor, R: torch.Tensor) -> dict:
        """
        Convenience: build 804-dim input from retinal arrays and classify.
        L, R: (800,) retinal arrays [0:400]=intensity, [400:800]=type
        Returns dict with 'probs', 'class_id', 'class_name'
        """
        type_L = L[400:]
        type_R = R[400:]
        type_all = torch.cat([type_L, type_R])  # (800,)

        # Aggregate pixel counts (same as v1)
        obs_px = ((torch.abs(type_all - 0.75) < 0.1).float().sum()).unsqueeze(0)
        ene_px = ((torch.abs(type_all - 0.5) < 0.1).float().sum()).unsqueeze(0)
        food_px = ((torch.abs(type_all - 1.0) < 0.15).float().sum()).unsqueeze(0)
        bound_px = ((torch.abs(type_all - 0.12) < 0.05).float().sum()).unsqueeze(0)

        x = torch.cat([type_all, obs_px, ene_px, food_px, bound_px])  # (804,)
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1).squeeze()
        class_id = int(probs.argmax().item())
        return {
            'probs': probs,
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'logits': logits.squeeze(),
        }

    def reset(self):
        self.v_hidden.zero_()
        self.v_out.zero_()

    def load_v1_weights(self, path: str):
        """Load v1 classifier checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if 'W_in' in state:
            self.W_in.load_state_dict(state['W_in'])
            self.W_out.load_state_dict(state['W_out'])
            if 'W_lat' in state:
                self.W_lat.data = state['W_lat'].to(self.device)
        elif 'state_dict' in state:
            self.load_state_dict(state['state_dict'], strict=False)
