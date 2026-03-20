"""
Spiking Classifier — LIF neuron readout for entity classification.

Replaces the standard nn.Linear classifier with a spiking neural network
that classifies retinal type-channel input into 5 entity categories.

Architecture:
  - Input: 804 dims (800 type pixels + 4 aggregate counts)
  - Hidden: 128 LIF neurons with lateral inhibition
  - Output: 5 LIF neurons (nothing, food, enemy, colleague, environment)
  - Winner = highest firing rate over integration window

Neuroscience: tectal classification in zebrafish uses populations of
size-selective neurons with competitive dynamics (Barker & Bhatt 2021).

Torch-based — spiking LIF neurons.
"""
import torch
import torch.nn as nn
import numpy as np


class SpikingClassifier(nn.Module):
    """LIF spiking classifier for entity recognition.

    Args:
        n_input: int — input dimensions (804)
        n_hidden: int — hidden LIF neurons
        n_classes: int — output classes
        tau: float — membrane time constant
        v_thresh: float — spike threshold
        n_integration: int — steps to integrate before readout
        device: str
    """

    def __init__(self, n_input=804, n_hidden=128, n_classes=5,
                 tau=0.8, v_thresh=0.5, n_integration=5, device="cpu"):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.tau = tau
        self.v_thresh = v_thresh
        self.n_integration = n_integration
        self.device = device

        # Input → hidden (feedforward)
        self.W_in = nn.Linear(n_input, n_hidden, bias=True)
        # Hidden → output (readout)
        self.W_out = nn.Linear(n_hidden, n_classes, bias=True)
        # Lateral inhibition in hidden layer
        self.W_lat = nn.Parameter(
            -0.1 * (torch.ones(n_hidden, n_hidden)
                     - torch.eye(n_hidden)))

        # LIF state
        self.v_hidden = torch.zeros(1, n_hidden, device=device)
        self.v_out = torch.zeros(1, n_classes, device=device)

        self.to(device)

    def forward(self, x):
        """Classify input via spiking dynamics.

        Args:
            x: tensor [1, 804] — type channel + aggregate counts

        Returns:
            logits: tensor [1, 5] — firing rates per class
            probs: tensor [1, 5] — softmax probabilities
        """
        spike_counts = torch.zeros(1, self.n_classes, device=self.device)

        for _ in range(self.n_integration):
            # Hidden layer: LIF with lateral inhibition
            I_in = self.W_in(x)
            I_lat = self.v_hidden @ self.W_lat
            self.v_hidden = (self.tau * self.v_hidden
                             + (1 - self.tau) * (I_in + I_lat))

            # Spike generation
            h_spikes = (self.v_hidden >= self.v_thresh).float()
            self.v_hidden = self.v_hidden * (1 - h_spikes)  # reset

            # Output layer
            I_out = self.W_out(h_spikes)
            self.v_out = self.tau * self.v_out + (1 - self.tau) * I_out

            o_spikes = (self.v_out >= self.v_thresh).float()
            self.v_out = self.v_out * (1 - o_spikes)
            spike_counts += o_spikes

        # Firing rate as logits
        logits = spike_counts / self.n_integration
        # Also add analog readout for gradient flow during training
        analog = self.W_out(torch.relu(self.W_in(x)))
        logits = logits + 0.5 * analog

        return logits

    def reset(self):
        self.v_hidden.zero_()
        self.v_out.zero_()

    def get_saveable_state(self):
        return {
            "W_in": self.W_in.state_dict(),
            "W_out": self.W_out.state_dict(),
            "W_lat": self.W_lat.data.clone(),
        }

    def load_saveable_state(self, state):
        self.W_in.load_state_dict(state["W_in"])
        self.W_out.load_state_dict(state["W_out"])
        self.W_lat.data = state["W_lat"].to(self.device)
