import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from zebra_v60.world.world_env import WorldEnv

class HebbianLearning:
    """
    Simple rate-based Hebbian learning algorithm for updating weights.
    Rule: Δw_ij = η * pre_i * post_j
    """
    def __init__(self, learning_rate=0.01):
        self.eta = learning_rate

    def update(self, W, pre_activity, post_activity):
        """
        Update weights using Hebbian rule.
        W: weight matrix (n_pre, n_post)
        pre_activity: pre-synaptic activities (batch, n_pre)
        post_activity: post-synaptic activities (batch, n_post)
        """
        with torch.no_grad():
            # Compute outer product: (n_pre, batch) @ (batch, n_post) -> (n_pre, n_post)
            delta_W = self.eta * (pre_activity.t() @ post_activity)
            W.add_(delta_W)

class SpikeBasedHebbianLearning:
    """
    Optimized spike-based Hebbian learning using STDP (Spike-Timing Dependent Plasticity).
    Uses vectorized operations for efficiency.
    """
    def __init__(self, learning_rate=0.01, tau_plus=20.0, tau_minus=20.0, A_plus=0.1, A_minus=0.1):
        self.eta = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus

    def update(self, W, pre_spike_times, post_spike_times):
        """
        Optimized STDP update using vectorized computations.
        W: weight matrix (n_pre, n_post)
        pre_spike_times: list of tensors, each (n_spikes,) for pre neurons
        post_spike_times: list of tensors, each (n_spikes,) for post neurons
        """
        with torch.no_grad():
            n_pre, n_post = W.shape
            delta_W = torch.zeros_like(W)

            for i in range(n_pre):
                pre_times = pre_spike_times[i]  # (n_pre_spikes,)
                if len(pre_times) == 0:
                    continue
                for j in range(n_post):
                    post_times = post_spike_times[j]  # (n_post_spikes,)
                    if len(post_times) == 0:
                        continue

                    # Compute all pairwise time differences: (n_pre_spikes, n_post_spikes)
                    dt = post_times.unsqueeze(0) - pre_times.unsqueeze(1)  # post - pre

                    # LTP: pre before post (dt > 0)
                    ltp_mask = dt > 0
                    delta_ltp = torch.where(ltp_mask, self.A_plus * torch.exp(-dt / self.tau_plus), 0.0).sum()

                    # LTD: post before pre (dt < 0)
                    ltd_mask = dt < 0
                    delta_ltd = torch.where(ltd_mask, -self.A_minus * torch.exp(dt / self.tau_minus), 0.0).sum()

                    delta_W[i, j] = self.eta * (delta_ltp + delta_ltd)

            W.add_(delta_W)

# Training Set and Loop Example
class ZebrafishTrainingDataset:
    """
    Simple training dataset for zebrafish vision-behavior tasks.
    Generates food positions and expected behaviors.
    """
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        for _ in range(self.n_samples):
            # Random food position
            x = np.random.uniform(-100, 100)
            y = np.random.uniform(-75, 75)
            food_pos = (x, y)

            # Compute expected turn direction based on angle
            angle = np.arctan2(y, x)
            if angle < -np.pi/4:
                expected_turn = -1  # left
            elif angle > np.pi/4:
                expected_turn = 1   # right
            else:
                expected_turn = 0   # forward

            data.append((food_pos, expected_turn))
        return data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]

def train_zebrafish_model(model, dataset, hebbian_learner, epochs=10, device="cpu"):
    """
    Training loop using Hebbian learning.
    model: ZebrafishSNN_v60
    dataset: ZebrafishTrainingDataset
    hebbian_learner: HebbianLearning or SpikeBasedHebbianLearning
    """
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for food_pos, target_turn in dataset:
            world = WorldEnv()
            world.food = [food_pos]

            model.reset()
            pos = np.array([0., 0.])
            heading = 0.0

            out = model.forward(pos, heading, world)
            motor = out["motor"]
            pred_turn = (motor[0, :100].mean() - motor[0, 100:].mean()).item()

            # Supervised loss (optional, for fine-tuning)
            loss = (pred_turn - target_turn)**2
            total_loss += loss

            # Hebbian update (example: update OT_F to PT_L)
            if isinstance(hebbian_learner, HebbianLearning):
                pre_act = model.OT_F.v  # post from OT_F
                post_act = model.PT_L.v  # pre to PT_L? Wait, adjust as needed
                hebbian_learner.update(model.PT_L.W, pre_act, post_act)
            # For STDP, collect spike times and update

        print(f"Epoch {epoch+1}: Loss {total_loss / len(dataset):.4f}")

# Example usage
if __name__ == "__main__":
    # Rate-based example
    n_pre = 10
    n_post = 5
    batch_size = 32

    W = torch.randn(n_pre, n_post)
    pre = torch.randn(batch_size, n_pre)
    post = torch.randn(batch_size, n_post)

    hebb = HebbianLearning(learning_rate=0.001)
    hebb.update(W, pre, post)

    print("Rate-based weights updated.")

    # Spike-based example
    pre_spike_times = [torch.tensor([1.0, 3.0]) for _ in range(n_pre)]
    post_spike_times = [torch.tensor([2.0, 4.0]) for _ in range(n_post)]

    stdp = SpikeBasedHebbianLearning(learning_rate=0.001)
    stdp.update(W, pre_spike_times, post_spike_times)

    print("Spike-based weights updated with optimized STDP.")

    # Training dataset example
    dataset = ZebrafishTrainingDataset(n_samples=100)
    print(f"Generated {len(dataset)} training samples.")