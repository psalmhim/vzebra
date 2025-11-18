import torch
import torch.nn as nn

class PrecisionUnit(nn.Module):
    def __init__(self, size, device="cpu", alpha=0.02, beta=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(size, device=device))
        self.alpha = alpha
        self.beta = beta   # desired error level

    def compute_pi(self):
        return torch.sigmoid(self.gamma)

    def update_precision(self, error):
        with torch.no_grad():
            mag = error.abs().mean()
            dgamma = self.alpha * (mag - self.beta)
            self.gamma += dgamma
