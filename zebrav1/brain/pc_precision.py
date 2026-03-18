import torch
import torch.nn as nn


class PrecisionUnit(nn.Module):
    def __init__(self, size, device="cpu", alpha=0.05, beta=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(size, device=device))
        self.alpha = alpha      # learning rate (was 0.02, now 0.05 for faster adaptation)
        self.beta = beta        # desired error level
        self.prev_mag = 0.0     # for novelty detection

    def compute_pi(self):
        return torch.sigmoid(self.gamma)

    def update_precision(self, error):
        """Update precision based on prediction error.
        High error → increase gamma → increase precision (attend more).
        Low error → decrease gamma → decrease precision (relax).
        Sudden changes (novelty) → sharp precision drop then recovery.
        """
        with torch.no_grad():
            mag = float(error.abs().mean())
            # Novelty: sudden change in error magnitude
            novelty = abs(mag - self.prev_mag)
            self.prev_mag = 0.8 * self.prev_mag + 0.2 * mag  # faster tracking (was 0.9/0.1)

            # Base precision update
            dgamma = self.alpha * (mag - self.beta)

            # Novelty-driven precision dip (surprise → temporary drop)
            if novelty > 0.03:  # lower threshold (was 0.05)
                dgamma -= 0.05 * novelty  # stronger penalty (was 0.03)

            self.gamma.data += dgamma

            # Clamp gamma to prevent runaway
            self.gamma.data.clamp_(-3.0, 3.0)
