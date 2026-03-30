"""
Spiking predictive network: predicts next-frame retinal input.

Architecture (spiking autoencoder):
  Encoder: retinal ON rates (100) → 80 spiking hidden → 32 latent (rate-coded)
  Decoder: 32 latent + 2 motor → 80 spiking hidden → 100 predicted retinal

Prediction error drives STDP-based weight updates.
Used for active inference: expected free energy includes prediction uncertainty.
Replaces the analytic InternalWorldModel with a learned spiking world model.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingPredictiveNet(nn.Module):
    def __init__(self, n_input=100, n_hidden=80, n_latent=32,
                 n_motor=2, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        # Encoder: input → hidden (spiking) → latent (rate)
        self.enc_hidden = IzhikevichLayer(n_hidden, 'RS', device)
        self.enc_hidden.i_tonic.fill_(-1.0)

        self.W_in_enc = nn.Linear(n_input, n_hidden, bias=False)
        self.W_enc_lat = nn.Linear(n_hidden, n_latent, bias=False)
        for W in [self.W_in_enc, self.W_enc_lat]:
            nn.init.xavier_uniform_(W.weight, gain=0.5)
            W.to(device)

        # Decoder: latent + motor → hidden (spiking) → predicted retinal
        self.dec_hidden = IzhikevichLayer(n_hidden, 'RS', device)
        self.dec_hidden.i_tonic.fill_(-1.0)

        self.W_lat_dec = nn.Linear(n_latent + n_motor, n_hidden, bias=False)
        self.W_dec_out = nn.Linear(n_hidden, n_input, bias=False)
        for W in [self.W_lat_dec, self.W_dec_out]:
            nn.init.xavier_uniform_(W.weight, gain=0.5)
            W.to(device)

        # State
        self.register_buffer('latent', torch.zeros(n_latent, device=device))
        self.register_buffer('prediction', torch.zeros(n_input, device=device))
        self.register_buffer('pred_error', torch.zeros(n_input, device=device))
        self.register_buffer('enc_rate', torch.zeros(n_hidden, device=device))
        self.register_buffer('dec_rate', torch.zeros(n_hidden, device=device))

        # Previous retinal input (for next-frame prediction)
        self.register_buffer('prev_retinal', torch.zeros(n_input, device=device))

        # Learning rate
        self.lr = 5e-4

    def _pool_input(self, retinal: torch.Tensor) -> torch.Tensor:
        """Pool retinal input to n_input size."""
        if retinal.shape[0] == self.n_input:
            return retinal
        # Manual index-based pooling (MPS doesn't support non-divisible adaptive pool)
        indices = torch.linspace(0, retinal.shape[0] - 1, self.n_input,
                                 device=self.device).long()
        return retinal[indices]

    @torch.no_grad()
    def encode(self, retinal: torch.Tensor) -> torch.Tensor:
        """Encode retinal input to latent representation."""
        x = self._pool_input(retinal)
        I_enc = self.W_in_enc(x.unsqueeze(0)).squeeze(0).detach()
        I_enc = I_enc * (6.0 / (I_enc.abs().mean() + 1e-8))

        for _ in range(20):  # reduced substeps
            self.enc_hidden(I_enc + torch.randn(self.n_hidden, device=self.device) * 0.3)

        self.enc_rate.copy_(self.enc_hidden.rate)
        latent = self.W_enc_lat(self.enc_hidden.rate.unsqueeze(0)).squeeze(0).detach()
        # Scale up before sigmoid for wider dynamic range
        latent = torch.sigmoid(latent * 8.0)  # bounded [0,1] with more spread
        self.latent.copy_(latent)
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, motor_cmd: torch.Tensor) -> torch.Tensor:
        """Decode latent + motor to predicted retinal."""
        z = torch.cat([latent, motor_cmd])
        I_dec = self.W_lat_dec(z.unsqueeze(0)).squeeze(0).detach()
        I_dec = I_dec * (6.0 / (I_dec.abs().mean() + 1e-8))

        for _ in range(20):  # reduced substeps
            self.dec_hidden(I_dec + torch.randn(self.n_hidden, device=self.device) * 0.3)

        self.dec_rate.copy_(self.dec_hidden.rate)
        pred = self.W_dec_out(self.dec_hidden.rate.unsqueeze(0)).squeeze(0).detach()
        pred = torch.sigmoid(pred * 3.0)
        self.prediction.copy_(pred)
        return pred

    @torch.no_grad()
    def forward(self, retinal_input: torch.Tensor,
                motor_cmd: torch.Tensor) -> dict:
        """
        Full forward pass: encode current input, decode prediction, compute error.
        retinal_input: (N,) current retinal ON rates
        motor_cmd: (2,) [turn, speed]
        Returns: prediction, prediction_error, surprise
        """
        x = self._pool_input(retinal_input)

        # Compute prediction error against previous prediction
        error = x - self.prediction
        self.pred_error.copy_(error)
        surprise = float((error ** 2).mean())

        # Update weights with prediction error (anti-Hebbian)
        if surprise > 0.01:
            # Decoder: reduce error by adjusting output weights
            dW_out = -self.lr * torch.outer(error, self.dec_rate)
            self.W_dec_out.weight.data.add_(dW_out)
            self.W_dec_out.weight.data.clamp_(-2.0, 2.0)

            # Encoder: adjust to better represent predictive features
            enc_error = self.W_dec_out.weight.data.T @ error
            dW_enc = self.lr * torch.outer(enc_error[:self.n_hidden], x)
            self.W_in_enc.weight.data.add_(dW_enc)
            self.W_in_enc.weight.data.clamp_(-2.0, 2.0)

        # Encode current → decode next prediction
        latent = self.encode(x)
        prediction = self.decode(latent, motor_cmd)

        # Store for next step comparison
        self.prev_retinal.copy_(x)

        return {
            'prediction': prediction,
            'pred_error': self.pred_error,
            'surprise': surprise,
            'latent': latent,
            'enc_sparsity': float((self.enc_rate > 0.01).float().mean()),
        }

    def get_uncertainty(self) -> float:
        """Prediction uncertainty for EFE epistemic term."""
        return float((self.pred_error ** 2).mean())

    def reset(self):
        self.enc_hidden.reset()
        self.dec_hidden.reset()
        self.latent.zero_()
        self.prediction.zero_()
        self.pred_error.zero_()
        self.enc_rate.zero_()
        self.dec_rate.zero_()
        self.prev_retinal.zero_()
