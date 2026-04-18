"""
Spiking pretectum: optokinetic response (OKR) and image stabilization.

Zebrafish pretectum anatomy:
  Area pretectalis (AF7/AF9): direction-selective (DS) neurons that
  detect whole-field retinal slip (optic flow).
    - Receives DS retinal ganglion cell (RGC) input contralaterally
    - Computes slip velocity: mismatch between eye velocity and world motion
    - Drives compensatory eye movements (OKR) to stabilize retinal image
    - Projects to tectum (spatial attention), thalamus (motion signals),
      and oculomotor nuclei (gaze correction)

  Bilateral organization: 30 neurons per hemisphere (60 total).
    Left pretectum ← right retina DS cells  (contralateral)
    Right pretectum ← left retina DS cells

  Direction selectivity index (DSI): measures tuning sharpness.
    DSI = (R_pref - R_null) / (R_pref + R_null)
    Healthy zebrafish DSI ≈ 0.6–0.9 (Kubo et al. 2014)

  Temporal low-pass filter (tau=0.8): smooths OKR to match slow
  compensatory eye dynamics (~1-5 Hz in larvae).

References:
  - Kubo et al. (2014) "Functional architecture of an optic flow-responsive
    area that drives horizontal eye movements in zebrafish" Neuron
  - Naumann et al. (2016) "From whole-brain data to functional circuit models"
    Current Opinion in Neurobiology
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingPretectum(nn.Module):
    def __init__(self, n_per_side=30, tau_okr=0.8, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_per_side = n_per_side
        self.n_total = n_per_side * 2
        self.tau_okr = tau_okr

        # Bilateral DS populations (EI not needed — pretectal neurons are
        # predominantly excitatory in zebrafish, inhibition is inter-hemispheric)
        self.L = IzhikevichLayer(n_per_side, 'RS', device)  # left pretectum
        self.R = IzhikevichLayer(n_per_side, 'RS', device)  # right pretectum
        self.L.i_tonic.fill_(-1.5)
        self.R.i_tonic.fill_(-1.5)

        # State buffers
        self.register_buffer('rate_L', torch.zeros(n_per_side, device=device))
        self.register_buffer('rate_R', torch.zeros(n_per_side, device=device))
        self.register_buffer('okr_velocity', torch.tensor(0.0, device=device))

        # OKR temporal filter state
        self._okr_filtered = 0.0
        # Direction selectivity tracking
        self._dsi = 0.0

    @torch.no_grad()
    def forward(self, ds_L: float, ds_R: float,
                eye_velocity: float = 0.0) -> dict:
        """
        Compute pretectal response to retinal slip.

        Parameters
        ----------
        ds_L : float
            Direction-selective RGC rate from LEFT retina.
            Feeds RIGHT pretectum (contralateral crossing at optic chiasm).
        ds_R : float
            Direction-selective RGC rate from RIGHT retina.
            Feeds LEFT pretectum (contralateral crossing at optic chiasm).
        eye_velocity : float
            Current gaze/eye velocity (from saccade module).
            Subtracted to compute retinal slip (efference copy cancellation).

        Returns
        -------
        dict with:
            'okr_velocity'  : float  — compensatory eye velocity command
            'rate_L'        : float  — mean left pretectum firing rate
            'rate_R'        : float  — mean right pretectum firing rate
            'retinal_slip'  : float  — detected image motion (L-R asymmetry)
            'dsi'           : float  — direction selectivity index
        """
        # Retinal slip = world motion - eye motion (efference copy cancellation)
        # Positive slip = image moving rightward on retina = world rotating left
        retinal_slip = (ds_R - ds_L) - eye_velocity * 0.5

        # Drive pretectal populations (contralateral: R retina → L pretectum)
        # Both populations receive baseline + slip-proportional drive
        I_L = torch.full((self.n_per_side,), 3.0 + ds_R * 15.0,
                         device=self.device)
        I_R = torch.full((self.n_per_side,), 3.0 + ds_L * 15.0,
                         device=self.device)

        # Run spiking dynamics (20 substeps, same as habenula)
        for _ in range(20):
            noise = torch.randn(self.n_per_side, device=self.device) * 0.5
            self.L(I_L + noise)
            noise = torch.randn(self.n_per_side, device=self.device) * 0.5
            self.R(I_R + noise)

        self.rate_L.copy_(self.L.rate)
        self.rate_R.copy_(self.R.rate)
        rate_L_mean = float(self.rate_L.mean())
        rate_R_mean = float(self.rate_R.mean())

        # OKR velocity: asymmetry between hemispheres → compensatory drive
        # L pretectum active (= right visual motion) → compensate leftward
        raw_okr = (rate_L_mean - rate_R_mean) * 2.0

        # Temporal low-pass filter (smooth pursuit dynamics)
        self._okr_filtered = (self.tau_okr * self._okr_filtered
                              + (1.0 - self.tau_okr) * raw_okr)
        okr_vel = max(-1.0, min(1.0, self._okr_filtered))
        self.okr_velocity.fill_(okr_vel)

        # Direction selectivity index
        r_pref = max(rate_L_mean, rate_R_mean)
        r_null = min(rate_L_mean, rate_R_mean)
        self._dsi = ((r_pref - r_null) / (r_pref + r_null + 1e-8))

        return {
            'okr_velocity': okr_vel,
            'rate_L': rate_L_mean,
            'rate_R': rate_R_mean,
            'retinal_slip': float(retinal_slip),
            'dsi': self._dsi,
        }

    def reset(self):
        self.L.reset()
        self.R.reset()
        self.rate_L.zero_()
        self.rate_R.zero_()
        self.okr_velocity.zero_()
        self._okr_filtered = 0.0
        self._dsi = 0.0
