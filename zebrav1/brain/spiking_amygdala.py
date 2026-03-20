"""
Spiking Amygdala — fear circuit with lateral and central nuclei.

Replaces the numpy leaky integrator with a spiking fear circuit:
  - Lateral amygdala (LA, 5 neurons): receives sensory threat input
  - Central amygdala (CeA, 5 neurons): drives fear output
  - Intercalated cells (ITC, 5 neurons): inhibitory gate (extinction)
  - Fear output: CeA firing rate → threat_arousal

Episodic fear conditioning: near-death events potentiate LA→CeA
synapses (LTP), creating lasting fear memory.

Neuroscience: zebrafish pallial circuits homologous to mammalian
amygdala mediate fear conditioning (Lal et al. 2018). Dm (medial
dorsal telencephalon) processes aversive stimuli.

Torch-based — spiking LIF neurons.
"""
import math
import torch
import torch.nn as nn
import numpy as np


class SpikingAmygdala(nn.Module):
    """Spiking fear circuit with episodic conditioning.

    Args:
        n_la: int — lateral amygdala neurons (sensory input)
        n_cea: int — central amygdala neurons (fear output)
        n_itc: int — intercalated cells (extinction gate)
        tau: float — membrane time constant
        device: str
    """

    def __init__(self, n_la=5, n_cea=5, n_itc=5,
                 tau=0.85, v_thresh=0.4,
                 proximity_range=200.0, retinal_gain=0.08,
                 device="cpu"):
        super().__init__()
        self.n_total = n_la + n_cea + n_itc
        self.tau = tau
        self.v_thresh = v_thresh
        self.proximity_range = proximity_range
        self.retinal_gain = retinal_gain
        self.device = device

        # Sensory input → LA
        self.W_sensory = nn.Linear(4, n_la, bias=False)  # [enemy_px, proximity, stress, gaze]

        # LA → CeA (fear pathway, potentiated by conditioning)
        self.W_la_cea = nn.Parameter(torch.ones(n_la, n_cea) * 0.5)

        # ITC → CeA (inhibitory, extinction)
        self.w_itc_cea = -0.3

        # LA → ITC (feeds extinction circuit)
        self.w_la_itc = 0.2

        # LIF state
        self.v_la = torch.zeros(n_la, device=device)
        self.v_cea = torch.zeros(n_cea, device=device)
        self.v_itc = torch.zeros(n_itc, device=device)

        # Episodic fear memory
        self.fear_baseline = 0.0
        self.near_death_count = 0

        # Output
        self.threat_arousal = 0.0

        self.to(device)

    @torch.no_grad()
    def step(self, enemy_pixels, pred_dist, stress, pred_facing_score=0.0):
        """Update spiking amygdala.

        Returns:
            threat_arousal: float [0, 1]
        """
        # Sensory features
        retinal = min(1.0, enemy_pixels * self.retinal_gain)
        proximity = max(0.0, 1.0 - pred_dist / self.proximity_range)
        gaze = pred_facing_score * proximity

        x = torch.tensor([retinal, proximity, stress, gaze],
                          dtype=torch.float32, device=self.device)

        # Lateral amygdala
        I_la = self.W_sensory(x)
        # Fear baseline adds persistent excitation after trauma
        I_la += self.fear_baseline * proximity
        noise = torch.randn_like(self.v_la) * 0.05
        self.v_la = self.tau * self.v_la + (1 - self.tau) * (I_la + noise)
        la_spikes = (self.v_la >= self.v_thresh).float()
        self.v_la *= (1 - la_spikes)

        # Intercalated cells (extinction gate)
        I_itc = la_spikes.mean() * self.w_la_itc
        self.v_itc = self.tau * self.v_itc + (1 - self.tau) * I_itc
        itc_spikes = (self.v_itc >= self.v_thresh).float()
        self.v_itc *= (1 - itc_spikes)

        # Central amygdala (fear output)
        I_cea = la_spikes @ self.W_la_cea  # LA → CeA (potentiated)
        I_cea += itc_spikes.mean() * self.w_itc_cea  # ITC inhibition
        self.v_cea = self.tau * self.v_cea + (1 - self.tau) * I_cea
        cea_spikes = (self.v_cea >= self.v_thresh).float()
        self.v_cea *= (1 - cea_spikes)

        cea_rate = float(cea_spikes.mean())

        # Episodic fear conditioning: near-death potentiates LA→CeA
        if proximity > 0.9 and pred_facing_score > 0.5:
            self.fear_baseline = min(0.3, self.fear_baseline + 0.03)
            self.near_death_count += 1
            # LTP: strengthen LA→CeA weights
            self.W_la_cea.data += 0.05 * torch.outer(la_spikes, cea_spikes)
            self.W_la_cea.data.clamp_(0.1, 2.0)

        # Threat arousal: smoothed CeA firing rate
        raw = max(cea_rate, retinal * 0.5, proximity * 0.3)
        if raw > self.threat_arousal:
            self.threat_arousal = 0.4 * self.threat_arousal + 0.6 * raw
        else:
            self.threat_arousal = max(
                self.fear_baseline * 0.3,
                self.threat_arousal * 0.75)

        return self.threat_arousal

    def get_diagnostics(self):
        return {
            "threat_arousal": self.threat_arousal,
            "fear_baseline": self.fear_baseline,
            "near_death_count": self.near_death_count,
            "la_cea_weight_mean": float(self.W_la_cea.data.mean()),
        }

    def reset(self):
        self.v_la.zero_()
        self.v_cea.zero_()
        self.v_itc.zero_()
        self.threat_arousal = 0.0

    def reset_full(self):
        self.reset()
        self.fear_baseline = 0.0
        self.near_death_count = 0
        self.W_la_cea.data.fill_(0.5)
