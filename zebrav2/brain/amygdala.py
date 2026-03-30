"""
Spiking amygdala — Izhikevich E/I fear circuit with episodic LTP.

Nuclei (zebrafish Dm homologue):
  LA  (Lateral, 20 neurons):  sensory threat input, RS excitatory
  CeA (Central, 20 neurons):  fear output, IB bursting
  ITC (Intercalated, 10 neurons): extinction gate, FS inhibitory

Episodic conditioning: near-death events potentiate LA→CeA via Hebbian LTP.
Fear baseline persists within episode, creating hypervigilance after trauma.
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE, SUBSTEPS
from zebrav2.brain.neurons import IzhikevichLayer


class SpikingAmygdalaV2(nn.Module):
    def __init__(self, n_la=20, n_cea=20, n_itc=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_la = n_la
        self.n_cea = n_cea
        self.n_itc = n_itc

        # Neuron populations
        self.LA = IzhikevichLayer(n_la, 'RS', device)
        self.CeA = IzhikevichLayer(n_cea, 'IB', device)
        self.ITC = IzhikevichLayer(n_itc, 'FS', device)

        # Sensory → LA projection: [enemy_px, proximity, stress, gaze]
        self.W_sensory = nn.Linear(4, n_la, bias=False)
        nn.init.xavier_uniform_(self.W_sensory.weight, gain=0.5)
        self.W_sensory.to(device)

        # LA → CeA (excitatory, potentiated by LTP)
        self.register_buffer('W_la_cea',
            torch.ones(n_cea, n_la, device=device) * 0.5)

        # LA → ITC (weak excitatory)
        self.register_buffer('W_la_itc',
            torch.ones(n_itc, n_la, device=device) * 0.2)

        # ITC → CeA (inhibitory extinction gate)
        self.register_buffer('W_itc_cea',
            torch.ones(n_cea, n_itc, device=device) * -0.3)

        # State
        self.threat_arousal = 0.0
        self.fear_baseline = 0.0
        self.near_death_count = 0

        # Sensory params
        self.retinal_gain = 0.08
        self.proximity_range = 200.0

    @torch.no_grad()
    def forward(self, enemy_pixels: float, pred_dist: float,
                stress: float = 0.0, pred_facing: float = 0.0) -> float:
        """
        Run spiking amygdala for one behavioral step.
        Returns threat_arousal [0, 1].
        """
        retinal = min(1.0, enemy_pixels * self.retinal_gain)
        proximity = max(0.0, 1.0 - pred_dist / self.proximity_range)
        gaze = pred_facing * proximity

        x = torch.tensor([retinal, proximity, stress, gaze],
                         dtype=torch.float32, device=self.device)

        # Sensory drive to LA
        I_la_base = self.W_sensory(x.unsqueeze(0)).squeeze(0) * 20.0
        I_la_base += self.fear_baseline * proximity * 10.0

        # Run substeps
        la_spike_acc = torch.zeros(self.n_la, device=self.device)
        cea_spike_acc = torch.zeros(self.n_cea, device=self.device)
        itc_spike_acc = torch.zeros(self.n_itc, device=self.device)

        for _ in range(20):  # reduced substeps
            # LA: sensory-driven
            la_sp = self.LA(I_la_base + torch.randn(self.n_la, device=self.device) * 1.0)
            la_spike_acc += la_sp

            # ITC: extinction gate (LA → ITC)
            I_itc = self.W_la_itc @ self.LA.rate * 15.0
            itc_sp = self.ITC(I_itc)
            itc_spike_acc += itc_sp

            # CeA: fear output (LA excitation - ITC inhibition)
            I_cea = (self.W_la_cea @ self.LA.rate * 20.0
                     + self.W_itc_cea @ self.ITC.rate * 15.0)
            cea_sp = self.CeA(I_cea)
            cea_spike_acc += cea_sp

        cea_rate = float(cea_spike_acc.mean()) / SUBSTEPS

        # Episodic LTP: near-death potentiates LA→CeA
        if proximity > 0.9 and pred_facing > 0.5:
            self.fear_baseline = min(0.3, self.fear_baseline + 0.03)
            self.near_death_count += 1
            # Hebbian LTP on LA→CeA
            la_active = (la_spike_acc > 0).float()
            cea_active = (cea_spike_acc > 0).float()
            self.W_la_cea += 0.05 * torch.outer(cea_active, la_active)
            self.W_la_cea.clamp_(0.1, 2.0)

        # Smoothed threat arousal
        raw = max(cea_rate, retinal * 0.5, proximity * 0.3)
        if raw > self.threat_arousal:
            self.threat_arousal = 0.4 * self.threat_arousal + 0.6 * raw
        else:
            self.threat_arousal = max(
                self.fear_baseline * 0.3,
                self.threat_arousal * 0.75)

        return self.threat_arousal

    def reset(self):
        self.LA.reset()
        self.CeA.reset()
        self.ITC.reset()
        self.threat_arousal = 0.0

    def reset_full(self):
        self.reset()
        self.fear_baseline = 0.0
        self.near_death_count = 0
        self.W_la_cea.fill_(0.5)
