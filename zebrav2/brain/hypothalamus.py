"""
Spiking hypothalamus: master homeostatic integrator.

Zebrafish hypothalamus (homologous to mammalian hypothalamus):
  - Preoptic area (POA): thermoregulation, social/reproductive behavior
  - Lateral hypothalamus (LH): feeding drive, arousal, orexin/hypocretin
  - Ventromedial hypothalamus (VMH): satiety, defensive aggression
  - Paraventricular nucleus (PVN): CRH release → HPI axis, oxytocin/vasopressin
  - Suprachiasmatic-like (SCN): circadian entrainment relay

The hypothalamus integrates:
  - Energy state (hunger/satiety) → feeding drive
  - Threat state (amygdala, cortisol) → HPI activation
  - Temperature → thermoregulatory behavior
  - Circadian phase → sleep/wake modulation
  - Social signals (oxytocin) → affiliative drive

Free Energy Principle:
  Generative model predicts bodily setpoints.  Allostatic error
  (deviation from setpoint) drives autonomic + behavioral correction.
  Precision weights reflect metabolic urgency.

Architecture:
  10 RS neurons: 2 LH (feeding), 2 VMH (satiety), 2 PVN (stress),
                 2 POA (thermoreg), 2 SCN (circadian relay)
  + 5-channel TwoCompColumn for homeostatic prediction
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingHypothalamus(nn.Module):
    def __init__(self, n_neurons=10, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-0.5)
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 5 homeostatic channels
        self.pc = TwoCompColumn(n_channels=5, n_per_ch=4, substeps=8, device=device)

        # Homeostatic setpoints
        self.hunger_setpoint = 0.25    # desired hunger level (low = fed)
        self.stress_setpoint = 0.10    # desired stress level (low = calm)
        self.temperature_setpoint = 26.0  # zebrafish comfort (°C)
        self.fatigue_setpoint = 0.20   # desired fatigue (low = rested)
        self.social_setpoint = 0.30    # desired social proximity

        # State variables
        self.feeding_drive = 0.0       # LH output: motivational drive to eat
        self.satiety = 0.0             # VMH output: inhibits feeding
        self.crh_release = 0.0         # PVN output: drives HPI cortisol
        self.thermoreg_drive = 0.0     # POA output: behavioral thermoregulation
        self.circadian_gate = 1.0      # SCN output: activity gating
        self.orexin = 0.0              # LH neuropeptide: arousal + feeding
        self.autonomic_output = 0.0    # integrated autonomic drive

        # Allostatic errors
        self.hunger_error = 0.0
        self.stress_error = 0.0
        self.thermal_error = 0.0
        self.fatigue_error = 0.0
        self.social_error = 0.0

        # FEP state
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

    @torch.no_grad()
    def forward(self, energy: float = 50.0, hunger: float = 0.0,
                stress: float = 0.0, fatigue: float = 0.0,
                temperature: float = 26.0, amygdala_alpha: float = 0.0,
                cortisol: float = 0.0, melatonin: float = 0.0,
                social_proximity: float = 0.0,
                food_odor: float = 0.0) -> dict:
        """
        energy: current energy level [0, 100]
        hunger: allostatic hunger [0, 1]
        stress: allostatic stress [0, 1]
        fatigue: allostatic fatigue [0, 1]
        temperature: water temperature (°C)
        amygdala_alpha: fear signal [0, 1]
        cortisol: HPI cortisol level [0, 1]
        melatonin: circadian melatonin [0, 1]
        social_proximity: proximity to conspecifics [0, 1]
        food_odor: olfactory food signal [0, 1]
        """
        # --- Allostatic error computation ---
        self.hunger_error = hunger - self.hunger_setpoint
        self.stress_error = stress - self.stress_setpoint
        self.thermal_error = abs(temperature - self.temperature_setpoint) / 10.0
        self.fatigue_error = fatigue - self.fatigue_setpoint
        self.social_error = self.social_setpoint - social_proximity

        # --- Lateral hypothalamus: feeding drive ---
        # Hunger + food odor → feeding motivation; inhibited by satiety, stress
        raw_feeding = max(0, self.hunger_error) * 2.0 + food_odor * 0.5
        stress_inhibit = 1.0 - min(1.0, stress * 0.7)  # stress suppresses appetite
        self.feeding_drive = min(1.0, raw_feeding * stress_inhibit)

        # Orexin: arousal neuropeptide, high when hungry + awake
        self.orexin = self.feeding_drive * (1.0 - melatonin) * 0.8
        self.orexin = min(1.0, self.orexin)

        # --- Ventromedial hypothalamus: satiety ---
        # Energy > 70 → satiety signal; recently eaten → satiety
        self.satiety = min(1.0, max(0, energy - 50) / 50.0)

        # --- Paraventricular nucleus: CRH → HPI axis ---
        # Amygdala fear + allostatic stress → CRH release
        self.crh_release = min(1.0,
                               amygdala_alpha * 0.6 + max(0, self.stress_error) * 0.4)
        # Cortisol negative feedback: high cortisol suppresses CRH
        self.crh_release *= max(0.2, 1.0 - cortisol * 0.6)

        # --- Preoptic area: thermoregulation ---
        # Behavioral thermoregulation drive (seek warm/cool areas)
        self.thermoreg_drive = min(1.0, self.thermal_error)

        # --- SCN relay: circadian gating ---
        # Activity gate: high during day (low melatonin), low at night
        self.circadian_gate = 0.3 + 0.7 * (1.0 - melatonin)

        # --- Integrated autonomic output ---
        self.autonomic_output = (
            self.feeding_drive * 0.3 +
            self.crh_release * 0.3 +
            self.thermoreg_drive * 0.2 +
            (1.0 - self.circadian_gate) * 0.2
        )

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        I[0] = self.feeding_drive * 12.0    # LH feeding +
        I[1] = self.orexin * 10.0           # LH orexin
        I[2] = self.satiety * 10.0          # VMH satiety +
        I[3] = (1.0 - self.satiety) * 8.0   # VMH hunger signal
        I[4] = self.crh_release * 15.0      # PVN CRH +
        I[5] = self.crh_release * 10.0      # PVN vasopressin
        I[6] = self.thermoreg_drive * 12.0  # POA warm +
        I[7] = self.thermoreg_drive * 8.0   # POA cool
        I[8] = self.circadian_gate * 10.0   # SCN day
        I[9] = (1.0 - self.circadian_gate) * 10.0  # SCN night

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction ---
        sensory = torch.tensor([
            hunger, stress, self.thermal_error, fatigue, social_proximity
        ], device=self.device, dtype=torch.float32)
        prediction = torch.tensor([
            self.hunger_setpoint, self.stress_setpoint,
            0.0,  # thermal setpoint error = 0
            self.fatigue_setpoint, self.social_setpoint
        ], device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'feeding_drive': self.feeding_drive,
            'satiety': self.satiety,
            'crh_release': self.crh_release,
            'thermoreg_drive': self.thermoreg_drive,
            'circadian_gate': self.circadian_gate,
            'orexin': self.orexin,
            'autonomic_output': self.autonomic_output,
            'hunger_error': self.hunger_error,
            'stress_error': self.stress_error,
            'thermal_error': self.thermal_error,
            'fatigue_error': self.fatigue_error,
            'social_error': self.social_error,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def get_efe_bias(self):
        """Hypothalamic bias on goal selection (lower = more attractive)."""
        return {
            'forage': -self.feeding_drive * 0.3,
            'flee': -self.crh_release * 0.2,
            'explore': -self.thermoreg_drive * 0.1,
            'social': -max(0, self.social_error) * 0.15,
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.feeding_drive = 0.0
        self.satiety = 0.0
        self.crh_release = 0.0
        self.thermoreg_drive = 0.0
        self.circadian_gate = 1.0
        self.orexin = 0.0
        self.autonomic_output = 0.0
        self.hunger_error = 0.0
        self.stress_error = 0.0
        self.thermal_error = 0.0
        self.fatigue_error = 0.0
        self.social_error = 0.0
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
