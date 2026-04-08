"""
Spiking insular cortex — Active Inference / Free Energy Principle (FEP).

Zebrafish homologue: dorsal pallium Dp (insula-like function).

FEP formulation:
  Belief update:  mu_x  += alpha * pi_x * epsilon_x
  Pred. error:    epsilon_x = actual_x - mu_x
  Precision:      pi_x  (inverse expected variance per channel)
  Spiking drive:  I_x = |epsilon_x| * gain   (encodes PE magnitude)
  Valence:        -sum_x(sign_x * pi_x * epsilon_x)  (signed surprise)
  Arousal:        sum_x(pi_x * epsilon_x^2)           (total surprise / F)
  EFE bias:       G(pi) = sum_x(pi_x*(2*eps_x*delta_x + delta_x^2))
"""
import math
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer

class InteroceptiveGenerativeModel:
    """
    Maintains Bayesian beliefs (mu) and precisions (pi) over three
    interoceptive channels.  Performs one gradient-descent step on
    variational free energy each call.
    """
    # Allostatic setpoints (prior means)
    SETPOINTS = {'hunger': 0.25, 'fatigue': 0.20, 'stress': 0.10}

    def __init__(self, alpha: float = 0.08):
        self.alpha = alpha           # belief-update learning rate

        # Predicted bodily states (beliefs, initialised at setpoints)
        self.mu_hunger  = self.SETPOINTS['hunger']
        self.mu_fatigue = self.SETPOINTS['fatigue']
        self.mu_stress  = self.SETPOINTS['stress']

        # Precision (confidence in prediction; higher = more sensitive to PE)
        self.pi_hunger  = 2.0
        self.pi_fatigue = 1.5
        self.pi_stress  = 2.5

        # Last computed prediction errors (signed)
        self.eps_hunger  = 0.0
        self.eps_fatigue = 0.0
        self.eps_stress  = 0.0

    def update(self, actual_hunger: float, actual_fatigue: float,
               actual_stress: float) -> tuple[float, float, float]:
        """
        Gradient descent on variational free energy (= prediction error).
        Returns signed prediction errors (epsilon) for each channel.
        """
        self.eps_hunger  = actual_hunger  - self.mu_hunger
        self.eps_fatigue = actual_fatigue - self.mu_fatigue
        self.eps_stress  = actual_stress  - self.mu_stress

        # Belief update (ascending prediction error → update mu)
        self.mu_hunger  += self.alpha * self.pi_hunger  * self.eps_hunger
        self.mu_fatigue += self.alpha * self.pi_fatigue * self.eps_fatigue
        self.mu_stress  += self.alpha * self.pi_stress  * self.eps_stress

        # Clamp beliefs to valid range
        self.mu_hunger  = max(0.0, min(1.0, self.mu_hunger))
        self.mu_fatigue = max(0.0, min(1.0, self.mu_fatigue))
        self.mu_stress  = max(0.0, min(1.0, self.mu_stress))

        return self.eps_hunger, self.eps_fatigue, self.eps_stress

    def reset(self):
        self.mu_hunger  = self.SETPOINTS['hunger']
        self.mu_fatigue = self.SETPOINTS['fatigue']
        self.mu_stress  = self.SETPOINTS['stress']
        self.eps_hunger = self.eps_fatigue = self.eps_stress = 0.0


# Expected delta-prediction-error per policy per channel [hunger, fatigue, stress]
# Negative = expected reduction in PE (good); positive = expected increase (bad)
_POLICY_EFE = {
    'forage':  (-0.30,  0.10,  0.05),   # reduces hunger PE; costs fatigue
    'flee':    ( 0.05,  0.15, -0.50),   # reduces stress PE; costs fatigue
    'explore': (-0.05, -0.10,  0.00),   # mild fatigue recovery; neutral hunger
    'social':  ( 0.00, -0.05, -0.15),   # reduces stress slightly via safety
}

# Signs for valence: +1 → positive PE means worse than expected (negative valence)
_VALENCE_SIGN = {'hunger': 1.0, 'fatigue': 0.6, 'stress': 1.0}


class SpikingInsularCortex(nn.Module):
    def __init__(self, n_per_channel: int = 10, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_ch = n_per_channel

        # Three spiking populations encode interoceptive prediction error
        self.hunger_pop = IzhikevichLayer(n_per_channel, 'RS', device)
        self.fatigue_pop = IzhikevichLayer(n_per_channel, 'RS', device)
        self.stress_pop  = IzhikevichLayer(n_per_channel, 'RS', device)

        # Subthreshold resting state — require PE-driven input to fire
        self.hunger_pop.i_tonic.fill_(-3.0)
        self.fatigue_pop.i_tonic.fill_(-3.0)
        self.stress_pop.i_tonic.fill_(-3.0)

        # Spike-rate buffers (public — brain_v2 may read them)
        self.register_buffer('hunger_rate',
                             torch.zeros(n_per_channel, device=device))
        self.register_buffer('fatigue_rate',
                             torch.zeros(n_per_channel, device=device))
        self.register_buffer('stress_rate',
                             torch.zeros(n_per_channel, device=device))

        # Generative model
        self.gen_model = InteroceptiveGenerativeModel()

        # Heart rate model
        self.heart_rate  = 2.0    # Hz (zebrafish baseline)
        self.heart_phase = 0.0
        self.heartbeat   = False
        self._step_count = 0

        # Emotional state attributes (read directly by brain_v2)
        self.valence = 0.0
        self.arousal = 0.0

    @torch.no_grad()
    def forward(self, energy: float, stress: float, fatigue: float,
                reward: float = 0.0, threat_acute: bool = False) -> dict:
        """Update interoceptive state via Active Inference (energy 0-100, stress/fatigue 0-1)."""
        self._step_count += 1

        # Convert energy to hunger (0 = full, 1 = starving)
        actual_hunger  = max(0.0, 1.0 - energy / 100.0)
        actual_fatigue = float(fatigue)
        actual_stress  = float(stress)

        # ── 1. Update generative model → compute prediction errors ────────────
        eps_h, eps_f, eps_s = self.gen_model.update(
            actual_hunger, actual_fatigue, actual_stress
        )
        pi_h = self.gen_model.pi_hunger
        pi_f = self.gen_model.pi_fatigue
        pi_s = self.gen_model.pi_stress

        # ── 2. Drive spiking populations with |PE| * gain ─────────────────────
        gain = 15.0
        I_h = torch.full((self.n_ch,), abs(eps_h) * gain, device=self.device)
        I_f = torch.full((self.n_ch,), abs(eps_f) * gain, device=self.device)
        I_s = torch.full((self.n_ch,), abs(eps_s) * gain, device=self.device)

        h_spikes = torch.zeros(self.n_ch, device=self.device)
        f_spikes = torch.zeros(self.n_ch, device=self.device)
        s_spikes = torch.zeros(self.n_ch, device=self.device)

        for _ in range(15):
            noise = lambda: torch.randn(self.n_ch, device=self.device) * 0.5
            h_spikes += self.hunger_pop(I_h + noise())
            f_spikes += self.fatigue_pop(I_f + noise())
            s_spikes += self.stress_pop(I_s + noise())

        self.hunger_rate.copy_(self.hunger_pop.rate)
        self.fatigue_rate.copy_(self.fatigue_pop.rate)
        self.stress_rate.copy_(self.stress_pop.rate)

        h_mean = float(self.hunger_rate.mean())
        f_mean = float(self.fatigue_rate.mean())
        s_mean = float(self.stress_rate.mean())

        # ── 3. Valence = signed precision-weighted PE ─────────────────────────
        # Positive PE (more stressed/hungry than predicted) → negative valence
        raw_valence = -(
            _VALENCE_SIGN['hunger']  * pi_h * eps_h +
            _VALENCE_SIGN['fatigue'] * pi_f * eps_f +
            _VALENCE_SIGN['stress']  * pi_s * eps_s
        )
        # Reward shifts valence upward
        raw_valence += reward * 1.5
        # Exponential smoothing
        self.valence = 0.85 * self.valence + 0.15 * raw_valence
        self.valence = max(-1.0, min(1.0, self.valence))

        # ── 4. Arousal = precision-weighted sum of squared PEs (FEP surprise) ──
        raw_arousal = pi_h * eps_h**2 + pi_f * eps_f**2 + pi_s * eps_s**2
        self.arousal = min(1.0, 0.9 * self.arousal + 0.1 * raw_arousal)

        # ── 5. Heart rate: driven by arousal (precision-weighted PE magnitude) ─
        hr_target = 2.0 + 4.0 * self.arousal + 1.0 * actual_stress
        if threat_acute and actual_stress > 0.5:
            hr_target = max(1.0, hr_target * 0.5)   # bradycardia (freezing)
        if actual_fatigue > 0.3 and actual_stress > 0.3:
            hr_target = min(8.0, hr_target * 1.3)   # tachycardia (flee)

        self.heart_rate = 0.9 * self.heart_rate + 0.1 * hr_target
        self.heart_rate = max(0.5, min(8.0, self.heart_rate))

        dt_step = 0.05  # ~50 ms per behavioural step
        self.heart_phase += self.heart_rate * dt_step * 2 * math.pi
        self.heartbeat = math.sin(self.heart_phase) > 0.9

        return {
            'hunger_rate': h_mean,
            'fatigue_rate': f_mean,
            'stress_rate': s_mean,
            'valence': self.valence,
            'arousal': self.arousal,
            'heart_rate': self.heart_rate,
            'heartbeat': self.heartbeat,
        }

    def get_allostatic_bias(self) -> dict:
        """EFE bias per policy: G(pi) ≈ sum_x(pi_x*(2*eps_x*delta_x + delta_x^2)), negated."""
        eps   = (self.gen_model.eps_hunger,
                 self.gen_model.eps_fatigue,
                 self.gen_model.eps_stress)
        pi    = (self.gen_model.pi_hunger,
                 self.gen_model.pi_fatigue,
                 self.gen_model.pi_stress)
        scale = 0.15

        def efe(policy_key: str) -> float:
            delta = _POLICY_EFE[policy_key]
            # Expected change in free energy (negative = expected reduction)
            g = sum(pi[i] * (2.0 * eps[i] * delta[i] + delta[i]**2)
                    for i in range(3))
            return -g * scale    # negate: lower free energy is better → positive bias

        return {
            'forage_bias':  efe('forage'),
            'flee_bias':    efe('flee'),
            'explore_bias': efe('explore'),
            'social_bias':  efe('social'),
        }

    def reset(self):
        self.hunger_pop.reset()
        self.fatigue_pop.reset()
        self.stress_pop.reset()
        self.hunger_rate.zero_()
        self.fatigue_rate.zero_()
        self.stress_rate.zero_()
        self.gen_model.reset()
        self.heart_rate  = 2.0
        self.heart_phase = 0.0
        self.heartbeat   = False
        self.valence     = 0.0
        self.arousal     = 0.0
        self._step_count = 0
