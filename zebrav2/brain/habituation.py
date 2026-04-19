"""
Short-term synaptic depression: sensory habituation.

Biology:
  Repeated identical stimuli → calcium-dependent vesicle depletion
  at RGC→tectum synapses → reduced postsynaptic response.

  Distinct from adaptation (which is receptor-level):
    - Habituation is synapse-specific (only the repeated input weakens)
    - Novel stimuli are NOT affected (dishabituation)
    - Recovery occurs with ~50-100 step time constant

  Zebrafish larvae habituate to repeated visual stimuli within
  10-30 presentations (Wolman et al. 2011, Roberts et al. 2011).

Model:
  Per-synapse depletion factor d ∈ [d_min, 1.0]:
    d_new = d * (1 - tau_depress * active) + tau_recover * (1 - d)
  Output current: I_out = I_in * d

  Dishabituation: when input pattern cosine distance > threshold,
  partially restore depleted synapses.

References:
  - Wolman et al. (2011) "Transient axonal glycoprotein-1 promotes
    habituation" Neuron
  - Roberts et al. (2011) "Habituation of the C-start response"
    J Neurosci (zebrafish startle habituation)
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE


class SynapticDepression(nn.Module):
    def __init__(self, n_synapses: int,
                 tau_depress: float = 0.02,
                 tau_recover: float = 0.01,
                 d_min: float = 0.3,
                 dishab_threshold: float = 0.3,
                 device=DEVICE):
        super().__init__()
        self.device = device
        self.n_synapses = n_synapses
        self.tau_depress = tau_depress
        self.tau_recover = tau_recover
        self.d_min = d_min
        self.dishab_threshold = dishab_threshold

        # Per-synapse depletion factor: 1.0 = full strength, d_min = depleted
        self.register_buffer('depletion',
                             torch.ones(n_synapses, device=device))
        # Previous input for novelty detection
        self.register_buffer('prev_input',
                             torch.zeros(n_synapses, device=device))

    def forward(self, I_ext: torch.Tensor) -> torch.Tensor:
        """Apply synaptic depression and update state.

        Parameters
        ----------
        I_ext : Tensor (n_synapses,) — presynaptic input current

        Returns
        -------
        Tensor (n_synapses,) — depressed output current
        """
        # Detect active synapses (above noise floor)
        active = (I_ext.abs() > 0.1).float()

        # Dishabituation check: is input pattern novel?
        inp_norm = I_ext.norm() + 1e-8
        prev_norm = self.prev_input.norm() + 1e-8
        if inp_norm > 0.5 and prev_norm > 0.5:
            cosine = torch.dot(I_ext, self.prev_input) / (inp_norm * prev_norm)
            if cosine < (1.0 - self.dishab_threshold):
                # Novel input — partially restore depleted synapses
                changed = (I_ext - self.prev_input).abs() > 0.5
                self.depletion[changed] = torch.clamp(
                    self.depletion[changed] + 0.3, max=1.0)

        # Update depletion: depress active synapses, recover inactive ones
        self.depletion.mul_(1.0 - self.tau_depress * active)
        self.depletion.add_(self.tau_recover * (1.0 - self.depletion))
        self.depletion.clamp_(self.d_min, 1.0)

        # Store for next step's novelty detection
        self.prev_input.copy_(I_ext.detach())

        return I_ext * self.depletion

    def get_habituation_level(self) -> float:
        """Mean depletion factor (1.0 = fresh, d_min = fully habituated)."""
        return float(self.depletion.mean())

    def reset(self):
        self.depletion.fill_(1.0)
        self.prev_input.zero_()
