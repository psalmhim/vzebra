"""
HPA (hypothalamic-pituitary-adrenal) axis — cortisol stress hormone.

Models chronic stress via amygdala activity integration:
  - stress_load accumulates when amygdala_alpha exceeds threshold
  - cortisol rises slowly from stress_load; decays on a ~300-step timescale
  - Three downstream effects (hippocampal shrinkage, allostatic sensitization,
    DA suppression / anhedonia) are read by brain_v2

Zebrafish have a functional HPI axis (hypothalamic-pituitary-interrenal) that
is the homologue of mammalian HPA. Cortisol is the primary glucocorticoid.
"""


class HPAAxis:
    """
    Hypothalamic-pituitary-adrenal (HPA / zebrafish HPI) axis.

    All state is plain Python floats — no torch tensors, no device dependency.
    Safe to checkpoint via state_dict() / load_state_dict().
    """

    def __init__(self):
        self.cortisol: float = 0.0       # chronic stress hormone [0, 1]
        self.stress_load: float = 0.0    # cumulative amygdala integral

        # Decay constants (EMA alphas per behavioral step)
        self._DECAY_CORTISOL: float = 0.997   # slow  (~300 steps to halve)
        self._DECAY_STRESS: float = 0.99      # medium (~100 steps to halve)

        # Accumulation parameters
        self._STRESS_THRESHOLD: float = 0.25  # amygdala_alpha above this accumulates
        self._CORTISOL_RISE: float = 0.003    # per-step cortisol increase when stressed

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(self, amygdala_alpha: float) -> None:
        """
        Call once per behavioral step.

        amygdala_alpha: current fear level from SpikingAmygdalaV2 [0, 1].
        Mutates cortisol and stress_load in place.
        """
        # Accumulate stress load when amygdala is above threshold
        excess = max(0.0, amygdala_alpha - self._STRESS_THRESHOLD)
        self.stress_load = self._DECAY_STRESS * self.stress_load + excess
        self.stress_load = min(self.stress_load, 1.0)

        # Cortisol rises when stress load is elevated, decays otherwise
        if self.stress_load > 0.1:
            self.cortisol = min(
                1.0,
                self.cortisol + self._CORTISOL_RISE * self.stress_load
            )
        else:
            self.cortisol = self._DECAY_CORTISOL * self.cortisol

    # ------------------------------------------------------------------
    # Downstream effects (read by brain_v2)
    # ------------------------------------------------------------------

    def place_lr_factor(self) -> float:
        """
        Hippocampal shrinkage: high cortisol reduces place cell learning rate.
        Returns multiplier in [0.4, 1.0].
        """
        return 1.0 - 0.6 * self.cortisol

    def amygdala_sensitization(self) -> float:
        """
        Allostatic load: chronic cortisol sensitizes amygdala fear responses.
        Returns multiplier in [1.0, 1.5].
        """
        return 1.0 + 0.5 * self.cortisol

    def da_suppression(self) -> float:
        """
        Anhedonia: cortisol suppresses dopamine signalling.
        Returns multiplier in [0.05, 1.0]; never reaches zero.
        """
        return max(0.05, 1.0 - 0.4 * self.cortisol)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all state (call at episode start)."""
        self.cortisol = 0.0
        self.stress_load = 0.0

    def state_dict(self) -> dict:
        return {
            'cortisol':     self.cortisol,
            'stress_load':  self.stress_load,
        }

    def load_state_dict(self, d: dict) -> None:
        self.cortisol    = float(d.get('cortisol', 0.0))
        self.stress_load = float(d.get('stress_load', 0.0))
