"""
Oxytocin (OXT) and vasopressin (AVP) system — social bonding and competition.

Zebrafish express isotocin (the teleost OXT homologue) and isotocin receptor
in the preoptic area; AVP-equivalent (vasotocin) is expressed in lateral
hypothalamus and habenula (Greenwood et al. 2008).

  OXT  — rises with social proximity; promotes trust/bonding, reduces fear
  AVP  — rises with crowded foraging; drives competition and territorial spacing

Both are plain Python floats; no torch dependency.
"""


class OxytocinSystem:
    """
    Oxytocin (OXT) / vasopressin (AVP) neuromodulatory system.

    Call update() once per behavioral step.
    Effects are read by brain_v2 as additive/multiplicative deltas on other
    systems (social_mem weights, amygdala gain, EFE goal bias).
    """

    def __init__(self):
        self.oxt: float = 0.0   # bonding/trust hormone [0, 1]
        self.avp: float = 0.0   # territorial/competition hormone [0, 1]

        # Decay: τ ~= 80 steps → alpha = exp(-1/80) ≈ 0.9876
        self._DECAY: float = 0.9876

        # Per-fish rise rate for OXT (clipped to 1.0)
        self._OXT_RISE_PER_FISH: float = 0.02
        # Per-crowder rise rate for AVP (clipped to 1.0)
        self._AVP_RISE_PER_CROWDER: float = 0.015
        # AVP only rises when competition is meaningful
        self._AVP_CROWDING_THRESHOLD: int = 2

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(self, n_nearby_fish: int, n_crowded_foragers: int) -> None:
        """
        Call once per behavioral step.

        n_nearby_fish:       number of conspecifics in proximity (any behaviour)
        n_crowded_foragers:  number of conspecifics foraging in the same patch
        """
        # OXT: social proximity drives bonding
        oxt_drive = self._OXT_RISE_PER_FISH * max(0, n_nearby_fish)
        self.oxt = min(1.0, self._DECAY * self.oxt + oxt_drive)

        # AVP: only rises when crowding exceeds threshold
        if n_crowded_foragers > self._AVP_CROWDING_THRESHOLD:
            avp_drive = self._AVP_RISE_PER_CROWDER * n_crowded_foragers
        else:
            avp_drive = 0.0
        self.avp = min(1.0, self._DECAY * self.avp + avp_drive)

    # ------------------------------------------------------------------
    # Downstream effects (read by brain_v2)
    # ------------------------------------------------------------------

    def social_trust_boost(self) -> float:
        """
        Additive boost to social_mem.w_food_cue when social contact is present.
        Returns value in [0.0, 0.3].
        """
        return 0.3 * self.oxt

    def competition_drive(self) -> float:
        """
        Additive increase to social_mem.w_competition when crowding is high.
        Returns value in [0.0, 0.2].
        """
        return 0.2 * self.avp

    def fear_extinction_factor(self) -> float:
        """
        Fractional reduction in amygdala_alpha when conspecifics are present.
        High OXT provides social buffering of fear responses.
        Returns value in [0.0, 0.5]; subtract from amygdala_alpha after clamping.
        """
        return 0.5 * self.oxt

    def social_efe_bias(self) -> float:
        """
        Additive bias on the SOCIAL goal EFE (negative = more preferred).
        High OXT makes social goals more attractive.
        Returns value in [-0.1, 0.0].
        """
        return -0.1 * self.oxt

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero both hormones (call at episode start)."""
        self.oxt = 0.0
        self.avp = 0.0

    def state_dict(self) -> dict:
        return {
            'oxt': self.oxt,
            'avp': self.avp,
        }

    def load_state_dict(self, d: dict) -> None:
        self.oxt = float(d.get('oxt', 0.0))
        self.avp = float(d.get('avp', 0.0))
