"""
RPE-gated Hebbian plasticity for the zebrafish SNN.

Implements three-factor learning:
    dW = eta * RPE * dopamine * outer(pre, post)

This mirrors post-natal synaptic refinement where dopamine reward-prediction
error gates Hebbian correlation learning.  Only trainable pathway weights
downstream of the frozen topographic map (OT_L, OT_R) are updated.

Layers updated:
  - OT_F  (fused bilateral → tectal features)
  - PT_L  (tectum → pretectum)
  - mot   (intent → motor commands)
  - eye   (intent → eye control)
  - DA    (intent → dopamine output)
"""

import torch


class HebbianPlasticity:
    """Online RPE-gated Hebbian learning for TwoComp SNN layers.

    Usage:
        hebb = HebbianPlasticity()
        # Each brain step:
        hebb.update(snn_model, rpe, dopa)
        # Diagnostics:
        hebb.get_stats()
    """

    def __init__(self, eta=5e-5, decay=0.9999, max_dw=0.002, max_w=1.0):
        self.eta = eta
        self.decay = decay
        self.max_dw = max_dw       # clamp per-element weight change
        self.max_w = max_w         # clamp absolute weight magnitude
        self._update_count = 0
        self._total_dw_norm = 0.0   # running diagnostic
        self._escape_success_count = 0
        self._escape_fail_count = 0

    def _clamp_weights(self, W):
        """Prevent weight explosion by clamping magnitude."""
        W.data.clamp_(-self.max_w, self.max_w)

    def record_escape_success(self):
        """Called when threat_arousal drops from >0.5 to <0.1."""
        self._escape_success_count += 1

    def record_escape_failure(self):
        """Called when fish is caught while under threat."""
        self._escape_fail_count += 1

    def get_saveable_state(self):
        """Return state for checkpoint persistence."""
        return {
            "update_count": self._update_count,
            "escape_success_count": self._escape_success_count,
            "escape_fail_count": self._escape_fail_count,
        }

    def load_saveable_state(self, state):
        """Restore state from checkpoint."""
        self._update_count = state.get("update_count", 0)
        self._escape_success_count = state.get("escape_success_count", 0)
        self._escape_fail_count = state.get("escape_fail_count", 0)

    def update(self, model, rpe, dopa, threat_boost=1.0):
        """Apply three-factor Hebbian update to trainable SNN weights.

        Args:
            model: ZebrafishSNN_v60 instance (must have _last_* cached).
            rpe:   reward-prediction error scalar.
            dopa:  dopamine level scalar.
            threat_boost: float — multiplier for eta during high-threat (max 3x).
        """
        dopa_mod = max(0.1, min(2.0, float(dopa)))
        rpe_clamped = max(-1.0, min(1.0, float(rpe)))
        effective_eta = self.eta * min(3.0, float(threat_boost))
        lr = effective_eta * rpe_clamped * dopa_mod

        # Skip if learning signal is negligible
        if abs(lr) < 1e-8:
            return

        total_dw = 0.0

        with torch.no_grad():
            # --- OT_F: bilateral fused → tectal features ---
            if hasattr(model, '_last_fused') and hasattr(model, '_last_oF'):
                pre = model._last_fused    # [1, 1200]
                post = model._last_oF      # [1, 800]
                dW = lr * (pre.t() @ post)  # [1200, 800]
                dW.clamp_(-self.max_dw, self.max_dw)
                model.OT_F.W.data += dW
                model.OT_F.W.data *= self.decay
                self._clamp_weights(model.OT_F.W)
                total_dw += dW.abs().sum().item()

            # --- PT_L: tectum → pretectum ---
            oF = getattr(model, '_last_oF', None)
            pt_v = model.PT_L.v
            if oF is not None:
                dW = lr * (oF.t() @ pt_v.sigmoid())  # [800, 400]
                dW.clamp_(-self.max_dw, self.max_dw)
                model.PT_L.W.data += dW
                model.PT_L.W.data *= self.decay
                self._clamp_weights(model.PT_L.W)
                total_dw += dW.abs().sum().item()

            # --- Output heads: intent → motor/eye/DA ---
            intent = model.PC_int.v  # [1, 30]

            # Motor
            mot_v = model.mot.v  # [1, 200]
            dW = lr * (intent.t() @ mot_v.sigmoid())  # [30, 200]
            dW.clamp_(-self.max_dw, self.max_dw)
            model.mot.W.data += dW
            model.mot.W.data *= self.decay
            self._clamp_weights(model.mot.W)
            total_dw += dW.abs().sum().item()

            # Eye
            eye_v = model.eye.v  # [1, 100]
            dW = lr * (intent.t() @ eye_v.sigmoid())  # [30, 100]
            dW.clamp_(-self.max_dw, self.max_dw)
            model.eye.W.data += dW
            model.eye.W.data *= self.decay
            self._clamp_weights(model.eye.W)
            total_dw += dW.abs().sum().item()

            # DA
            da_v = model.DA.v  # [1, 50]
            dW = lr * (intent.t() @ da_v.sigmoid())  # [30, 50]
            dW.clamp_(-self.max_dw, self.max_dw)
            model.DA.W.data += dW
            model.DA.W.data *= self.decay
            self._clamp_weights(model.DA.W)
            total_dw += dW.abs().sum().item()

        self._update_count += 1
        self._total_dw_norm = total_dw

    def get_stats(self):
        """Return diagnostics dict for the neural monitor."""
        return {
            "hebb_updates": self._update_count,
            "hebb_dw_norm": self._total_dw_norm,
            "escape_successes": self._escape_success_count,
            "escape_failures": self._escape_fail_count,
        }
