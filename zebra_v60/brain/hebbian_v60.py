"""
RPE-gated Hebbian plasticity for the zebrafish SNN.

Implements three-factor learning:
    dW = eta * RPE * dopamine * outer(pre, post)

Layers updated (feedforward W_FF):
  - OT_F, PT_L, mot, eye, DA

Feedback weights (anti-Hebbian W_FB):
  - OT_F.W_FB, PT_L.W_FB, PC_per.W_FB, PC_int.W_FB

Attention weights:
  - attention.W_att (goal → attention neurons)
"""

import torch


class HebbianPlasticity:
    """Online RPE-gated Hebbian learning for PredictiveTwoComp SNN layers."""

    def __init__(self, eta=5e-5, decay=0.9999, max_dw=0.002, max_w=1.0):
        self.eta = eta
        self.decay = decay
        self.max_dw = max_dw
        self.max_w = max_w
        self._update_count = 0
        self._total_dw_norm = 0.0
        self._escape_success_count = 0
        self._escape_fail_count = 0

    def _clamp_weights(self, W):
        """Prevent weight explosion by clamping magnitude."""
        W.data.clamp_(-self.max_w, self.max_w)

    def record_escape_success(self):
        self._escape_success_count += 1

    def record_escape_failure(self):
        self._escape_fail_count += 1

    def get_saveable_state(self):
        return {
            "update_count": self._update_count,
            "escape_success_count": self._escape_success_count,
            "escape_fail_count": self._escape_fail_count,
        }

    def load_saveable_state(self, state):
        self._update_count = state.get("update_count", 0)
        self._escape_success_count = state.get("escape_success_count", 0)
        self._escape_fail_count = state.get("escape_fail_count", 0)

    def update(self, model, rpe, dopa, threat_boost=1.0):
        """Apply three-factor Hebbian update to SNN weights.

        Updates feedforward (W_FF), feedback (W_FB), and attention weights.
        """
        dopa_mod = max(0.1, min(2.0, float(dopa)))
        rpe_clamped = max(-1.0, min(1.0, float(rpe)))
        effective_eta = self.eta * min(3.0, float(threat_boost))
        lr = effective_eta * rpe_clamped * dopa_mod

        if abs(lr) < 1e-8:
            return

        total_dw = 0.0

        with torch.no_grad():
            # === Feedforward weight updates (W_FF) ===

            # OT_F: bilateral fused → tectal features
            if hasattr(model, '_last_fused') and hasattr(model, '_last_oF'):
                pre = model._last_fused
                post = model._last_oF
                dW = lr * (pre.t() @ post)
                dW.clamp_(-self.max_dw, self.max_dw)
                model.OT_F.W_FF.data += dW
                model.OT_F.W_FF.data *= self.decay
                self._clamp_weights(model.OT_F.W_FF)
                total_dw += dW.abs().sum().item()

            # PT_L: tectum → pretectum
            oF = getattr(model, '_last_oF', None)
            pt_v = model.PT_L.v_s
            if oF is not None:
                dW = lr * (oF.t() @ pt_v.sigmoid())
                dW.clamp_(-self.max_dw, self.max_dw)
                model.PT_L.W_FF.data += dW
                model.PT_L.W_FF.data *= self.decay
                self._clamp_weights(model.PT_L.W_FF)
                total_dw += dW.abs().sum().item()

            # Motor
            intent = model.PC_int.v_s
            mot_v = model.mot.v_s
            dW = lr * (intent.t() @ mot_v.sigmoid())
            dW.clamp_(-self.max_dw, self.max_dw)
            model.mot.W_FF.data += dW
            model.mot.W_FF.data *= self.decay
            self._clamp_weights(model.mot.W_FF)
            total_dw += dW.abs().sum().item()

            # Eye
            eye_v = model.eye.v_s
            dW = lr * (intent.t() @ eye_v.sigmoid())
            dW.clamp_(-self.max_dw, self.max_dw)
            model.eye.W_FF.data += dW
            model.eye.W_FF.data *= self.decay
            self._clamp_weights(model.eye.W_FF)
            total_dw += dW.abs().sum().item()

            # DA
            da_v = model.DA.v_s
            dW = lr * (intent.t() @ da_v.sigmoid())
            dW.clamp_(-self.max_dw, self.max_dw)
            model.DA.W_FF.data += dW
            model.DA.W_FF.data *= self.decay
            self._clamp_weights(model.DA.W_FF)
            total_dw += dW.abs().sum().item()

            # === Feedback weight updates (anti-Hebbian W_FB) ===
            # dW_FB = -lr_fb * outer(fb_source, pred_error)
            # Reduces prediction error over time
            lr_fb = lr * 0.5  # conservative: half of feedforward rate

            # OT_F.W_FB: PT → OT_F
            if model.OT_F.W_FB is not None:
                fb_pre = model.PT_L.v_s      # [1, 400]
                pe = model.OT_F.pred_error    # [1, 800]
                dW = -lr_fb * (fb_pre.t() @ pe)
                dW.clamp_(-self.max_dw, self.max_dw)
                model.OT_F.W_FB.data += dW
                model.OT_F.W_FB.data *= self.decay
                self._clamp_weights(model.OT_F.W_FB)
                total_dw += dW.abs().sum().item()

            # PT_L.W_FB: PC_per → PT
            if model.PT_L.W_FB is not None:
                fb_pre = model.PC_per.v_s     # [1, 120]
                pe = model.PT_L.pred_error    # [1, 400]
                dW = -lr_fb * (fb_pre.t() @ pe)
                dW.clamp_(-self.max_dw, self.max_dw)
                model.PT_L.W_FB.data += dW
                model.PT_L.W_FB.data *= self.decay
                self._clamp_weights(model.PT_L.W_FB)
                total_dw += dW.abs().sum().item()

            # PC_per.W_FB: PC_int → PC_per
            if model.PC_per.W_FB is not None:
                fb_pre = model.PC_int.v_s     # [1, 30]
                pe = model.PC_per.pred_error  # [1, 120]
                dW = -lr_fb * (fb_pre.t() @ pe)
                dW.clamp_(-self.max_dw, self.max_dw)
                model.PC_per.W_FB.data += dW
                model.PC_per.W_FB.data *= self.decay
                self._clamp_weights(model.PC_per.W_FB)
                total_dw += dW.abs().sum().item()

            # PC_int.W_FB: DA → PC_int
            if model.PC_int.W_FB is not None:
                fb_pre = model.DA.v_s         # [1, 50]
                pe = model.PC_int.pred_error  # [1, 30]
                dW = -lr_fb * (fb_pre.t() @ pe)
                dW.clamp_(-self.max_dw, self.max_dw)
                model.PC_int.W_FB.data += dW
                model.PC_int.W_FB.data *= self.decay
                self._clamp_weights(model.PC_int.W_FB)
                total_dw += dW.abs().sum().item()

            # === Attention weight update ===
            if hasattr(model, 'attention') and model.attention._last_goal is not None:
                att = model.attention
                goal = att._last_goal            # [1, 4]
                m_att = att.m                    # [1, 8]
                dW = lr * 0.5 * (goal.t() @ m_att)
                dW.clamp_(-self.max_dw, self.max_dw)
                att.W_att.data += dW
                att.W_att.data.clamp_(-0.5, 0.5)

        self._update_count += 1
        self._total_dw_norm = total_dw

    def get_stats(self):
        return {
            "hebb_updates": self._update_count,
            "hebb_dw_norm": self._total_dw_norm,
            "escape_successes": self._escape_success_count,
            "escape_failures": self._escape_fail_count,
        }
