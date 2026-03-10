"""
RPE-gated Hebbian plasticity for the zebrafish SNN.

Implements three-factor learning:
    dW = eta * RPE * dopamine * outer(pre, post)

Layers updated (feedforward W_FF):
  - OT_F, PT_L, mot, eye, DA

Feedback weights (PE-driven anti-Hebbian W_FB):
  - OT_F.W_FB, PT_L.W_FB, PC_per.W_FB, PC_int.W_FB
  - Updated every step via update_feedback(), independent of RPE

Attention weights:
  - attention.W_att (goal → attention neurons)
"""

import torch


class HebbianPlasticity:
    """Online RPE-gated Hebbian learning for PredictiveTwoComp SNN layers."""

    WARMUP_STEPS = 50  # ramp learning rate over first N steps

    def __init__(self, eta=5e-5, decay=0.9999, max_dw=0.002, max_w=1.0):
        self.eta = eta
        self.decay = decay
        self.max_dw = max_dw
        self.max_w = max_w
        self._update_count = 0
        self._total_dw_norm = 0.0
        self._fb_dw_norm = 0.0
        self._escape_success_count = 0
        self._escape_fail_count = 0

    def _clamp_weights(self, W):
        """Prevent weight explosion by clamping magnitude."""
        W.data.clamp_(-self.max_w, self.max_w)

    def _warmup_scale(self):
        """Linear warmup: 0→1 over first WARMUP_STEPS updates."""
        if self._update_count >= self.WARMUP_STEPS:
            return 1.0
        return self._update_count / self.WARMUP_STEPS

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
        """Apply three-factor Hebbian update to feedforward (W_FF) and
        attention weights. RPE-gated: only updates when reward signal present.
        """
        dopa_mod = max(0.1, min(2.0, float(dopa)))
        rpe_clamped = max(-1.0, min(1.0, float(rpe)))
        effective_eta = self.eta * min(3.0, float(threat_boost))
        warmup = self._warmup_scale()
        lr = effective_eta * rpe_clamped * dopa_mod * warmup

        self._update_count += 1

        if abs(lr) < 1e-8:
            self._total_dw_norm = 0.0
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

            # === Attention weight update ===
            if hasattr(model, 'attention') and model.attention._last_goal is not None:
                att = model.attention
                goal = att._last_goal            # [1, 4]
                m_att = att.m                    # [1, 8]
                dW = lr * 0.5 * (goal.t() @ m_att)
                dW.clamp_(-self.max_dw, self.max_dw)
                att.W_att.data += dW
                att.W_att.data.clamp_(-0.5, 0.5)

        self._total_dw_norm = total_dw

    def update_feedback(self, model, eta_fb=2e-4, max_dw_fb=0.003,
                        max_w_fb=0.15):
        """PE-driven anti-Hebbian update for feedback weights (W_FB).

        Runs every step, independent of RPE/reward. Directly minimizes
        prediction error by adjusting W_FB so that V_a tracks V_s.

        dW_FB = -eta_fb * outer(fb_source, pred_error)

        W_FB is clamped to [-max_w_fb, max_w_fb] (tighter than W_FF)
        to prevent feedback overshoot in non-stationary environments.
        """
        fb_total = 0.0

        with torch.no_grad():
            fb_pairs = [
                (model.OT_F, model.PT_L.v_s),      # PT → OT_F
                (model.PT_L, model.PC_per.v_s),     # PC_per → PT
                (model.PC_per, model.PC_int.v_s),   # PC_int → PC_per
                (model.PC_int, model.DA.v_s),       # DA → PC_int
            ]

            for layer, fb_source in fb_pairs:
                if layer.W_FB is None:
                    continue
                pe = layer.pred_error
                if pe is None:
                    continue

                # Direct anti-Hebbian: push W_FB to reduce PE
                dW = -eta_fb * (fb_source.t() @ pe)
                dW.clamp_(-max_dw_fb, max_dw_fb)
                layer.W_FB.data += dW
                layer.W_FB.data *= self.decay
                # Tight clamp for W_FB to prevent feedback overshoot
                layer.W_FB.data.clamp_(-max_w_fb, max_w_fb)
                fb_total += dW.abs().sum().item()

        self._fb_dw_norm = fb_total

    def get_stats(self):
        return {
            "hebb_updates": self._update_count,
            "hebb_dw_norm": self._total_dw_norm,
            "hebb_fb_dw_norm": self._fb_dw_norm,
            "escape_successes": self._escape_success_count,
            "escape_failures": self._escape_fail_count,
        }
