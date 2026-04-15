"""
Biologically accurate Retinal Ganglion Cell encoding.

Types per eye:
  ON-sustained  (400) — DoG center-excitatory RFs, clamp(conv(I, dog_on), 0, 1)
  OFF-transient (400) — DoG center-inhibitory RFs + EMA temporal smoothing
  Looming       (100) — angular expansion rate dθ/dt (Temizer 2015)
  Direction-sel (100) — Reichardt correlator with proper 1-step delay

Free Energy Principle:
  OFF cells ARE prediction error neurons (Rao & Ballard 1999):
    OFF = prev_intensity - current = temporal PE (luminance decrease).
  We add explicit retinal prediction error (spatial + temporal):
    PE_L/R = predicted_ON - actual_ON (top-down minus bottom-up).
  Retinal free energy drives surprise signaling to tectum.
  Precision: based on signal-to-noise of the retinal image.

Input : L, R each (800,): [:400] intensity (0-1), [400:] type channel
Output: dict with L_on/off/loom/ds, R_on/off/loom/ds, *_fused, loom_trigger,
        uv_prey_L, uv_prey_R, free_energy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from zebrav2.spec import DEVICE, N_RET_PER_TYPE, N_RET_LOOM, N_RET_DS
from zebrav2.brain.two_comp_column import TwoCompColumn


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_dog_kernel(n_pixels: int, center_sigma: float, surround_sigma: float,
                     device: torch.device) -> torch.Tensor:
    """1-D Difference-of-Gaussians kernel (ON polarity: center+, surround-)."""
    x = torch.arange(n_pixels, dtype=torch.float32, device=device) - n_pixels // 2
    center   = torch.exp(-x ** 2 / (2 * center_sigma ** 2))
    surround = torch.exp(-x ** 2 / (2 * surround_sigma ** 2))
    center   = center   / center.sum()
    surround = surround / surround.sum()
    dog = center - 0.5 * surround   # ON: bright centre excites
    return dog  # shape (n_pixels,)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class RetinaV2(nn.Module):
    """
    Rate-coded RGC output per eye.  True spiking lives in tectal layers
    downstream; here we emit firing-rate proxies in [0, 1].
    """

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device  = device
        self.n_on    = N_RET_PER_TYPE   # 400
        self.n_off   = N_RET_PER_TYPE   # 400
        self.n_loom  = N_RET_LOOM       # 100
        self.n_ds    = N_RET_DS         # 100

        # -- DoG kernels (shared for both eyes, not trained) ------------------
        # ON kernel: bright centre response
        dog_on = _make_dog_kernel(400, center_sigma=3.0, surround_sigma=9.0, device=device)
        # OFF kernel: dark centre response (negate ON)
        dog_off = -dog_on

        # Store as (1, 1, 400) for F.conv1d
        self.register_buffer('dog_on',  dog_on.view(1, 1, 400))
        self.register_buffer('dog_off', dog_off.view(1, 1, 400))

        # -- Previous intensity frames (for OFF EMA and DS delay) -------------
        self.register_buffer('prev_intensity_L', torch.zeros(400, device=device))
        self.register_buffer('prev_intensity_R', torch.zeros(400, device=device))

        # -- OFF EMA buffers (2-frame temporal smoothing) ---------------------
        self.register_buffer('off_ema_L', torch.zeros(400, device=device))
        self.register_buffer('off_ema_R', torch.zeros(400, device=device))

        # -- Direction-selective delay buffers (Reichardt 1-step) -------------
        self.register_buffer('delay_L', torch.zeros(400, device=device))
        self.register_buffer('delay_R', torch.zeros(400, device=device))

        # -- Looming: angular size history (one scalar per eye) ---------------
        self.register_buffer('theta_prev_L', torch.zeros(1, device=device))
        self.register_buffer('theta_prev_R', torch.zeros(1, device=device))

        # -- FEP: two-compartment temporal prediction (Lee et al. 2026) --------
        # 4 channels: L_on, R_on, L_off, R_off
        # Apical = previous frame (temporal prediction)
        # Soma = current frame (sensory evidence)
        # Bias = precision (contrast-dependent attention)
        self.pc = TwoCompColumn(n_channels=4, n_per_ch=4, substeps=8, device=device)
        self.register_buffer('prev_retinal', torch.zeros(4, device=device))
        self.free_energy = 0.0
        self.surprise_L = 0.0
        self.surprise_R = 0.0
        self.retinal_precision = 1.0

    # ------------------------------------------------------------------
    # DoG convolution helper
    # ------------------------------------------------------------------

    def _apply_dog(self, intensity: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Apply a 1-D DoG kernel to a (400,) intensity strip.
        Returns (400,) clamped to [0, 1].
        Padding='same' keeps spatial resolution identical.
        """
        x = intensity.view(1, 1, 400)          # (B=1, C=1, W=400)
        pad = kernel.shape[-1] // 2            # 200  → same-size output
        out = F.conv1d(x, kernel, padding=pad) # (1, 1, 400)
        # conv1d with even-length kernel over-pads by 1; trim to 400
        out = out[..., :400]
        return out.squeeze().clamp(0.0, 1.0)   # (400,)

    # ------------------------------------------------------------------
    # Per-eye encoding
    # ------------------------------------------------------------------

    def _encode_eye(
        self,
        intensity:    torch.Tensor,   # (400,) current frame
        type_ch:      torch.Tensor,   # (400,) type channel
        prev_int:     torch.Tensor,   # (400,) previous frame
        off_ema:      torch.Tensor,   # (400,) EMA state
        delay_buf:    torch.Tensor,   # (400,) one-step-old intensity
        theta_prev:   torch.Tensor,   # (1,)   previous angular size
        enemy_frac:   float,          # enemy pixel fraction [0,1]
    ):
        """
        Returns (on_rate, off_rate, loom_rate, ds_rate, loom_signal,
                 uv_prey, new_theta, new_off_ema)
        All in-place writes are avoided so MPS leaf-tensor rules hold.
        """

        # ---- ON cells: DoG center-excitatory --------------------------------
        on_rate = self._apply_dog(intensity, self.dog_on)   # (400,)
        # Pool to n_on — kernel already keeps 400 pixels, shape matches
        on_rate = F.adaptive_avg_pool1d(
            on_rate.unsqueeze(0).unsqueeze(0), self.n_on).squeeze()  # (400,)

        # ---- OFF cells: DoG center-inhibitory + EMA -------------------------
        # Temporal OFF transient (luminance decrease → positive)
        off_raw = torch.clamp((prev_int - intensity) * 2.0, 0.0, 1.0)  # (400,)
        # EMA smoothing over 2 frames (new value, not in-place on leaf)
        new_off_ema = 0.5 * off_ema + 0.5 * off_raw                    # (400,)
        off_rate = F.adaptive_avg_pool1d(
            new_off_ema.unsqueeze(0).unsqueeze(0), self.n_off).squeeze()  # (400,)

        # ---- Looming: angular expansion rate dθ/dt --------------------------
        # theta_now: normalised angular extent of enemy [0, 1]
        #   enemy_frac * 0.5 maps full-field enemy → 0.5
        theta_now   = torch.tensor([enemy_frac * 0.5], device=self.device).clamp(0.0, 1.0)
        theta_dot   = theta_now - theta_prev          # expansion per step
        # Fire when expanding fast AND already visible (size > 2% of field)
        loom_signal = torch.clamp(theta_dot * 20.0, 0.0, 1.0) * (theta_now > 0.02).float()
        loom_rate   = loom_signal.expand(self.n_loom)  # (100,)

        # ---- Direction-selective: Reichardt correlator ----------------------
        # Use the PREVIOUS frame (delay_buf) to correlate with current
        ds_raw  = torch.clamp(intensity * delay_buf, 0.0, 1.0)          # (400,)
        ds_rate = F.adaptive_avg_pool1d(
            ds_raw.unsqueeze(0).unsqueeze(0), self.n_ds).squeeze()       # (100,)
        # delay_buf is updated by caller AFTER this call (correct 1-step lag)

        # ---- UV prey detection (UV cones ~360 nm) ---------------------------
        # Zebrafish UV cones detect small dark food objects against bright bg.
        # Proxy: low-intensity pixels that carry a food-type label (type ≈ 1.0)
        uv_drive   = (intensity < 0.3).float() * (type_ch > 0.7).float()  # (400,)
        uv_prey    = float(uv_drive.sum().item() / 10.0)                   # scalar

        return on_rate, off_rate, loom_rate, ds_rate, loom_signal, uv_prey, theta_now, new_off_ema

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        L: torch.Tensor,
        R: torch.Tensor,
        entity_info: dict = None,
    ) -> dict:
        """
        L, R : (800,) — [:400] intensity, [400:] type channel
        entity_info : {'enemy': float}  enemy pixel fraction in [0, 1]
        """
        if entity_info is None:
            entity_info = {}

        enemy_frac = float(entity_info.get('enemy', 0.0))

        L_int,  R_int  = L[:400], R[:400]
        L_type, R_type = L[400:], R[400:]

        # ---- Encode left eye -----------------------------------------------
        (L_on, L_off, L_loom, L_ds,
         loom_sig_L, uv_prey_L,
         new_theta_L, new_off_ema_L) = self._encode_eye(
            intensity   = L_int,
            type_ch     = L_type,
            prev_int    = self.prev_intensity_L,
            off_ema     = self.off_ema_L,
            delay_buf   = self.delay_L,
            theta_prev  = self.theta_prev_L,
            enemy_frac  = enemy_frac,
        )

        # ---- Encode right eye ----------------------------------------------
        (R_on, R_off, R_loom, R_ds,
         loom_sig_R, uv_prey_R,
         new_theta_R, new_off_ema_R) = self._encode_eye(
            intensity   = R_int,
            type_ch     = R_type,
            prev_int    = self.prev_intensity_R,
            off_ema     = self.off_ema_R,
            delay_buf   = self.delay_R,
            theta_prev  = self.theta_prev_R,
            enemy_frac  = enemy_frac,
        )

        # ---- Update all history buffers (after encoding, not before) --------
        self.prev_intensity_L.copy_(L_int)
        self.prev_intensity_R.copy_(R_int)

        self.off_ema_L.copy_(new_off_ema_L)
        self.off_ema_R.copy_(new_off_ema_R)

        # DS delay: update AFTER use (proper 1-step lag)
        self.delay_L.copy_(L_int)
        self.delay_R.copy_(R_int)

        self.theta_prev_L.copy_(new_theta_L)
        self.theta_prev_R.copy_(new_theta_R)

        # ---- Bilateral fused signals ----------------------------------------
        on_fused   = (L_on   + R_on)   / 2.0               # (400,)
        off_fused  = (L_off  + R_off)  / 2.0               # (400,)
        loom_fused = torch.max(L_loom, R_loom)             # (100,) take max
        ds_fused   = (L_ds   + R_ds)   / 2.0               # (100,)

        # Looming trigger: expansion rate > 0.3 (≈ 1.5°/step, Temizer 2015)
        loom_trigger = bool(
            (loom_sig_L.item() > 0.3) or (loom_sig_R.item() > 0.3)
        )

        # --- FEP: two-compartment temporal prediction (Lee et al. 2026) ---
        # Sensory: current ON/OFF rate means per eye
        sensory = torch.tensor([
            float(L_on.mean()), float(R_on.mean()),
            float(L_off.mean()), float(R_off.mean()),
        ], device=self.device)
        # Prediction: previous frame (temporal prediction)
        prediction = self.prev_retinal.clone()
        # Precision modulation from image contrast
        contrast_L = float(L_int.std()) + 1e-8
        contrast_R = float(R_int.std()) + 1e-8
        att = torch.tensor([contrast_L, contrast_R, contrast_L, contrast_R],
                            device=self.device) * 5.0
        self.pc.set_attention(att)
        # Run two-compartment column
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.surprise_L = float(pe[0] ** 2 + pe[2] ** 2)  # L_on + L_off
        self.surprise_R = float(pe[1] ** 2 + pe[3] ** 2)  # R_on + R_off
        self.retinal_precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']
        # Update temporal prediction
        self.prev_retinal.copy_(sensory)

        return {
            'L_on':    L_on,    'R_on':    R_on,
            'L_off':   L_off,   'R_off':   R_off,
            'L_loom':  L_loom,  'R_loom':  R_loom,
            'L_ds':    L_ds,    'R_ds':    R_ds,
            'on_fused':    on_fused,
            'off_fused':   off_fused,
            'loom_fused':  loom_fused,
            'ds_fused':    ds_fused,
            'loom_trigger': loom_trigger,
            'uv_prey_L':   uv_prey_L,
            'uv_prey_R':   uv_prey_R,
            # FEP outputs
            'surprise_L':  self.surprise_L,
            'surprise_R':  self.surprise_R,
            'retinal_precision': self.retinal_precision,
            'free_energy': self.free_energy,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Zero all temporal state buffers (call at episode start)."""
        self.prev_intensity_L.zero_()
        self.prev_intensity_R.zero_()
        self.off_ema_L.zero_()
        self.off_ema_R.zero_()
        self.delay_L.zero_()
        self.delay_R.zero_()
        self.theta_prev_L.zero_()
        self.theta_prev_R.zero_()
        self.prev_retinal.zero_()
        self.pc.reset()
        self.free_energy = 0.0
        self.surprise_L = 0.0
        self.surprise_R = 0.0
        self.retinal_precision = 1.0
