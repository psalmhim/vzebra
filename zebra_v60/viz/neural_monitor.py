"""
Real-time neural activity visualization for the zebrafish SNN.

Renders to an off-screen pygame.Surface (does NOT create its own window).
The caller is responsible for blitting get_surface() onto a display.

Shows:
  - Retina L/R intensity heatmaps (20x20)
  - OT_L / OT_R tectal heatmaps (20x30)
  - OT_F fused heatmap (40x20)
  - Spike raster (PC_int + subsampled PT_L, 100 timesteps)
  - Classification bars (5 classes)
  - Neuromodulatory signal bars (DA, RPE, pi_OT, pi_PC, CMS, F_vis)
"""

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def _build_viridis_lut():
    """Build a 256-entry viridis-like colormap as numpy array [256, 3]."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = int(255 * max(0, min(1, -0.3 + 1.5 * t * t)))
        g = int(255 * max(0, min(1, 0.1 + 0.9 * t - 0.3 * t * t)))
        b = int(255 * max(0, min(1, 0.5 + 0.5 * np.sin(
            np.pi * (0.4 + 0.6 * t)))))
        lut[i] = [r, g, b]
    return lut


def _build_hot_lut():
    """Build a 256-entry hot colormap [256, 3]."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = int(255 * min(1.0, t * 2.5))
        g = int(255 * max(0, min(1.0, (t - 0.4) * 2.5)))
        b = int(255 * max(0, min(1.0, (t - 0.7) * 3.3)))
        lut[i] = [r, g, b]
    return lut


class NeuralMonitor:
    """Off-screen neural activity renderer.

    Renders to an internal pygame.Surface. Does NOT own a display window.

    Usage:
        monitor = NeuralMonitor()
        monitor.update(snn_out, diagnostics)
        monitor.render()
        surface = monitor.get_surface()  # blit this onto your display
    """

    WIDTH = 500
    HEIGHT = 600
    BG_COLOR = (15, 15, 25)
    LABEL_COLOR = (200, 200, 200)
    GRID_COLOR = (40, 40, 55)

    RASTER_HISTORY = 100
    RASTER_NEURONS = 50  # 30 PC_int + 20 subsampled PT_L
    SPIKE_THRESHOLD = 0.15

    def __init__(self):
        if not HAS_PYGAME:
            raise ImportError(
                "pygame is required for NeuralMonitor. "
                "Install with: pip install pygame")

        # Ensure pygame and font subsystem are initialized
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        # Off-screen surface — no display.set_mode here
        self.surface = pygame.Surface((self.WIDTH, self.HEIGHT))

        self.font_small = pygame.font.SysFont("monospace", 10)
        self.font_med = pygame.font.SysFont("monospace", 11, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 13, bold=True)

        self._viridis = _build_viridis_lut()
        self._hot = _build_hot_lut()

        self._raster_buf = np.zeros(
            (self.RASTER_HISTORY, self.RASTER_NEURONS), dtype=np.float32)
        self._raster_ptr = 0

        self._cls_names = ["Nothing", "Food", "Enemy", "Conspec", "Environ"]
        self._cls_colors = [
            (100, 100, 100), (50, 200, 50), (220, 50, 50),
            (50, 150, 220), (180, 180, 80),
        ]

        self._neuro_names = ["DA", "RPE", "pi_OT", "pi_PC", "CMS", "F_vis"]
        self._neuro_colors = [
            (255, 200, 50), (255, 100, 100), (100, 200, 255),
            (100, 255, 200), (200, 150, 255), (255, 150, 100),
        ]

        self._frame = 0
        self._snn_out = {}
        self._diagnostics = {}

    def _tensor_to_np(self, t):
        if hasattr(t, 'detach'):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def _heatmap(self, arr, tw, th, lut=None):
        """2D array [0,1] -> scaled pygame Surface."""
        if lut is None:
            lut = self._viridis
        arr = np.clip(arr, 0, 1)
        indices = (arr * 255).astype(np.uint8)
        rgb = lut[indices]
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        return pygame.transform.scale(surf, (tw, th))

    def _label(self, text, x, y, color=None, font=None):
        if color is None:
            color = self.LABEL_COLOR
        if font is None:
            font = self.font_small
        self.surface.blit(font.render(text, True, color), (x, y))

    def _bar(self, x, y, w, h, value, max_val, color, label=""):
        pygame.draw.rect(self.surface, (30, 30, 40), (x, y, w, h))
        frac = np.clip(abs(value) / max(abs(max_val), 1e-8), 0, 1)
        fill_w = int(w * frac)
        if value < 0:
            dim = tuple(max(0, c // 2) for c in color)
            pygame.draw.rect(
                self.surface, dim, (x + w - fill_w, y, fill_w, h))
        else:
            pygame.draw.rect(self.surface, color, (x, y, fill_w, h))
        pygame.draw.rect(self.surface, self.GRID_COLOR, (x, y, w, h), 1)
        if label:
            self._label(label, x + 2, y + 1)
        val_s = f"{value:+.2f}" if abs(value) < 10 else f"{value:+.1f}"
        self._label(val_s, x + w + 3, y + 1)

    def get_surface(self):
        """Return the rendered off-screen surface for compositing."""
        return self.surface

    def update(self, snn_out, diagnostics=None):
        """Push new neural data."""
        self._snn_out = snn_out
        self._diagnostics = diagnostics or {}

        intent_np = self._tensor_to_np(
            snn_out.get("intent", np.zeros((1, 30))))
        pt_np = self._tensor_to_np(
            snn_out.get("pt", np.zeros((1, 400))))

        intent_1d = intent_np.flatten()[:30]
        pt_1d = pt_np.flatten()
        pt_sub = pt_1d[::20][:20] if len(pt_1d) >= 20 else pt_1d[:20]

        row = np.zeros(self.RASTER_NEURONS, dtype=np.float32)
        row[:len(intent_1d)] = np.abs(intent_1d)
        row[30:30 + len(pt_sub)] = np.abs(pt_sub)

        self._raster_buf[self._raster_ptr] = row
        self._raster_ptr = (self._raster_ptr + 1) % self.RASTER_HISTORY
        self._frame += 1

    def render(self):
        """Draw everything onto the internal surface."""
        self.surface.fill(self.BG_COLOR)
        snn = self._snn_out
        diag = self._diagnostics
        W = self.WIDTH  # 500

        # ── Row 1: Retina L, Retina R (y: 0-120) ──
        hm_sz = 100  # heatmap display size
        y1 = 16
        self._label("RET L", 6, 2, font=self.font_med)
        retL = self._tensor_to_np(snn.get("retL", np.zeros((1, 400))))
        retL_2d = retL.flatten()[:400].reshape(20, 20)
        mx = retL_2d.max() + 1e-8
        self.surface.blit(self._heatmap(retL_2d / mx, hm_sz, hm_sz,
                                        self._hot), (6, y1))

        self._label("RET R", 6 + hm_sz + 8, 2, font=self.font_med)
        retR = self._tensor_to_np(snn.get("retR", np.zeros((1, 400))))
        retR_2d = retR.flatten()[:400].reshape(20, 20)
        mx = retR_2d.max() + 1e-8
        self.surface.blit(self._heatmap(retR_2d / mx, hm_sz, hm_sz,
                                        self._hot), (6 + hm_sz + 8, y1))

        # OT_L, OT_R (smaller, right side of row 1)
        sm = 70
        ox = 6 + 2 * (hm_sz + 8)
        self._label("OT_L", ox, 2, font=self.font_med)
        oL = self._tensor_to_np(snn.get("oL", np.zeros((1, 600))))
        oL_2d = np.abs(oL.flatten()[:600].reshape(20, 30))
        mx = oL_2d.max() + 1e-8
        self.surface.blit(self._heatmap(oL_2d / mx, sm, hm_sz,
                                        self._viridis), (ox, y1))

        self._label("OT_R", ox + sm + 4, 2, font=self.font_med)
        oR = self._tensor_to_np(snn.get("oR", np.zeros((1, 600))))
        oR_2d = np.abs(oR.flatten()[:600].reshape(20, 30))
        mx = oR_2d.max() + 1e-8
        self.surface.blit(self._heatmap(oR_2d / mx, sm, hm_sz,
                                        self._viridis), (ox + sm + 4, y1))

        # ── Row 2: OT_F fused + PT_L + PC_per (y: 125-230) ──
        y2 = 125
        self._label("OT_F FUSED", 6, y2 - 12, font=self.font_med)
        oF = self._tensor_to_np(snn.get("oF", np.zeros((1, 800))))
        oF_2d = np.abs(oF.flatten()[:800].reshape(40, 20))
        mx = oF_2d.max() + 1e-8
        self.surface.blit(self._heatmap(oF_2d / mx, 200, 80,
                                        self._viridis), (6, y2))

        self._label("PT_L", 215, y2 - 12, font=self.font_med)
        pt = self._tensor_to_np(snn.get("pt", np.zeros((1, 400))))
        pt_2d = np.abs(pt.flatten()[:400].reshape(20, 20))
        mx = pt_2d.max() + 1e-8
        self.surface.blit(self._heatmap(pt_2d / mx, 80, 80,
                                        self._viridis), (215, y2))

        self._label("PC_per", 305, y2 - 12, font=self.font_med)
        per = self._tensor_to_np(snn.get("per", np.zeros((1, 120))))
        per_2d = np.abs(per.flatten()[:120].reshape(12, 10))
        mx = per_2d.max() + 1e-8
        self.surface.blit(self._heatmap(per_2d / mx, 80, 80,
                                        self._viridis), (305, y2))

        # Goal + energy (right of row 2)
        goal = diag.get("goal", 2)
        gn = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
        gc = [(50, 200, 50), (220, 50, 50), (100, 150, 255), (0, 180, 170)]
        self._label(f"GOAL: {gn[goal]}", 400, y2, color=gc[goal],
                    font=self.font_title)
        energy = diag.get("energy", 100.0)
        ef = np.clip(energy / 100.0, 0, 1)
        ec = (50, 200, 50) if ef > 0.3 else (220, 100, 50)
        pygame.draw.rect(self.surface, (30, 30, 40), (400, y2 + 20, 90, 10))
        pygame.draw.rect(self.surface, ec, (400, y2 + 20, int(90 * ef), 10))
        self._label(f"{energy:.0f}%", 400, y2 + 33)

        # Epistemic
        vae_d = diag.get("vae", {})
        ep = vae_d.get("epistemic_per_goal", [0, 0, 0])
        if any(v != 0 for v in ep):
            self._label("EPISTEMIC", 400, y2 + 48, font=self.font_med)
            ep_labels = ["F", "L", "E", "S"]
            for i, (lbl, val) in enumerate(
                    zip(ep_labels[:len(ep)], ep)):
                self._label(f"{lbl}:{val:.2f}", 400 + i * 34, y2 + 62,
                            color=gc[min(i, len(gc) - 1)])

        # ── Row 3: Spike raster (y: 215-330) ──
        y3 = 220
        self._label("SPIKE RASTER (PC_int + PT_L, 100 steps)",
                    6, y3 - 12, font=self.font_med)

        rw = W - 20
        rh = 100

        ordered = np.zeros_like(self._raster_buf)
        ptr = self._raster_ptr
        ordered[:self.RASTER_HISTORY - ptr] = self._raster_buf[ptr:]
        ordered[self.RASTER_HISTORY - ptr:] = self._raster_buf[:ptr]

        raster_img = np.zeros(
            (self.RASTER_NEURONS, self.RASTER_HISTORY, 3), dtype=np.uint8)
        for ni in range(self.RASTER_NEURONS):
            for ti in range(self.RASTER_HISTORY):
                val = ordered[ti, ni]
                if abs(val) > self.SPIKE_THRESHOLD:
                    intensity = min(255, int(abs(val) * 400))
                    if ni < 30:
                        raster_img[ni, ti] = [0, intensity, intensity]
                    else:
                        raster_img[ni, ti] = [intensity // 2, intensity, 0]

        rs = pygame.surfarray.make_surface(
            np.transpose(raster_img, (1, 0, 2)))
        rs = pygame.transform.scale(rs, (rw, rh))
        self.surface.blit(rs, (10, y3))
        pygame.draw.rect(self.surface, self.GRID_COLOR,
                         (10, y3, rw, rh), 1)
        self._label("PC_int", rw - 30, y3 + 2, color=(0, 200, 200))
        self._label("PT_L", rw - 30, y3 + rh - 14, color=(128, 200, 0))

        # ── Row 4: Classification + Neuromodulatory (y: 340-590) ──
        y4 = 335

        # Classification
        self._label("CLASSIFICATION", 6, y4 - 12, font=self.font_med)
        cls_probs = diag.get("cls_probs", np.zeros(5))
        bw, bh = 130, 15
        for i, (name, color) in enumerate(
                zip(self._cls_names, self._cls_colors)):
            by = y4 + i * (bh + 3)
            self._bar(6, by, bw, bh, float(cls_probs[i]), 1.0, color, name)

        # Neuromodulatory
        self._label("NEUROMODULATORY", 260, y4 - 12, font=self.font_med)
        neuro_vals = [
            diag.get("dopa", 0.0),
            diag.get("rpe", 0.0),
            float(snn.get("pi_OT", 0.5)) if not isinstance(
                diag.get("pi_OT"), (int, float)) else diag.get("pi_OT", 0.5),
            float(snn.get("pi_PC", 0.5)) if not isinstance(
                diag.get("pi_PC"), (int, float)) else diag.get("pi_PC", 0.5),
            diag.get("cms", 0.0),
            diag.get("F_visual", 0.0),
        ]
        neuro_max = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
        bw2 = 150
        for i, (name, color, val, mx) in enumerate(
                zip(self._neuro_names, self._neuro_colors,
                    neuro_vals, neuro_max)):
            by = y4 + i * (bh + 3)
            self._bar(260, by, bw2, bh, float(val), mx, color, name)

        # Frame counter
        self._label(f"frame {self._frame}", W - 80, self.HEIGHT - 14,
                    color=(80, 80, 100))

    def close(self):
        """No-op — we don't own the display."""
        pass
