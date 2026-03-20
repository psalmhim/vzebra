"""
Real-time neural activity visualization for the zebrafish SNN.

Renders to an off-screen pygame.Surface (does NOT create its own window).
The caller is responsible for blitting get_surface() onto a display.

Shows:
  - Retina L/R intensity heatmaps (20x20)
  - OT_L / OT_R tectal heatmaps (20x30)
  - OT_F fused heatmap (40x20)
  - PT_L + PC_per heatmaps
  - Output heads: Motor L/R, Eye, DA heatmaps
  - Spike raster (PC_int + PT_L + Motor + Eye + DA, 100 timesteps)
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
    HEIGHT = 920
    BG_COLOR = (15, 15, 25)
    LABEL_COLOR = (200, 200, 200)
    GRID_COLOR = (40, 40, 55)

    RASTER_HISTORY = 100
    RASTER_NEURONS = 80  # 30 PC_int + 20 PT_L + 10 Motor + 10 Eye + 10 DA
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

        # PC_int (30 neurons)
        intent_np = self._tensor_to_np(
            snn_out.get("intent", np.zeros((1, 30))))
        intent_1d = intent_np.flatten()[:30]

        # PT_L subsampled (20 from 400)
        pt_np = self._tensor_to_np(
            snn_out.get("pt", np.zeros((1, 400))))
        pt_1d = pt_np.flatten()
        pt_sub = pt_1d[::20][:20] if len(pt_1d) >= 20 else pt_1d[:20]

        # Motor subsampled (10 from 200)
        mot_np = self._tensor_to_np(
            snn_out.get("motor", np.zeros((1, 200))))
        mot_1d = mot_np.flatten()
        mot_sub = mot_1d[::20][:10] if len(mot_1d) >= 20 else mot_1d[:10]

        # Eye subsampled (10 from 100)
        eye_np = self._tensor_to_np(
            snn_out.get("eye", np.zeros((1, 100))))
        eye_1d = eye_np.flatten()
        eye_sub = eye_1d[::10][:10] if len(eye_1d) >= 10 else eye_1d[:10]

        # DA subsampled (10 from 50)
        da_np = self._tensor_to_np(
            snn_out.get("DA", np.zeros((1, 50))))
        da_1d = da_np.flatten()
        da_sub = da_1d[::5][:10] if len(da_1d) >= 5 else da_1d[:10]

        row = np.zeros(self.RASTER_NEURONS, dtype=np.float32)
        row[:30] = np.abs(intent_1d)
        row[30:50] = np.abs(pt_sub)
        row[50:60] = np.abs(mot_sub)
        row[60:70] = np.abs(eye_sub)
        row[70:80] = np.abs(da_sub)

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

        # Overlay entity markers on retinal heatmaps
        # Type channel values: Food=1.0, Enemy=0.5, Colleague=0.25
        retL_full = self._tensor_to_np(
            snn.get("retL_full", np.zeros((1, 800)))).flatten()
        retR_full = self._tensor_to_np(
            snn.get("retR_full", np.zeros((1, 800)))).flatten()
        self._draw_retina_entities(retL_full, 6, y1, hm_sz)
        self._draw_retina_entities(retR_full, 6 + hm_sz + 8, y1, hm_sz)

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

        # Entity legend (below retinal heatmaps)
        ly = y1 + hm_sz + 2
        pygame.draw.circle(self.surface, (255, 50, 50), (12, ly + 4), 3)
        self._label("Enemy", 18, ly, color=(255, 100, 100))
        pygame.draw.circle(self.surface, (50, 200, 255), (72, ly + 4), 3)
        self._label("Colleague", 78, ly, color=(100, 200, 255))
        pygame.draw.circle(self.surface, (50, 220, 50), (148, ly + 4), 2)
        self._label("Food", 154, ly, color=(100, 220, 100))

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
        self._label("SPIKE RASTER (80 neurons, 100 steps)",
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
                        # PC_int — cyan
                        raster_img[ni, ti] = [0, intensity, intensity]
                    elif ni < 50:
                        # PT_L — lime
                        raster_img[ni, ti] = [intensity // 2, intensity, 0]
                    elif ni < 60:
                        # Motor — orange
                        raster_img[ni, ti] = [intensity,
                                              intensity * 2 // 3, 0]
                    elif ni < 70:
                        # Eye — magenta
                        raster_img[ni, ti] = [intensity, 0,
                                              intensity * 3 // 4]
                    else:
                        # DA — yellow
                        raster_img[ni, ti] = [intensity, intensity, 0]

        rs = pygame.surfarray.make_surface(
            np.transpose(raster_img, (1, 0, 2)))
        rs = pygame.transform.scale(rs, (rw, rh))
        self.surface.blit(rs, (10, y3))
        pygame.draw.rect(self.surface, self.GRID_COLOR,
                         (10, y3, rw, rh), 1)
        # Raster group labels (right-aligned)
        lx = rw - 30
        self._label("PC_int", lx, y3 + 2, color=(0, 200, 200))
        self._label("PT_L", lx, y3 + 24, color=(128, 200, 0))
        self._label("Motor", lx, y3 + 46, color=(255, 170, 0))
        self._label("Eye", lx, y3 + 64, color=(255, 0, 190))
        self._label("DA", lx, y3 + 82, color=(255, 255, 0))

        # ── Row 5: Output Heads — Motor L/R, Eye, DA (y: 330-450) ──
        y5 = 340
        self._label("OUTPUT HEADS", 6, y5 - 12, font=self.font_med)

        # Motor L (neurons 0-99)
        self._label("MOT L", 6, y5 - 1, font=self.font_small)
        mot = self._tensor_to_np(snn.get("motor", np.zeros((1, 200))))
        mot_1d = np.abs(mot.flatten())
        motL_2d = mot_1d[:100].reshape(10, 10)
        mx = motL_2d.max() + 1e-8
        self.surface.blit(self._heatmap(motL_2d / mx, 90, 80,
                                        self._hot), (6, y5 + 10))

        # Motor R (neurons 100-199)
        self._label("MOT R", 106, y5 - 1, font=self.font_small)
        motR_2d = mot_1d[100:200].reshape(10, 10)
        mx = motR_2d.max() + 1e-8
        self.surface.blit(self._heatmap(motR_2d / mx, 90, 80,
                                        self._hot), (106, y5 + 10))

        # Eye (100 neurons)
        self._label("EYE", 210, y5 - 1, font=self.font_small)
        eye = self._tensor_to_np(snn.get("eye", np.zeros((1, 100))))
        eye_2d = np.abs(eye.flatten()[:100].reshape(10, 10))
        mx = eye_2d.max() + 1e-8
        self.surface.blit(self._heatmap(eye_2d / mx, 80, 80,
                                        self._viridis), (210, y5 + 10))

        # DA (50 neurons)
        self._label("DA", 305, y5 - 1, font=self.font_small)
        da = self._tensor_to_np(snn.get("DA", np.zeros((1, 50))))
        da_2d = np.abs(da.flatten()[:50].reshape(10, 5))
        mx = da_2d.max() + 1e-8
        self.surface.blit(self._heatmap(da_2d / mx, 80, 40,
                                        self._viridis), (305, y5 + 10))

        # Motor activity summary bars (right side)
        mot_L_act = float(mot_1d[:100].mean())
        mot_R_act = float(mot_1d[100:].mean())
        self._label("L/R balance", 400, y5, font=self.font_small)
        self._bar(400, y5 + 12, 90, 10, mot_L_act, 0.5,
                  (255, 140, 0), "L")
        self._bar(400, y5 + 26, 90, 10, mot_R_act, 0.5,
                  (255, 140, 0), "R")
        eye_act = float(np.abs(eye.flatten()).mean())
        da_act = float(np.abs(da.flatten()).mean())
        self._bar(400, y5 + 44, 90, 10, eye_act, 0.5,
                  (255, 0, 190), "Eye")
        self._bar(400, y5 + 58, 90, 10, da_act, 0.5,
                  (255, 255, 0), "DA")

        # ── Row 4: Classification + Neuromodulatory (y: 465-760) ──
        y4 = 465

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

        # ── Row 5b: Predictive Coding (y: 575-660) ──
        y5b = 575
        self._label("PREDICTIVE CODING", 6, y5b - 12, font=self.font_med)

        # Per-layer prediction error bars
        pe_names = ["PE OT_F", "PE PT", "PE PCper", "PE PCint"]
        pe_keys = ["pe_OTF", "pe_PT", "pe_PC_per", "pe_PC_int"]
        pe_color = (255, 120, 60)
        for i, (name, key) in enumerate(zip(pe_names, pe_keys)):
            val = diag.get(key, 0.0)
            self._bar(6, y5b + i * 18, 160, 14, val, 0.5, pe_color, name)

        # Attention state (8 neurons, 2 per goal)
        att = diag.get("att_signals", None)
        if att is not None:
            import numpy as _np
            if hasattr(att, 'cpu'):
                att_np = att.cpu().numpy().flatten()[:8]
            else:
                att_np = _np.array(att).flatten()[:8]
            goal_colors = [(46, 204, 113), (231, 76, 60),
                           (52, 152, 219), (26, 188, 156)]
            goal_labels = ["F", "L", "E", "S"]
            att_x0 = 200
            self._label("ATTENTION", att_x0, y5b - 12, font=self.font_med)
            bar_w = 60
            for g in range(4):
                v0 = abs(float(att_np[g * 2]))
                v1 = abs(float(att_np[g * 2 + 1]))
                avg = (v0 + v1) / 2.0
                bx = att_x0 + g * (bar_w + 8)
                self._bar(bx, y5b, bar_w, 14, avg, 0.1,
                          goal_colors[g], goal_labels[g])
            # Total prediction error
            total_pe = sum(diag.get(k, 0.0) for k in pe_keys)
            self._label(f"PE total: {total_pe:.3f}",
                        att_x0, y5b + 24, color=(255, 180, 100))

        # ── Row 6: Hebbian Learning + Escape Stats (y: 680-760) ──
        y6 = 680
        self._label("HEBBIAN LEARNING", 6, y6 - 12, font=self.font_med)

        dw_norm = diag.get("hebb_dw_norm", 0.0)
        self._bar(6, y6, 200, 14, dw_norm, 0.1,
                  (180, 100, 255), "dW norm")

        rpe_val = diag.get("rpe", 0.0)
        dopa_val = diag.get("dopa", 0.5)
        threat_boost = diag.get("threat_boost", 1.0)
        lr_eff = 5e-5 * abs(rpe_val) * max(0.1, dopa_val) * threat_boost
        self._bar(6, y6 + 20, 200, 14, lr_eff, 0.001,
                  (100, 200, 180), "LR eff")

        hebb_updates = diag.get("hebb_updates", 0)
        self._label(f"Updates: {hebb_updates}", 220, y6,
                    color=(180, 180, 200))

        escape_s = diag.get("escape_successes", 0)
        escape_f = diag.get("escape_failures", 0)
        esc_color = (100, 220, 100) if escape_s >= escape_f else (220, 100, 100)
        self._label(f"Escapes: {escape_s}S/{escape_f}F", 220, y6 + 16,
                    color=esc_color)

        pc_diag = diag.get("place_cells", {})
        fear_count = pc_diag.get("fear_memory_count", 0)
        max_risk = pc_diag.get("max_risk", 0.0)
        self._label(f"Fear: {fear_count} memories  risk:{max_risk:.2f}",
                    220, y6 + 32, color=(255, 150, 100))

        if threat_boost > 1.1:
            bc = min(255, int(100 + 155 * (threat_boost - 1.0) / 2.0))
            self._label(f"BOOST x{threat_boost:.1f}", 400, y6,
                        color=(bc, 50, 50), font=self.font_med)

        # ── Row 7: Steps 37-41 Sensorimotor Extensions (y: 740-870) ──
        y7 = 740
        self._label("SENSORIMOTOR EXTENSIONS", 6, y7 - 12,
                     font=self.font_med)

        # Vestibular (Step 37): balance meter
        vest = diag.get("vestibular", {})
        balance = vest.get("balance", 1.0)
        pitch = vest.get("pitch", 0.0)
        bal_color = (50, 200, 50) if balance > 0.8 else (220, 150, 50)
        self._bar(6, y7, 90, 12, balance, 1.0, bal_color, "Balance")
        self._bar(6, y7 + 16, 90, 12, pitch, 0.5, (100, 150, 220), "Pitch")

        # Spinal CPG (Step 38): L/R oscillation
        cpg = diag.get("spinal_cpg", {})
        cpg_L = cpg.get("v_L", 0.0)
        cpg_R = cpg.get("v_R", 0.0)
        phase = cpg.get("phase", 0.0)
        self._bar(110, y7, 70, 12, cpg_L, 1.0, (100, 220, 100), "CPG L")
        self._bar(110, y7 + 16, 70, 12, cpg_R, 1.0, (100, 220, 100), "CPG R")
        self._label(f"ph:{phase:.2f}", 110, y7 + 32, color=(150, 200, 150))

        # Color vision (Step 39): 4-channel bars
        cv = diag.get("color_vision", {})
        cv_names = ["UV", "B", "G", "R"]
        cv_colors = [(128, 0, 255), (50, 100, 255), (50, 200, 50), (220, 50, 50)]
        cv_keys = ["mean_uv", "mean_blue", "mean_green", "mean_red"]
        for i, (n, c, k) in enumerate(zip(cv_names, cv_colors, cv_keys)):
            v = cv.get(k, 0.0)
            self._bar(220 + i * 50, y7, 40, 12, v, 0.5, c, n)

        # Circadian (Step 40): phase dial
        circ = diag.get("circadian", {})
        c_phase = circ.get("phase", 0.0)
        import math as _m
        cx_c, cy_c = 430, y7 + 15
        r_c = 14
        pygame.draw.circle(self.surface, (40, 40, 60), (cx_c, cy_c), r_c)
        # Day half (yellow), night half (blue)
        pygame.draw.arc(self.surface, (180, 180, 50),
                        (cx_c - r_c, cy_c - r_c, 2 * r_c, 2 * r_c),
                        _m.pi / 2, 3 * _m.pi / 2, 2)
        pygame.draw.arc(self.surface, (40, 60, 120),
                        (cx_c - r_c, cy_c - r_c, 2 * r_c, 2 * r_c),
                        -_m.pi / 2, _m.pi / 2, 2)
        # Hand
        hx = cx_c + int(r_c * 0.8 * _m.cos(2 * _m.pi * c_phase - _m.pi / 2))
        hy = cy_c + int(r_c * 0.8 * _m.sin(2 * _m.pi * c_phase - _m.pi / 2))
        pygame.draw.line(self.surface, (255, 255, 200),
                         (cx_c, cy_c), (hx, hy), 2)
        self._label("Circ", 450, y7, color=(180, 180, 100))

        # Proprioception (Step 41): speed PE + collision
        prop = diag.get("proprioception", {})
        spd_pe = prop.get("speed_pe", 0.0)
        collision = prop.get("collision", 0.0)
        effort = prop.get("effort", 0.0)
        self._bar(220, y7 + 18, 100, 12, spd_pe, 0.5,
                  (200, 100, 255), "Spd PE")
        self._bar(330, y7 + 18, 80, 12, effort, 1.0,
                  (255, 200, 100), "Effort")
        if collision > 0.1:
            self._label("COLLISION!", 420, y7 + 18,
                         color=(255, 50, 50), font=self.font_med)

        # Lateral line (Step 32)
        ll = diag.get("lateral_line", {})
        rear_wake = ll.get("rear_wake_intensity", 0.0)
        total_flow = ll.get("total_flow", 0.0)
        self._bar(6, y7 + 50, 100, 12, rear_wake, 5.0,
                  (50, 200, 200), "LL rear")
        self._bar(120, y7 + 50, 100, 12, total_flow, 20.0,
                  (50, 200, 200), "LL total")

        # Olfaction (Step 35)
        olf = diag.get("olfaction", {})
        food_odour = olf.get("total_food_odour", 0.0)
        alarm = olf.get("total_alarm", 0.0)
        self._bar(240, y7 + 50, 90, 12, food_odour, 5.0,
                  (50, 200, 50), "Smell")
        self._bar(340, y7 + 50, 80, 12, alarm, 2.0,
                  (220, 50, 50), "Alarm")

        # Habenula (Step 36)
        hab = diag.get("habenula", {})
        helpless = hab.get("helplessness", 0.0)
        self._bar(430, y7 + 50, 60, 12, helpless, 1.0,
                  (180, 50, 180), "Help")

        # Cerebellum (Step 33)
        cb = diag.get("cerebellum", {})
        cb_pe = cb.get("prediction_error", 0.0)
        self._bar(6, y7 + 68, 100, 12, min(cb_pe, 5.0), 5.0,
                  (255, 180, 50), "CB PE")
        cb_turn = cb.get("turn_correction", 0.0)
        self._bar(120, y7 + 68, 100, 12, cb_turn, 0.1,
                  (255, 180, 50), "CB trn")

        # Insula / Heart rate (Step 42)
        ins = diag.get("insula", {})
        hr = diag.get("heart_rate", 0.3)
        arousal = ins.get("arousal", 0.0)
        fear = ins.get("fear", 0.0)
        valence = ins.get("valence", 0.0)

        # Heart rate: pulsing red circle
        hr_x, hr_y = 260, y7 + 68
        hr_radius = int(6 + 8 * hr)
        # Pulse effect: size oscillates with HR frequency
        import math as _m
        pulse = 0.5 + 0.5 * _m.sin(self._frame * hr * 0.5)
        pr = int(hr_radius * (0.8 + 0.4 * pulse))
        hr_color = (int(150 + 105 * hr), int(40 * (1 - hr)), int(40 * (1 - hr)))
        pygame.draw.circle(self.surface, hr_color, (hr_x, hr_y + 6), pr)
        pygame.draw.circle(self.surface, (200, 60, 60), (hr_x, hr_y + 6), pr, 1)
        self._label(f"HR {hr:.0%}", hr_x + pr + 4, hr_y, color=(220, 80, 80))

        # Arousal/fear/valence bars
        self._bar(320, y7 + 68, 50, 12, arousal, 1.0,
                  (220, 120, 50), "Arsl")
        self._bar(380, y7 + 68, 50, 12, fear, 1.0,
                  (220, 50, 50), "Fear")
        val_color = (50, 180, 50) if valence > 0 else (180, 50, 50)
        self._bar(440, y7 + 68, 50, 12, valence, 1.0, val_color, "Val")

        # Frame counter
        self._label(f"frame {self._frame}", W - 80, self.HEIGHT - 16,
                    color=(80, 80, 100))

    def _draw_retina_entities(self, ret_full, hm_x, hm_y, hm_sz):
        """Overlay colored markers on a retinal heatmap for detected entities.

        Scans the type channel (pixels 400-799) and draws small markers:
          - Enemy (type ≈ 0.5):     red circle
          - Colleague (type ≈ 0.25): cyan circle
          - Food (type ≈ 1.0):      green dot
        """
        type_ch = ret_full[400:800]  # 400 type-channel pixels
        n_px = len(type_ch)
        # Map pixel index → (col, row) in 20×20 grid
        scale = hm_sz / 20.0
        for i in range(n_px):
            tv = type_ch[i]
            if tv < 0.05:
                continue
            col = i % 20
            row = i // 20
            cx = int(hm_x + col * scale + scale * 0.5)
            cy = int(hm_y + row * scale + scale * 0.5)
            r = max(2, int(scale * 0.4))
            if abs(tv - 0.5) < 0.08:
                # Enemy — red
                pygame.draw.circle(self.surface, (255, 50, 50), (cx, cy), r)
            elif abs(tv - 0.25) < 0.08:
                # Colleague — cyan
                pygame.draw.circle(self.surface, (50, 200, 255), (cx, cy), r)
            elif abs(tv - 1.0) < 0.08:
                # Food — green dot (smaller)
                pygame.draw.circle(
                    self.surface, (50, 220, 50), (cx, cy), max(1, r - 1))

    def close(self):
        """No-op — we don't own the display."""
        pass
