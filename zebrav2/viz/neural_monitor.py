"""
Real-time neural activity monitor for ZebrafishBrainV2.

Renders to an off-screen pygame.Surface. Caller blits get_surface() onto display.

Shows:
  - Retina L/R intensity heatmaps
  - Tectum SFGS-b/d, SGC, SO rate heatmaps
  - Thalamus TC/TRN rates
  - Pallium-S / Pallium-D rate heatmaps
  - Spike raster (tectum E + pallium E + amygdala CeA, 100 timesteps)
  - Basal ganglia D1/D2/GPi bars
  - Classification bars (5 classes)
  - Neuromodulatory bars (DA, NA, 5-HT, ACh)
  - Allostatic bars (hunger, fatigue, stress)
  - Predator model state (belief dist, intent, TTC)
  - Goal + energy + free energy
  - Place cell activity map
"""
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def _build_viridis_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = int(255 * max(0, min(1, -0.3 + 1.5 * t * t)))
        g = int(255 * max(0, min(1, 0.1 + 0.9 * t - 0.3 * t * t)))
        b = int(255 * max(0, min(1, 0.5 + 0.5 * np.sin(np.pi * (0.4 + 0.6 * t)))))
        lut[i] = [r, g, b]
    return lut


def _build_hot_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = int(255 * min(1.0, t * 2.5))
        g = int(255 * max(0, min(1.0, (t - 0.4) * 2.5)))
        b = int(255 * max(0, min(1.0, (t - 0.7) * 3.3)))
        lut[i] = [r, g, b]
    return lut


class NeuralMonitorV2:
    """Off-screen neural activity renderer for ZebrafishBrainV2."""

    WIDTH = 500
    HEIGHT = 700
    BG = (15, 15, 25)
    LABEL = (200, 200, 200)
    GRID = (40, 40, 55)
    RASTER_HISTORY = 100
    RASTER_NEURONS = 80

    GOAL_NAMES = ["FORAGE", "FLEE", "EXPLORE", "SOCIAL"]
    GOAL_COLORS = [(50, 200, 50), (220, 50, 50), (100, 150, 255), (0, 180, 170)]
    CLS_NAMES = ["Nothing", "Food", "Enemy", "Conspec", "Environ"]
    CLS_COLORS = [(100, 100, 100), (50, 200, 50), (220, 50, 50),
                  (50, 150, 220), (180, 180, 80)]

    def __init__(self):
        if not HAS_PYGAME:
            raise ImportError("pygame required for NeuralMonitorV2")
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self.surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_sm = pygame.font.SysFont("monospace", 10)
        self.font_md = pygame.font.SysFont("monospace", 11, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 13, bold=True)

        self._viridis = _build_viridis_lut()
        self._hot = _build_hot_lut()

        self._raster_buf = np.zeros((self.RASTER_HISTORY, self.RASTER_NEURONS),
                                    dtype=np.float32)
        self._raster_ptr = 0
        self._frame = 0

    def _t2np(self, t):
        if hasattr(t, 'detach'):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    def _heatmap(self, arr, tw, th, lut=None):
        if lut is None:
            lut = self._viridis
        arr = np.clip(arr, 0, 1)
        idx = (arr * 255).astype(np.uint8)
        rgb = lut[idx]
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        return pygame.transform.scale(surf, (tw, th))

    def _lbl(self, text, x, y, color=None, font=None):
        color = color or self.LABEL
        font = font or self.font_sm
        self.surface.blit(font.render(text, True, color), (x, y))

    def _bar(self, x, y, w, h, value, max_val, color, label=""):
        pygame.draw.rect(self.surface, (30, 30, 40), (x, y, w, h))
        frac = np.clip(abs(value) / max(abs(max_val), 1e-8), 0, 1)
        fw = int(w * frac)
        if value < 0:
            dim = tuple(max(0, c // 2) for c in color)
            pygame.draw.rect(self.surface, dim, (x + w - fw, y, fw, h))
        else:
            pygame.draw.rect(self.surface, color, (x, y, fw, h))
        pygame.draw.rect(self.surface, self.GRID, (x, y, w, h), 1)
        if label:
            self._lbl(label, x + 2, y + 1)
        self._lbl(f"{value:+.2f}" if abs(value) < 10 else f"{value:+.1f}",
                  x + w + 3, y + 1)

    def get_surface(self):
        return self.surface

    def update(self, brain, step_out, env_info=None):
        """Push new data from brain and step output."""
        self._brain = brain
        self._out = step_out
        self._info = env_info or {}
        self._frame += 1

        # Build raster row from v2 spiking layers
        row = np.zeros(self.RASTER_NEURONS, dtype=np.float32)

        # Tectum SFGS-b E spikes (20 subsampled from ~281)
        sfgsb_rate = self._t2np(brain.tectum.sfgs_b.get_rate_e())
        n = len(sfgsb_rate)
        step_s = max(1, n // 20)
        row[:20] = sfgsb_rate[::step_s][:20]

        # Pallium-S E rates (20 subsampled from ~375)
        pals_rate = self._t2np(brain.pallium.rate_s)
        n = len(pals_rate)
        step_s = max(1, n // 20)
        row[20:40] = pals_rate[::step_s][:20]

        # Pallium-D E rates (10 subsampled from ~150)
        pald_rate = self._t2np(brain.pallium.rate_d)
        n = len(pald_rate)
        step_s = max(1, n // 10)
        row[40:50] = pald_rate[::step_s][:10]

        # Amygdala CeA rates (10 subsampled from 20)
        cea_rate = self._t2np(brain.amygdala.CeA.rate)
        row[50:60] = cea_rate[::2][:10]

        # BG D1 rates (10 subsampled from 120)
        d1_rate = self._t2np(brain.bg.d1_rate)
        row[60:70] = d1_rate[::12][:10]

        # Thalamus TC rates (10 subsampled from 80)
        tc_rate = self._t2np(brain.thalamus.tc_rate)
        row[70:80] = tc_rate[::8][:10]

        self._raster_buf[self._raster_ptr] = row
        self._raster_ptr = (self._raster_ptr + 1) % self.RASTER_HISTORY

    def render(self):
        """Draw everything onto the internal surface."""
        self.surface.fill(self.BG)
        brain = self._brain
        out = self._out
        info = self._info
        W = self.WIDTH

        # ── Row 1: Retina L/R (y: 0-120) ──
        y1 = 16
        hm = 100
        self._lbl("RETINA L", 6, 2, font=self.font_md)
        L = self._t2np(brain.retina.prev_intensity_L)
        L_2d = L[:400].reshape(20, 20)
        mx = L_2d.max() + 1e-8
        self.surface.blit(self._heatmap(L_2d / mx, hm, hm, self._hot), (6, y1))

        self._lbl("RETINA R", hm + 14, 2, font=self.font_md)
        R = self._t2np(brain.retina.prev_intensity_R)
        R_2d = R[:400].reshape(20, 20)
        mx = R_2d.max() + 1e-8
        self.surface.blit(self._heatmap(R_2d / mx, hm, hm, self._hot), (hm + 14, y1))

        # Tectum SFGS-b / SFGS-d (small heatmaps)
        sm = 70
        ox = 2 * (hm + 8)
        self._lbl("SFGS-b", ox, 2, font=self.font_md)
        sb = self._t2np(brain.tectum.sfgs_b.get_rate_e())
        n = len(sb)
        side = int(np.sqrt(n)) or 1
        sb_2d = sb[:side * side].reshape(side, -1)
        mx = sb_2d.max() + 1e-8
        self.surface.blit(self._heatmap(sb_2d / mx, sm, hm // 2), (ox, y1))

        self._lbl("SFGS-d", ox + sm + 4, 2, font=self.font_md)
        sd = self._t2np(brain.tectum.sfgs_d.get_rate_e())
        n = len(sd)
        side = int(np.sqrt(n)) or 1
        sd_2d = sd[:side * side].reshape(side, -1)
        mx = sd_2d.max() + 1e-8
        self.surface.blit(self._heatmap(sd_2d / mx, sm, hm // 2), (ox + sm + 4, y1))

        # SGC + SO (below tectum pair)
        self._lbl("SGC", ox, y1 + hm // 2 + 4, font=self.font_sm)
        sgc = self._t2np(brain.tectum.sgc.get_rate_e())
        n = len(sgc)
        side = int(np.sqrt(n)) or 1
        sgc_2d = sgc[:side * side].reshape(side, -1)
        mx = sgc_2d.max() + 1e-8
        self.surface.blit(self._heatmap(sgc_2d / mx, sm // 2, sm // 2),
                          (ox, y1 + hm // 2 + 14))

        self._lbl("SO", ox + sm // 2 + 4, y1 + hm // 2 + 4, font=self.font_sm)
        so = self._t2np(brain.tectum.so.get_rate_e())
        n = len(so)
        side = int(np.sqrt(n)) or 1
        so_2d = so[:side * side].reshape(side, -1)
        mx = so_2d.max() + 1e-8
        self.surface.blit(self._heatmap(so_2d / mx, sm // 2, sm // 2),
                          (ox + sm // 2 + 4, y1 + hm // 2 + 14))

        # ── Row 2: Goal + Energy + Pallium (y: 125-220) ──
        y2 = 125
        goal = out.get('goal', 2)
        gc = self.GOAL_COLORS[goal]
        self._lbl(f"GOAL: {self.GOAL_NAMES[goal]}", 6, y2, color=gc, font=self.font_lg)
        energy = brain.energy
        ef = np.clip(energy / 100.0, 0, 1)
        ec = (50, 200, 50) if ef > 0.3 else (220, 100, 50)
        pygame.draw.rect(self.surface, (30, 30, 40), (6, y2 + 18, 120, 10))
        pygame.draw.rect(self.surface, ec, (6, y2 + 18, int(120 * ef), 10))
        self._lbl(f"Energy: {energy:.0f}%", 6, y2 + 30)

        # Free energy
        fe = out.get('free_energy', 0.0)
        self._lbl(f"F: {fe:.4f}", 6, y2 + 42)
        loom = "LOOMING!" if out.get('looming', False) else ""
        if loom:
            self._lbl(loom, 80, y2 + 42, color=(255, 50, 50), font=self.font_md)

        # Pallium-S / Pallium-D heatmaps
        self._lbl("PAL-S", 150, y2 - 12, font=self.font_md)
        ps = self._t2np(brain.pallium.rate_s)
        n = len(ps)
        side = int(np.sqrt(n)) or 1
        ps_2d = ps[:side * side].reshape(side, -1)
        mx = ps_2d.max() + 1e-8
        self.surface.blit(self._heatmap(ps_2d / mx, 80, 80), (150, y2))

        self._lbl("PAL-D", 240, y2 - 12, font=self.font_md)
        pd = self._t2np(brain.pallium.rate_d)
        n = len(pd)
        side = int(np.sqrt(n)) or 1
        pd_2d = pd[:side * side].reshape(side, -1)
        mx = pd_2d.max() + 1e-8
        self.surface.blit(self._heatmap(pd_2d / mx, 60, 80), (240, y2))

        # TC / TRN
        self._lbl("TC", 315, y2 - 12, font=self.font_md)
        tc = self._t2np(brain.thalamus.tc_rate)
        tc_2d = tc.reshape(-1, 1)
        mx = tc_2d.max() + 1e-8
        self.surface.blit(self._heatmap(tc_2d / mx, 20, 80), (315, y2))

        self._lbl("TRN", 340, y2 - 12, font=self.font_md)
        trn = self._t2np(brain.thalamus.trn_rate)
        trn_2d = trn.reshape(-1, 1)
        mx = trn_2d.max() + 1e-8
        self.surface.blit(self._heatmap(trn_2d / mx, 15, 80), (340, y2))

        # Pred model state
        pm = brain.pred_model
        self._lbl(f"Pred: d={brain._pred_dist_gt:.0f} int={pm.intent:.2f}",
                  370, y2, font=self.font_sm)
        self._lbl(f"vis={'Y' if pm.visible else 'N'} ago={pm.steps_since_seen}",
                  370, y2 + 12, font=self.font_sm)
        self._lbl(f"Amy: {brain.amygdala_alpha:.2f}", 370, y2 + 24,
                  color=(220, 120, 80))

        # ── Row 3: Spike raster (y: 220-330) ──
        y3 = 220
        self._lbl("SPIKE RASTER (80 neurons, 100 steps)", 6, y3 - 12,
                  font=self.font_md)
        rw = W - 20
        rh = 100

        ordered = np.zeros_like(self._raster_buf)
        ptr = self._raster_ptr
        ordered[:self.RASTER_HISTORY - ptr] = self._raster_buf[ptr:]
        ordered[self.RASTER_HISTORY - ptr:] = self._raster_buf[:ptr]

        raster_img = np.zeros((self.RASTER_NEURONS, self.RASTER_HISTORY, 3),
                              dtype=np.uint8)
        for ni in range(self.RASTER_NEURONS):
            for ti in range(self.RASTER_HISTORY):
                val = ordered[ti, ni]
                if abs(val) > 0.01:
                    intensity = min(255, int(abs(val) * 600))
                    if ni < 20:      # Tectum SFGS-b — cyan
                        raster_img[ni, ti] = [0, intensity, intensity]
                    elif ni < 40:    # Pallium-S — lime
                        raster_img[ni, ti] = [intensity // 2, intensity, 0]
                    elif ni < 50:    # Pallium-D — blue
                        raster_img[ni, ti] = [50, 50, intensity]
                    elif ni < 60:    # Amygdala CeA — red
                        raster_img[ni, ti] = [intensity, 30, 30]
                    elif ni < 70:    # BG D1 — orange
                        raster_img[ni, ti] = [intensity, intensity * 2 // 3, 0]
                    else:            # Thalamus TC — yellow
                        raster_img[ni, ti] = [intensity, intensity, 0]

        rs = pygame.surfarray.make_surface(np.transpose(raster_img, (1, 0, 2)))
        rs = pygame.transform.scale(rs, (rw, rh))
        self.surface.blit(rs, (10, y3))
        pygame.draw.rect(self.surface, self.GRID, (10, y3, rw, rh), 1)

        lx = rw - 50
        self._lbl("SFGS-b", lx, y3 + 2, color=(0, 200, 200))
        self._lbl("Pal-S", lx, y3 + 16, color=(128, 200, 0))
        self._lbl("Pal-D", lx, y3 + 30, color=(80, 80, 255))
        self._lbl("CeA", lx, y3 + 44, color=(255, 80, 80))
        self._lbl("D1", lx, y3 + 58, color=(255, 170, 0))
        self._lbl("TC", lx, y3 + 72, color=(255, 255, 0))

        # ── Row 4: BG bars + Classification (y: 340-460) ──
        y4 = 340
        bw, bh = 130, 13

        # Basal ganglia
        self._lbl("BASAL GANGLIA", 6, y4 - 12, font=self.font_md)
        bg = brain.bg
        self._bar(6, y4, bw, bh, float(bg.d1_rate.mean()), 1.0,
                  (255, 170, 0), "D1")
        self._bar(6, y4 + bh + 3, bw, bh, float(bg.d2_rate.mean()), 1.0,
                  (0, 170, 255), "D2")
        self._bar(6, y4 + 2 * (bh + 3), bw, bh, float(bg.gpi_rate.mean()), 1.0,
                  (180, 80, 220), "GPi")
        self._bar(6, y4 + 3 * (bh + 3), bw, bh, out.get('bg_gate', 0.5), 1.0,
                  (50, 220, 50), "Gate")

        # Classification
        self._lbl("CLASSIFICATION", 260, y4 - 12, font=self.font_md)
        cls_probs = self._t2np(brain._cls_probs) if hasattr(brain, '_cls_probs') else np.zeros(5)
        for i, (name, color) in enumerate(zip(self.CLS_NAMES, self.CLS_COLORS)):
            by = y4 + i * (bh + 3)
            self._bar(260, by, bw, bh, float(cls_probs[i]), 1.0, color, name)

        # ── Row 5: Neuromod + Allostasis (y: 470-580) ──
        y5 = 470
        self._lbl("NEUROMODULATION", 6, y5 - 12, font=self.font_md)
        nm = brain.neuromod
        nm_data = [
            ("DA", nm.DA.item(), (255, 200, 50)),
            ("NA", nm.NA.item(), (100, 200, 255)),
            ("5-HT", nm.HT5.item(), (200, 100, 255)),
            ("ACh", nm.ACh.item(), (100, 255, 150)),
        ]
        for i, (name, val, col) in enumerate(nm_data):
            self._bar(6, y5 + i * (bh + 3), bw, bh, val, 1.0, col, name)

        # Allostasis
        self._lbl("ALLOSTASIS", 260, y5 - 12, font=self.font_md)
        allo = brain.allostasis
        allo_data = [
            ("Hunger", allo.hunger, (255, 140, 50)),
            ("Fatigue", allo.fatigue, (100, 180, 255)),
            ("Stress", allo.stress, (255, 80, 80)),
            ("Urgency", allo.urgency, (220, 220, 50)),
        ]
        for i, (name, val, col) in enumerate(allo_data):
            self._bar(260, y5 + i * (bh + 3), bw, bh, val, 1.0, col, name)

        # ── Row 6: Goal probs + Place cells (y: 550-650) ──
        y6 = 550
        self._lbl("GOAL PROBABILITIES", 6, y6 - 12, font=self.font_md)
        gp = self._t2np(brain.goal_probs) if brain.goal_probs is not None else np.zeros(4)
        for i, (name, col) in enumerate(zip(self.GOAL_NAMES, self.GOAL_COLORS)):
            self._bar(6, y6 + i * (bh + 3), bw, bh, float(gp[i]), 1.0, col, name)

        # Prediction error
        pe = self._t2np(brain.pallium.pred_error)
        pe_mean = float(np.mean(pe ** 2))
        self._lbl(f"PE²={pe_mean:.5f}", 6, y6 + 4 * (bh + 3) + 2)

        # Place cell mini-map
        self._lbl("PLACE CELLS", 260, y6 - 12, font=self.font_md)
        pc = brain.place
        pc_rate = self._t2np(pc.rate)
        cx = self._t2np(pc.cx)
        cy = self._t2np(pc.cy)
        # Draw minimap 120x90
        mw, mh = 120, 90
        mx0, my0 = 260, y6
        pygame.draw.rect(self.surface, (20, 20, 35), (mx0, my0, mw, mh))
        for i in range(len(cx)):
            if pc_rate[i] > 0.01:
                px = int(cx[i] / 800.0 * mw) + mx0
                py = int(cy[i] / 600.0 * mh) + my0
                intensity = min(255, int(pc_rate[i] * 500))
                r = max(1, int(pc_rate[i] * 4))
                pygame.draw.circle(self.surface, (intensity, intensity // 2, 0),
                                   (px, py), r)
        # Food/risk overlay
        food_val = float(pc.food_rate.mean())
        risk_val = float(pc.risk_rate.mean())
        self._lbl(f"food={food_val:.3f} risk={risk_val:.3f}",
                  260, y6 + mh + 2, font=self.font_sm)
        theta = pc.theta_phase
        self._lbl(f"theta={theta:.2f}rad", 260, y6 + mh + 14, font=self.font_sm)

        # ── Row 7: Step info (y: 670-700) ──
        y7 = 670
        eaten = info.get('total_eaten', 0)
        step = self._frame
        self._lbl(f"Step: {step}  Food: {eaten}  CMS: {brain.cms:.2f}",
                  6, y7, font=self.font_md)
        turn = out.get('turn', 0)
        speed = out.get('speed', 0)
        self._lbl(f"Turn: {turn:+.3f}  Speed: {speed:.2f}", 6, y7 + 16,
                  font=self.font_md)

        # Predator model confidence
        self._lbl(f"Pred intent: {brain.pred_model.intent:.2f}  "
                  f"vis: {'Y' if brain.pred_model.visible else 'N'}",
                  6, y7 + 32, font=self.font_sm)
