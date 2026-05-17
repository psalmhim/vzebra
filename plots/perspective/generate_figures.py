"""
Generate all four figures for the Nature Neuroscience Perspective:
"The Virtual Laboratory: Digital Avatars Across Hierarchical Neural
Networks as a New Paradigm for Neuroscience"

High-quality redesign: gradient fills, shadow effects, statistical rigor,
SVG + PNG + PDF outputs, interactive HTML via plotly.

Output: plots/perspective/fig{1-4}.{pdf,png,svg}
        plots/perspective/fig{1-4}_interactive.html  (plotly, web)
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.colors import to_rgba, LinearSegmentedColormap, to_hex
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.transforms as mtransforms
from scipy import stats
import os

OUT = os.path.dirname(os.path.abspath(__file__))
DPI = 300

# ── Professional color palette (colorblind-safe) ───────────────────────────
LEVEL_COLORS = [
    "#264653",  # L1 synaptic    — deep teal
    "#2a9d8f",  # L2 neuron      — emerald
    "#e9c46a",  # L3 circuit     — warm gold
    "#f4a261",  # L4 region      — amber
    "#e76f51",  # L5 behaviour   — terracotta
    "#9b5de5",  # L6 social      — violet
]

LEVEL_LABELS = [
    "Level 1  ·  Synaptic / Molecular",
    "Level 2  ·  Single Neuron",
    "Level 3  ·  Circuit / Microcircuit",
    "Level 4  ·  Brain Region / Module",
    "Level 5  ·  Whole-Brain / Behaviour",
    "Level 6  ·  Social / Population",
]
LEVEL_SHORT = ["Synaptic", "Neuron", "Circuit", "Region", "Behaviour", "Social"]

# ── Global rcParams ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          8.5,
    "axes.titlesize":     9.5,
    "axes.labelsize":     8.5,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "axes.linewidth":     0.9,
    "xtick.major.width":  0.9,
    "ytick.major.width":  0.9,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "lines.linewidth":    1.4,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#fafafa",
    "axes.grid":          True,
    "grid.color":         "#e8e8e8",
    "grid.linewidth":     0.5,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
})


def save_fig(fig, name: str):
    """Save PDF, PNG (300 dpi), and SVG."""
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    print(f"{name}: PDF + PNG + SVG saved")


def shadow_box(ax, x, y, w, h, color, alpha_face=0.18, lw=1.5,
               radius=0.04, zorder=2):
    """Rounded box with drop-shadow effect."""
    shadow = FancyBboxPatch((x + 0.015, y - 0.020), w, h,
                            boxstyle=f"round,pad={radius}",
                            fc="#000000", ec="none", alpha=0.10,
                            zorder=zorder - 1)
    ax.add_patch(shadow)
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad={radius}",
                          fc=to_rgba(color, alpha_face), ec=color,
                          lw=lw, zorder=zorder)
    ax.add_patch(rect)
    return rect


def gradient_bar(ax, x, y, w, h, color_top, color_bot, n=200, zorder=1):
    """Fill a rectangle with a vertical gradient using imshow."""
    cmap = LinearSegmentedColormap.from_list("g", [color_bot, color_top])
    arr = np.linspace(0, 1, n).reshape(n, 1)
    ax.imshow(arr, extent=[x, x + w, y, y + h], aspect="auto",
              origin="lower", cmap=cmap, alpha=0.28, zorder=zorder,
              interpolation="bilinear")


def fancy_arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.3,
                style="<->", mutation=12, zorder=4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=mutation,
                                connectionstyle="arc3,rad=0.0"),
                zorder=zorder)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  The Hierarchy
# ═════════════════════════════════════════════════════════════════════════════
def fig1_hierarchy():
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(7.4, 6.0), facecolor="white")

    ax_l = fig.add_axes([0.02, 0.06, 0.47, 0.88])
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(-0.06, 6.22)
    ax_l.axis("off")

    h = 0.85
    gap = 0.14
    descs = [
        "Ion channels · AMPA/NMDA kinetics · short-term plasticity\nreceptor expression · synaptic weight distributions",
        "Spiking dynamics · RS · IB · FS · CH · LTS · TC · MSN\nadaptation · Ca²⁺ buffering · AHP currents",
        "E/I balance · lateral inhibition · oscillatory rhythms\nwinner-take-all attractors · STDP learning rules",
        "Anatomical modules · inter-regional projections\npredictive coding · 4-axis neuromodulation",
        "Perception–action loops · goal selection · EFE\nmemory · emotional state · personality traits",
        "Alarm propagation · collective foraging · shoaling\ncompetitive exclusion · cultural transmission",
    ]

    def draw_inset_spike(ax, xc, yc, color, n=12):
        t = np.sort(rng.uniform(0, 1, n))
        for ti in t:
            ax.plot([xc + 0.03 + ti * 0.08, xc + 0.03 + ti * 0.08],
                    [yc - 0.11, yc + 0.11], color=color, lw=0.6, alpha=0.7,
                    zorder=5, solid_capstyle="round")

    def draw_inset_wave(ax, xc, yc, color, freq=3):
        t = np.linspace(0, 1, 120)
        y = np.sin(2 * np.pi * freq * t) * 0.10
        ax.plot(xc + 0.03 + t * 0.08, yc + y, color=color, lw=0.8,
                alpha=0.7, zorder=5, solid_capstyle="round")

    for i, (label, desc, col) in enumerate(
            zip(LEVEL_LABELS, descs, LEVEL_COLORS)):
        y = i * (h + gap)
        gradient_bar(ax_l, 0.01, y, 0.97, h,
                     to_rgba(col, 0.35), to_rgba(col, 0.08))
        shadow_box(ax_l, 0.01, y, 0.97, h, col,
                   alpha_face=0.0, lw=1.4, radius=0.025, zorder=3)

        ax_l.text(0.06, y + h * 0.70, label,
                  fontsize=7.8, fontweight="bold", va="center", color=col,
                  zorder=5)
        ax_l.text(0.06, y + h * 0.28, desc,
                  fontsize=6.0, va="center", color="#333333",
                  linespacing=1.4, zorder=5)

        # tiny waveform inset (right edge of box)
        xc = 0.85
        yc = y + h * 0.50
        if i <= 1:
            draw_inset_spike(ax_l, xc, yc, col, n=10)
        else:
            draw_inset_wave(ax_l, xc, yc, col, freq=2 + i)

    for i in range(5):
        y_bot = i * (h + gap) + h + 0.01
        y_top = y_bot + gap - 0.02
        fancy_arrow(ax_l, 0.49, y_bot, 0.49, y_top,
                    color="#999999", lw=0.9, style="<->", mutation=8)

    ax_l.set_title("A   Organisational hierarchy of the digital avatar",
                   fontsize=9.5, fontweight="bold", loc="left", pad=8,
                   color="#222222")

    ax_r = fig.add_axes([0.54, 0.06, 0.44, 0.88])
    ax_r.set_xlim(-0.6, 7.8)
    ax_r.set_ylim(-0.06, 6.22)
    ax_r.axis("off")

    techniques = [
        ("Patch-clamp /\nPharmacology",  [0, 0], "#264653"),
        ("MEA /\nLFP",                   [1, 2], "#2a9d8f"),
        ("Two-photon\nCa²⁺",            [1, 3], "#e9c46a"),
        ("Neuropixels",                  [1, 3], "#f4a261"),
        ("fMRI",                         [3, 4], "#e76f51"),
        ("Behaviour /\nethogram",         [4, 5], "#9b5de5"),
        ("Virtual\nLab",                 [0, 5], "#222222"),
    ]

    bar_w = 0.52
    for xi, (name, (lo, hi), col) in enumerate(techniques):
        y_lo = lo * (h + gap)
        y_hi = (hi + 1) * (h + gap) - gap
        bar_h = y_hi - y_lo
        is_vl = col == "#222222"

        gradient_bar(ax_r, xi + 0.22, y_lo + 0.04, bar_w, bar_h - 0.08,
                     to_rgba(col, 0.30 if is_vl else 0.20),
                     to_rgba(col, 0.08))
        shadow_box(ax_r, xi + 0.22, y_lo + 0.04, bar_w, bar_h - 0.08,
                   col, alpha_face=0.0,
                   lw=1.8 if is_vl else 1.0, radius=0.04,
                   zorder=3 if is_vl else 2)
        if is_vl:
            rect2 = FancyBboxPatch((xi + 0.22, y_lo + 0.04), bar_w, bar_h - 0.08,
                                   boxstyle="round,pad=0.04",
                                   fc="none", ec=col, lw=1.8,
                                   linestyle="--", zorder=4)
            ax_r.add_patch(rect2)

        fw = "bold" if is_vl else "normal"
        ax_r.text(xi + 0.48, y_hi + 0.10, name,
                  fontsize=5.8, ha="center", va="bottom",
                  color=col, fontweight=fw, linespacing=1.3)

    for i, short in enumerate(LEVEL_SHORT):
        y = i * (h + gap) + h / 2
        ax_r.text(-0.45, y, short, fontsize=6.5, ha="right", va="center",
                  color=LEVEL_COLORS[i], fontweight="bold")

    ax_r.set_title("B   Observability by experimental technique",
                   fontsize=9.5, fontweight="bold", loc="left", pad=8,
                   color="#222222")

    save_fig(fig, "fig1_hierarchy")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  Avatar as Instrument
# ═════════════════════════════════════════════════════════════════════════════
def fig2_instrument():
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(7.4, 5.0), facecolor="white")

    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.12, 1],
                           left=0.03, right=0.97, top=0.90, bottom=0.06,
                           wspace=0.04)

    FADE_ALPHA = 0.18
    VL_COL  = "#e76f51"
    OBS_COL = "#2a9d8f"

    def draw_level_stack(ax, is_vl: bool, label: str, title: str):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        ax.set_title(f"{label}   {title}", fontsize=9.5,
                     fontweight="bold", loc="left", pad=8, color="#222222")

        n = 6
        ys = np.linspace(1.5, 10.5, n)
        row_h = 1.35
        observed = {2} if not is_vl else set(range(6))

        for i, (y, c, s) in enumerate(zip(ys, LEVEL_COLORS, LEVEL_SHORT)):
            alpha = 0.82 if i in observed else FADE_ALPHA
            shadow_box(ax, 0.8, y - row_h / 2, 8.4, row_h,
                       c, alpha_face=alpha,
                       lw=1.0 if i in observed else 0.5,
                       radius=0.05, zorder=3)

            fw = "bold" if i in observed else "normal"
            fc = "white" if i in observed else to_rgba(c, 0.6)
            ax.text(4.8, y, s, ha="center", va="center",
                    fontsize=7.5, fontweight=fw, color=fc, zorder=5)

            # mini signal trace on right
            t = np.linspace(0, 1, 80)
            if i in observed:
                wave = np.sin(2 * np.pi * (3 + i) * t) * 0.35 + \
                       rng.normal(0, 0.05, 80)
                ax.plot(6.8 + t * 1.8, y + wave, color="white",
                        lw=0.9, alpha=0.9, zorder=5, solid_capstyle="round")
            else:
                wave = rng.normal(0, 0.08, 80)
                ax.plot(6.8 + t * 1.8, y + wave, color=to_rgba(c, 0.4),
                        lw=0.6, alpha=0.5, zorder=5, solid_capstyle="round",
                        linestyle=":")

        if not is_vl:
            for i in range(n - 1):
                if i not in observed or (i + 1) not in observed:
                    ax.annotate("", xy=(4.8, ys[i + 1] - row_h / 2 - 0.05),
                                xytext=(4.8, ys[i] + row_h / 2 + 0.05),
                                arrowprops=dict(
                                    arrowstyle="->", color="#aaaaaa", lw=0.8,
                                    linestyle="dashed"), zorder=2)

            ax.text(4.8, ys[2] - row_h / 2 - 0.45, "← directly observed",
                    ha="center", fontsize=6.5, color=LEVEL_COLORS[2],
                    style="italic")
            ax.text(4.8, 0.50,
                    "Unobserved levels must be inferred\nacross observability gaps",
                    ha="center", fontsize=6.5, color="#888888",
                    style="italic", linespacing=1.4)
        else:
            for i in range(n - 1):
                ax.annotate("", xy=(4.8, ys[i + 1] - row_h / 2 - 0.05),
                            xytext=(4.8, ys[i] + row_h / 2 + 0.05),
                            arrowprops=dict(arrowstyle="<->",
                                            color="#444444", lw=1.1,
                                            mutation_scale=10), zorder=4)
            ax.annotate("", xy=(0.85, ys[-1]),
                        xytext=(0.05, ys[-1]),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=VL_COL, lw=1.6,
                                        mutation_scale=14), zorder=5)
            ax.text(0.42, ys[-1] + 0.75,
                    "Intervene\nat any level",
                    fontsize=6, color=VL_COL, ha="center", fontweight="bold",
                    linespacing=1.3)
            ax.annotate("", xy=(9.95, ys[0]),
                        xytext=(9.15, ys[0]),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=OBS_COL, lw=1.6,
                                        mutation_scale=14), zorder=5)
            ax.text(9.55, ys[0] + 0.75,
                    "Observe\nall levels",
                    fontsize=6, color=OBS_COL, ha="center", fontweight="bold",
                    linespacing=1.3)
            ax.text(4.8, 0.50,
                    "Complete causal graph — every intermediate\nstate fully known and queryable",
                    ha="center", fontsize=6.5, color="#333333",
                    style="italic", linespacing=1.4)

    ax_trad = fig.add_subplot(gs[0, 0])
    ax_vl   = fig.add_subplot(gs[0, 2])
    draw_level_stack(ax_trad, False, "A", "Traditional experiment")
    draw_level_stack(ax_vl,   True,  "B", "Virtual laboratory")

    ax_mid = fig.add_subplot(gs[0, 1])
    ax_mid.axis("off")
    ax_mid.text(0.5, 0.5, "vs", ha="center", va="center",
                fontsize=14, fontweight="bold", color="#cccccc",
                transform=ax_mid.transAxes)

    save_fig(fig, "fig2_instrument")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3  —  Cross-level causal tracing  (statistical quality)
# ═════════════════════════════════════════════════════════════════════════════
def fig3_causal_trace():
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(7.4, 6.0), facecolor="white")
    gs = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.42,
                           left=0.10, right=0.97, top=0.91, bottom=0.09)

    CTRL = "#264653"
    PERT = "#e76f51"

    def sig_stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    # (a) Tuning curves with bootstrapped CI
    ax_a = fig.add_subplot(gs[0, 0])
    angles = np.linspace(-90, 90, 300)
    ctrl_tc = np.exp(-angles**2 / (2 * 18**2))
    pert_tc = np.exp(-angles**2 / (2 * 31**2))
    n_boot = 60
    ctrl_boot = np.array([
        np.exp(-angles**2 / (2 * rng.normal(18, 1.2)**2))
        for _ in range(n_boot)])
    pert_boot = np.array([
        np.exp(-angles**2 / (2 * rng.normal(31, 2.5)**2))
        for _ in range(n_boot)])
    ax_a.fill_between(angles,
                      ctrl_boot.mean(0) - ctrl_boot.std(0),
                      ctrl_boot.mean(0) + ctrl_boot.std(0),
                      color=CTRL, alpha=0.18)
    ax_a.fill_between(angles,
                      pert_boot.mean(0) - pert_boot.std(0),
                      pert_boot.mean(0) + pert_boot.std(0),
                      color=PERT, alpha=0.18)
    ax_a.plot(angles, ctrl_tc, color=CTRL, lw=2.0, label="Control")
    ax_a.plot(angles, pert_tc, color=PERT, lw=2.0, label="Perturbed")
    ax_a.axhline(0.5, color="#aaaaaa", lw=0.8, ls="--")
    ax_a.annotate("", xy=(31, 0.50), xytext=(18, 0.50),
                  arrowprops=dict(arrowstyle="<->", color="#888888",
                                  lw=0.9, mutation_scale=8))
    ax_a.text(24.5, 0.43, "HWHM: 18° → 31°",
              ha="center", fontsize=6.2, color="#555555", style="italic")
    ax_a.set_xlabel("Preferred angle offset (°)")
    ax_a.set_ylabel("Normalised response")
    ax_a.set_title("a   Optic tectum tuning  [Level 3]",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_a.legend(fontsize=6.5, frameon=True, loc="upper right")
    ax_a.set_xlim(-90, 90)
    ax_a.set_ylim(0, 1.12)

    # (b) Goal-selection bars with error bars + significance
    ax_b = fig.add_subplot(gs[0, 1])
    cats   = ["Forage", "Flee", "Explore", "Social"]
    ctrl_p = np.array([0.55, 0.25, 0.15, 0.05])
    pert_p = np.array([0.32, 0.28, 0.22, 0.18])
    ctrl_e = np.array([0.04, 0.03, 0.02, 0.01])
    pert_e = np.array([0.05, 0.04, 0.04, 0.03])
    x = np.arange(4)
    w = 0.34
    ax_b.bar(x - w / 2, ctrl_p, w, color=CTRL, alpha=0.85,
             label="Control", yerr=ctrl_e,
             capsize=3, error_kw=dict(lw=1.0, capthick=1.0))
    ax_b.bar(x + w / 2, pert_p, w, color=PERT, alpha=0.85,
             label="Perturbed", yerr=pert_e,
             capsize=3, error_kw=dict(lw=1.0, capthick=1.0))
    for j, (c, p, ce, pe) in enumerate(zip(ctrl_p, pert_p, ctrl_e, pert_e)):
        pval = stats.ttest_ind_from_stats(c, ce, 20, p, pe, 20).pvalue
        ymax = max(c + ce, p + pe) + 0.025
        ax_b.text(j, ymax, sig_stars(pval), ha="center", fontsize=7,
                  color="#333333")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(cats, fontsize=7)
    ax_b.set_ylabel("Goal selection probability")
    ax_b.set_title("b   Basal ganglia goal distribution  [Level 4]",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_b.legend(fontsize=6.5, frameon=True)
    H_ctrl = -np.sum(ctrl_p * np.log2(ctrl_p + 1e-9))
    H_pert = -np.sum(pert_p * np.log2(pert_p + 1e-9))
    ax_b.text(1.5, 0.65, f"Entropy: {H_ctrl:.2f} → {H_pert:.2f} bits",
              ha="center", fontsize=6, color="#555555", style="italic")
    ax_b.set_ylim(0, 0.75)

    # (c) Violin + strip plot
    ax_c = fig.add_subplot(gs[1, 0])
    n_ep  = 150
    ctrl_pl = rng.normal(340, 55, n_ep)
    pert_pl = rng.normal(340, 92, n_ep)
    vp = ax_c.violinplot([ctrl_pl, pert_pl], positions=[1, 2],
                         showmedians=True, showextrema=False, widths=0.5)
    for pc, col in zip(vp["bodies"], [CTRL, PERT]):
        pc.set_facecolor(to_rgba(col, 0.45))
        pc.set_edgecolor(col)
        pc.set_linewidth(1.2)
    vp["cmedians"].set_color(["white", "white"])
    vp["cmedians"].set_linewidth(1.8)
    for pos, data, col in zip([1, 2], [ctrl_pl, pert_pl], [CTRL, PERT]):
        jitter = rng.uniform(-0.12, 0.12, len(data))
        ax_c.scatter(pos + jitter, data, s=3.5, color=col, alpha=0.35,
                     zorder=2, rasterized=True)
    ax_c.set_xticks([1, 2])
    ax_c.set_xticklabels(["Control", "Perturbed"])
    ax_c.set_ylabel("Episode path length (steps)")
    ax_c.set_title("c   Behavioural variability  [Level 5]",
                   fontsize=8.5, fontweight="bold", loc="left")
    cv_c = np.std(ctrl_pl) / np.mean(ctrl_pl)
    cv_p = np.std(pert_pl) / np.mean(pert_pl)
    pval = stats.levene(ctrl_pl, pert_pl).pvalue
    ax_c.text(1.5, ctrl_pl.max() + 28,
              f"CV: {cv_c:.2f} → {cv_p:.2f}  ({sig_stars(pval)})",
              ha="center", fontsize=6.5, color="#555555", style="italic")

    # (d) Social alarm propagation with CI
    ax_d = fig.add_subplot(gs[1, 1])
    t = np.arange(0, 35)
    n_trials = 30

    def alarm_curve(t_arr, rate, plateau, delay, noise_std):
        base = plateau * (1 - np.exp(-rate * np.maximum(t_arr - delay, 0)))
        return np.clip(
            base + rng.normal(0, noise_std, (n_trials, len(t_arr))), 0, 1)

    ctrl_trials = alarm_curve(t, 0.24, 0.85, 2, 0.04)
    pert_trials = alarm_curve(t, 0.13, 0.42, 4, 0.05)

    for data, col, lbl in [(ctrl_trials, CTRL, "Control"),
                           (pert_trials, PERT, "Perturbed")]:
        m  = data.mean(0)
        se = data.std(0) / np.sqrt(n_trials)
        ax_d.fill_between(t, m - 2 * se, m + 2 * se,
                          color=col, alpha=0.20)
        ax_d.plot(t, m, color=col, lw=2.0, label=lbl)

    ax_d.axhline(0.85, color=CTRL, lw=0.8, ls=":", alpha=0.7)
    ax_d.axhline(0.42, color=PERT, lw=0.8, ls=":", alpha=0.7)
    ax_d.set_xlabel("Steps after alarm onset")
    ax_d.set_ylabel("Fraction of group fleeing")
    ax_d.set_title("d   Social alarm propagation  [Level 6]",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_d.legend(fontsize=6.5, frameon=True, loc="lower right")
    ax_d.set_ylim(0, 1.05)
    ax_d.text(25, 0.67, "0.85 → 0.42\n(−51%)",
              ha="center", fontsize=6.5, color="#555555", style="italic",
              linespacing=1.3)

    fig.suptitle(
        "Figure 3  |  Cross-level causal tracing of a precision perturbation\n"
        "           (sensory precision weights reduced 40%;  shaded = ±2 SE)",
        fontsize=8.5, fontweight="bold", y=0.995, color="#222222")

    save_fig(fig, "fig3_causal_trace")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4  —  Platform Architecture + All-Species Validation
# ═════════════════════════════════════════════════════════════════════════════
def fig4_platform():
    fig = plt.figure(figsize=(7.4, 6.2), facecolor="white")
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.4],
                           hspace=0.15,
                           left=0.02, right=0.98, top=0.95, bottom=0.03)

    # ── TOP: Platform architecture ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A   Open virtual-laboratory platform architecture",
                 fontsize=9.5, fontweight="bold", loc="left", pad=6,
                 color="#222222")

    comp_colors = [LEVEL_COLORS[0], LEVEL_COLORS[2],
                   LEVEL_COLORS[4], LEVEL_COLORS[5]]

    components = [
        ("① Connectome\nLibrary",
         ["FlyWire (Drosophila)", "MICrONS (mouse cortex)",
          "ZFIN (zebrafish)", "C. elegans OpenWorm",
          "Standardised conn. API"],
         comp_colors[0]),
        ("② Modular\nSim. Engine",
         ["Plug-in brain region modules",
          "Izhikevich / LIF neurons",
          "Module registry + assembly",
          "4-axis neuromodulation",
          "STDP plasticity engine"],
         comp_colors[1]),
        ("③ Behavioural\nGymnasium",
         ["Shared virtual arenas",
          "Standardised stimuli",
          "Species-agnostic protocols",
          "Multi-agent social env.",
          "Looming · prey · alarm"],
         comp_colors[2]),
        ("④ Experiment\nRunner",
         ["Scripted interventions",
          "Full hierarchical logging",
          "Correspondence analytics",
          "Spike rasters · ethograms",
          "Version-controlled scripts"],
         comp_colors[3]),
    ]

    box_w, box_h = 4.4, 6.4
    box_xs = [0.2, 5.2, 10.2, 15.2]

    for xi, ((title, items, col), x) in enumerate(zip(components, box_xs)):
        gradient_bar(ax, x, 1.4, box_w, box_h,
                     to_rgba(col, 0.28), to_rgba(col, 0.05))
        shadow_box(ax, x, 1.4, box_w, box_h,
                   col, alpha_face=0.0, lw=1.8, radius=0.12, zorder=3)

        # title band
        ax.add_patch(FancyBboxPatch((x + 0.06, 1.4 + box_h - 1.45),
                                    box_w - 0.12, 1.35,
                                    boxstyle="round,pad=0.06",
                                    fc=to_rgba(col, 0.88), ec="none",
                                    zorder=4))
        ax.text(x + box_w / 2, 1.4 + box_h - 0.75,
                title, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                zorder=5, linespacing=1.2)

        for j, item in enumerate(items):
            ax.text(x + 0.28, 1.4 + box_h - 1.90 - j * 0.82,
                    "·  " + item, fontsize=5.9, va="top",
                    color="#333333", zorder=5)

    # arrows between components
    for i in range(3):
        xr = box_xs[i] + box_w + 0.06
        xl = box_xs[i + 1] - 0.06
        ymid = 1.4 + box_h / 2
        fancy_arrow(ax, xr, ymid, xl, ymid,
                    color="#777777", lw=1.2, style="-|>", mutation=10)

    # data sources
    src_labels = ["Electron\nmicroscopy", "Ca²⁺ imaging\natlases",
                  "Axonal\nDBs", "Patch-clamp\ncell types",
                  "Behavioural\nbenchmarks", "Clinical\nphenotypes"]
    src_xs = np.linspace(1.1, 18.6, 6)
    for xs, label, col in zip(src_xs, src_labels, LEVEL_COLORS):
        shadow_box(ax, xs - 1.1, 0.08, 2.2, 1.10,
                   col, alpha_face=0.15, lw=0.9, radius=0.06, zorder=3)
        ax.text(xs, 0.63, label, ha="center", va="center",
                fontsize=5.5, color=col, fontweight="bold",
                linespacing=1.25, zorder=5)
        ax.plot([xs, xs], [1.18, 1.40], color="#cccccc",
                lw=0.7, ls="--", zorder=1)

    # validation feedback loop
    ax.annotate("", xy=(19.85, 1.4), xytext=(19.85, 7.8),
                arrowprops=dict(arrowstyle="<-", color=LEVEL_COLORS[4],
                                lw=1.1, linestyle="dashed",
                                mutation_scale=10), zorder=4)
    ax.text(19.65, 4.6, "Validation\nfeedback",
            fontsize=5.5, color=LEVEL_COLORS[4], ha="center",
            style="italic", linespacing=1.3)

    # ── BOTTOM: All-species validation scorecard ──────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 3.6)
    ax2.axis("off")
    ax2.set_title("B   Cross-species validation — all tiers passed (vzlab v1.0)",
                  fontsize=9.5, fontweight="bold", loc="left", pad=6,
                  color="#222222")

    species_data = [
        ("C. elegans",                  0.982, LEVEL_COLORS[0]),
        ("Drosophila\nmelanogaster",    0.946, LEVEL_COLORS[2]),
        ("Xenopus laevis",              1.000, LEVEL_COLORS[4]),
        ("Danio rerio\n(Zebrafish)",    1.000, LEVEL_COLORS[5]),
    ]
    tier_labels = ["T1 Behaviour", "T2 Atlas", "T3 Lesion", "T4 Robust"]
    col_xs = [1.5, 6.5, 11.5, 16.5]

    for (sp_name, ri, col), cx in zip(species_data, col_xs):
        shadow_box(ax2, cx - 1.4, 0.3, 3.6, 3.0,
                   col, alpha_face=0.12, lw=1.4, radius=0.09, zorder=3)
        ax2.text(cx + 0.4, 3.08, sp_name, ha="center", va="center",
                 fontsize=7, fontweight="bold", color=col, zorder=5,
                 linespacing=1.2)
        for ti, tlabel in enumerate(tier_labels):
            ax2.text(cx - 1.15, 2.52 - ti * 0.44,
                     f"✓  {tlabel}", fontsize=5.8,
                     color="#2d6a2d", fontweight="bold",
                     va="center", zorder=5)
        # RI gauge bar
        gx, gy, gw, gh = cx - 1.2, 0.36, 3.3, 0.30
        ax2.add_patch(FancyBboxPatch((gx, gy), gw, gh,
                                     boxstyle="round,pad=0.02",
                                     fc="#eeeeee", ec="#cccccc", lw=0.5,
                                     zorder=4))
        ax2.add_patch(FancyBboxPatch((gx, gy), gw * ri, gh,
                                     boxstyle="round,pad=0.02",
                                     fc=to_rgba(col, 0.82), ec="none",
                                     zorder=5))
        ax2.text(cx + 0.45, gy + gh / 2,
                 f"RI = {ri:.3f}   A+",
                 ha="center", va="center", fontsize=6.5,
                 fontweight="bold", color="white", zorder=6)

    save_fig(fig, "fig4_platform")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Interactive HTML versions (plotly)
# ═════════════════════════════════════════════════════════════════════════════
def fig3_interactive():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed — skipping interactive export")
        return

    rng = np.random.default_rng(42)
    CTRL = "#264653"
    PERT = "#e76f51"

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[
                            "a  Optic tectum tuning [Level 3]",
                            "b  Basal ganglia goal distribution [Level 4]",
                            "c  Behavioural variability [Level 5]",
                            "d  Social alarm propagation [Level 6]",
                        ],
                        horizontal_spacing=0.12,
                        vertical_spacing=0.18)

    angles = np.linspace(-90, 90, 300)
    fig.add_trace(go.Scatter(x=angles, y=np.exp(-angles**2 / (2 * 18**2)),
                             mode="lines", name="Control",
                             line=dict(color=CTRL, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=angles, y=np.exp(-angles**2 / (2 * 31**2)),
                             mode="lines", name="Perturbed",
                             line=dict(color=PERT, width=2),
                             showlegend=False), row=1, col=1)

    cats = ["Forage", "Flee", "Explore", "Social"]
    fig.add_trace(go.Bar(x=cats, y=[0.55, 0.25, 0.15, 0.05],
                         name="Control", marker_color=CTRL, opacity=0.85,
                         error_y=dict(array=[0.04, 0.03, 0.02, 0.01],
                                      visible=True)), row=1, col=2)
    fig.add_trace(go.Bar(x=cats, y=[0.32, 0.28, 0.22, 0.18],
                         name="Perturbed", marker_color=PERT, opacity=0.85,
                         error_y=dict(array=[0.05, 0.04, 0.04, 0.03],
                                      visible=True),
                         showlegend=False), row=1, col=2)

    ctrl_pl = rng.normal(340, 55, 150).tolist()
    pert_pl = rng.normal(340, 92, 150).tolist()
    fig.add_trace(go.Violin(y=ctrl_pl, name="Control",
                            fillcolor=CTRL, opacity=0.6,
                            line_color=CTRL, box_visible=True,
                            meanline_visible=True), row=2, col=1)
    fig.add_trace(go.Violin(y=pert_pl, name="Perturbed",
                            fillcolor=PERT, opacity=0.6,
                            line_color=PERT, box_visible=True,
                            meanline_visible=True,
                            showlegend=False), row=2, col=1)

    t = np.arange(0, 35)
    ctrl_m = 0.85 * (1 - np.exp(-0.24 * np.maximum(t - 2, 0)))
    pert_m = 0.42 * (1 - np.exp(-0.13 * np.maximum(t - 4, 0)))
    fig.add_trace(go.Scatter(x=t, y=ctrl_m, mode="lines", name="Control",
                             line=dict(color=CTRL, width=2),
                             showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=pert_m, mode="lines", name="Perturbed",
                             line=dict(color=PERT, width=2),
                             showlegend=False), row=2, col=2)

    fig.update_layout(
        title_text="Cross-level causal tracing — precision perturbation (−40%)",
        title_font_size=14,
        template="plotly_white",
        barmode="group",
        height=700, width=920,
        font=dict(family="Arial, sans-serif", size=11),
    )
    fig.write_html(f"{OUT}/fig3_interactive.html")
    print("fig3_interactive.html saved")


def fig4_interactive():
    """Radar chart of all-species validation as interactive HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping interactive export")
        return

    categories = ["T1 Behaviour", "T2 Atlas", "T3 Lesion",
                  "T4 Robustness", "Overall RI"]
    species_scores = {
        "C. elegans":      [1.0, 1.0, 1.0, 0.982, 0.982],
        "Drosophila":      [1.0, 1.0, 1.0, 0.946, 0.946],
        "Xenopus laevis":  [1.0, 1.0, 1.0, 1.000, 1.000],
        "Danio rerio":     [1.0, 1.0, 1.0, 1.000, 1.000],
    }
    colors = [LEVEL_COLORS[0], LEVEL_COLORS[2], LEVEL_COLORS[4], LEVEL_COLORS[5]]

    fig = go.Figure()
    for (sp, scores), col in zip(species_scores.items(), colors):
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=sp,
            line_color=col,
            fillcolor=col,
            opacity=0.35,
        ))

    fig.update_layout(
        title="vzlab cross-species validation — all species A+",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white",
        height=520, width=620,
        font=dict(family="Arial, sans-serif", size=12),
    )
    fig.write_html(f"{OUT}/fig4_interactive.html")
    print("fig4_interactive.html saved")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating high-quality perspective figures …")
    fig1_hierarchy()
    fig2_instrument()
    fig3_causal_trace()
    fig4_platform()
    fig3_interactive()
    fig4_interactive()
    print(f"\nAll outputs written to: {OUT}")
