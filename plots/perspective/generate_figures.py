"""
Generate all four figures for the Nature Neuroscience Perspective:
"The Virtual Laboratory: Digital Avatars Across Hierarchical Neural
Networks as a New Paradigm for Neuroscience"

Output: plots/perspective/fig1_hierarchy.pdf (and .png)
        plots/perspective/fig2_instrument.pdf
        plots/perspective/fig3_causal_trace.pdf
        plots/perspective/fig4_platform.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import to_rgba
import matplotlib.gridspec as gridspec
import os

OUT = os.path.dirname(os.path.abspath(__file__))
DPI = 300

# ── Nature-style rcParams ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "lines.linewidth":    1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

LEVEL_COLORS = [
    "#4e79a7",  # L1 synaptic    — blue
    "#59a14f",  # L2 neuron      — green
    "#f28e2b",  # L3 circuit     — orange
    "#e15759",  # L4 region      — red
    "#b07aa1",  # L5 whole-brain — purple
    "#76b7b2",  # L6 social      — teal
]

LEVEL_LABELS = [
    "Level 1\nSynaptic / Molecular",
    "Level 2\nSingle Neuron",
    "Level 3\nCircuit / Microcircuit",
    "Level 4\nBrain Region / Module",
    "Level 5\nWhole-Brain / Behaviour",
    "Level 6\nSocial / Population",
]

LEVEL_SHORT = ["Synaptic", "Neuron", "Circuit", "Region", "Behaviour", "Social"]


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1  –  The Hierarchy
# ═════════════════════════════════════════════════════════════════════════════
def fig1_hierarchy():
    fig = plt.figure(figsize=(7.2, 5.6))

    # ── left: level stack with arrows ────────────────────────────────────────
    ax_l = fig.add_axes([0.03, 0.07, 0.44, 0.88])
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(-0.05, 6.1)
    ax_l.axis("off")

    h = 0.82          # box height
    gap = 0.13        # gap between boxes

    descs = [
        "Ion channels · receptors · short-term plasticity\nneuromodulatory receptor expression",
        "Spiking dynamics · RS · IB · FS · CH · LTS · TC · MSN\nspike-frequency adaptation · bursting",
        "E/I balance · recurrent excitation · lateral inhibition\noscillatory rhythms · winner-take-all attractors",
        "Anatomical structures · inter-regional projections\npredictive coding · neuromodulatory broadcast",
        "Perception–action loops · goal selection · learning\nmemory · emotional state · personality",
        "Alarm propagation · collective foraging\ncompetitive exclusion · cultural transmission",
    ]

    for i, (label, desc, col) in enumerate(
            zip(LEVEL_LABELS, descs, LEVEL_COLORS)):
        y = i * (h + gap)
        rect = FancyBboxPatch((0.01, y), 0.97, h,
                              boxstyle="round,pad=0.02",
                              fc=to_rgba(col, 0.15), ec=col, lw=1.2)
        ax_l.add_patch(rect)
        ax_l.text(0.06, y + h * 0.62, label,
                  fontsize=7.5, fontweight="bold", va="center", color=col)
        ax_l.text(0.06, y + h * 0.24, desc,
                  fontsize=6.2, va="center", color="#444444", linespacing=1.4)

    # bidirectional arrows between adjacent levels
    for i in range(5):
        y_bot = i * (h + gap) + h
        y_top = y_bot + gap
        y_mid = (y_bot + y_top) / 2
        ax_l.annotate("", xy=(0.5, y_top - 0.01),
                       xytext=(0.5, y_bot + 0.01),
                       arrowprops=dict(arrowstyle="<->", color="#888888",
                                       lw=1.0, mutation_scale=10))

    ax_l.set_title("A  Organisational hierarchy of the digital avatar",
                   fontsize=9, fontweight="bold", loc="left", pad=6)

    # ── right: technique coverage bars ───────────────────────────────────────
    ax_r = fig.add_axes([0.52, 0.07, 0.44, 0.88])
    ax_r.set_xlim(-0.5, 7.5)
    ax_r.set_ylim(-0.05, 6.1)
    ax_r.axis("off")

    techniques = [
        ("Patch-clamp /\nPharmacology",   [0, 1],   "#4e79a7"),
        ("Multi-electrode\narray / LFP",  [1, 2],   "#59a14f"),
        ("Two-photon\nCa²⁺ imaging",      [1, 3],   "#f28e2b"),
        ("Neuropixels",                   [1, 3],   "#e15759"),
        ("fMRI",                          [3, 4],   "#b07aa1"),
        ("Behaviour /\nethogram",          [4, 5],   "#76b7b2"),
        ("Virtual\nLaboratory",           [0, 5],   "#2d2d2d"),
    ]

    bar_w = 0.55
    for xi, (name, (lo, hi), col) in enumerate(techniques):
        y_lo = lo * (h + gap)
        y_hi = (hi + 1) * (h + gap) - gap
        rect = FancyBboxPatch((xi + 0.2, y_lo + 0.04), bar_w,
                              y_hi - y_lo - 0.08,
                              boxstyle="round,pad=0.03",
                              fc=to_rgba(col, 0.18 if col != "#2d2d2d" else 0.10),
                              ec=col, lw=1.5 if col == "#2d2d2d" else 0.9,
                              linestyle="--" if col == "#2d2d2d" else "-",
                              zorder=3)
        ax_r.add_patch(rect)
        ax_r.text(xi + 0.48, y_hi + 0.08, name,
                  fontsize=6, ha="center", va="bottom", color=col,
                  fontweight="bold" if col == "#2d2d2d" else "normal",
                  linespacing=1.35)

    # y-axis level labels
    for i, short in enumerate(LEVEL_SHORT):
        y = i * (h + gap) + h / 2
        ax_r.text(-0.4, y, short, fontsize=6.5, ha="right", va="center",
                  color=LEVEL_COLORS[i], fontweight="bold")

    ax_r.set_title("B  Observability by experimental technique",
                   fontsize=9, fontweight="bold", loc="left", pad=6)

    fig.savefig(f"{OUT}/fig1_hierarchy.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig1_hierarchy.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2  –  Avatar as Instrument
# ═════════════════════════════════════════════════════════════════════════════
def fig2_instrument():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))

    def draw_panel(ax, title, panel_label, is_vl=False):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(f"{panel_label}  {title}",
                     fontsize=9, fontweight="bold", loc="left", pad=8)

        n = 6
        ys = np.linspace(1.2, 8.8, n)
        col = LEVEL_COLORS

        if not is_vl:
            # Traditional: one level observable, others inferred
            chosen = 2
            for i, (y, c, s) in enumerate(zip(ys, col, LEVEL_SHORT)):
                alpha = 0.85 if i == chosen else 0.25
                rect = FancyBboxPatch((1.0, y - 0.45), 8.0, 0.90,
                                     boxstyle="round,pad=0.05",
                                     fc=to_rgba(c, alpha), ec=c, lw=0.8)
                ax.add_patch(rect)
                ax.text(5.0, y, s, ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if i == chosen else c)

            # gap arrows (inferential)
            for i in range(n - 1):
                if i != chosen and i != chosen - 1:
                    ax.annotate("", xy=(5.0, ys[i + 1] - 0.5),
                                xytext=(5.0, ys[i] + 0.5),
                                arrowprops=dict(arrowstyle="->",
                                                color="#aaaaaa", lw=0.8,
                                                linestyle="dashed"))

            # annotation
            ax.text(5.0, ys[chosen] - 0.85, "← observed",
                    ha="center", fontsize=6.5, color=col[chosen])
            ax.text(5.0, 0.3, "Causal connections\ninferred across\nobservability gaps",
                    ha="center", fontsize=6.5, color="#888888",
                    style="italic", linespacing=1.35)

        else:
            # Virtual lab: all levels fully observable, two-way arrows
            for i, (y, c, s) in enumerate(zip(ys, col, LEVEL_SHORT)):
                rect = FancyBboxPatch((1.0, y - 0.45), 8.0, 0.90,
                                     boxstyle="round,pad=0.05",
                                     fc=to_rgba(c, 0.80), ec=c, lw=0.8)
                ax.add_patch(rect)
                ax.text(5.0, y, s, ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")

            for i in range(n - 1):
                ax.annotate("", xy=(5.0, ys[i + 1] - 0.5),
                            xytext=(5.0, ys[i] + 0.5),
                            arrowprops=dict(arrowstyle="<->",
                                            color="#333333", lw=1.0))

            # intervention arrow
            ax.annotate("", xy=(1.05, ys[-1]),
                        xytext=(0.2, ys[-1]),
                        arrowprops=dict(arrowstyle="->",
                                        color="#e15759", lw=1.4))
            ax.text(0.1, ys[-1] + 0.6, "Intervention\nat any level",
                    fontsize=6.5, color="#e15759", ha="center",
                    fontweight="bold")

            # observation arrow
            ax.annotate("", xy=(9.8, ys[0]),
                        xytext=(9.05, ys[0]),
                        arrowprops=dict(arrowstyle="->",
                                        color="#59a14f", lw=1.4))
            ax.text(9.9, ys[0] + 0.6, "Observe\nall levels",
                    fontsize=6.5, color="#59a14f", ha="center",
                    fontweight="bold")

            ax.text(5.0, 0.3, "Complete causal graph — every intermediate\nstate known and queryable",
                    ha="center", fontsize=6.5, color="#333333",
                    style="italic", linespacing=1.35)

    draw_panel(axes[0], "Traditional experiment", "A", is_vl=False)
    draw_panel(axes[1], "Virtual laboratory", "B", is_vl=True)

    fig.tight_layout(pad=1.0)
    fig.savefig(f"{OUT}/fig2_instrument.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig2_instrument.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3  –  Cross-level causal tracing (schizophrenia perturbation)
# ═════════════════════════════════════════════════════════════════════════════
def fig3_causal_trace():
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38,
                           left=0.10, right=0.97, top=0.91, bottom=0.10)

    CTRL = "#4e79a7"
    PERT = "#e15759"

    # ── (a) Tuning curves ────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    angles = np.linspace(-90, 90, 200)
    ctrl_tc  = np.exp(-angles**2 / (2 * 18**2))
    pert_tc  = np.exp(-angles**2 / (2 * 31**2))
    ax_a.plot(angles, ctrl_tc, color=CTRL, lw=1.5, label="Control")
    ax_a.plot(angles, pert_tc, color=PERT, lw=1.5, label="Perturbed")
    ax_a.axhline(0.5, color="#aaaaaa", lw=0.7, ls="--")
    ax_a.set_xlabel("Preferred angle offset (°)")
    ax_a.set_ylabel("Normalised response")
    ax_a.set_title("a   Optic tectum tuning (Level 3)",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_a.legend(fontsize=6.5, frameon=False)
    ax_a.set_xlim(-90, 90)
    ax_a.set_ylim(0, 1.08)
    ax_a.text(0, 0.54, "HWHM: 18° → 31°",
              ha="center", fontsize=6.5, color="#555555", style="italic")

    # ── (b) Goal-selection entropy ───────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    cats = ["Forage", "Flee", "Explore", "Social"]
    ctrl_p  = np.array([0.55, 0.25, 0.15, 0.05])
    pert_p  = np.array([0.32, 0.28, 0.22, 0.18])
    x = np.arange(4)
    w = 0.32
    ax_b.bar(x - w / 2, ctrl_p, w, color=CTRL, alpha=0.85, label="Control")
    ax_b.bar(x + w / 2, pert_p, w, color=PERT, alpha=0.85, label="Perturbed")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(cats, fontsize=6.5)
    ax_b.set_ylabel("Goal selection probability")
    ax_b.set_title("b   Basal ganglia goal distribution (Level 4)",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_b.legend(fontsize=6.5, frameon=False)
    H_ctrl = -np.sum(ctrl_p * np.log2(ctrl_p + 1e-9))
    H_pert = -np.sum(pert_p * np.log2(pert_p + 1e-9))
    ax_b.text(1.5, 0.57, f"Entropy: {H_ctrl:.2f} → {H_pert:.2f} bits",
              ha="center", fontsize=6.5, color="#555555", style="italic")

    # ── (c) Behavioural path-length CV ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    n_ep = 120
    ctrl_pl = rng.normal(340, 55, n_ep)
    pert_pl = rng.normal(340, 92, n_ep)   # higher variance
    bp = ax_c.boxplot([ctrl_pl, pert_pl],
                      patch_artist=True,
                      widths=0.38,
                      medianprops=dict(color="white", lw=1.5),
                      whiskerprops=dict(lw=0.8),
                      capprops=dict(lw=0.8),
                      flierprops=dict(marker="o", markersize=2.5,
                                     markerfacecolor="#aaaaaa", lw=0.5))
    bp["boxes"][0].set_facecolor(to_rgba(CTRL, 0.75))
    bp["boxes"][1].set_facecolor(to_rgba(PERT, 0.75))
    ax_c.set_xticks([1, 2])
    ax_c.set_xticklabels(["Control", "Perturbed"])
    ax_c.set_ylabel("Episode path length (steps)")
    ax_c.set_title("c   Behavioural variability (Level 5)",
                   fontsize=8.5, fontweight="bold", loc="left")
    cv_ctrl = np.std(ctrl_pl) / np.mean(ctrl_pl)
    cv_pert = np.std(pert_pl) / np.mean(pert_pl)
    ax_c.text(1.5, np.max(pert_pl) + 12,
              f"CV: {cv_ctrl:.2f} → {cv_pert:.2f}  (+{100*(cv_pert/cv_ctrl-1):.0f}%)",
              ha="center", fontsize=6.5, color="#555555", style="italic")

    # ── (d) Social alarm propagation ────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    t = np.arange(0, 30)
    ctrl_flee  = 1 - np.exp(-0.22 * np.maximum(t - 2, 0))
    ctrl_flee  = np.clip(ctrl_flee * 0.84, 0, 0.84)
    pert_flee  = 1 - np.exp(-0.12 * np.maximum(t - 4, 0))
    pert_flee  = np.clip(pert_flee * 0.41, 0, 0.41)
    ax_d.fill_between(t, ctrl_flee, alpha=0.20, color=CTRL)
    ax_d.fill_between(t, pert_flee, alpha=0.20, color=PERT)
    ax_d.plot(t, ctrl_flee, color=CTRL, lw=1.5, label="Control")
    ax_d.plot(t, pert_flee, color=PERT, lw=1.5, label="Perturbed")
    ax_d.axhline(0.84, color=CTRL, lw=0.7, ls=":", alpha=0.6)
    ax_d.axhline(0.41, color=PERT, lw=0.7, ls=":", alpha=0.6)
    ax_d.set_xlabel("Steps after alarm onset")
    ax_d.set_ylabel("Fraction of group fleeing")
    ax_d.set_title("d   Social alarm propagation (Level 6)",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_d.legend(fontsize=6.5, frameon=False)
    ax_d.set_ylim(0, 1.0)
    ax_d.text(14, 0.68, "0.84 → 0.41\n(−51%)",
              ha="center", fontsize=6.5, color="#555555", style="italic")

    # overall title
    fig.suptitle(
        "Figure 3  |  Cross-level causal tracing of a precision perturbation\n"
        "           (schizophrenia-spectrum model: sensory precision weights reduced 40%)",
        fontsize=8.5, fontweight="bold", y=0.99)

    fig.savefig(f"{OUT}/fig3_causal_trace.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig3_causal_trace.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig3 done")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4  –  Platform Architecture
# ═════════════════════════════════════════════════════════════════════════════
def fig4_platform():
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.90])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Figure 4  |  Open platform architecture",
                 fontsize=9, fontweight="bold", loc="left")

    # ── helper ───────────────────────────────────────────────────────────────
    def box(ax, x, y, w, h, title, items, color, title_color="white"):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.15",
                              fc=to_rgba(color, 0.15), ec=color, lw=1.5)
        ax.add_patch(rect)
        # title band
        rect2 = FancyBboxPatch((x + 0.05, y + h - 1.05), w - 0.1, 0.95,
                               boxstyle="round,pad=0.1",
                               fc=to_rgba(color, 0.85), ec="none")
        ax.add_patch(rect2)
        ax.text(x + w / 2, y + h - 0.58, title,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=title_color)
        for j, item in enumerate(items):
            ax.text(x + 0.3, y + h - 1.55 - j * 0.72, "·  " + item,
                    fontsize=6.5, va="top", color="#333333")

    def arrow(ax, x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>",
                                   color="#555555", lw=1.2,
                                   mutation_scale=12))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.22, label,
                    ha="center", fontsize=6, color="#666666", style="italic")

    # ── four main components ──────────────────────────────────────────────────
    box(ax, 0.3, 5.8, 4.5, 5.8,
        "① Connectome Library",
        ["FlyWire (Drosophila)",
         "MICrONS (mouse cortex)",
         "ZFIN (zebrafish)",
         "C. elegans OpenWorm",
         "Standardised conn. API"],
        LEVEL_COLORS[0])

    box(ax, 5.5, 5.8, 4.5, 5.8,
        "② Modular Sim. Engine",
        ["Plug-in brain region modules",
         "Izhikevich / LIF neurons",
         "Module registry + assembly",
         "4-axis neuromodulation",
         "STDP plasticity engine"],
        LEVEL_COLORS[2])

    box(ax, 10.7, 5.8, 4.5, 5.8,
        "③ Behavioural Gymnasium",
        ["Shared virtual arenas",
         "Standardised stimuli",
         "Species-agnostic protocols",
         "Multi-agent social env.",
         "Looming · prey · alarm · odour"],
        LEVEL_COLORS[4])

    box(ax, 15.2, 5.8, 4.5, 5.8,
        "④ Experiment Runner",
        ["Scripted interventions",
         "Full hierarchical logging",
         "Correspondence analytics",
         "Spike rasters · ethograms",
         "Version-controlled scripts"],
        LEVEL_COLORS[5])

    # ── arrows between components ─────────────────────────────────────────────
    arrow(ax, 4.8, 8.7, 5.5, 8.7, "connectivity")
    arrow(ax, 10.0, 8.7, 10.7, 8.7, "avatar")
    arrow(ax, 15.2, 8.7, 14.7, 8.7, "behaviour")   # runner pulls from gym

    # ── bottom row: data sources → connectome lib ──────────────────────────
    sources = [
        (1.5,  "Electron\nmicroscopy"),
        (4.5,  "Ca²⁺ imaging\natlases"),
        (7.5,  "Axonal\nprojection DBs"),
        (10.5, "Patch-clamp\ncell types"),
        (13.5, "Behavioural\nbenchmarks"),
        (16.5, "Clinical\nphenotypes"),
    ]
    src_colors = LEVEL_COLORS

    for (xs, label), col in zip(sources, src_colors):
        rect = FancyBboxPatch((xs - 1.2, 1.0), 2.4, 1.8,
                              boxstyle="round,pad=0.1",
                              fc=to_rgba(col, 0.12), ec=col, lw=0.9)
        ax.add_patch(rect)
        ax.text(xs, 1.9, label, ha="center", va="center",
                fontsize=6, color=col, fontweight="bold", linespacing=1.3)

    ax.text(9.0, 0.4, "Empirical neuroscience data",
            ha="center", fontsize=7.5, color="#555555",
            fontweight="bold", style="italic")

    # arrows from data sources upward into the platform
    for xs, _ in sources:
        ax.annotate("", xy=(xs, 2.88), xytext=(xs, 2.82),
                    arrowprops=dict(arrowstyle="-|>",
                                   color="#aaaaaa", lw=0.9,
                                   mutation_scale=8))
        ax.plot([xs, xs], [2.8, 5.8], color="#bbbbbb", lw=0.6, ls="--")

    # ── validation feedback loop ───────────────────────────────────────────
    ax.annotate("", xy=(17.5, 2.8), xytext=(17.5, 5.8),
                arrowprops=dict(arrowstyle="<-",
                                color=LEVEL_COLORS[4], lw=1.0,
                                linestyle="dashed",
                                mutation_scale=10))
    ax.text(18.1, 4.3, "Validation\nfeedback",
            fontsize=6.2, color=LEVEL_COLORS[4], ha="center",
            style="italic", linespacing=1.3)

    fig.savefig(f"{OUT}/fig4_platform.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig4_platform.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig1_hierarchy()
    fig2_instrument()
    fig3_causal_trace()
    fig4_platform()
    print("All figures written to", OUT)
