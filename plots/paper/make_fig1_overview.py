#!/usr/bin/env python3
"""
Figure 1 — Nature-style overview figure for the Virtual Zebrafish manuscript.
Layout:
  A (left, full height) : Behavioral arena + sensor annotations
  B (top-right)         : Brain architecture (5 functional layers)
  C (bottom-right left) : Active inference loop
  D (bottom-right right): Decision rationality benchmark

Run from project root:
  .venv/bin/python plots/paper/make_fig1_overview.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, Ellipse, Circle,
                                 Wedge, Polygon, FancyArrowPatch)
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.colors as mcolors
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Nature typography
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 0.9,
    'patch.linewidth': 0.5,
    'pdf.fonttype': 42,   # editable text in Illustrator
    'ps.fonttype': 42,
})

# ---------------------------------------------------------------------------
# Palette (colorblind-safe, muted Nature style)
# ---------------------------------------------------------------------------
P = {
    # Brain regions
    'retina':     '#4393c3',   # sky blue
    'tectum':     '#2166ac',   # deep blue
    'pallium':    '#d6604d',   # coral
    'bg':         '#35978f',   # teal-green
    'habenula':   '#bf812d',   # amber-brown
    'cerebellum': '#762a83',   # purple
    'brainstem':  '#4d9221',   # dark green
    'neuromod':   '#f4a582',   # peach

    # Goals
    'forage':  '#2166ac',
    'flee':    '#d6604d',
    'explore': '#35978f',
    'social':  '#762a83',

    # Environment
    'fish':    '#1a6fad',
    'pred':    '#b2182b',
    'food':    '#1a7a1a',
    'rock':    '#777777',
    'water':   '#e0f3f8',
    'arena_edge': '#74add1',

    # UI
    'text':      '#222222',
    'subtext':   '#555555',
    'arrow':     '#555555',
    'bg_panel':  '#fafafa',
    'border':    '#cccccc',
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(7.5, 5.2), dpi=300, facecolor='white')

gs_outer = GridSpec(1, 2, figure=fig,
                    left=0.02, right=0.99, top=0.97, bottom=0.05,
                    wspace=0.10, width_ratios=[1.0, 1.5])

gs_right = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[0, 1],
                                   hspace=0.32, wspace=0.28,
                                   height_ratios=[1.1, 1.0])

ax_a = fig.add_subplot(gs_outer[0, 0])
ax_b = fig.add_subplot(gs_right[0, :])     # top row, full width
ax_c = fig.add_subplot(gs_right[1, 0])
ax_d = fig.add_subplot(gs_right[1, 1])

for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.set_aspect('auto')
    ax.axis('off')


def panel_tag(ax, letter):
    ax.text(-0.04, 1.03, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left', color='black')


panel_tag(ax_a, 'A')
panel_tag(ax_b, 'B')
panel_tag(ax_c, 'C')
panel_tag(ax_d, 'D')

# ---------------------------------------------------------------------------
# PANEL A — Behavioral Arena
# ---------------------------------------------------------------------------
ax = ax_a
ax.set_xlim(0, 10)
ax.set_ylim(0, 15.5)

ax.text(5.0, 15.2, 'Virtual Zebrafish Environment',
        ha='center', va='top', fontsize=7, fontweight='bold', color=P['text'])

# Water arena background
arena_rect = FancyBboxPatch((0.3, 0.6), 9.4, 13.5,
    boxstyle='round,pad=0.2',
    facecolor=P['water'], edgecolor=P['arena_edge'],
    linewidth=1.0, zorder=0)
ax.add_patch(arena_rect)

# Rocks (irregular polygons)
rng = np.random.default_rng(7)
def draw_rock(ax, cx, cy, r, seed):
    rs = np.random.default_rng(seed)
    th = np.sort(rs.uniform(0, 2*np.pi, 6))
    rr = r * rs.uniform(0.65, 1.0, 6)
    xs = cx + rr * np.cos(th)
    ys = cy + rr * np.sin(th)
    ax.add_patch(Polygon(list(zip(xs, ys)), closed=True,
                         facecolor=P['rock'], edgecolor='#555555',
                         linewidth=0.4, alpha=0.75, zorder=1))

draw_rock(ax, 2.2, 2.2, 0.55, 1)
draw_rock(ax, 8.3, 2.8, 0.50, 2)
draw_rock(ax, 7.8, 11.5, 0.48, 3)
draw_rock(ax, 1.8, 11.0, 0.52, 4)

# Food items with olfactory glow
food_positions = [(4.0, 9.5), (6.8, 8.2), (3.2, 5.5)]
for fx, fy in food_positions:
    for gr, ga in [(1.4, 0.06), (0.9, 0.12), (0.5, 0.20)]:
        ax.add_patch(Circle((fx, fy), gr, facecolor=P['food'],
                            edgecolor='none', alpha=ga, zorder=1))
    ax.add_patch(Circle((fx, fy), 0.2, facecolor=P['food'],
                        edgecolor='#0f4f0f', linewidth=0.5, zorder=3))

# Fish (center-upper of arena)
fx0, fy0 = 5.5, 7.0
heading = 38.0  # degrees
cos_h, sin_h = np.cos(np.radians(heading)), np.sin(np.radians(heading))

# Visual field cone (±55 deg, range ~4 units)
visual_range = 4.0
v_lo, v_hi = heading - 55, heading + 55
wedge_v = Wedge((fx0, fy0), visual_range, v_lo, v_hi,
                facecolor=P['tectum'], edgecolor=P['tectum'],
                alpha=0.10, linewidth=0.5, linestyle='--', zorder=2)
ax.add_patch(wedge_v)
# Solid outline of cone
for ang in [v_lo, v_hi]:
    ax.plot([fx0, fx0 + visual_range*np.cos(np.radians(ang))],
            [fy0, fy0 + visual_range*np.sin(np.radians(ang))],
            color=P['tectum'], linewidth=0.5, linestyle='--', zorder=2, alpha=0.6)

# Lateral line range (dashed circle)
ax.add_patch(Circle((fx0, fy0), 2.1, facecolor='none',
                    edgecolor='#5ba3c9', linewidth=0.6,
                    linestyle=':', zorder=2, alpha=0.7))

# Olfactory detection range (looser dashed)
ax.add_patch(Circle((fx0, fy0), 3.5, facecolor='none',
                    edgecolor=P['food'], linewidth=0.5,
                    linestyle=(0, (3,5)), zorder=2, alpha=0.45))

# Fish body (elongated ellipse)
body = Ellipse((fx0, fy0), 1.0, 0.44, angle=heading,
               facecolor=P['fish'], edgecolor='#0d3f70',
               linewidth=0.7, zorder=5, alpha=0.93)
ax.add_patch(body)
# Dorsal stripe
ax.plot([fx0 - 0.35*cos_h, fx0 + 0.35*cos_h],
        [fy0 - 0.35*sin_h, fy0 + 0.35*sin_h],
        color='#1a5d9e', linewidth=1.2, zorder=6, alpha=0.5)
# Eye
ex, ey = fx0 + 0.30*cos_h, fy0 + 0.30*sin_h
ax.add_patch(Circle((ex, ey), 0.075, facecolor='#111111', zorder=7))
ax.add_patch(Circle((ex+0.02, ey+0.02), 0.028, facecolor='white', zorder=8))
# Tail fin
tx, ty = fx0 - 0.52*cos_h, fy0 - 0.52*sin_h
perp = np.array([-sin_h, cos_h])
tail_pts = np.array([
    [tx, ty],
    [tx - 0.3*cos_h + 0.22*perp[0], ty - 0.3*sin_h + 0.22*perp[1]],
    [tx - 0.3*cos_h - 0.22*perp[0], ty - 0.3*sin_h - 0.22*perp[1]],
])
ax.add_patch(Polygon(tail_pts, facecolor=P['fish'], edgecolor='#0d3f70',
                     linewidth=0.5, zorder=4))

# Predator (bold red triangle, bottom-right)
px0, py0 = 8.2, 2.2
pred_size = 0.62
pred_pts = np.array([
    [px0, py0 + pred_size],
    [px0 - pred_size*0.8, py0 - pred_size*0.5],
    [px0 + pred_size*0.8, py0 - pred_size*0.5],
])
ax.add_patch(Polygon(pred_pts, facecolor=P['pred'], edgecolor='#7a0010',
                     linewidth=0.6, zorder=5, alpha=0.92))
ax.add_patch(Circle((px0, py0 + 0.1), 0.08, facecolor='#ffcc00', zorder=6))

# --- Annotation arrows (clean, labeled) ---
def annotate_sensor(ax, label, xy, xytext, color):
    ax.annotate(label, xy=xy, xytext=xytext,
                xycoords='data', textcoords='data',
                fontsize=5.2, color=color, va='center', ha='center',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=0.6, mutation_scale=7))

annotate_sensor(ax, 'Visual field\n(±55°, 3-channel\ncolor)',
                xy=(fx0 + 2.8*cos_h, fy0 + 2.8*sin_h),
                xytext=(8.6, 10.8), color=P['tectum'])

annotate_sensor(ax, 'Lateral line\n(24 neurons,\n200 px range)',
                xy=(fx0 - 1.5, fy0 + 1.3),
                xytext=(1.0, 9.8), color='#3d8ab5')

annotate_sensor(ax, 'Olfactory\ngradient\n(28 neurons)',
                xy=(4.0, 8.7),
                xytext=(1.2, 7.0), color=P['food'])

annotate_sensor(ax, 'Virtual\nzebrafish\n(7,316+ neurons)',
                xy=(fx0, fy0),
                xytext=(8.2, 6.2), color=P['fish'])

annotate_sensor(ax, 'Predator\n(5-state AI)',
                xy=(px0, py0 + 0.6),
                xytext=(6.5, 3.6), color=P['pred'])

annotate_sensor(ax, 'Food item\n+ glow = odor',
                xy=(food_positions[2][0], food_positions[2][1] + 0.95),
                xytext=(1.5, 4.5), color=P['food'])

# Scale bar
ax.plot([0.6, 1.6], [0.9, 0.9], color='#444444', linewidth=1.2, solid_capstyle='butt')
ax.text(1.1, 1.1, '100 px', ha='center', va='bottom', fontsize=4.8, color='#444444')

# ---------------------------------------------------------------------------
# PANEL B — Brain Architecture
# ---------------------------------------------------------------------------
ax = ax_b
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

ax.text(5, 9.75, 'Biologically Grounded Brain Architecture (48 modules · 7,316+ neurons)',
        ha='center', va='top', fontsize=6.5, fontweight='bold', color=P['text'])

# 5 functional layers (bottom → top = input → output)
layers = [
    # (y_bottom, label, detail, color)
    (1.0, 'Sensory Periphery',
     'Retina 256 · Color vision · Olfaction 28 · Lateral line 24 · Vestibular · Nociception · Auditory',
     P['retina']),
    (2.7, 'Subcortical Processing',
     'Optic tectum 3,200 (8 layers/hemi) · Amygdala · Cerebellum · Habenula · Hypothalamus · CPG 32',
     P['tectum']),
    (4.4, 'Predictive Coding',
     'Thalamus 256 · Pallium 2,780 · Anti-Hebbian W_FB · VAE world model (batch ELBO)',
     P['pallium']),
    (6.1, 'Learning & Neuromodulation',
     'RL critic (TD + eligibility traces) · DA-gated STDP · Four-axis neuromod (DA · NA · 5-HT · ACh)',
     P['neuromod']),
    (7.8, 'Action–Goal Selection',
     'EFE minimization → WTA attractor → FORAGE / FLEE / EXPLORE / SOCIAL → Reticulospinal',
     P['bg']),
]

box_x0, box_w, box_h = 0.2, 9.4, 1.35
for (yb, label, detail, color) in layers:
    rect = FancyBboxPatch((box_x0, yb), box_w, box_h,
                          boxstyle='round,pad=0.12',
                          facecolor=color, edgecolor='white',
                          linewidth=0.6, alpha=0.88, zorder=2)
    ax.add_patch(rect)
    ax.text(box_x0 + 0.2, yb + 0.90, label,
            fontsize=5.8, fontweight='bold', color='white',
            va='top', zorder=3)
    ax.text(box_x0 + 0.2, yb + 0.52, detail,
            fontsize=4.8, color='white', alpha=0.95,
            va='top', zorder=3)

# Upward arrows between layers
arrow_kw = dict(arrowstyle='->', color=P['arrow'],
                lw=0.9, mutation_scale=8)
arrow_xs = [1.5, 3.5, 5.5, 7.5]
for (yb, label, detail, color), ax_x in zip(layers[:-1],
                                              [5.0, 5.0, 5.0, 5.0]):
    y_top = yb + box_h
    y_next = layers[layers.index((yb, label, detail, color)) + 1][0]
    ax.annotate('', xy=(ax_x, y_next), xytext=(ax_x, y_top),
                arrowprops=arrow_kw, zorder=1)

# Feedback arrow (right side, top → bottom)
fbx = box_x0 + box_w + 0.18
ax.plot([box_x0 + box_w, fbx], [layers[-1][0] + box_h/2]*2,
        color='#aaaaaa', lw=0.7)
ax.plot([box_x0 + box_w, fbx], [layers[0][0] + box_h/2]*2,
        color='#aaaaaa', lw=0.7)
ax.annotate('', xy=(fbx, layers[0][0] + box_h/2),
            xytext=(fbx, layers[-1][0] + box_h/2),
            arrowprops=dict(arrowstyle='->', color='#aaaaaa',
                            lw=0.7, mutation_scale=7))
ax.text(fbx + 0.08,
        (layers[0][0] + layers[-1][0]) / 2 + box_h/2,
        'sensory\nfeedback', fontsize=4.2, color='#aaaaaa',
        va='center', rotation=90)

# Neuron-count legend (right side, inner)
ax.text(9.55, 5.0, '3,200\nneurons', fontsize=4, color=P['tectum'],
        ha='center', va='center', alpha=0.0)  # hidden — kept for reference

# ---------------------------------------------------------------------------
# PANEL C — Active Inference Loop
# ---------------------------------------------------------------------------
ax = ax_c
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_facecolor(P['bg_panel'])

ax.text(5, 9.6, 'Active Inference Loop',
        ha='center', va='top', fontsize=6.5, fontweight='bold', color=P['text'])

# 5-node circular diagram
cx, cy, r_c = 5.0, 5.2, 2.8
node_r = 1.0

# Node definitions: (label_lines, angle_deg, color)
nodes = [
    (['Sensory', 'input'],                90,   P['retina']),
    (['Perceptual', 'inference', '(↓PE)'], 18,   P['tectum']),
    (['EFE', 'computation', '(4 goals)'], -54,  P['pallium']),
    (['Goal', 'selection', '(WTA)'],      -126, P['bg']),
    (['Motor', 'action'],                  162,  P['brainstem']),
]

positions = []
for (lines, ang_deg, color) in nodes:
    ang = np.radians(ang_deg)
    nx, ny = cx + r_c * np.cos(ang), cy + r_c * np.sin(ang)
    positions.append((nx, ny))
    circ = Circle((nx, ny), node_r, facecolor=color, edgecolor='white',
                  linewidth=0.9, zorder=4, alpha=0.90)
    ax.add_patch(circ)
    label = '\n'.join(lines)
    ax.text(nx, ny, label, ha='center', va='center', fontsize=4.8,
            color='white', fontweight='bold', zorder=5, multialignment='center')

# Curved arrows between nodes (clockwise)
for i in range(len(nodes)):
    p1 = np.array(positions[i])
    p2 = np.array(positions[(i + 1) % len(nodes)])
    direction = (p2 - p1)
    dist = np.linalg.norm(direction)
    d = direction / dist
    start = p1 + d * node_r
    end   = p2 - d * node_r
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=P['arrow'],
                                lw=0.9, mutation_scale=9,
                                connectionstyle='arc3,rad=0.22'),
                zorder=3)

# Central label
ax.text(cx, cy, 'Minimize\nFree Energy\n$\\mathcal{F}$',
        ha='center', va='center', fontsize=5.5, color='#444444',
        style='italic', zorder=3)

# 4 goal policy boxes below
goal_info = [
    ('FORAGE', P['forage']),
    ('FLEE',   P['flee']),
    ('EXPLORE', P['explore']),
    ('SOCIAL', P['social']),
]
gx_centers = [1.5, 3.6, 6.4, 8.5]
gy = 0.85
for (label, color), gx in zip(goal_info, gx_centers):
    rect = FancyBboxPatch((gx - 0.88, gy - 0.32), 1.76, 0.64,
                          boxstyle='round,pad=0.08',
                          facecolor=color, edgecolor='white',
                          linewidth=0.4, alpha=0.88, zorder=3)
    ax.add_patch(rect)
    ax.text(gx, gy + 0.01, label, ha='center', va='center',
            fontsize=4.5, color='white', fontweight='bold', zorder=4)

# Arrow from EFE node down to goal boxes
efe_pos = positions[2]
ax.annotate('', xy=(5.0, 1.55), xytext=(efe_pos[0], efe_pos[1] - node_r),
            arrowprops=dict(arrowstyle='->', color=P['bg'],
                            lw=0.7, mutation_scale=7,
                            linestyle='dashed'),
            zorder=3)
ax.text(5.8, 2.3, 'WTA\nattractor', fontsize=4.2, color=P['bg'],
        va='center', ha='left')

# ---------------------------------------------------------------------------
# PANEL D — Decision Rationality Benchmark
# ---------------------------------------------------------------------------
ax = ax_d
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

ax.text(5, 9.6, 'Decision Rationality Benchmark',
        ha='center', va='top', fontsize=6.5, fontweight='bold', color=P['text'])

scenarios = [
    ('A', 'Safe food', 74,  P['pallium']),
    ('B', 'Predator', 100,  P['flee']),
    ('C', 'Starvation', 100, P['tectum']),
    ('D', 'Foraging', 78,  P['bg']),
    ('E', 'Exploration', 67, P['cerebellum']),
]
avg_score = 84

bar_x0 = 2.6
bar_max = 6.8
bar_h   = 0.70
ys = [8.15, 6.85, 5.55, 4.25, 2.95]

for (sc_id, sc_name, score, color), yb in zip(scenarios, ys):
    bw = score / 100 * bar_max
    # Bar background (light gray rail)
    ax.add_patch(FancyBboxPatch((bar_x0, yb), bar_max, bar_h,
                                boxstyle='round,pad=0.04',
                                facecolor='#e8e8e8', edgecolor='none',
                                alpha=0.7, zorder=1))
    # Filled bar
    ax.add_patch(FancyBboxPatch((bar_x0, yb), bw, bar_h,
                                boxstyle='round,pad=0.04',
                                facecolor=color, edgecolor='white',
                                linewidth=0.3, alpha=0.88, zorder=2))
    # Scenario ID
    ax.text(bar_x0 - 0.15, yb + bar_h/2, sc_id,
            ha='right', va='center', fontsize=5.5,
            fontweight='bold', color=P['text'])
    # Scenario name
    ax.text(bar_x0 - 0.15, yb + bar_h/2 - 0.55, sc_name,
            ha='right', va='center', fontsize=4.2, color=P['subtext'])
    # Score
    ax.text(bar_x0 + bw + 0.12, yb + bar_h/2,
            f'{score}/100', ha='left', va='center',
            fontsize=5.2, color=P['text'])

# Average dashed line
avg_x = bar_x0 + (avg_score / 100) * bar_max
ax.plot([avg_x, avg_x], [ys[-1] - 0.15, ys[0] + bar_h + 0.15],
        color='#333333', linewidth=1.0, linestyle='--', zorder=4)
ax.text(avg_x + 0.12, ys[0] + bar_h + 0.25,
        f'avg {avg_score}/100', fontsize=4.8, color='#333333', va='bottom')

# Summary stats box at bottom
stats = [
    ('Multi-seed', '10/10 complete  (500 ± 0 steps)'),
    ('Food intake', '11.1 ± 2.8 items  ·  0/10 caught'),
    ('Classifier', '96.2% accuracy (spiking SNN)'),
    ('Ablation',   'Only olfaction reduces survival (−7%)'),
]
bx0, bx1 = 0.3, 9.7
by_top = ys[-1] - 0.6
box_bg = FancyBboxPatch((bx0, by_top - 1.75), bx1 - bx0, 1.80,
                         boxstyle='round,pad=0.15',
                         facecolor='#f2f2f2', edgecolor='#cccccc',
                         linewidth=0.5, zorder=1)
ax.add_patch(box_bg)
for i, (key, val) in enumerate(stats):
    y_pos = by_top - 0.25 - i * 0.40
    ax.text(0.55, y_pos, f'▪ {key}:',
            fontsize=4.8, fontweight='bold', color=P['tectum'],
            va='top', ha='left')
    ax.text(2.9, y_pos, val,
            fontsize=4.8, color=P['text'], va='top', ha='left')

# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(__file__), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), 'fig0_intro_overview.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
