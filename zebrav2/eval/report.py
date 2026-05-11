"""Report generation: JSON + per-tier figures + radar chart summary."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from zebrav2.eval.base import EvalResult

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


TIER_LABELS = {1: 'Neuron', 2: 'Circuit', 3: 'System', 4: 'Behavior', 5: 'Social'}
TIER_COLORS = {1: '#4C72B0', 2: '#55A868', 3: '#C44E52', 4: '#8172B2', 5: '#CCB974'}


class ReportBuilder:
    def __init__(self, results: list[EvalResult], output_dir: Path):
        self.results = results
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def build(self) -> Path:
        self._save_json()
        if HAS_MPL:
            self._save_tier_figures()
            self._save_radar()
            self._save_summary_figure()
        return self.out / 'eval_report.json'

    # ── JSON ──────────────────────────────────────────────────────────────────
    def _save_json(self):
        data = {
            'summary': self._summary(),
            'by_tier': self._by_tier_summary(),
            'results': [self._result_to_dict(r) for r in self.results],
        }
        path = self.out / 'eval_report.json'
        path.write_text(json.dumps(data, indent=2))

    def _result_to_dict(self, r: EvalResult) -> dict:
        return {
            'name': r.name, 'tier': r.tier, 'tags': r.tags,
            'purpose': r.purpose, 'method': r.method, 'reference': r.reference,
            'metrics': r.metrics, 'pass_criteria': {k: list(v) for k, v in r.pass_criteria.items()},
            'passed': r.passed, 'skipped': r.skipped, 'skip_reason': r.skip_reason,
            'duration_s': round(r.duration_s, 2), 'notes': r.notes,
        }

    def _summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        skipped = sum(1 for r in self.results if r.skipped)
        return {'total': total, 'passed': passed, 'failed': total - passed - skipped,
                'skipped': skipped, 'pass_rate': passed / max(1, total - skipped)}

    def _by_tier_summary(self) -> dict:
        by_tier = defaultdict(list)
        for r in self.results:
            by_tier[r.tier].append(r)
        out = {}
        for tier, results in sorted(by_tier.items()):
            p = sum(1 for r in results if r.passed)
            s = sum(1 for r in results if r.skipped)
            out[str(tier)] = {'label': TIER_LABELS.get(tier, f'Tier {tier}'),
                               'total': len(results), 'passed': p, 'skipped': s,
                               'pass_rate': p / max(1, len(results) - s)}
        return out

    # ── Figures ───────────────────────────────────────────────────────────────
    def _save_tier_figures(self):
        by_tier = defaultdict(list)
        for r in self.results:
            by_tier[r.tier].append(r)
        for tier, results in sorted(by_tier.items()):
            self._tier_table_figure(tier, results)

    def _tier_table_figure(self, tier: int, results: list[EvalResult]):
        if not results:
            return
        color = TIER_COLORS.get(tier, '#888888')
        label = TIER_LABELS.get(tier, f'Tier {tier}')

        fig, ax = plt.subplots(figsize=(14, max(3, len(results) * 0.55 + 1.5)))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.set_title(f'Tier {tier}: {label} Evaluation', fontsize=13, fontweight='bold',
                     color=color, pad=10)

        # Column headers
        cols = ['Eval', 'Status', 'Key Metrics', 'Reference', 'Duration']
        col_x = [0.01, 0.30, 0.37, 0.72, 0.93]
        col_w = [0.28, 0.06, 0.34, 0.20, 0.07]
        row_h = 0.85 / (len(results) + 1)
        y0 = 0.92

        for ci, (cname, cx) in enumerate(zip(cols, col_x)):
            ax.text(cx, y0, cname, fontsize=8, fontweight='bold', va='top',
                    color='white', bbox=dict(facecolor=color, pad=3, lw=0))

        for ri, r in enumerate(results):
            y = y0 - (ri + 1) * row_h
            bg = '#f5f5f5' if ri % 2 == 0 else 'white'
            ax.add_patch(mpatches.Rectangle((0, y - row_h * 0.9), 1, row_h * 0.9,
                                             color=bg, zorder=0))

            status = 'SKIP' if r.skipped else ('✓ PASS' if r.passed else '✗ FAIL')
            scolor = '#888888' if r.skipped else ('#2ca02c' if r.passed else '#d62728')

            top3 = '  '.join(f'{k}={v:.3g}' for k, v in list(r.metrics.items())[:3])

            ax.text(col_x[0], y, r.name, fontsize=7.5, va='top', color='#333333')
            ax.text(col_x[1], y, status, fontsize=7.5, va='top', color=scolor, fontweight='bold')
            ax.text(col_x[2], y, top3, fontsize=6.5, va='top', color='#555555', family='monospace')
            ax.text(col_x[3], y, r.reference[:30], fontsize=6.5, va='top', color='#666666', style='italic')
            ax.text(col_x[4], y, f'{r.duration_s:.1f}s', fontsize=6.5, va='top', color='#666666')

        plt.tight_layout()
        path = self.out / f'tier{tier}_{label.lower()}.pdf'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def _save_radar(self):
        by_tier = defaultdict(list)
        for r in self.results:
            by_tier[r.tier].append(r)
        tiers = sorted(by_tier)
        if len(tiers) < 3:
            return

        labels = [f'T{t}\n{TIER_LABELS.get(t,"")}' for t in tiers]
        rates = []
        for t in tiers:
            rs = by_tier[t]
            p = sum(1 for r in rs if r.passed)
            n = sum(1 for r in rs if not r.skipped)
            rates.append(p / max(1, n))

        angles = np.linspace(0, 2 * np.pi, len(tiers), endpoint=False).tolist()
        rates_c = rates + [rates[0]]; angles_c = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles_c, rates_c, 'o-', color='#4C72B0', linewidth=2)
        ax.fill(angles_c, rates_c, alpha=0.25, color='#4C72B0')
        ax.set_thetagrids(np.degrees(angles), labels, fontsize=10)
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
        ax.set_title('Evaluation Pass Rate by Tier', fontsize=12, fontweight='bold', pad=20)

        fig.savefig(self.out / 'eval_radar.pdf', bbox_inches='tight')
        plt.close(fig)

    def _save_summary_figure(self):
        by_tier = defaultdict(list)
        for r in self.results:
            by_tier[r.tier].append(r)

        tiers = sorted(by_tier)
        passed = [sum(1 for r in by_tier[t] if r.passed) for t in tiers]
        failed = [sum(1 for r in by_tier[t] if not r.passed and not r.skipped) for t in tiers]
        skipped = [sum(1 for r in by_tier[t] if r.skipped) for t in tiers]
        labels = [f'T{t} {TIER_LABELS.get(t,"")}' for t in tiers]
        x = np.arange(len(tiers))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x, passed,  label='Pass',   color='#2ca02c')
        ax.bar(x, failed,  label='Fail',   color='#d62728', bottom=passed)
        ax.bar(x, skipped, label='Skip',   color='#aec7e8',
               bottom=[p + f for p, f in zip(passed, failed)])
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('# Evaluations'); ax.set_title('vzebra v2 — Evaluation Summary')
        ax.legend(loc='upper right')
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        fig.savefig(self.out / 'eval_summary.pdf', bbox_inches='tight')
        plt.close(fig)
