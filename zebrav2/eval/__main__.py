"""
Usage:
  python -m zebrav2.eval                     # all tiers
  python -m zebrav2.eval --tier 1 2          # tier 1 and 2 only
  python -m zebrav2.eval --tag retina        # all evals tagged 'retina'
  python -m zebrav2.eval --list              # list all registered plugins
  python -m zebrav2.eval --out /tmp/eval     # custom output dir
"""
import argparse, sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog='python -m zebrav2.eval',
                                     description='vzebra v2 evaluation pipeline')
    parser.add_argument('--tier', nargs='+', type=int, help='Tier(s) to run (1-5)')
    parser.add_argument('--tag',  nargs='+', type=str, help='Filter by tag')
    parser.add_argument('--out',  type=str,  default='plots/eval', help='Output directory')
    parser.add_argument('--list', action='store_true', help='List plugins and exit')
    parser.add_argument('--no-brain', action='store_true', help='Skip tests requiring full brain')
    args = parser.parse_args()

    from zebrav2.eval.registry import EvalRegistry
    from zebrav2.eval.pipeline import Pipeline
    from zebrav2.eval.report import ReportBuilder
    from zebrav2.eval.base import EvalContext

    # Auto-discover all plugins
    reg = EvalRegistry.get()
    reg.discover()

    if args.list:
        plugins = reg.get_ordered()
        print(f'{"T":>2}  {"Name":<45} {"Tags":<30} {"Requires"}')
        print('-' * 100)
        for p in plugins:
            print(f'{p.tier:>2}  {p.name:<45} {",".join(p.tags):<30} {",".join(p.requires)}')
        sys.exit(0)

    max_tier = max(args.tier) if args.tier else None
    tags = args.tag if args.tag else None

    # Build brain if any tier-3+ test will run
    brain = None
    if not args.no_brain and (max_tier is None or max_tier >= 3):
        print('Loading ZebrafishBrainV2...')
        import torch
        from zebrav2.brain.brain_v2 import ZebrafishBrainV2
        from zebrav2.spec import DEVICE
        device = 'cpu'   # deterministic eval
        brain = ZebrafishBrainV2(device=device)
        # Try loading latest checkpoint (prefer ckpt_latest.pt, else newest numbered)
        ckpt_dir = Path('zebrav2/checkpoints')
        ckpt_path = ckpt_dir / 'ckpt_latest.pt'
        if not ckpt_path.exists():
            pts = sorted(ckpt_dir.glob('ckpt_round_*.pt'))
            if pts:
                ckpt_path = pts[-1]
        if ckpt_path.exists():
            from zebrav2.engine.checkpoint import CheckpointManager
            try:
                cm = CheckpointManager()
                cm.load(brain, str(ckpt_path))
                print(f'  loaded {ckpt_path}')
            except Exception as e:
                print(f'  checkpoint load failed: {e} — using fresh weights')
        device = 'cpu'
    else:
        import torch
        device = 'cpu'

    out_dir = Path(args.out)
    ctx = EvalContext(brain=brain, device='cpu', output_dir=out_dir)

    pipeline = Pipeline(reg)
    results = pipeline.run(ctx, max_tier=max_tier, tags=tags, verbose=True)

    print('\nBuilding report...')
    report_path = ReportBuilder(results, out_dir).build()
    print(f'Report saved → {report_path}')


if __name__ == '__main__':
    main()
