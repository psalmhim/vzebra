"""Assembly-line execution of registered eval plugins."""
from __future__ import annotations
import time
from zebrav2.eval.base import EvalContext, EvalResult, EvalPlugin
from zebrav2.eval.registry import EvalRegistry


class Pipeline:
    def __init__(self, registry: EvalRegistry | None = None):
        self.registry = registry or EvalRegistry.get()

    def run(self, ctx: EvalContext,
            max_tier: int | None = None,
            tags: list[str] | None = None,
            verbose: bool = True) -> list[EvalResult]:

        plugins = self.registry.get_ordered(max_tier=max_tier, tags=tags)
        failed_names: set[str] = set()

        tier_header = -1
        for plugin in plugins:
            if verbose and plugin.tier != tier_header:
                tier_header = plugin.tier
                tier_labels = {1:'NEURON', 2:'CIRCUIT', 3:'SYSTEM', 4:'BEHAVIOR', 5:'SOCIAL'}
                print(f'\n── Tier {plugin.tier}: {tier_labels.get(plugin.tier,"?")} ──')

            # Check upstream dependencies
            blocking = [r for r in plugin.requires if r in failed_names]
            if blocking:
                reason = f'upstream failed: {blocking}'
                result = EvalResult.skipped_result(plugin, reason)
                ctx.results.append(result)
                failed_names.add(plugin.name)
                if verbose:
                    print(f'  SKIP  {plugin.name:<40} ({reason})')
                continue

            if verbose:
                print(f'  RUN   {plugin.name:<40}', end='', flush=True)

            t0 = time.time()
            try:
                result = plugin.run(ctx)
            except Exception as e:
                result = EvalResult.skipped_result(plugin, f'exception: {e}')
                result.skipped = False   # it ran, it errored
                result.notes = str(e)
                import traceback; result.artifacts['traceback'] = traceback.format_exc()
            result.duration_s = time.time() - t0

            ctx.results.append(result)
            ctx.artifacts[plugin.name] = result.artifacts

            if result.skipped:
                failed_names.add(plugin.name)
            elif not result.passed:
                failed_names.add(plugin.name)

            if verbose:
                status = 'SKIP' if result.skipped else ('PASS' if result.passed else 'FAIL')
                key_metrics = '  '.join(f'{k}={v:.3g}' for k, v in list(result.metrics.items())[:3])
                print(f'  {status}  {key_metrics}  [{result.duration_s:.1f}s]')

        # Summary
        total = len(ctx.results)
        passed = sum(1 for r in ctx.results if r.passed)
        skipped = sum(1 for r in ctx.results if r.skipped)
        failed = total - passed - skipped
        if verbose:
            print(f'\n{"="*60}')
            print(f'EVAL COMPLETE: {passed}/{total-skipped} passed  |  {skipped} skipped  |  {failed} failed')

        return ctx.results
