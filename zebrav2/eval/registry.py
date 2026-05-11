"""Self-registering plugin registry with auto-discovery."""
from __future__ import annotations
import importlib, pkgutil
from pathlib import Path
from collections import defaultdict

class EvalRegistry:
    _instance: 'EvalRegistry | None' = None

    def __init__(self):
        self._plugins: dict[str, 'EvalPlugin'] = {}

    @classmethod
    def get(cls) -> 'EvalRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, plugin: 'EvalPlugin'):
        self._plugins[plugin.name] = plugin

    def discover(self, package: str = 'zebrav2.eval.plugins'):
        """Auto-import all modules in the plugins package."""
        pkg = importlib.import_module(package)
        pkg_path = Path(pkg.__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
            importlib.import_module(f'{package}.{module_name}')

    def get_ordered(self, max_tier: int | None = None,
                    tags: list[str] | None = None) -> list['EvalPlugin']:
        """Topologically sort plugins by requires[], filter by tier/tags."""
        plugins = list(self._plugins.values())
        if max_tier is not None:
            plugins = [p for p in plugins if p.tier <= max_tier]
        if tags:
            tag_set = set(tags)
            plugins = [p for p in plugins if tag_set & set(p.tags)]

        # Kahn topological sort
        name_set = {p.name for p in plugins}
        adj = defaultdict(list)
        in_deg = {p.name: 0 for p in plugins}
        for p in plugins:
            for req in p.requires:
                if req in name_set:
                    adj[req].append(p.name)
                    in_deg[p.name] += 1

        queue = [p for p in plugins if in_deg[p.name] == 0]
        queue.sort(key=lambda p: (p.tier, p.name))
        ordered = []
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            for nxt_name in sorted(adj[node.name]):
                in_deg[nxt_name] -= 1
                if in_deg[nxt_name] == 0:
                    nxt = self._plugins[nxt_name]
                    if nxt in plugins:
                        queue.append(nxt)
        return ordered

    def all_plugins(self) -> dict[str, 'EvalPlugin']:
        return dict(self._plugins)


def register(cls):
    """Class decorator — registers plugin on import."""
    from zebrav2.eval.base import EvalPlugin
    instance = cls()
    EvalRegistry.get().register(instance)
    return cls
