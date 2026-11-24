"""Graph version registry (Stage 0 continuous updating, scaffold version).

In-memory store for graph versions with an edge-diff changelog. The full
Stage 0 loop (drift-triggered re-discovery + refutation promotion gate) is
Phase 2 work; this registry is the versioning surface it will write to.
"""
from __future__ import annotations

from .schemas import TwinGraph


def _edge_tuples(graph: TwinGraph) -> set[tuple]:
    return {(e.source, e.target, int(e.lag)) for e in graph.edges}


class GraphRegistry:
    def __init__(self):
        self._versions: dict[int, TwinGraph] = {}
        self._next = 1

    def register(self, graph: TwinGraph) -> int:
        version = self._next
        self._next += 1
        stored = graph.model_copy(deep=True)
        stored.graph_version = version
        self._versions[version] = stored
        return version

    def get(self, version: int) -> TwinGraph | None:
        return self._versions.get(version)

    def list_versions(self) -> list[int]:
        return sorted(self._versions)

    def diff(self, v1: int, v2: int) -> dict:
        g1 = _edge_tuples(self._versions[v1])
        g2 = _edge_tuples(self._versions[v2])
        fmt = lambda s: [{"source": a, "target": b, "lag": lag} for (a, b, lag) in sorted(s)]
        return {
            "from_version": v1,
            "to_version": v2,
            "added": fmt(g2 - g1),
            "removed": fmt(g1 - g2),
            "unchanged_count": len(g1 & g2),
        }
