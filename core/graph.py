"""
Similarity graph for Atlas.

Obsidian's graph draws an edge wherever you typed a ``[[wikilink]]``. Atlas
indexes files you never linked — code, PDFs, spreadsheets — so the edges have
to be inferred. They come from the embeddings that are already in ChromaDB:
two files are connected when their vectors are close.

Three things this has to get right.

**Aggregate chunks into files first.** The collection stores up to 12 chunks
per file. Left as-is, the graph would be a hairball of 2,457 chunk-nodes with
the strongest edges running between chunks of the *same* document — visually
dominant and completely uninformative. Nodes are files; a file's vector is the
mean of its chunk vectors, renormalised.

**Sparsify, don't threshold alone.** A plain "connect everything above 0.72"
gives a graph whose density depends entirely on how similar the corpus happens
to be — one tightly-themed folder becomes a solid blob. Taking each node's top
*k* neighbours *and* applying a floor keeps degree bounded, so the layout stays
readable whether there are 50 files or 50,000.

**Never materialise the full N×N matrix.** At 10,000 files a dense float32
similarity matrix is 400 MB. The comparison runs in row blocks, keeping only
each block's top-k, so peak memory is O(block × N) — a few MB — regardless of
corpus size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:                                   # pragma: no cover
    np = None                                          # type: ignore

# Cosine floor. Below this, "similar" is mostly embedding noise — two unrelated
# English documents sit around 0.3-0.5 with most sentence embedders.
DEFAULT_THRESHOLD = 0.72
# Neighbours kept per node. Obsidian graphs read well at this density; above
# ~10 the layout turns into a ball.
DEFAULT_MAX_NEIGHBOURS = 6
# Rows compared at once. Bounds peak memory to roughly block × N × 4 bytes.
BLOCK_ROWS = 512
# Rows sampled when inferring a threshold from the data.
SAMPLE_ROWS = 512
# Fraction of nodes an inferred threshold should leave with at least one edge.
TARGET_CONNECTED = 0.65


def suggest_threshold(matrix: Any, *, sample: int = SAMPLE_ROWS,
                      target_connected: float = TARGET_CONNECTED) -> float:
    """Infer a cosine floor from the corpus rather than hardcoding one.

    A fixed threshold is a guess about the embedding model. Different models
    put "clearly related documents" at wildly different cosines — 0.85 for one,
    0.45 for another — so a constant that looks right on nomic-embed renders an
    empty map on mxbai, and the user sees a broken feature rather than a tuning
    problem.

    This samples nodes, takes each one's single best neighbour, and returns the
    similarity at which ``target_connected`` of them would keep that edge. The
    result is a floor for the corpus at hand, whatever model produced it.
    """
    if np is None or matrix is None or len(matrix) < 3:
        return DEFAULT_THRESHOLD

    n = matrix.shape[0]
    idx = (np.arange(n) if n <= sample
           else np.linspace(0, n - 1, sample).astype(int))
    sims = matrix[idx] @ matrix.T
    for local, global_i in enumerate(idx):
        sims[local, global_i] = -1.0

    best = sims.max(axis=1)
    best = best[np.isfinite(best)]
    if best.size == 0:
        return DEFAULT_THRESHOLD

    # The (1 - target) quantile: at this value, `target` of sampled nodes have
    # a best-neighbour at least this similar.
    value = float(np.quantile(best, 1.0 - target_connected))
    # Keep it sane: never so low that everything connects to everything, never
    # so high that the map is dust.
    return round(max(0.30, min(0.95, value)), 3)


@dataclass
class GraphNode:
    id: str
    label: str
    group: str = ""
    ext: str = ""
    path: str = ""
    size: int = 0
    degree: int = 0
    mtime: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "group": self.group,
            "ext": self.ext, "path": self.path, "size": self.size,
            "degree": self.degree, "mtime": self.mtime,
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float

    def to_json(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target,
                "weight": round(self.weight, 4)}


@dataclass
class Graph:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    truncated: bool = False
    note: str = ""
    threshold: float = DEFAULT_THRESHOLD

    def to_json(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_json() for n in self.nodes],
            "edges": [e.to_json() for e in self.edges],
            "truncated": self.truncated,
            "note": self.note,
            "threshold": round(self.threshold, 3),
            "stats": {
                "nodes": len(self.nodes), "edges": len(self.edges),
                "isolated": sum(1 for n in self.nodes if n.degree == 0),
            },
        }


# ── Aggregation ─────────────────────────────────────────────────────────────


def aggregate_chunks(
    vectors: Sequence[Sequence[float]],
    metadatas: Sequence[Dict[str, Any]],
) -> Tuple[List[str], Any, Dict[str, Dict[str, Any]]]:
    """Collapse chunk vectors into one unit vector per file.

    Returns ``(paths, matrix, meta_by_path)``. The matrix rows are L2-normalised
    so a dot product *is* cosine similarity — which is what lets the blocked
    comparison below be a single matmul.
    """
    if np is None:                                     # pragma: no cover
        raise RuntimeError("numpy is required to build the Atlas graph")

    by_path: Dict[str, List[int]] = {}
    meta_by_path: Dict[str, Dict[str, Any]] = {}
    for i, meta in enumerate(metadatas):
        path = (meta or {}).get("path")
        if not path:
            continue
        by_path.setdefault(path, []).append(i)
        # First chunk wins for display metadata; they're identical per file.
        meta_by_path.setdefault(path, dict(meta or {}))

    if not by_path:
        return [], np.zeros((0, 0), dtype="float32"), {}

    arr = np.asarray(vectors, dtype="float32")
    paths = sorted(by_path)
    rows = np.vstack([arr[idx].mean(axis=0) for idx in (by_path[p] for p in paths)])

    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    # A zero vector would divide by zero and poison the whole matrix with NaN,
    # which then silently compares equal to nothing and drops the node.
    norms[norms == 0] = 1.0
    return paths, (rows / norms).astype("float32"), meta_by_path


# ── Edge construction ───────────────────────────────────────────────────────


def top_k_edges(
    matrix: Any,
    ids: Sequence[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_neighbours: int = DEFAULT_MAX_NEIGHBOURS,
    block_rows: int = BLOCK_ROWS,
) -> List[GraphEdge]:
    """Each node's strongest neighbours above ``threshold``, deduplicated.

    Edges are undirected: A→B and B→A collapse to one, keeping the higher
    weight. Because top-k is per row, an edge survives if *either* endpoint
    ranks the other highly — which is what stops a popular hub from crowding
    out a small cluster's internal links.

    Note what this does and doesn't bound. Total edges are hard-capped at
    ``n * max_neighbours``. Individual *degree* is not: a file that many others
    consider their nearest neighbour accumulates edges from all of them. That's
    deliberate — those hubs are the most informative thing on the map, and
    clamping them would hide the structure the graph exists to show.
    """
    n = len(ids)
    if n < 2 or matrix.shape[0] != n:
        return []

    best: Dict[Tuple[str, str], float] = {}
    k = min(max_neighbours, n - 1)

    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        # (block × dim) @ (dim × n) -> (block × n). Never the full N×N.
        sims = matrix[start:stop] @ matrix.T

        # Remove self-similarity, which is always 1.0 and would take a slot.
        for local, global_i in enumerate(range(start, stop)):
            sims[local, global_i] = -1.0

        # argpartition is O(n) per row vs O(n log n) for a full sort; we only
        # need the top k, not their order.
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k] if k > 0 else None
        if idx is None:
            continue

        for local, global_i in enumerate(range(start, stop)):
            for j in idx[local]:
                weight = float(sims[local, j])
                if weight < threshold:
                    continue
                a, b = ids[global_i], ids[int(j)]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                if weight > best.get(key, -1.0):
                    best[key] = weight

    return [GraphEdge(a, b, w) for (a, b), w in best.items()]


# ── Grouping ────────────────────────────────────────────────────────────────


def group_for(path: str, roots: Optional[Sequence[str]] = None) -> str:
    """A colour group for a node: the first path segment below an indexed root.

    Folder is a better grouping than a clustering algorithm here — it's stable
    between rebuilds, it's what the user already recognises, and it doesn't
    shuffle colours every time a file changes.
    """
    if not path:
        return "other"
    normalised = path.replace("\\", "/")
    for root in sorted(roots or [], key=len, reverse=True):
        r = str(root).replace("\\", "/").rstrip("/")
        if r and normalised.startswith(r + "/"):
            rest = normalised[len(r) + 1:]
            head = rest.split("/", 1)[0]
            root_name = r.rsplit("/", 1)[-1]
            # A file sitting directly in the root has no subfolder to name.
            return f"{root_name}/{head}" if "/" in rest else root_name
    return normalised.rsplit("/", 2)[-2] if "/" in normalised else "other"


# ── Assembly ────────────────────────────────────────────────────────────────


def build_graph(
    vectors: Sequence[Sequence[float]],
    metadatas: Sequence[Dict[str, Any]],
    *,
    roots: Optional[Sequence[str]] = None,
    threshold: Any = DEFAULT_THRESHOLD,
    max_neighbours: int = DEFAULT_MAX_NEIGHBOURS,
    max_nodes: int = 4000,
) -> Graph:
    """Chunk vectors and metadata in, a renderable graph out.

    ``max_nodes`` bounds what the browser is asked to lay out. When it bites,
    the most recently modified files are kept and ``truncated`` is set — a
    silent cap would read as "this is everything you have", which is exactly
    the failure mode worth avoiding in a tool whose whole promise is showing
    you your own files.
    """
    paths, matrix, meta_by_path = aggregate_chunks(vectors, metadatas)
    if not paths:
        return Graph(note="nothing indexed yet")

    truncated = False
    if len(paths) > max_nodes:
        ranked = sorted(paths, key=lambda p: meta_by_path[p].get("mtime", 0),
                        reverse=True)[:max_nodes]
        keep = set(ranked)
        keep_idx = [i for i, p in enumerate(paths) if p in keep]
        matrix = matrix[keep_idx]
        paths = [paths[i] for i in keep_idx]
        truncated = True

    # "auto" infers the floor from this corpus — see suggest_threshold.
    auto = isinstance(threshold, str) and threshold.lower() == "auto"
    effective = suggest_threshold(matrix) if auto else float(threshold)

    edges = top_k_edges(matrix, paths, threshold=effective,
                        max_neighbours=max_neighbours)

    degree: Dict[str, int] = {}
    for e in edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1

    nodes = []
    for path in paths:
        meta = meta_by_path.get(path, {})
        nodes.append(GraphNode(
            id=path,
            label=meta.get("name") or path.rsplit("/", 1)[-1],
            group=group_for(path, roots),
            ext=meta.get("ext", ""),
            path=path,
            size=int(meta.get("size", 0) or 0),
            degree=degree.get(path, 0),
            mtime=float(meta.get("mtime", 0) or 0),
        ))

    notes = []
    if truncated:
        notes.append(f"showing the {len(nodes)} most recently modified files")
    if auto:
        notes.append(f"link threshold auto-set to {effective:.2f} for this corpus")
    if nodes and not edges:
        # Never leave the user staring at unconnected dots with no explanation.
        notes.append("nothing was similar enough to link — lower the threshold")
    return Graph(nodes=nodes, edges=edges, truncated=truncated,
                 note="; ".join(notes), threshold=effective)


__all__ = [
    "build_graph", "aggregate_chunks", "top_k_edges", "group_for",
    "suggest_threshold",
    "Graph", "GraphNode", "GraphEdge",
    "DEFAULT_THRESHOLD", "DEFAULT_MAX_NEIGHBOURS",
]
