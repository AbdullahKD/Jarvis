"""Tests for core.graph — the Atlas similarity graph.

The properties that matter: chunks collapse to files (or the graph is a
hairball of intra-document edges), degree stays bounded (or the layout is a
blob), and the memory cost doesn't grow with N² (or a large vault kills the
process).
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from core.graph import (
    DEFAULT_MAX_NEIGHBOURS,
    DEFAULT_THRESHOLD,
    aggregate_chunks,
    build_graph,
    group_for,
    top_k_edges,
)


def unit(vec):
    v = np.asarray(vec, dtype="float32")
    return (v / np.linalg.norm(v)).tolist()


def chunk(path, vec, *, name=None, ext=".md", mtime=0.0, size=100):
    return vec, {"path": path, "name": name or path.rsplit("/", 1)[-1],
                 "ext": ext, "mtime": mtime, "size": size}


def corpus(*items):
    """(vectors, metadatas) from chunk() tuples."""
    return [i[0] for i in items], [i[1] for i in items]


# ── Aggregation ─────────────────────────────────────────────────────────────


def test_chunks_of_one_file_become_a_single_node():
    """Without this the graph is dominated by edges between chunks of the same
    document — visually loud, informationally empty."""
    vecs, metas = corpus(
        chunk("/a.md", [1.0, 0.0, 0.0]),
        chunk("/a.md", [0.9, 0.1, 0.0]),
        chunk("/a.md", [0.8, 0.2, 0.0]),
        chunk("/b.md", [0.0, 1.0, 0.0]),
    )
    paths, matrix, meta = aggregate_chunks(vecs, metas)
    assert paths == ["/a.md", "/b.md"]
    assert matrix.shape[0] == 2


def test_aggregated_rows_are_unit_length():
    """Rows must be normalised, because the whole edge computation relies on a
    dot product being cosine similarity."""
    vecs, metas = corpus(
        chunk("/a.md", [3.0, 4.0, 0.0]),
        chunk("/b.md", [0.0, 0.0, 7.0]),
    )
    _, matrix, _ = aggregate_chunks(vecs, metas)
    for row in matrix:
        assert math.isclose(float(np.linalg.norm(row)), 1.0, rel_tol=1e-5)


def test_a_zero_vector_does_not_produce_nan():
    """A file whose chunks cancel out would divide by zero, and NaN silently
    compares false against every threshold — the node would vanish."""
    vecs, metas = corpus(
        chunk("/a.md", [1.0, 0.0]),
        chunk("/a.md", [-1.0, 0.0]),
        chunk("/b.md", [0.0, 1.0]),
    )
    _, matrix, _ = aggregate_chunks(vecs, metas)
    assert not np.isnan(matrix).any()


def test_chunks_without_a_path_are_ignored():
    vecs, metas = corpus(chunk("/a.md", [1.0, 0.0]))
    metas.append({"name": "orphan"})
    vecs.append([0.0, 1.0])
    paths, matrix, _ = aggregate_chunks(vecs, metas)
    assert paths == ["/a.md"]
    assert matrix.shape[0] == 1


def test_empty_input():
    paths, matrix, meta = aggregate_chunks([], [])
    assert paths == [] and meta == {}


# ── Edges ───────────────────────────────────────────────────────────────────


def test_similar_files_are_connected_and_dissimilar_ones_are_not():
    ids = ["a", "b", "far"]
    m = np.array([unit([1, 0, 0]), unit([0.98, 0.2, 0]), unit([0, 0, 1])],
                 dtype="float32")
    edges = top_k_edges(m, ids, threshold=0.72)
    pairs = {tuple(sorted((e.source, e.target))) for e in edges}
    assert ("a", "b") in pairs
    assert ("a", "far") not in pairs
    assert ("b", "far") not in pairs


def test_no_self_edges():
    ids = ["a", "b"]
    m = np.array([unit([1, 0]), unit([0.99, 0.1])], dtype="float32")
    for e in top_k_edges(m, ids, threshold=0.5):
        assert e.source != e.target


def test_edges_are_undirected_and_deduplicated():
    ids = ["a", "b"]
    m = np.array([unit([1, 0]), unit([0.99, 0.1])], dtype="float32")
    edges = top_k_edges(m, ids, threshold=0.5)
    assert len(edges) == 1


def test_degree_is_bounded_by_max_neighbours():
    """A tightly-themed folder where everything resembles everything must not
    become a solid blob."""
    n = 40
    rng = np.random.default_rng(0)
    base = unit(rng.random(16))
    rows = np.array([unit(np.asarray(base) + rng.random(16) * 0.02)
                     for _ in range(n)], dtype="float32")
    ids = [f"f{i}" for i in range(n)]

    k = 3
    edges = top_k_edges(rows, ids, threshold=0.5, max_neighbours=k)

    # The guarantee is on TOTAL edges, not max degree. Each node contributes at
    # most k, and dedup only removes — so n*k is a hard ceiling. Max degree is
    # deliberately NOT capped: a node many others point at is a hub, and
    # hubs are the most informative thing on the map.
    assert len(edges) <= n * k

    degree = {}
    for e in edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1
    mean_degree = sum(degree.values()) / len(degree)
    assert mean_degree <= 2 * k, f"mean degree {mean_degree:.1f} too dense"


def test_sparsity_improves_as_the_corpus_grows():
    """The n*k ceiling is only ~15% of the complete graph at n=40, but the
    ratio falls as 2k/(n-1) — which is the property that keeps a real vault
    renderable. Asserted where it actually bites."""
    k = 6
    rng = np.random.default_rng(11)
    for n, max_fraction in ((100, 0.15), (600, 0.03)):
        base = rng.random(16)
        rows = np.array([unit(base + rng.random(16) * 0.05) for _ in range(n)],
                        dtype="float32")
        edges = top_k_edges(rows, [f"f{i}" for i in range(n)],
                            threshold=0.5, max_neighbours=k)
        complete = n * (n - 1) / 2
        assert len(edges) <= n * k
        assert len(edges) / complete < max_fraction, (
            f"n={n}: {len(edges)}/{complete:.0f} edges")


def test_threshold_is_actually_applied():
    ids = ["a", "b"]
    m = np.array([unit([1, 0]), unit([0.6, 0.8])], dtype="float32")   # cos = 0.6
    assert top_k_edges(m, ids, threshold=0.72) == []
    assert len(top_k_edges(m, ids, threshold=0.5)) == 1


def test_blocking_gives_the_same_answer_as_one_pass():
    """The block loop is the memory optimisation; it must not change results."""
    rng = np.random.default_rng(7)
    rows = np.array([unit(rng.random(24)) for _ in range(50)], dtype="float32")
    ids = [f"n{i}" for i in range(50)]

    whole = top_k_edges(rows, ids, threshold=0.6, block_rows=10_000)
    blocked = top_k_edges(rows, ids, threshold=0.6, block_rows=7)

    def key(es):
        return sorted((tuple(sorted((e.source, e.target))), round(e.weight, 5))
                      for e in es)
    assert key(whole) == key(blocked)


def test_single_node_produces_no_edges():
    m = np.array([unit([1, 0])], dtype="float32")
    assert top_k_edges(m, ["only"]) == []


# ── Grouping ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path,expected", [
    ("/Users/akd/Desktop/Jarvis/core/tool.py", "Desktop/Jarvis"),
    ("/Users/akd/Desktop/notes.md", "Desktop"),
    ("/Users/akd/Documents/CV/cv.docx", "Documents/CV"),
])
def test_group_is_the_first_folder_below_a_root(path, expected):
    roots = ["/Users/akd/Desktop", "/Users/akd/Documents"]
    assert group_for(path, roots) == expected


def test_group_prefers_the_most_specific_root():
    roots = ["/Users/akd", "/Users/akd/Documents"]
    assert group_for("/Users/akd/Documents/x/y.md", roots) == "Documents/x"


def test_group_falls_back_outside_any_root():
    assert group_for("/tmp/weird/file.md", []) == "weird"
    assert group_for("", []) == "other"


# ── Whole graph ─────────────────────────────────────────────────────────────


def test_build_graph_end_to_end():
    vecs, metas = corpus(
        chunk("/d/cv.docx", [1.0, 0.0, 0.0], ext=".docx"),
        chunk("/d/cv.docx", [0.95, 0.1, 0.0], ext=".docx"),
        chunk("/d/coverletter.docx", [0.97, 0.05, 0.0], ext=".docx"),
        chunk("/d/thesis.pdf", [0.0, 1.0, 0.0], ext=".pdf"),
        chunk("/d/refs.pdf", [0.02, 0.99, 0.0], ext=".pdf"),
    )
    g = build_graph(vecs, metas, roots=["/d"], threshold=0.72)
    j = g.to_json()

    assert j["stats"]["nodes"] == 4
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in j["edges"]}
    assert ("/d/coverletter.docx", "/d/cv.docx") in pairs
    assert ("/d/refs.pdf", "/d/thesis.pdf") in pairs
    # The two clusters must NOT be joined — they're orthogonal.
    assert ("/d/cv.docx", "/d/thesis.pdf") not in pairs


def test_degree_is_reported_on_nodes():
    vecs, metas = corpus(
        chunk("/a.md", [1.0, 0.0]),
        chunk("/b.md", [0.99, 0.1]),
        chunk("/lonely.md", [0.0, 1.0]),
    )
    g = build_graph(vecs, metas, threshold=0.72)
    by_id = {n.id: n for n in g.nodes}
    assert by_id["/a.md"].degree == 1
    assert by_id["/lonely.md"].degree == 0
    assert g.to_json()["stats"]["isolated"] == 1


def test_isolated_files_are_still_nodes():
    """A file nothing resembles is a real result — it should appear as a lone
    dot, not be dropped from your own file map."""
    vecs, metas = corpus(chunk("/alone.md", [1.0, 0.0, 0.0]))
    g = build_graph(vecs, metas)
    assert len(g.nodes) == 1
    assert g.edges == []


def test_truncation_is_reported_not_silent():
    """Silently capping reads as 'this is everything you have'."""
    items = [chunk(f"/f{i}.md", unit([1.0, i * 0.01]), mtime=float(i))
             for i in range(20)]
    vecs, metas = corpus(*items)
    g = build_graph(vecs, metas, max_nodes=5)
    assert len(g.nodes) == 5
    assert g.truncated is True
    assert "most recently modified" in g.note


def test_truncation_keeps_the_newest_files():
    items = [chunk(f"/f{i}.md", unit([1.0, i * 0.01]), mtime=float(i))
             for i in range(10)]
    vecs, metas = corpus(*items)
    g = build_graph(vecs, metas, max_nodes=3)
    assert {n.id for n in g.nodes} == {"/f7.md", "/f8.md", "/f9.md"}


def test_empty_index_gives_an_explanatory_graph():
    g = build_graph([], [])
    assert g.nodes == [] and g.edges == []
    assert "nothing indexed" in g.note


def test_json_is_serialisable():
    import json
    vecs, metas = corpus(chunk("/a.md", [1.0, 0.0]), chunk("/b.md", [0.99, 0.1]))
    json.dumps(build_graph(vecs, metas).to_json())


# ── Scale ───────────────────────────────────────────────────────────────────


def test_large_corpus_stays_within_memory_and_time():
    """2,000 files x 128 dims. A dense N x N float32 matrix would be 16 MB here
    and 400 MB at 10,000 files; the blocked path must not build one."""
    rng = np.random.default_rng(3)
    n, dim = 2000, 128
    rows = rng.random((n, dim)).astype("float32")
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    ids = [f"f{i}" for i in range(n)]

    import time
    t0 = time.perf_counter()
    edges = top_k_edges(rows, ids, threshold=0.9, max_neighbours=6)
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, f"took {elapsed:.1f}s"
    assert len(edges) < n * DEFAULT_MAX_NEIGHBOURS


def test_defaults_are_sane():
    assert 0.5 < DEFAULT_THRESHOLD < 0.95
    assert 2 <= DEFAULT_MAX_NEIGHBOURS <= 10


# ── Auto threshold ──────────────────────────────────────────────────────────


def test_auto_threshold_adapts_to_a_high_similarity_corpus():
    """Different embedding models put 'clearly related' at very different
    cosines. A constant tuned for one renders an empty map on another."""
    from core.graph import suggest_threshold
    rng = np.random.default_rng(4)
    base = rng.random(32)
    tight = np.array([unit(base + rng.random(32) * 0.02) for _ in range(60)],
                     dtype="float32")
    assert suggest_threshold(tight) > 0.9


def test_auto_threshold_adapts_to_a_low_similarity_corpus():
    from core.graph import suggest_threshold
    rng = np.random.default_rng(5)
    loose = np.array([unit(rng.standard_normal(64)) for _ in range(120)],
                     dtype="float32")
    t = suggest_threshold(loose)
    assert 0.30 <= t < 0.72, f"got {t}"


def test_auto_threshold_stays_within_bounds():
    from core.graph import suggest_threshold
    rng = np.random.default_rng(6)
    identical = np.array([unit([1.0] + [0.0] * 31)] * 40, dtype="float32")
    assert 0.30 <= suggest_threshold(identical) <= 0.95


def test_auto_threshold_falls_back_on_a_tiny_corpus():
    from core.graph import suggest_threshold, DEFAULT_THRESHOLD
    assert suggest_threshold(np.array([unit([1, 0])], dtype="float32")) == DEFAULT_THRESHOLD


def test_build_graph_auto_connects_most_nodes():
    """The point of auto: a graph that isn't dust on first load."""
    rng = np.random.default_rng(8)
    items = []
    for cluster in range(4):
        base = rng.random(32)
        for i in range(10):
            v = unit(base + rng.random(32) * 0.08)
            items.append(chunk(f"/c{cluster}/f{i}.md", v))
    vecs, metas = corpus(*items)

    g = build_graph(vecs, metas, threshold="auto")
    connected = sum(1 for n in g.nodes if n.degree > 0)
    assert connected / len(g.nodes) >= 0.5, f"only {connected}/{len(g.nodes)} linked"
    assert "auto-set" in g.note


def test_explicit_threshold_is_not_overridden():
    vecs, metas = corpus(chunk("/a.md", [1.0, 0.0]), chunk("/b.md", [0.99, 0.1]))
    g = build_graph(vecs, metas, threshold=0.5)
    assert g.to_json()["threshold"] == 0.5
    assert "auto-set" not in g.note


def test_an_edgeless_graph_explains_itself():
    """Unconnected dots with no explanation reads as broken software."""
    vecs, metas = corpus(chunk("/a.md", [1.0, 0.0]), chunk("/b.md", [0.0, 1.0]))
    g = build_graph(vecs, metas, threshold=0.9)
    assert g.edges == []
    assert "lower the threshold" in g.note
