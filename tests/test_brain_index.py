"""Integration tests for the Atlas indexer and the graph built from it.

Real ChromaDB, real file walking, real chunking — only the embedding model is
faked. That combination is what makes these worth having: the exclusion rules
and the chunk→file→graph pipeline are exactly the parts where a plausible-
looking change silently does the wrong thing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
np = pytest.importorskip("numpy")

from core.graph import build_graph
from tools.brain import INDEXABLE_EXTS, SKIP_DIRS, Brain, _skip_dir


# ── Exclusion rules ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "node_modules", ".git", "__pycache__", "venv", ".venv",
    "venv.icloud.bak",          # the one that motivated pattern matching
    "my-virtualenv", "backups.bak", "stuff.old", "Photos.icloud",
    "Packages", "Pods", "vendor", "target", "DerivedData",
    "TextMesh Pro",             # the one currently polluting the index
    "Plugins", "ThirdParty", "External", "site-packages",
])
def test_vendored_and_junk_directories_are_pruned(name):
    assert _skip_dir(name) is True, f"{name!r} would be indexed"


@pytest.mark.parametrize("name", [
    "core", "tools", "Documents", "Desktop", "MyGame", "Assets",
    "src", "notes", "dissertation", "Coursework",
])
def test_real_work_directories_are_kept(name):
    """The exclusions must not be so eager they drop the user's own work.
    `Assets` in particular stays — it's where a Unity developer's own code
    lives; only the vendored subfolders under it are pruned."""
    assert _skip_dir(name) is False, f"{name!r} would be skipped"


def test_skip_is_case_insensitive_for_patterns():
    assert _skip_dir("VENV") is True
    assert _skip_dir("Node_Modules") is True


# ── Indexing ────────────────────────────────────────────────────────────────


@pytest.fixture
def brain(monkeypatch, tmp_path, fake_llm):
    """A Brain backed by a temp ChromaDB and the deterministic fake embedder."""
    import chromadb
    from chromadb.config import Settings

    import tools.brain as brain_mod

    monkeypatch.setattr(brain_mod, "BRAIN_DIR", tmp_path / "brain")
    (tmp_path / "brain").mkdir(parents=True, exist_ok=True)

    b = Brain.__new__(Brain)
    b.llm = fake_llm
    b.chroma = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"),
        settings=Settings(anonymized_telemetry=False))
    b._collections = {}
    b.status = {}
    return b


def make_tree(root: Path) -> Path:
    """A small corpus with two obvious themes and some vendored noise."""
    (root / "work").mkdir(parents=True)
    (root / "work" / "cv.md").write_text(
        "Abdullah Khan Durrani curriculum vitae. Software engineer. "
        "Experience building assistants and data pipelines.")
    (root / "work" / "coverletter.md").write_text(
        "Cover letter. Abdullah Khan Durrani, software engineer, applying with "
        "experience building assistants and data pipelines.")
    (root / "cooking").mkdir()
    (root / "cooking" / "curry.md").write_text(
        "Chicken karahi recipe. Tomatoes, ginger, green chilli, coriander. "
        "Cook on high heat.")
    (root / "cooking" / "biryani.md").write_text(
        "Chicken biryani recipe. Rice, tomatoes, ginger, coriander, saffron. "
        "Layer and steam.")

    # Vendored noise that must never be indexed.
    vendor = root / "game" / "Assets" / "TextMesh Pro"
    vendor.mkdir(parents=True)
    (vendor / "TMP_Text.cs").write_text("public class TMP_Text {}" * 40)
    node = root / "site" / "node_modules" / "left-pad"
    node.mkdir(parents=True)
    (node / "index.js").write_text("module.exports = function(){}" * 40)
    venv = root / "venv.icloud.bak" / "lib"
    venv.mkdir(parents=True)
    (venv / "thing.py").write_text("import os" * 40)
    return root


async def test_index_walks_real_files(brain, tmp_path):
    root = make_tree(tmp_path / "corpus")
    st = await brain.index_paths("test_files", [str(root)])
    assert st["indexed"] == 4, f"expected the 4 real files, got {st['indexed']}"


async def test_vendored_files_never_reach_the_index(brain, tmp_path):
    """The reason for the whole exclusion change: TextMesh Pro, node_modules
    and a stray backup venv were being embedded and diluting search."""
    root = make_tree(tmp_path / "corpus")
    await brain.index_paths("test_files", [str(root)])
    got = brain.vectors("test_files")
    paths = {m["path"] for m in got["metadatas"]}
    assert not any("TextMesh Pro" in p for p in paths)
    assert not any("node_modules" in p for p in paths)
    assert not any("venv" in p for p in paths)
    assert len(paths) == 4


async def test_reindex_is_incremental_by_default(brain, tmp_path):
    root = make_tree(tmp_path / "corpus")
    await brain.index_paths("test_files", [str(root)])
    second = await brain.index_paths("test_files", [str(root)])
    assert second["indexed"] == 0, "unchanged files were re-embedded"
    assert second["skipped"] == 4


async def test_rebuild_drops_previously_indexed_files(brain, tmp_path):
    """Incremental indexing only adds and updates. After a skip-list change,
    a rebuild is the only thing that removes what shouldn't be there."""
    root = make_tree(tmp_path / "corpus")
    await brain.index_paths("test_files", [str(root)])

    # Simulate the pre-fix state: something indexed that no longer qualifies.
    stray = root / "work" / "stray.md"
    stray.write_text("temporary note about nothing in particular")
    await brain.index_paths("test_files", [str(root)])
    assert len(brain.vectors("test_files")["metadatas"]) > 0
    before = {m["path"] for m in brain.vectors("test_files")["metadatas"]}
    assert str(stray) in before

    stray.unlink()
    rebuilt = await brain.index_paths("test_files", [str(root)], rebuild=True)
    after = {m["path"] for m in brain.vectors("test_files")["metadatas"]}
    assert str(stray) not in after
    assert rebuilt["indexed"] == 4


async def test_drop_clears_collection_and_manifest(brain, tmp_path):
    root = make_tree(tmp_path / "corpus")
    await brain.index_paths("test_files", [str(root)])
    assert brain._safe_count("test_files") > 0

    brain.drop("test_files")
    assert brain._safe_count("test_files") == 0
    assert brain._load_manifest("test_files") == {}


async def test_vectors_returns_embeddings_not_none(brain, tmp_path):
    """Chroma omits embeddings from get() unless explicitly included, and
    returns None rather than raising — an easy silent failure."""
    root = make_tree(tmp_path / "corpus")
    await brain.index_paths("test_files", [str(root)])
    got = brain.vectors("test_files")
    assert got["embeddings"] is not None
    assert len(got["embeddings"]) == len(got["metadatas"]) > 0
    assert len(got["embeddings"][0]) > 0


async def test_vectors_on_an_empty_source(brain):
    got = brain.vectors("nothing")
    assert got == {"embeddings": [], "metadatas": []}


# ── Index → graph ───────────────────────────────────────────────────────────


async def test_graph_from_a_real_index(brain, tmp_path):
    root = make_tree(tmp_path / "corpus")
    await brain.index_paths("test_files", [str(root)])
    got = brain.vectors("test_files")

    g = build_graph(got["embeddings"], got["metadatas"],
                    roots=[str(root)], threshold=0.0, max_neighbours=3)
    j = g.to_json()

    # One node per file, never one per chunk.
    assert j["stats"]["nodes"] == 4
    assert {n["label"] for n in j["nodes"]} == {
        "cv.md", "coverletter.md", "curry.md", "biryani.md"}


async def test_graph_groups_follow_folders(brain, tmp_path):
    root = make_tree(tmp_path / "corpus")
    await brain.index_paths("test_files", [str(root)])
    got = brain.vectors("test_files")
    g = build_graph(got["embeddings"], got["metadatas"], roots=[str(root)])

    groups = {n.label: n.group for n in g.nodes}
    assert groups["cv.md"] == groups["coverletter.md"]
    assert groups["curry.md"] == groups["biryani.md"]
    assert groups["cv.md"] != groups["curry.md"]


async def test_graph_json_survives_serialisation(brain, tmp_path):
    import json
    root = make_tree(tmp_path / "corpus")
    await brain.index_paths("test_files", [str(root)])
    got = brain.vectors("test_files")
    payload = build_graph(got["embeddings"], got["metadatas"]).to_json()
    assert json.loads(json.dumps(payload))["stats"]["nodes"] == 4


async def test_a_high_threshold_yields_an_all_isolated_graph(brain, tmp_path):
    """Turning the slider up must degrade to lone dots, not to an error — and
    the UI needs `isolated` to tell the user why the map looks empty."""
    root = make_tree(tmp_path / "corpus")
    await brain.index_paths("test_files", [str(root)])
    got = brain.vectors("test_files")
    g = build_graph(got["embeddings"], got["metadatas"], threshold=0.999)
    assert g.to_json()["stats"]["isolated"] == 4
    assert g.edges == []
