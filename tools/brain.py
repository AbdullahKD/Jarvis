"""
Jarvis Brain — shared local knowledge index.

A thin, reusable layer over JARVIS's existing ChromaDB + Ollama embeddings that
powers Atlas (files), Codex (Obsidian notes) and Recall (activity). Everything
stays local: vectors live in the same CHROMA_DIR the memory agent already uses,
and nothing leaves the machine.

Design notes
------------
* One Chroma collection per source ("atlas_files", "codex_notes",
  "recall_activity") so a re-index of one source never touches another.
* Embeddings go through the same OllamaClient the rest of Jarvis uses; if the
  embed model is unavailable it falls back to a deterministic hash vector, and
  because Chroma also stores the raw document text we can always fall back to a
  keyword ($contains) search. Search therefore degrades gracefully instead of
  failing.
* Indexing is incremental: a JSON manifest maps path -> mtime so unchanged files
  are skipped on re-index, and stale chunks are deleted before re-adding.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config.llm_client import OllamaClient
from config.settings import CHROMA_DIR, DATA_DIR

# ── Config ───────────────────────────────────────────────────────────────────
BRAIN_DIR = Path(DATA_DIR) / "brain"
try:
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

MAX_FILE_BYTES = 6 * 1024 * 1024          # skip files larger than 6 MB
MAX_TEXT_CHARS = 24_000                    # cap extracted text per file
CHUNK_CHARS = 1_200
CHUNK_OVERLAP = 150
MAX_CHUNKS_PER_FILE = 12
EMBED_CONCURRENCY = 4

# Extensions we can extract text from.
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".htm", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".sh", ".bash",
    ".zsh", ".sql", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs",
    ".rb", ".php", ".swift", ".kt", ".r", ".m", ".pl", ".lua", ".vue", ".svelte",
}
DOC_EXTS = {".pdf", ".docx"}
INDEXABLE_EXTS = TEXT_EXTS | DOC_EXTS

# Directories we never descend into.
#
# Exact names only — see SKIP_PATTERNS below for the ones an exact list can't
# catch.
SKIP_DIRS = {
    "node_modules", ".git", ".svn", "__pycache__", ".venv", "venv",
    ".cache", ".Trash", "Library", ".npm", ".gradle",
    ".idea", ".vscode", "dist", "build", ".next", ".DS_Store", "site-packages",
    ".pytest_cache", ".mypy_cache", "chroma", "brain",
    # Third-party code you didn't write. Every one of these files gets
    # embedded and then competes with your own work in search results — a
    # single Unity project contributes thousands of .cs files, and .cs is in
    # TEXT_EXTS.
    "Packages",            # Unity package manager
    "Pods",                # CocoaPods
    "vendor", "vendored",  # Go / PHP / Ruby
    "target",              # Rust / Maven
    "DerivedData",         # Xcode
    "TextMesh Pro",        # the one currently in the index
    "Plugins", "ThirdParty", "External",
    "bower_components", ".terraform",
}

# Substring and suffix matches, for directory names an exact list can't catch.
# `venv.icloud.bak` is the case that motivated this: 15,268 Python files in the
# repo, matching neither "venv" nor ".bak" exactly.
SKIP_PATTERNS = ("venv", "virtualenv", "node_modules", ".egg-info",
                 "site-packages")
SKIP_SUFFIXES = (".bak", ".old", ".orig", ".backup", ".icloud")


def _skip_dir(name: str) -> bool:
    """Whether to prune a directory during the walk."""
    if name in SKIP_DIRS or name.startswith("."):
        return True
    low = name.lower()
    return (any(pat in low for pat in SKIP_PATTERNS)
            or low.endswith(SKIP_SUFFIXES))


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def _chunks(text: str) -> List[str]:
    text = text[:MAX_TEXT_CHARS]
    out: List[str] = []
    i = 0
    while i < len(text) and len(out) < MAX_CHUNKS_PER_FILE:
        out.append(text[i:i + CHUNK_CHARS])
        i += CHUNK_CHARS - CHUNK_OVERLAP
    return [c for c in out if c.strip()]


def extract_text(path: Path) -> str:
    """Best-effort text extraction. Returns '' on anything unreadable."""
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_EXTS:
            return path.read_text(errors="ignore")[:MAX_TEXT_CHARS]
        if suffix == ".pdf":
            import pdfplumber
            parts = []
            with pdfplumber.open(str(path)) as pdf:
                for pg in pdf.pages[:30]:
                    parts.append(pg.extract_text() or "")
            return "\n".join(parts)[:MAX_TEXT_CHARS]
        if suffix == ".docx":
            import docx
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs)[:MAX_TEXT_CHARS]
    except Exception:
        return ""
    return ""


class Brain:
    """Singleton-ish holder for the Chroma client and per-source collections."""

    _instance: Optional["Brain"] = None

    def __init__(self) -> None:
        self.llm = OllamaClient()
        self.chroma = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: Dict[str, Any] = {}
        # per-source indexing status, surfaced to the UI
        self.status: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get(cls) -> "Brain":
        if cls._instance is None:
            cls._instance = Brain()
        return cls._instance

    def collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = self.chroma.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    # ── manifest (incremental index bookkeeping) ─────────────────────────────
    def _manifest_path(self, source: str) -> Path:
        return BRAIN_DIR / f"{source}_manifest.json"

    def _load_manifest(self, source: str) -> Dict[str, Any]:
        try:
            return json.loads(self._manifest_path(source).read_text())
        except Exception:
            return {}

    def _save_manifest(self, source: str, data: Dict[str, Any]) -> None:
        try:
            self._manifest_path(source).write_text(json.dumps(data))
        except Exception as exc:  # noqa: BLE001
            print(f"Brain: manifest save failed ({source}): {exc}")

    # ── embedding ────────────────────────────────────────────────────────────
    async def _embed(self, text: str) -> List[float]:
        try:
            emb = await asyncio.wait_for(self.llm.embed(text), timeout=12.0)
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            return [float(x) for x in emb]
        except Exception:
            return self.llm._hash_embed(text)

    # ── indexing ─────────────────────────────────────────────────────────────
    async def index_paths(self, source: str, roots: List[str],
                          exts: Optional[set] = None,
                          rebuild: bool = False) -> Dict[str, Any]:
        """Walk `roots`, (re)index changed files into `source`. Idempotent.

        With ``rebuild=True`` the collection and manifest are dropped first.
        Needed after the skip list changes: incremental indexing only ever adds
        and updates, so files that were embedded before an exclusion existed
        stay in the collection forever and keep polluting search results.
        """
        exts = exts or INDEXABLE_EXTS
        if rebuild:
            self.drop(source)
        col = self.collection(source)
        manifest = {} if rebuild else self._load_manifest(source)
        st = {
            "running": True, "indexed": 0, "skipped": 0, "errors": 0,
            "seen": 0, "removed": 0, "current": "", "roots": roots,
            "started": time.time(), "finished": None,
        }
        self.status[source] = st
        sem = asyncio.Semaphore(EMBED_CONCURRENCY)
        seen_paths = set()

        async def _do_file(fp: Path) -> None:
            async with sem:
                try:
                    stat = fp.stat()
                    key = str(fp)
                    seen_paths.add(key)
                    st["seen"] += 1
                    st["current"] = fp.name
                    prev = manifest.get(key)
                    if prev and abs(prev.get("mtime", 0) - stat.st_mtime) < 1:
                        st["skipped"] += 1
                        return
                    text = extract_text(fp)
                    if not text.strip():
                        st["skipped"] += 1
                        manifest[key] = {"mtime": stat.st_mtime, "ids": prev.get("ids", []) if prev else []}
                        return
                    # drop old chunks for this file first
                    if prev and prev.get("ids"):
                        try:
                            col.delete(ids=prev["ids"])
                        except Exception:
                            pass
                    fid = _sha(key)
                    chunks = _chunks(text)
                    ids, docs, metas, embs = [], [], [], []
                    for i, ch in enumerate(chunks):
                        ids.append(f"{fid}_{i}")
                        docs.append(ch)
                        metas.append({
                            "path": key, "name": fp.name, "ext": fp.suffix.lower(),
                            "dir": str(fp.parent), "mtime": stat.st_mtime,
                            "size": stat.st_size, "source": source, "chunk": i,
                        })
                        embs.append(await self._embed(ch))
                    if ids:
                        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                        manifest[key] = {"mtime": stat.st_mtime, "ids": ids}
                        st["indexed"] += 1
                except Exception as exc:  # noqa: BLE001
                    st["errors"] += 1
                    print(f"Brain: index error {fp}: {exc}")

        tasks = []
        for root in roots:
            rp = Path(os.path.expanduser(root))
            if not rp.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(rp):
                dirnames[:] = [d for d in dirnames if not _skip_dir(d)]
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    if fp.suffix.lower() not in exts:
                        continue
                    try:
                        if fp.stat().st_size > MAX_FILE_BYTES:
                            continue
                    except Exception:
                        continue
                    tasks.append(_do_file(fp))
                    if len(tasks) >= 40:
                        await asyncio.gather(*tasks)
                        tasks = []
        if tasks:
            await asyncio.gather(*tasks)

        # prune manifest entries for files that vanished
        for gone in [k for k in manifest if k not in seen_paths]:
            try:
                if manifest[gone].get("ids"):
                    col.delete(ids=manifest[gone]["ids"])
                    st["removed"] += 1
            except Exception:
                pass
            manifest.pop(gone, None)

        self._save_manifest(source, manifest)
        st["running"] = False
        st["finished"] = time.time()
        st["count"] = self._safe_count(source)
        return st

    def drop(self, source: str) -> None:
        """Delete a source's collection and manifest, so the next index starts
        from nothing."""
        try:
            self.chroma.delete_collection(source)
        except Exception as exc:  # noqa: BLE001
            print(f"Brain: could not drop collection {source}: {exc}")
        self._collections.pop(source, None)
        try:
            path = self._manifest_path(source)
            if path.exists():
                path.unlink()
        except Exception as exc:  # noqa: BLE001
            print(f"Brain: could not remove manifest for {source}: {exc}")

    def _safe_count(self, source: str) -> int:
        try:
            return self.collection(source).count()
        except Exception:
            return 0

    def vectors(self, source: str) -> Dict[str, Any]:
        """Every stored embedding and its metadata, for graph building.

        Chroma omits embeddings from get() unless asked, which is easy to miss
        and returns None rather than raising.
        """
        col = self.collection(source)
        if col.count() == 0:
            return {"embeddings": [], "metadatas": []}
        got = col.get(include=["embeddings", "metadatas"])
        # `x or []` would be wrong here: Chroma returns embeddings as a NumPy
        # array, and bool() on a multi-element array raises ValueError rather
        # than being falsy. Explicit None checks only.
        embeddings = got.get("embeddings")
        metadatas = got.get("metadatas")
        return {
            "embeddings": [] if embeddings is None else list(embeddings),
            "metadatas": [] if metadatas is None else list(metadatas),
        }

    # ── search ───────────────────────────────────────────────────────────────
    async def search(self, source: str, query: str, k: int = 12) -> List[Dict[str, Any]]:
        col = self.collection(source)
        if col.count() == 0 or not query.strip():
            return []
        results: List[Dict[str, Any]] = []
        seen_paths = set()
        try:
            emb = await self._embed(query)
            res = col.query(query_embeddings=[emb], n_results=min(k * 2, 40),
                            include=["documents", "metadatas", "distances"])
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                p = meta.get("path", "")
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                results.append({
                    "path": p, "name": meta.get("name"), "ext": meta.get("ext"),
                    "dir": meta.get("dir"), "mtime": meta.get("mtime"),
                    "source": meta.get("source"), "score": round(1 - float(dist), 3),
                    "snippet": (doc or "").strip().replace("\n", " ")[:260],
                })
                if len(results) >= k:
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"Brain: semantic search failed, keyword fallback: {exc}")
        # keyword fallback / supplement
        if len(results) < k:
            try:
                res = col.query(query_texts=[query], n_results=k,
                                where_document={"$contains": query.split()[0]},
                                include=["documents", "metadatas"])
                for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                    p = meta.get("path", "")
                    if p in seen_paths:
                        continue
                    seen_paths.add(p)
                    results.append({
                        "path": p, "name": meta.get("name"), "ext": meta.get("ext"),
                        "dir": meta.get("dir"), "mtime": meta.get("mtime"),
                        "source": meta.get("source"), "score": None,
                        "snippet": (doc or "").strip().replace("\n", " ")[:260],
                    })
            except Exception:
                pass
        return results[:k]


def get_brain() -> Brain:
    return Brain.get()
