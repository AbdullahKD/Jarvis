"""
finex/chroma_store.py — Persistent PDF text store for FinEx

Stores uploaded financial report text so it survives server restarts.

Storage layers (first available wins):
  1. ChromaDB  — semantic chunk search (preferred, needs `chromadb` package)
  2. SQLite    — simple full-text store (always available, no extra deps)

Usage:
    from finex.chroma_store import pdf_store

    # Store after PDF extraction
    pdf_store.save(company="Bestway Cement", filename="annual_2025.pdf", text=raw_text)

    # Retrieve for LLM context
    text = pdf_store.load(company="Bestway Cement")

    # Semantic search (returns relevant chunks)
    chunks = pdf_store.search(company="Bestway Cement", query="Who is the CEO?", n=5)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

# ── Storage path ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent  # /Users/.../Jarvis
_DATA = _HERE / "data"
_DATA.mkdir(exist_ok=True)
_SQLITE_PATH = _DATA / "finex_pdfs.db"
_CHROMA_DIR  = _DATA / "chroma" / "finex"


# ── SQLite backend (always available) ─────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_SQLITE_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finex_pdfs (
            company  TEXT NOT NULL,
            filename TEXT NOT NULL DEFAULT '',
            text     TEXT NOT NULL,
            uploaded_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (company)
        )
    """)
    conn.commit()
    return conn


def _sqlite_save(company: str, filename: str, text: str):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO finex_pdfs (company, filename, text, uploaded_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(company) DO UPDATE SET
            filename   = excluded.filename,
            text       = excluded.text,
            uploaded_at = excluded.uploaded_at
    """, (company, filename, text))
    conn.commit()
    conn.close()


def _sqlite_load(company: str) -> str:
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT text FROM finex_pdfs WHERE company = ?", (company,)
        ).fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception:
        return ""


def _sqlite_list() -> List[dict]:
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT company, filename, uploaded_at FROM finex_pdfs ORDER BY uploaded_at DESC"
        ).fetchall()
        conn.close()
        return [{"company": r[0], "filename": r[1], "uploaded_at": r[2]} for r in rows]
    except Exception:
        return []


def _sqlite_delete(company: str):
    try:
        conn = _get_conn()
        conn.execute("DELETE FROM finex_pdfs WHERE company = ?", (company,))
        conn.commit()
        conn.close()
    except Exception:
        pass


# ── ChromaDB backend (semantic search, optional) ──────────────────────────────

_chroma_collection = None


def _get_chroma():
    """Return ChromaDB collection, or None if unavailable."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    try:
        import chromadb
        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(
            name="finex_documents",
            metadata={"hnsw:space": "cosine"},
        )
        return _chroma_collection
    except Exception:
        return None


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    if not text:
        return []
    # Try to split on paragraph boundaries first
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # If single paragraph is too long, hard-split it
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def _chroma_save(company: str, filename: str, text: str):
    col = _get_chroma()
    if col is None:
        return

    try:
        # Remove existing docs for this company
        existing = col.get(where={"company": company})
        if existing["ids"]:
            col.delete(ids=existing["ids"])

        chunks = _chunk_text(text)
        if not chunks:
            return

        ids       = [f"{company}__{i}" for i in range(len(chunks))]
        metadatas = [
            {"company": company, "filename": filename, "chunk_idx": i}
            for i in range(len(chunks))
        ]
        col.add(documents=chunks, metadatas=metadatas, ids=ids)
    except Exception:
        pass  # Silently fall back to SQLite


def _chroma_search(company: str, query: str, n: int = 5) -> List[str]:
    col = _get_chroma()
    if col is None:
        return []
    try:
        results = col.query(
            query_texts=[query],
            n_results=min(n, col.count()),
            where={"company": company},
        )
        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d]
    except Exception:
        return []


# ── Public façade ──────────────────────────────────────────────────────────────

class FinExPdfStore:
    """
    Persistent store for FinEx PDF texts.
    Writes to both SQLite (full text) and ChromaDB (chunked, semantic search).
    Reads from SQLite for full-text; ChromaDB for semantic queries.
    """

    def save(self, company: str, filename: str, text: str):
        """Persist PDF text. Overwrites previous entry for the same company."""
        _sqlite_save(company, filename, text)
        _chroma_save(company, filename, text)

    def load(self, company: str) -> str:
        """Return the full raw text of the last PDF uploaded for this company."""
        return _sqlite_load(company)

    def search(self, company: str, query: str, n: int = 5) -> List[str]:
        """
        Semantic search within the company's PDF.
        Returns up to n relevant text chunks.
        Falls back to a substring excerpt from SQLite if ChromaDB unavailable.
        """
        # Try ChromaDB first
        chunks = _chroma_search(company, query, n)
        if chunks:
            return chunks

        # Fallback: return first 8000 chars as a single block
        text = _sqlite_load(company)
        if not text:
            return []
        return [text[:8000]]

    def list_docs(self) -> List[dict]:
        """List all stored PDFs with company, filename, and upload timestamp."""
        return _sqlite_list()

    def delete(self, company: str):
        """Remove a company's PDF from both stores."""
        _sqlite_delete(company)
        col = _get_chroma()
        if col:
            try:
                existing = col.get(where={"company": company})
                if existing["ids"]:
                    col.delete(ids=existing["ids"])
            except Exception:
                pass

    def has_pdf(self, company: str) -> bool:
        """Return True if a PDF has been stored for this company."""
        return bool(self.load(company))


# Singleton
pdf_store = FinExPdfStore()
