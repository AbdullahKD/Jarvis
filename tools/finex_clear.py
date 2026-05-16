#!/usr/bin/env python3
"""
finex_clear.py — Safely clear FinEx data.

Clears one or all of the three stores FinEx writes to:
  1. Postgres `financials` table       (structured row-per-period)
  2. SQLite  `data/finex_pdfs.db`      (raw PDF text)
  3. Chroma  `data/chroma/finex/`      (semantic chunks)

Plus invalidates any in-process caches if you're running this inside the server
shell (pointless from a one-shot CLI but harmless).

USAGE
─────
  # Wipe EVERYTHING (asks for confirmation)
  python3 -m tools.finex_clear --all

  # Wipe a single company
  python3 -m tools.finex_clear --company "Tesco 25'"

  # See what's there without deleting
  python3 -m tools.finex_clear --list

  # Skip the confirmation prompt (for scripting)
  python3 -m tools.finex_clear --all --yes
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Make sure project root is importable when run as a script from anywhere
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SQLITE_PATH = ROOT / "data" / "finex_pdfs.db"
CHROMA_DIR  = ROOT / "data" / "chroma" / "finex"


# ── helpers ──────────────────────────────────────────────────────────────

def _confirm(prompt: str, force: bool) -> bool:
    if force:
        return True
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


def _connect_pg():
    """Returns an autocommit psycopg2 connection or None on failure."""
    try:
        import psycopg2
        conn = psycopg2.connect(dbname="finance_db", user="akd", host="localhost")
        conn.set_session(autocommit=True)
        return conn
    except Exception as exc:
        print(f"  ⚠ Postgres unreachable: {exc}")
        return None


# ── listing ──────────────────────────────────────────────────────────────

def list_state() -> None:
    print(" Current FinEx state")
    print(" ────────────────────")

    # Postgres
    conn = _connect_pg()
    if conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT company, COUNT(*) AS rows, MIN(year), MAX(year) "
                "FROM financials GROUP BY company ORDER BY company"
            )
            rows = cur.fetchall()
            if not rows:
                print(" Postgres: (empty)")
            else:
                print(f" Postgres: {sum(r[1] for r in rows)} rows across {len(rows)} compan{'y' if len(rows)==1 else 'ies'}")
                for company, n, yfrom, yto in rows:
                    span = f"{yfrom}" if yfrom == yto else f"{yfrom}–{yto}"
                    print(f"    • {company:30s}  {n} row(s)  ({span})")
        finally:
            cur.close()
            conn.close()

    # SQLite
    if SQLITE_PATH.exists():
        try:
            sc = sqlite3.connect(str(SQLITE_PATH))
            rows = sc.execute(
                "SELECT company, length(text) FROM finex_pdfs ORDER BY company"
            ).fetchall()
            sc.close()
            if rows:
                print(f" SQLite PDF text store ({SQLITE_PATH.name}): {len(rows)} entries")
                for company, n in rows:
                    print(f"    • {company:30s}  {n:>10,} chars")
            else:
                print(" SQLite: (empty)")
        except Exception as exc:
            print(f"  ⚠ SQLite read failed: {exc}")
    else:
        print(" SQLite: (file missing)")

    # Chroma
    if CHROMA_DIR.exists():
        size_mb = sum(p.stat().st_size for p in CHROMA_DIR.rglob("*") if p.is_file()) / 1e6
        n_files = sum(1 for _ in CHROMA_DIR.rglob("*") if _.is_file())
        print(f" Chroma:  {n_files} files, {size_mb:.1f} MB at {CHROMA_DIR}")
    else:
        print(" Chroma:  (directory missing)")


# ── delete-all ───────────────────────────────────────────────────────────

def clear_all(force: bool = False) -> None:
    if not _confirm("This will delete ALL FinEx data (Postgres + SQLite + Chroma). Continue?", force):
        print(" Aborted.")
        return

    # Postgres
    conn = _connect_pg()
    if conn:
        cur = conn.cursor()
        try:
            cur.execute("TRUNCATE TABLE financials RESTART IDENTITY")
            print(" ✓ Postgres: TRUNCATE financials")
        except Exception as exc:
            print(f" ✗ Postgres truncate failed: {exc}")
        finally:
            cur.close()
            conn.close()

    # SQLite
    if SQLITE_PATH.exists():
        try:
            SQLITE_PATH.unlink()
            print(f" ✓ SQLite : removed {SQLITE_PATH.name}")
        except Exception as exc:
            print(f" ✗ SQLite removal failed: {exc}")
    else:
        print(" • SQLite : (already absent)")

    # Chroma
    if CHROMA_DIR.exists():
        try:
            shutil.rmtree(CHROMA_DIR)
            print(f" ✓ Chroma : removed {CHROMA_DIR}")
        except Exception as exc:
            print(f" ✗ Chroma removal failed: {exc}")
    else:
        print(" • Chroma : (already absent)")

    # In-process caches (only relevant if called from inside the server)
    try:
        import finex.LLM_SQL as ls
        for cache in ("_context_cache", "_hr_context_cache", "_META_CACHE", "_AVAILABLE_CACHE"):
            d = getattr(ls, cache, None)
            if isinstance(d, dict):
                d.clear()
        print(" ✓ Cleared in-process FinEx caches (best-effort)")
    except Exception:
        pass

    print("\n Done. Restart the FastAPI server to drop any other in-memory state.")


# ── delete-single-company ────────────────────────────────────────────────

def clear_company(name: str, force: bool = False) -> None:
    if not name.strip():
        print(" ✗ Empty company name; nothing to do.")
        return
    if not _confirm(f"Delete ALL FinEx data for company '{name}'?", force):
        print(" Aborted.")
        return

    # Postgres — parameterised, so apostrophes are safe
    conn = _connect_pg()
    if conn:
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM financials WHERE company = %s", (name,))
            print(f" ✓ Postgres: deleted {cur.rowcount} row(s) for company={name!r}")
        except Exception as exc:
            print(f" ✗ Postgres delete failed: {exc}")
        finally:
            cur.close()
            conn.close()

    # SQLite + Chroma — use the existing pdf_store helper (handles both)
    try:
        from finex.chroma_store import pdf_store
        pdf_store.delete(name)
        print(" ✓ SQLite + Chroma: deleted entries for that company")
    except Exception as exc:
        print(f" ✗ chroma_store.delete failed: {exc}")

    # Caches
    try:
        import finex.LLM_SQL as ls
        if hasattr(ls, "invalidate_cache"):
            ls.invalidate_cache(name)
            print(" ✓ Invalidated in-process caches for that company")
    except Exception:
        pass

    print("\n Done. The server will reflect this on the next request.")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="finex-clear",
        description="Clear FinEx data (Postgres + SQLite + Chroma).",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Wipe all FinEx data")
    g.add_argument("--company", metavar="NAME", help="Wipe one company by name")
    g.add_argument("--list", action="store_true", help="Just show what's currently stored")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = ap.parse_args()

    if args.list:
        list_state()
    elif args.all:
        clear_all(force=args.yes)
    elif args.company:
        clear_company(args.company, force=args.yes)


if __name__ == "__main__":
    main()
