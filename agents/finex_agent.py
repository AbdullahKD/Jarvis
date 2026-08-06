"""
FinEx Agent — Financial Statement Expert Sub-Agent
Wraps the HBL Financial Analysis chatbot (finex/) as an async agent
integrated into the Jarvis FastAPI server.

The underlying engine (LLM_SQL.py) uses synchronous subprocess Ollama calls,
so every call is dispatched to a thread executor to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional

# Thread pool — FinEx LLM calls are blocking (HTTP, but still synchronous).
# Bumped from 2 → 8 (configurable via env) so a slow Ollama call doesn't freeze
# the whole FinEx capacity. Per-call timeout is applied via asyncio.wait_for.
_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get("FINEX_WORKERS", "8")),
    thread_name_prefix="finex",
)

# Per-question wall-clock budget. Ollama itself has a 120s HTTP timeout below,
# so this is a soft cap to free the event loop / give the user a clean error.
_CHAT_TIMEOUT_S = float(os.environ.get("FINEX_CHAT_TIMEOUT_S", "100"))


def _import_engine():
    """Lazy import so missing psycopg2 doesn't break Jarvis startup."""
    from finex.LLM_SQL import answer, store_pdf_text, invalidate_cache, LEVEL_LABELS
    return answer, store_pdf_text, invalidate_cache, LEVEL_LABELS


def _import_db():
    from finex.db_query import run_query, get_all_years
    from finex.db_insert import insert_financials
    return run_query, get_all_years, insert_financials


def _import_extract():
    from finex.extract_pdf import extract_text_only, extract_financials_intelligent
    return extract_text_only, extract_financials_intelligent


class FinExAgent:
    """
    Async wrapper around the FinEx financial statement Q&A engine.

    Capabilities:
    - Upload a PDF financial statement → extract + store in Postgres
    - Ask questions at 6 levels of sophistication (L1 retrieval → L6 strategic)
    - List available companies and periods
    """

    def __init__(self):
        self._ready = False
        self._error: Optional[str] = None
        try:
            _import_engine()
            self._ready = True
            print("💹 FinExAgent ready — financial statement Q&A enabled")
            # One-time schema migration up-front (cheap, idempotent) so it's not
            # repeated on every insert.
            try:
                _, _, insert_fn = _import_db()
                from finex.db_insert import create_schema
                create_schema()
            except Exception as exc:
                print(f"💹 FinEx schema init warning (non-fatal): {exc}")
            # Best-effort: pre-warm Ollama in a background thread so first
            # question doesn't pay cold-start cost.
            try:
                import threading
                from finex.LLM_SQL import warm_model
                threading.Thread(target=warm_model, name="finex-warm", daemon=True).start()
            except Exception:
                pass
        except ImportError as exc:
            self._error = str(exc)
            print(f"💹 FinExAgent unavailable: {exc} (run: pip install psycopg2-binary)")

    # ── Public async API ────────────────────────────────────────────────────

    async def chat(
        self,
        question: str,
        company: str = "Bestway Cement",
        history: Optional[List[Dict[str, str]]] = None,
        level=None,
    ) -> Dict[str, Any]:
        """
        Answer a financial question about a company.

        ``level`` is an optional explicit override (1–6, or "auto"/None for the
        deterministic router). Returns dict with keys:
        answer, level, level_label, question
        """
        if not self._ready:
            return {
                "answer": f"FinEx is not available: {self._error}",
                "level": 0,
                "level_label": "Error",
                "question": question,
            }

        # Build history string from last 6 messages, capped at ~1500 chars
        # so it never starves the system prompt out of the model's context window.
        _HIST_BUDGET = 1500
        history_str = ""
        if history:
            for msg in history[-6:]:
                role = "User" if msg.get("role") == "user" else "Analyst"
                history_str += f"{role}: {msg.get('content', '')}\n\n"
            if len(history_str) > _HIST_BUDGET:
                # Drop oldest turns first; keep the tail
                history_str = "…\n" + history_str[-_HIST_BUDGET:]

        loop = asyncio.get_event_loop()
        answer_fn, _, _, LEVEL_LABELS = _import_engine()

        def _run():
            resp, lvl, label = answer_fn(question, company, history_str, level)
            return resp, lvl, label

        try:
            resp_text, level, label = await asyncio.wait_for(
                loop.run_in_executor(_executor, _run),
                timeout=_CHAT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return {
                "answer": (
                    f"FinEx timed out after {_CHAT_TIMEOUT_S:.0f}s. "
                    "The model may be loading or under load — try again in a moment."
                ),
                "level": 0,
                "level_label": "Timeout",
                "question": question,
            }
        except Exception as exc:
            return {
                "answer": f"FinEx error: {exc}",
                "level": 0,
                "level_label": "Error",
                "question": question,
            }

        return {
            "answer": resp_text,
            "level": level,
            "level_label": label,
            "question": question,
        }

    async def upload_pdf(
        self, pdf_path: str, company: str = "Bestway Cement"
    ) -> Dict[str, Any]:
        """
        Extract financials from a PDF and store in the database.
        Returns extraction summary including validation results.
        """
        if not self._ready:
            return {"success": False, "error": self._error}

        import re
        loop = asyncio.get_event_loop()
        _, extract_fn = _import_extract()
        _, _, insert_fn = _import_db()
        _, store_pdf_text, invalidate_cache, _ = _import_engine()

        def _run():
            try:
                result = extract_fn(pdf_path)
            except Exception as exc:
                import traceback
                return {"success": False, "error": f"PDF extraction failed: {exc}", "traceback": traceback.format_exc()}

            current = result.get("current", {})
            prior = result.get("prior", {})
            meta = result.get("metadata", {})
            raw_text = result.get("raw_text", "")

            if not current:
                return {"success": False, "error": "No financial data could be extracted from this PDF. Ensure it contains a P&L, Balance Sheet, or Cash Flow statement."}

            import os as _os
            try:
                store_pdf_text(company, raw_text, filename=_os.path.basename(pdf_path))
                invalidate_cache(company)
            except Exception as exc:
                pass  # non-fatal — data still inserted into Postgres

            period_str = meta.get("period_current") or "31 December 2025"
            year_match = re.search(r"\d{4}", period_str)
            year = int(year_match.group()) if year_match else 2025
            period_label = f"FY{year}"

            # Currency / unit metadata from extractor
            currency   = meta.get("currency",   "Unknown")
            unit_label = meta.get("unit_label",  "millions (assumed)")

            try:
                insert_fn(
                    current,
                    company=company,
                    year=year,
                    period=period_label,
                    currency=currency,
                    unit_label=unit_label,
                )
            except Exception as exc:
                import traceback
                return {"success": False, "error": f"Database insert failed: {exc}", "traceback": traceback.format_exc()}

            prior_inserted = False
            if prior:
                try:
                    prior_str = meta.get("period_prior") or str(year - 1)
                    pm = re.search(r"\d{4}", prior_str)
                    prior_year = int(pm.group()) if pm else year - 1
                    insert_fn(
                        prior,
                        company=company,
                        year=prior_year,
                        period=f"FY{prior_year}",
                        currency=currency,
                        unit_label=unit_label,
                    )
                    prior_inserted = True
                except Exception:
                    pass  # prior year insert failing is non-fatal

            validation = meta.get("validation", {})
            return {
                "success": True,
                "company": company,
                "period": period_label,
                "fields_extracted": len(current),
                "prior_year_extracted": prior_inserted,
                "validation": {
                    "checks_passed": len(validation.get("passed", [])),
                    "checks_failed": len(validation.get("failed", [])),
                    "warnings": validation.get("warnings", []),
                },
            }

        try:
            return await loop.run_in_executor(_executor, _run)
        except Exception as exc:
            import traceback
            return {"success": False, "error": f"Unexpected error: {exc}", "traceback": traceback.format_exc()}

    async def list_companies(self) -> Dict[str, Any]:
        """Return all companies and periods stored in the database."""
        if not self._ready:
            return {"success": False, "companies": [], "error": self._error}

        loop = asyncio.get_event_loop()
        run_query, _, _ = _import_db()

        def _run():
            result = run_query(
                "SELECT DISTINCT company, year, period FROM financials ORDER BY company, year DESC"
            )
            if "error" in result:
                return {"success": False, "companies": [], "error": result["error"]}
            return {
                "success": True,
                "companies": [
                    {"company": r[0], "year": r[1], "period": r[2]}
                    for r in result["rows"]
                ],
            }

        return await loop.run_in_executor(_executor, _run)
