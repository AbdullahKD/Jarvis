import os
import threading
import psycopg2
from psycopg2 import pool as _pg_pool
from finex.db_insert import get_connection

# ── Connection pool ────────────────────────────────────────────────────────────
# Borrow/return connections instead of opening a fresh socket per query.
# A ThreadedConnectionPool is safe for our ThreadPoolExecutor-driven workload.
_POOL = None
_POOL_LOCK = threading.Lock()
_POOL_MIN  = int(os.environ.get("FINEX_PG_POOL_MIN", "1"))
_POOL_MAX  = int(os.environ.get("FINEX_PG_POOL_MAX", "8"))


def _build_pool():
    """Construct the pool lazily against the same local Postgres instance as
    db_insert.get_connection — keep the two in sync."""
    return _pg_pool.ThreadedConnectionPool(
        minconn=_POOL_MIN,
        maxconn=_POOL_MAX,
        dbname="finance_db",
        user="akd",
        host="localhost",
    )


def _get_pool():
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = _build_pool()
    return _POOL


class _PooledConn:
    """Context manager: borrow a conn from the pool, set autocommit, return on exit."""
    def __init__(self):
        self.conn = None

    def __enter__(self):
        self.conn = _get_pool().getconn()
        try:
            self.conn.set_session(autocommit=True)
        except Exception:
            pass
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if self.conn is None:
            return
        try:
            # If the connection is broken, drop it from the pool entirely
            broken = exc_type is not None and isinstance(exc, psycopg2.Error)
            _get_pool().putconn(self.conn, close=broken)
        except Exception:
            try:
                self.conn.close()
            except Exception:
                pass


MONETARY_COLS = {
    "revenue", "gross_profit", "operating_profit", "profit_before_tax",
    "net_profit", "cost_of_goods_sold", "operating_expenses", "depreciation",
    "finance_cost", "tax_expense", "total_assets", "current_assets",
    "non_current_assets", "cash_balance", "trade_receivables", "inventory",
    "total_liabilities", "current_liabilities", "non_current_liabilities",
    "total_equity", "share_capital", "long_term_debt", "gross_turnover",
    "operating_cashflow", "investing_cashflow", "financing_cashflow",
}

# Per-share figures are NOT bulk monetary amounts and must never be scaled.
# dividend_per_share used to sit in MONETARY_COLS above, so a £1.41 dividend
# was rendered "£1.41k" — and because every long-form level (L3-L6 and DETAIL)
# reads its context from here while L1 formats through LLM_SQL._fmt_value,
# which has always treated per-share fields correctly, the same company could
# report two different dividends depending on which level answered.
# LLM_SQL._NO_SCALE_FIELDS is the matching set; keep the two in step.
PER_SHARE_COLS = {"eps", "dividend_per_share"}

_SKIP_COLS = {"id"}


def get_conn():
    """Back-compat shim: return a pooled connection. Caller must NOT close()."""
    return _get_pool().getconn()


def _fresh_conn():
    """Back-compat shim. Prefer _PooledConn() for new code."""
    c = _get_pool().getconn()
    try:
        c.set_session(autocommit=True)
    except Exception:
        pass
    return c


def run_query(sql: str, params=None):
    try:
        with _PooledConn() as conn:
            cur = conn.cursor()
            try:
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)
                colnames = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall() if cur.description else []
                return {"columns": colnames, "rows": rows}
            finally:
                cur.close()
    except Exception as e:
        return {"error": str(e)}


def get_all_years(company: str = "Bestway Cement") -> list:
    try:
        with _PooledConn() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT DISTINCT year FROM financials WHERE company = %s ORDER BY year DESC",
                    (company,)
                )
                rows = cur.fetchall()
                return [r[0] for r in rows]
            finally:
                cur.close()
    except Exception:
        return []


def get_financial_context(company: str = "Bestway Cement", years: list = None) -> str:
    """
    Returns structured financial data for LLM prompts.
    Values are stored as absolute currency amounts (raw PDF number × scale_factor).
    """
    if years is None:
        years = get_all_years(company)

    if not years:
        return "No financial data found in database."

    try:
        with _PooledConn() as conn:
            cur = conn.cursor()
            try:
                placeholders = ", ".join(["%s"] * len(years))
                cur.execute(
                    f"SELECT * FROM financials WHERE company = %s AND year IN ({placeholders}) ORDER BY year DESC, period",
                    [company] + list(years)
                )
                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
            finally:
                cur.close()
    except Exception as e:
        return f"DB error: {e}"

    if not rows:
        return f"No financial data found for {company}."

    sym_map = {"GBP": "£", "USD": "$", "EUR": "€", "PKR": "PKR",
               "AED": "AED", "SAR": "SAR"}

    lines = [f"=== Financial Data: {company} ===\n"]

    for row in rows:
        record = dict(zip(colnames, row))
        currency   = record.get("currency")  or "Unknown"
        unit_label = record.get("unit_label") or ""
        sym = sym_map.get(currency, currency if currency != "Unknown" else "")

        lines.append(f"--- Period: {record.get('period', record['year'])} ---")
        lines.append(f"  [Source unit: {unit_label or 'unknown'}  |  Stored as: absolute {currency}]")

        for col in colnames:
            if col in _SKIP_COLS or col in ("currency", "unit_label"):
                continue
            val = record.get(col)
            if val is None:
                continue
            if isinstance(val, float):
                if col in PER_SHARE_COLS:
                    # Explicitly labelled: unlabelled, the model has no way to
                    # tell a per-share figure from a bulk amount.
                    lines.append(f"  {col}: {sym}{val:,.2f} per share")
                elif col in MONETARY_COLS:
                    if abs(val) >= 1_000_000_000:
                        human = f"({sym}{val/1_000_000_000:,.2f}bn)"
                    elif abs(val) >= 1_000_000:
                        human = f"({sym}{val/1_000_000:,.2f}m)"
                    elif abs(val) >= 1_000:
                        human = f"({sym}{val/1_000:,.2f}k)"
                    else:
                        human = ""
                    lines.append(f"  {col}: {sym}{val:,.2f} {human}")
                else:
                    lines.append(f"  {col}: {val:,.4f}")
            else:
                lines.append(f"  {col}: {val}")

        lines.append("")

    return "\n".join(lines)


def format_result(result: dict) -> str:
    if "error" in result:
        return f"Error: {result['error']}"
    cols = result["columns"]
    rows = result["rows"]
    if not rows:
        return "No results found."
    lines = [" | ".join(cols)]
    lines.append("-" * 60)
    for row in rows:
        lines.append(" | ".join(str(v) if v is not None else "N/A" for v in row))
    return "\n".join(lines)
